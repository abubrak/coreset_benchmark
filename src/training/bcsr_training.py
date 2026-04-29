"""
BCSR (Bilevel Coreset Selection with Reweighting) 训练模块

实现基于双层优化的coreset选择训练方法，使用Neumann系列近似进行隐式微分。
- 内层优化：加权SGD训练代理模型
- 外层优化：使用平滑top-K正则化更新样本权重
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Callable


class BCSRTraining:
    """
    BCSR训练类 - 实现双层优化框架用于coreset选择

    算法核心：
    1. 内层：在加权数据上训练代理模型
    2. 外层：使用Neumann系列近似计算雅可比矩阵，更新样本权重
    3. 使用平滑top-K正则化促进稀疏权重分布
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 5.0,
        learning_rate_inner: float = None,
        learning_rate_outer: float = None,
        inner_epochs: int = 1,
        num_inner_steps: int = None,
        outer_steps: int = 5,
        num_outer_steps: int = None,
        beta: float = 0.1,
        lmbda: float = 0.0,
        device: str = 'cpu',
        kernel_fn: Optional[Callable] = None
    ):
        """
        初始化BCSR训练器

        参数:
            model: PyTorch模型（代理模型）
            lr: 权重学习率 (默认: 5.0)
            learning_rate_inner: 内层学习率（兼容性参数，优先使用lr）
            learning_rate_outer: 外层学习率（兼容性参数，优先使用lr）
            inner_epochs: 内层训练轮数 (默认: 1)
            num_inner_steps: 内层步数（兼容性参数，同inner_epochs）
            outer_steps: 外层优化步数 (默认: 5)
            num_outer_steps: 外层步数（兼容性参数，同outer_steps）
            beta: 平滑top-K正则化系数 (默认: 0.1)
            lmbda: L2正则化系数（保留用于兼容性）
            device: 设备 ('cpu' 或 'cuda')
            kernel_fn: 核函数（保留用于兼容性）
        """
        self.model = model.to(device)
        # 兼容新旧参数名
        self.lr = learning_rate_outer if learning_rate_outer is not None else lr
        self.lr_p = learning_rate_inner if learning_rate_inner is not None else 0.01
        self.inner_epochs = num_inner_steps if num_inner_steps is not None else inner_epochs
        self.outer_steps = num_outer_steps if num_outer_steps is not None else outer_steps
        self.beta = beta
        self.lmbda = lmbda
        self.kernel_fn = kernel_fn
        self.device = device

    def train_inner(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        sample_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        内层优化：在加权数据上训练代理模型

        使用加权SGD更新模型参数，权重高的样本对梯度贡献更大。

        参数:
            X: 训练数据，形状 (n_samples, n_features)
            y: 训练标签，形状 (n_samples,)
            sample_weights: 样本权重，形状 (n_samples,)

        返回:
            loss: 最终的内层损失值
        """
        self.model.train()
        loss = float('inf')

        for _ in range(self.inner_epochs):
            # 创建优化器
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr_p)
            optimizer.zero_grad()

            # 转移数据到设备
            X = X.to(self.device).float()
            y = y.to(self.device).long()
            sample_weights = sample_weights.to(self.device).float().detach()

            # 前向传播
            output = self.model(X)

            # 计算加权交叉熵损失
            loss = torch.mean(
                sample_weights * F.cross_entropy(output, y, reduction='none')
            )

            # 反向传播并更新参数
            loss.backward()
            optimizer.step()
            self.model.zero_grad()

        return loss

    def _projection_onto_simplex(
        self,
        v: torch.Tensor
    ) -> torch.Tensor:
        """
        将向量投影到单纯形（非负且和为1）

        使用(Duchi et al., 2008)的算法，确保权重满足：
        - w_i >= 0（非负）
        - sum(w_i) = 1（和为1）

        参数:
            v: 输入向量，形状 (n,)

        返回:
            w: 投影后的向量，满足w >= 0且sum(w) = 1
        """
        n = v.shape[0]

        # 确保非负
        v = torch.clamp(v, min=0.0)

        # 如果和为0，返回均匀分布
        if v.sum() == 0:
            return torch.ones(n, device=v.device) / n

        # 排序
        u = torch.sort(v, descending=True)[0]

        # 找到rho（向量化）
        cumsum = torch.cumsum(u, dim=0)
        rho_mask = u - (cumsum - 1.0) / torch.arange(1, n + 1, device=v.device, dtype=torch.float32) > 0
        rho = rho_mask.sum().item()

        # 计算阈值
        theta = (cumsum[rho - 1] - 1.0) / rho

        # 投影
        w = torch.clamp(v - theta, min=0.0)

        # 数值稳定性检查
        if w.sum() == 0:
            w = torch.ones(n, device=v.device) / n
        else:
            w = w / w.sum()

        return w

    def _update_sample_weights(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        sample_weights: torch.Tensor,
        topk: int,
        epsilon: float = 1e-3
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        使用Neumann系列近似更新样本权重（双层优化的关键）

        算法步骤：
        1. 计算外层损失（平滑top-K正则化）
        2. 计算外层梯度 dL_outer/dθ
        3. 使用Neumann系列近似计算雅可比矩阵 dθ*/dw
        4. 更新权重：w <- w - lr * dL_outer/dw

        Neumann系列：
        雅可比矩阵通过求解隐式方程获得：
        dθ*/dw ≈ -Σ_{t=0}^{T} (I - η*H)^t * g

        其中：
        - H = ∂²L_inner/∂θ²（海森矩阵）
        - g = ∂²L_inner/∂θ∂w（混合导数）
        - T=3（系列迭代次数）

        参数:
            X: 训练数据，形状 (n_samples, n_features)
            y: 训练标签，形状 (n_samples,)
            sample_weights: 当前样本权重，形状 (n_samples,)
            topk: 选择的coreset大小
            epsilon: 平滑top-K的小常数

        返回:
            sample_weights: 更新后的样本权重
            jacobian: 雅可比矩阵（用于调试）
            loss_outer: 外层损失值
        """
        # 生成平滑top-K的随机噪声
        z = torch.randn(topk, device=self.device)

        # 计算外层损失：平均损失 - beta * 平滑top-K正则化
        loss_outer = F.cross_entropy(self.model(X), y, reduction='none')

        # 获取top-k权重（用于正则化）
        topk_weights, _ = sample_weights.topk(topk)

        # 外层损失：L_outer = mean(loss) - beta * (topk_weights + epsilon*z).sum()
        # 这鼓励权重集中在top-k个样本上
        loss_outer_avg = torch.mean(loss_outer) - self.beta * (topk_weights + epsilon * z).sum()

        # 第1步：计算外层梯度 v0 = dL_outer/dθ
        d_theta = torch.autograd.grad(
            loss_outer_avg,
            self.model.parameters(),
            allow_unused=True
        )
        v_0 = d_theta

        # 第2步：计算内层梯度（用于Neumann系列）
        # 使用softmax归一化的权重来计算内层损失
        # 确保sample_weights需要梯度
        if not sample_weights.requires_grad:
            sample_weights = sample_weights.requires_grad_(True)

        loss_inner = torch.mean(
            F.softmax(sample_weights, dim=-1) *
            F.cross_entropy(self.model(X), y, reduction='none')
        )

        # 计算内层梯度（需要计算图，用于高阶微分）
        grads_theta = torch.autograd.grad(
            loss_inner,
            self.model.parameters(),
            create_graph=True  # 关键：创建计算图以支持二阶导数
        )

        # 第3步：计算G_theta = θ - η * ∇_θ L_inner
        # 这是内层更新后的参数（未实际执行）
        G_theta = []
        for p, g in zip(self.model.parameters(), grads_theta):
            if g is None:
                G_theta.append(None)
            else:
                G_theta.append(p - self.lr_p * g)

        # 第4步：Neumann系列近似（T=3次迭代）
        # 计算 v_Q = Σ_{t=0}^{T-1} (I - ηH)^t * v_0
        # 这是雅可比-向量积，用于隐式微分
        v_Q = v_0
        for _ in range(3):
            # 计算 (I - ηH) * v_0
            # 通过自动微分实现：∂G_theta/∂θ * v_0
            v_new = torch.autograd.grad(
                G_theta,
                self.model.parameters(),
                grad_outputs=v_0,
                retain_graph=True
            )
            # 分离梯度以避免计算图过大
            v_0 = [i.detach() for i in v_new]

            # 累加到v_Q
            for i in range(len(v_0)):
                v_Q[i].add_(v_0[i].detach())

        # 第5步：计算雅可比矩阵 dθ*/dw
        # 使用链式法则：dL_outer/dw = dL_outer/dθ * dθ*/dw
        # 通过隐式微分：jacobian = -∂(∇_θ L_inner)/∂w * v_Q
        jacobian = -torch.autograd.grad(
            grads_theta,
            sample_weights,
            grad_outputs=v_Q
        )[0]

        # 第6步：更新权重
        with torch.no_grad():
            sample_weights -= self.lr * jacobian

        return sample_weights, jacobian, loss_outer

    def train_outer(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        sample_weights: torch.Tensor,
        topk: int
    ) -> Tuple[torch.Tensor, Dict]:
        """
        外层优化：多次迭代更新样本权重

        参数:
            X: 训练数据，形状 (n_samples, n_features)
            y: 训练标签，形状 (n_samples,)
            sample_weights: 初始样本权重，形状 (n_samples,)
            topk: 选择的coreset大小

        返回:
            sample_weights: 最终的样本权重
            info: 包含训练信息的字典
        """
        info = {
            'weights_history': [],
            'jacobian_norm': [],
            'loss_outer_history': []
        }

        for outer_step in range(self.outer_steps):
            # 1. 内层优化：训练代理模型
            _ = self.train_inner(X, y, sample_weights)

            # 2. 外层优化：更新样本权重
            sample_weights, jacobian, loss_outer = self._update_sample_weights(
                X, y, sample_weights, topk
            )

            # 3. 投影到单纯形（确保权重非负且和为1）
            sample_weights = self._projection_onto_simplex(sample_weights)

            # 记录历史
            info['weights_history'].append(sample_weights.detach().clone().cpu().numpy())
            info['jacobian_norm'].append(torch.norm(jacobian).item())
            info['loss_outer_history'].append(torch.mean(loss_outer).item())

        return sample_weights, info

    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        n_samples: int,
        topk: int
    ) -> Tuple[torch.Tensor, Dict]:
        """
        完整的BCSR训练流程

        参数:
            X: 训练数据，形状 (n_samples, n_features)
            y: 训练标签，形状 (n_samples,)
            n_samples: 样本数量（与X.shape[0]相同，用于接口兼容）
            topk: 选择的coreset大小

        返回:
            weights: 最终的样本权重，形状 (n_samples,)
            info: 包含训练信息的字典
        """
        # 初始化权重为均匀分布
        weights = torch.ones(n_samples, device=self.device) / n_samples

        # 外层优化
        weights, info = self.train_outer(X, y, weights, topk)

        # 添加兼容性键
        info['outer_loss'] = info['loss_outer_history']

        return weights, info


# 为了兼容性，添加numpy导入
import numpy as np
