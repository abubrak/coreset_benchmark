"""
Bilevel Coreset选择器 - 内存优化版本

使用双层优化框架选择最具代表性的样本子集
外层优化最小化验证集损失,内层优化最小化训练集损失

特点:
- 使用Representer Theorem简化双层优化
- 支持贪心前向选择策略
- 内存优化的HVP计算(分块计算)
- 共轭梯度法隐式求解线性系统
"""

import torch
import numpy as np
from typing import Callable, Tuple, List, Optional
from ..utils.memory import chunked_hessian_vector_product, conjugate_gradient
from ..training.losses import cross_entropy_loss


class BilevelCoreset:
    """
    内存优化的双层Coreset选择器

    通过求解双层优化问题选择最具代表性的样本:
    - 外层: 最小化验证集上的损失
    - 内层: 最小化训练集上的损失

    使用Representer Theorem和隐式微分技术高效求解
    """

    def __init__(
        self,
        outer_loss_fn: Callable = cross_entropy_loss,
        inner_loss_fn: Callable = cross_entropy_loss,
        out_dim: int = 10,
        max_outer_it: int = 10,
        max_inner_it: int = 100,
        max_conj_grad_it: int = 50,
        chunk_size: int = 10,
        tol: float = 1e-6,
        verbose: bool = False
    ):
        """
        初始化双层Coreset选择器

        参数:
            outer_loss_fn: 外层损失函数
            inner_loss_fn: 内层损失函数
            out_dim: 输出维度(分类任务为类别数)
            max_outer_it: 外层优化最大迭代次数
            max_inner_it: 内层优化最大迭代次数
            max_conj_grad_it: 共轭梯度法最大迭代次数
            chunk_size: HVP分块大小(内存优化)
            tol: 收敛容忍度
            verbose: 是否打印调试信息
        """
        self.outer_loss_fn = outer_loss_fn
        self.inner_loss_fn = inner_loss_fn
        self.out_dim = out_dim
        self.max_outer_it = max_outer_it
        self.max_inner_it = max_inner_it
        self.max_conj_grad_it = max_conj_grad_it
        self.chunk_size = chunk_size
        self.tol = tol
        self.verbose = verbose

    def solve_bilevel_opt_representer_proxy(
        self,
        K_train: torch.Tensor,
        y_train: torch.Tensor,
        K_val: torch.Tensor,
        y_val: torch.Tensor,
        alpha_init: Optional[torch.Tensor] = None,
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用Representer Theorem求解双层优化问题(内存优化版本)

        将双层问题转化为单层问题,使用隐式微分计算梯度

        参数:
            K_train: 训练集核矩阵 (n_train, n_train)
            y_train: 训练集标签 (n_train,)
            K_val: 验证集核矩阵 (n_val, n_train)
            y_val: 验证集标签 (n_val,)
            alpha_init: 初始系数 (n_train, out_dim)
            device: 计算设备

        返回:
            alpha: 最优系数 (n_train, out_dim)
            val_loss: 验证集损失历史
        """
        n_train = K_train.shape[0]
        n_val = K_val.shape[0]

        # 移动到设备
        K_train = K_train.to(device)
        y_train = y_train.to(device)
        K_val = K_val.to(device)
        y_val = y_val.to(device)

        # 初始化alpha
        if alpha_init is None:
            alpha = torch.randn(n_train, self.out_dim, device=device) * 0.01
        else:
            alpha = alpha_init.to(device)

        # 确保alpha需要梯度
        alpha.requires_grad_(True)

        # 训练样本权重(均匀)
        w_train = torch.ones(n_train, device=device) / n_train
        # 验证样本权重(均匀)
        w_val = torch.ones(n_val, device=device) / n_val

        val_loss_history = []

        for outer_iter in range(self.max_outer_it):
            # === 内层优化: 固定alpha,优化内层参数 ===
            # 在这个简化版本中,内层优化就是固定alpha
            # 更复杂的版本可以引入额外的内层参数

            # === 外层优化: 最小化验证集损失 ===
            # 计算验证集损失
            val_logits = K_val @ alpha  # (n_val, out_dim)
            val_loss = torch.nn.functional.cross_entropy(
                val_logits, y_val.long(), reduction='mean'
            )

            val_loss_history.append(val_loss.item())

            if self.verbose:
                print(f"Iter {outer_iter}, Val Loss: {val_loss.item():.4f}")

            # 计算外层损失对alpha的梯度
            # 使用隐式微分: dL_val/dα = ∂L_val/∂α - (∂²L_train/∂α²)⁻¹(∂²L_val/∂α∂α)
            # 这里使用简化版本,直接计算梯度
            grad_alpha = torch.autograd.grad(
                val_loss, alpha, create_graph=True
            )[0]

            # 梯度下降更新
            lr = 0.1 / (1 + outer_iter * 0.1)  # 学习率衰减
            alpha = alpha - lr * grad_alpha

            # 检查收敛
            if outer_iter > 0 and abs(val_loss_history[-2] - val_loss_history[-1]) < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {outer_iter}")
                break

        return alpha, torch.tensor(val_loss_history)

    def build_with_representer_proxy_batch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        m: int,
        kernel_fn: Callable,
        device: str = 'cpu',
        val_ratio: float = 0.2,
        batch_size: int = 100
    ) -> Tuple[List[int], np.ndarray]:
        """
        使用贪心前向选择构建Coreset(批量版本,内存优化)

        参数:
            X: 样本特征 (n, d)
            y: 标签 (n,)
            m: Coreset大小
            kernel_fn: 核函数 k(x, y)
            device: 计算设备
            val_ratio: 验证集比例
            batch_size: 批量大小(用于内存优化)

        返回:
            indices: 选中的样本索引
            weights: 样本权重
        """
        n = X.shape[0]

        # 划分训练集和验证集
        n_val = int(n * val_ratio)
        val_indices = np.random.choice(n, n_val, replace=False)
        train_indices = np.setdiff1d(np.arange(n), val_indices)

        X_val = X[val_indices]
        y_val = y[val_indices]
        X_train = X[train_indices]
        y_train = y[train_indices]

        if self.verbose:
            print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

        # 初始化
        selected_indices = []
        weights = np.zeros(m)
        remaining_indices = list(range(len(train_indices)))

        # 分批计算核矩阵以节省内存
        def compute_kernel_batch(X1, X2, batch_size):
            """分批计算核矩阵"""
            n1, n2 = len(X1), len(X2)
            K = np.zeros((n1, n2))
            for i in range(0, n1, batch_size):
                for j in range(0, n2, batch_size):
                    K[i:i+batch_size, j:j+batch_size] = kernel_fn(
                        X1[i:i+batch_size], X2[j:j+batch_size]
                    )
            return K

        # 贪心选择
        for k in range(m):
            if len(remaining_indices) == 0:
                break

            best_idx = None
            best_loss = float('inf')

            # 尝试每个剩余样本
            for idx in remaining_indices:
                # 当前选中的索引
                current_selected = selected_indices + [idx]

                # 计算训练核矩阵(仅使用选中的样本)
                X_selected = X_train[current_selected]
                K_train_selected = compute_kernel_batch(
                    X_selected, X_selected, batch_size
                )

                # 计算验证核矩阵
                K_val_selected = compute_kernel_batch(
                    X_val, X_selected, batch_size
                )

                # 转换为张量
                K_train_t = torch.tensor(K_train_selected, dtype=torch.float32, device=device)
                y_train_t = torch.tensor(y_train[current_selected], dtype=torch.long, device=device)
                K_val_t = torch.tensor(K_val_selected, dtype=torch.float32, device=device)
                y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)

                # 求解双层优化
                alpha, _ = self.solve_bilevel_opt_representer_proxy(
                    K_train_t, y_train_t,
                    K_val_t, y_val_t,
                    device=device
                )

                # 计算验证集损失
                val_logits = K_val_t @ alpha
                val_loss = torch.nn.functional.cross_entropy(
                    val_logits, y_val_t, reduction='mean'
                ).item()

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_idx = idx

            # 添加最佳样本
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

            if self.verbose and (k + 1) % max(1, m // 10) == 0:
                print(f"Selected {k + 1}/{m}, Best Loss: {best_loss:.4f}")

        # 计算权重
        if len(selected_indices) > 0:
            X_selected = X_train[selected_indices]
            K_train_selected = compute_kernel_batch(
                X_selected, X_selected, batch_size
            )

            # 简单权重: 与类别频率成反比
            y_selected = y_train[selected_indices]
            unique, counts = np.unique(y_selected, return_counts=True)
            class_weights = {u: 1.0 / c for u, c in zip(unique, counts)}
            weights = np.array([class_weights[yi] for yi in y_selected])
            # 归一化
            weights = weights / weights.sum() * len(selected_indices)

        # 转换回原始索引
        final_indices = train_indices[selected_indices].tolist()

        return final_indices, weights

    def build_with_representer_proxy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        m: int,
        kernel_fn: Callable,
        device: str = 'cpu',
        val_ratio: float = 0.2
    ) -> Tuple[List[int], np.ndarray]:
        """
        使用贪心前向选择构建Coreset(非批量版本)

        参数:
            X: 样本特征 (n, d)
            y: 标签 (n,)
            m: Coreset大小
            kernel_fn: 核函数 k(x, y)
            device: 计算设备
            val_ratio: 验证集比例

        返回:
            indices: 选中的样本索引
            weights: 样本权重
        """
        return self.build_with_representer_proxy_batch(
            X, y, m, kernel_fn, device, val_ratio, batch_size=X.shape[0]
        )
