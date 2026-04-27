"""
BCSR (Bilevel Coreset Selection with Reweighting) 训练模块

实现基于双层优化的coreset选择训练方法，包含：
- 内层优化：训练代理模型（surrogate model）
- 外层优化：更新样本权重
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Optional, Callable
import numpy as np

from .losses import cross_entropy_loss, accuracy


class BCSRTraining:
    """
    BCSR训练类

    实现双层优化框架用于coreset选择：
    - 内层：在加权数据上训练代理模型
    - 外层：基于验证损失更新样本权重
    """

    def __init__(
        self,
        model: nn.Module,
        kernel_fn: Optional[Callable] = None,
        learning_rate_inner: float = 0.01,
        learning_rate_outer: float = 0.1,
        num_inner_steps: int = 50,
        num_outer_steps: int = 20,
        lmbda: float = 0.0,
        device: str = 'cpu'
    ):
        """
        初始化BCSR训练器

        参数:
            model: PyTorch模型
            kernel_fn: 核函数（如果使用核方法）
            learning_rate_inner: 内层优化学习率
            learning_rate_outer: 外层优化学习率
            num_inner_steps: 内层优化步数
            num_outer_steps: 外层优化步数
            lmbda: L2正则化系数
            device: 设备 ('cpu' 或 'cuda')
        """
        self.model = model.to(device)
        self.kernel_fn = kernel_fn
        self.lr_inner = learning_rate_inner
        self.lr_outer = learning_rate_outer
        self.num_inner_steps = num_inner_steps
        self.num_outer_steps = num_outer_steps
        self.lmbda = lmbda
        self.device = device

        # 初始化优化器
        self.optimizer_inner = optim.SGD(
            self.model.parameters(),
            lr=learning_rate_inner
        )

        # 存储训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'weights': []
        }

    def train_inner(
        self,
        train_loader,
        weights: torch.Tensor,
        val_loader: Optional = None
    ) -> Dict[str, float]:
        """
        内层优化：在加权数据上训练代理模型

        参数:
            train_loader: 训练数据加载器
            weights: 样本权重，形状为 (n_samples,)
            val_loader: 验证数据加载器（可选）

        返回:
            metrics: 包含训练和验证指标的字典
        """
        self.model.train()

        train_losses = []
        train_accs = []

        weight_idx = 0

        for epoch in range(self.num_inner_steps):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                batch_size = data.size(0)

                # 获取当前批次的权重
                batch_weights = weights[weight_idx:weight_idx + batch_size]
                weight_idx += batch_size

                # 前向传播
                if self.kernel_fn is not None:
                    # 使用核方法
                    with torch.no_grad():
                        features = self.model(data)
                    # 这里简化处理，实际需要更复杂的核方法实现
                    output = self.model(data)
                else:
                    output = self.model(data)

                # 计算加权损失
                loss = nn.functional.cross_entropy(
                    output, target, reduction='none'
                )
                weighted_loss = (loss * batch_weights).mean()

                # 反向传播
                self.optimizer_inner.zero_grad()
                weighted_loss.backward()
                self.optimizer_inner.step()

                # 统计
                epoch_loss += weighted_loss.item() * batch_size
                pred = output.argmax(dim=1)
                epoch_correct += (pred == target).sum().item()
                epoch_total += batch_size

            # 计算平均指标
            avg_loss = epoch_loss / epoch_total
            avg_acc = epoch_correct / epoch_total

            train_losses.append(avg_loss)
            train_accs.append(avg_acc)

            # 重置权重索引
            weight_idx = 0

        # 验证
        val_metrics = {}
        if val_loader is not None:
            val_metrics = self._evaluate(val_loader)

        # 组合指标
        metrics = {
            'train_loss': train_losses[-1] if train_losses else 0.0,
            'train_acc': train_accs[-1] if train_accs else 0.0,
            **val_metrics
        }

        return metrics

    def train_outer(
        self,
        train_loader,
        val_loader,
        n_samples: int
    ) -> torch.Tensor:
        """
        外层优化：基于验证损失更新样本权重

        参数:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            n_samples: 训练样本总数

        返回:
            weights: 更新后的样本权重，形状为 (n_samples,)
        """
        # 初始化权重为均匀分布
        weights = torch.ones(n_samples, device=self.device) / n_samples

        for outer_step in range(self.num_outer_steps):
            # 内层优化：训练代理模型
            train_metrics = self.train_inner(train_loader, weights, val_loader)

            # 计算验证损失和梯度
            self.model.eval()
            val_loss = 0.0
            weight_idx = 0
            sample_gradients = []

            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(train_loader):
                    data = data.to(self.device)
                    target = target.to(self.device)
                    batch_size = data.size(0)

                    # 获取当前批次的权重
                    batch_weights = weights[weight_idx:weight_idx + batch_size]

                    # 前向传播
                    output = self.model(data)

                    # 计算每个样本的损失
                    sample_losses = nn.functional.cross_entropy(
                        output, target, reduction='none'
                    )

                    # 计算验证损失（在验证集上）
                    for val_data, val_target in val_loader:
                        val_data = val_data.to(self.device)
                        val_target = val_target.to(self.device)
                        val_output = self.model(val_data)
                        val_loss_batch = nn.functional.cross_entropy(
                            val_output, val_target
                        )
                        val_loss += val_loss_batch.item()
                        break  # 只计算一个batch作为示例

                    # 计算样本梯度（简化版本）
                    # 实际应该使用自动微分计算 d(val_loss)/d(weights)
                    sample_grad = sample_losses.detach()
                    sample_gradients.append(sample_grad)

                    weight_idx += batch_size

            # 更新权重（梯度上升，因为我们要最大化验证集性能）
            weight_idx = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                batch_size = data.size(0)
                batch_grad = sample_gradients[batch_idx]

                # 更新权重
                weights[weight_idx:weight_idx + batch_size] -= \
                    self.lr_outer * batch_grad

                weight_idx += batch_size

            # 投影到单纯形（权重非负且和为1）
            weights = self._projection_onto_simplex(weights)

            # 记录历史
            self.history['train_loss'].append(train_metrics['train_loss'])
            self.history['train_acc'].append(train_metrics['train_acc'])
            self.history['val_loss'].append(train_metrics.get('val_loss', 0.0))
            self.history['val_acc'].append(train_metrics.get('val_acc', 0.0))
            self.history['weights'].append(weights.clone().cpu().numpy())

        return weights

    def _projection_onto_simplex(self, v: torch.Tensor) -> torch.Tensor:
        """
        将向量投影到单纯形（非负且和为1）

        参数:
            v: 输入向量

        返回:
            投影后的向量
        """
        # 确保非负
        v = torch.clamp(v, min=0)

        # 如果和为0，返回均匀分布
        if v.sum() == 0:
            return torch.ones_like(v) / len(v)

        # 归一化
        v = v / v.sum()

        return v

    def _evaluate(self, data_loader) -> Dict[str, float]:
        """
        评估模型

        参数:
            data_loader: 数据加载器

        返回:
            metrics: 包含验证指标的字典
        """
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device)
                target = target.to(self.device)

                output = self.model(data)
                loss = nn.functional.cross_entropy(output, target)

                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += data.size(0)

        metrics = {
            'val_loss': total_loss / total,
            'val_acc': correct / total
        }

        return metrics

    def train(
        self,
        train_loader,
        val_loader,
        n_samples: int
    ) -> torch.Tensor:
        """
        完整的训练流程

        参数:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            n_samples: 训练样本总数

        返回:
            weights: 最终的样本权重
        """
        print(f"开始BCSR训练，共{self.num_outer_steps}个外层迭代")

        weights = self.train_outer(train_loader, val_loader, n_samples)

        print("BCSR训练完成")
        print(f"最终训练损失: {self.history['train_loss'][-1]:.4f}")
        print(f"最终训练准确率: {self.history['train_acc'][-1]:.4f}")
        print(f"最终验证损失: {self.history['val_loss'][-1]:.4f}")
        print(f"最终验证准确率: {self.history['val_acc'][-1]:.4f}")

        return weights
