"""
BCSR (Bilevel Coreset Selection with Reweighting) Coreset选择模块

实现基于双层优化的coreset选择方法，通过优化样本权重来选择最有价值的样本。
"""

import torch
import numpy as np
from typing import Tuple, Dict, Optional, Callable
import warnings


class BCSRCoreset:
    """
    BCSR Coreset选择类

    使用双层优化框架选择coreset：
    1. 训练阶段：通过内层和外层优化学习样本权重
    2. 选择阶段：根据权重选择top-k样本作为coreset
    """

    def __init__(
        self,
        kernel_fn: Optional[Callable] = None,
        learning_rate_inner: float = 0.01,
        learning_rate_outer: float = 0.1,
        num_inner_steps: int = 50,
        num_outer_steps: int = 20,
        lmbda: float = 0.0,
        device: str = 'cpu',
        random_state: Optional[int] = None
    ):
        """
        初始化BCSR coreset选择器

        参数:
            kernel_fn: 核函数（可选）
            learning_rate_inner: 内层优化学习率
            learning_rate_outer: 外层优化学习率
            num_inner_steps: 内层优化步数
            num_outer_steps: 外层优化步数
            lmbda: L2正则化系数
            device: 设备 ('cpu' 或 'cuda')
            random_state: 随机种子
        """
        self.kernel_fn = kernel_fn
        self.lr_inner = learning_rate_inner
        self.lr_outer = learning_rate_outer
        self.num_inner_steps = num_inner_steps
        self.num_outer_steps = num_outer_steps
        self.lmbda = lmbda
        self.device = device
        self.random_state = random_state

        # 存储结果
        self.weights_ = None
        self.selected_indices_ = None

        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

    def projection_onto_simplex(self, v: np.ndarray) -> np.ndarray:
        """
        将向量投影到单纯形（非负且和为1）

        使用(Duchi et al., 2008)的算法

        参数:
            v: 输入向量，形状为 (n,)

        返回:
            w: 投影后的向量，满足 w >= 0 且 sum(w) = 1
        """
        n = len(v)
        v = v.astype(np.float64)

        # 确保非负
        v = np.maximum(v, 0)

        # 如果和为0，返回均匀分布
        if v.sum() == 0:
            return np.ones(n) / n

        # 排序
        u = np.sort(v)[::-1]

        # 找到rho
        rho = 0
        cumsum = 0
        for j in range(n):
            cumsum += u[j]
            if u[j] - (cumsum - 1) / (j + 1) > 0:
                rho = j + 1

        # 计算阈值
        theta = (np.sum(u[:rho]) - 1) / rho

        # 投影
        w = np.maximum(v - theta, 0)

        # 数值稳定性检查
        if w.sum() == 0:
            w = np.ones(n) / n
        else:
            w = w / w.sum()

        return w

    def projection_onto_simplex_torch(self, v: torch.Tensor) -> torch.Tensor:
        """
        将向量投影到单纯形（PyTorch 版，GPU 兼容）

        使用(Duchi et al., 2008)的算法

        参数:
            v: 输入向量，形状为 (n,)

        返回:
            w: 投影后的向量，满足 w >= 0 且 sum(w) = 1
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

    def coreset_select(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        coreset_size: int,
        model: Optional[torch.nn.Module] = None,
        validation_split: float = 0.2,
        batch_size: int = 128
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        使用BCSR方法选择coreset

        参数:
            X: 训练数据，形状为 (n_samples, n_features) 或 (n_samples, C, H, W)
            y: 标签，形状为 (n_samples,)
            coreset_size: coreset大小
            model: PyTorch模型（用于基于梯度的方法）
            validation_split: 验证集比例
            batch_size: 批次大小

        返回:
            selected_X: 选择的coreset数据
            selected_y: 选择的coreset标签
            info: 包含选择信息的字典
        """
        n_samples = X.shape[0]

        if coreset_size > n_samples:
            warnings.warn(
                f"coreset_size ({coreset_size}) 大于样本数 ({n_samples})，"
                f"将使用所有样本"
            )
            coreset_size = n_samples

        print(f"开始BCSR coreset选择，从{n_samples}个样本中选择{coreset_size}个")

        # 方法1: 如果提供了模型，使用基于双层优化的方法
        if model is not None:
            weights = self._optimize_weights_with_model(
                X, y, model, validation_split, batch_size
            )
        # 方法2: 否则使用简化的基于核的方法
        else:
            weights = self._optimize_weights_kernel(
                X, y, validation_split
            )

        # 根据权重选择top-k样本
        top_k_indices = np.argsort(-weights)[:coreset_size]

        # 提取选择的样本
        if isinstance(X, torch.Tensor):
            selected_X = X[top_k_indices].cpu().numpy()
        else:
            selected_X = X[top_k_indices]

        if isinstance(y, torch.Tensor):
            selected_y = y[top_k_indices].cpu().numpy()
        else:
            selected_y = y[top_k_indices]

        # 存储结果
        self.weights_ = weights
        self.selected_indices_ = top_k_indices

        # 统计信息
        info = {
            'method': 'BCSR',
            'n_samples': n_samples,
            'coreset_size': coreset_size,
            'weights_mean': float(np.mean(weights)),
            'weights_std': float(np.std(weights)),
            'weights_min': float(np.min(weights)),
            'weights_max': float(np.max(weights)),
            'selected_indices': top_k_indices,
            'all_weights': weights
        }

        print(f"BCSR coreset选择完成")
        print(f"权重统计: 均值={info['weights_mean']:.4f}, "
              f"标准差={info['weights_std']:.4f}")
        print(f"权重范围: [{info['weights_min']:.4f}, {info['weights_max']:.4f}]")

        return selected_X, selected_y, info

    def _optimize_weights_with_model(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        model: torch.nn.Module,
        validation_split: float,
        batch_size: int
    ) -> np.ndarray:
        """
        使用模型进行双层优化来学习权重

        参数:
            X: 训练数据
            y: 标签
            model: PyTorch模型
            validation_split: 验证集比例
            batch_size: 批次大小

        返回:
            weights: 学习到的样本权重
        """
        from torch.utils.data import TensorDataset, DataLoader, random_split

        # 划分训练集和验证集
        dataset = TensorDataset(X, y)
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size

        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        # 导入BCSR训练器
        from ..training.bcsr_training import BCSRTraining

        # 创建训练器
        trainer = BCSRTraining(
            model=model,
            kernel_fn=self.kernel_fn,
            learning_rate_inner=self.lr_inner,
            learning_rate_outer=self.lr_outer,
            num_inner_steps=self.num_inner_steps,
            num_outer_steps=self.num_outer_steps,
            lmbda=self.lmbda,
            device=self.device
        )

        # 训练并获取权重
        weights = trainer.train(train_loader, val_loader, n_samples=train_size)

        # 转换为numpy数组
        weights = weights.detach().cpu().numpy()

        return weights

    def _optimize_weights_kernel(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        validation_split: float
    ) -> np.ndarray:
        """
        使用核方法进行简化的权重优化（GPU 版本）

        这是一个简化版本，不使用完整的双层优化，而是基于样本的多样性和重要性

        参数:
            X: 训练数据（torch.Tensor，在 device 上）
            y: 标签（torch.Tensor，在 device 上）
            validation_split: 验证集比例

        返回:
            weights: 学习到的样本权重（numpy 数组）
        """
        # 确保 X, y 在正确的设备上
        device = self.device
        if isinstance(X, torch.Tensor):
            X_t = X.to(device)
        else:
            X_t = torch.from_numpy(X).float().to(device)

        if isinstance(y, torch.Tensor):
            y_t = y.to(device)
        else:
            y_t = torch.from_numpy(y).long().to(device)

        n_samples = X_t.shape[0]

        # 展平图像数据
        if X_t.ndim > 2:
            X_flat = X_t.view(n_samples, -1)
        else:
            X_flat = X_t

        # 归一化（PyTorch）
        X_norm = X_flat / (torch.norm(X_flat, dim=1, keepdim=True) + 1e-8)

        # 计算RBF核矩阵（PyTorch，GPU）
        gamma = 1.0 / X_flat.shape[1]
        K = self._compute_rbf_kernel_torch(X_norm, gamma)

        # 计算类别平衡权重（PyTorch）
        unique_labels = torch.unique(y_t)
        counts = torch.tensor([(y_t == label).sum() for label in unique_labels],
                              dtype=torch.float32, device=device)
        class_weights = 1.0 / counts

        # 为每个样本分配类别权重
        sample_class_weights = torch.zeros(n_samples, device=device)
        for label, weight in zip(unique_labels, class_weights):
            sample_class_weights[y_t == label] = weight

        # 计算多样性权重（基于核矩阵）
        diversity_scores = K.mean(dim=1)

        # 组合权重
        weights = sample_class_weights * diversity_scores

        # 归一化到单纯形（PyTorch 版）
        weights = self.projection_onto_simplex_torch(weights)

        return weights.cpu().numpy()

    def _compute_rbf_kernel_torch(
        self,
        X: torch.Tensor,
        gamma: float
    ) -> torch.Tensor:
        """
        计算RBF核矩阵（PyTorch 版，GPU 兼容）

        参数:
            X: 数据矩阵，形状为 (n_samples, n_features)
            gamma: RBF核参数

        返回:
            K: 核矩阵，形状为 (n_samples, n_samples)
        """
        # 计算欧几里得距离平方
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x^T*y
        X_norm = torch.sum(X ** 2, dim=1)
        dist_sq = X_norm.unsqueeze(1) + X_norm.unsqueeze(0) - 2.0 * (X @ X.T)

        # 确保非负
        dist_sq = torch.clamp(dist_sq, min=0.0)

        # 计算RBF核
        K = torch.exp(-gamma * dist_sq)

        return K

    def _compute_rbf_kernel(
        self,
        X: np.ndarray,
        gamma: float
    ) -> np.ndarray:
        """
        计算RBF核矩阵

        参数:
            X: 数据矩阵，形状为 (n_samples, n_features)
            gamma: RBF核参数

        返回:
            K: 核矩阵，形状为 (n_samples, n_samples)
        """
        n_samples = X.shape[0]

        # 计算欧几里得距离平方
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x^T*y
        X_norm = np.sum(X ** 2, axis=1)
        dist_sq = X_norm[:, np.newaxis] + X_norm[np.newaxis, :] - 2 * X @ X.T

        # 确保非负
        dist_sq = np.maximum(dist_sq, 0)

        # 计算RBF核
        K = np.exp(-gamma * dist_sq)

        return K

    def get_weights(self) -> Optional[np.ndarray]:
        """
        获取学习到的样本权重

        返回:
            weights: 样本权重，如果尚未训练则返回None
        """
        return self.weights_

    def get_selected_indices(self) -> Optional[np.ndarray]:
        """
        获取选择的样本索引

        返回:
            indices: 选择的样本索引，如果尚未训练则返回None
        """
        return self.selected_indices_
