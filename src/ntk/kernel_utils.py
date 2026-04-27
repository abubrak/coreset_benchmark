"""
核函数工具

提供常用的核函数实现以及与PyTorch模型集成的核函数工厂。
"""

import torch
import torch.nn as nn
from typing import Callable, Optional, Union, Tuple
import functools


def rbf_kernel(
    x1: torch.Tensor,
    x2: torch.Tensor,
    length_scale: float = 1.0,
    variance: float = 1.0
) -> torch.Tensor:
    """
    径向基函数（RBF）核，也称为高斯核

    K(x, y) = variance * exp(-||x - y||^2 / (2 * length_scale^2))

    参数:
        x1: 第一个输入，形状为 [..., n_features]
        x2: 第二个输入，形状为 [..., n_features]
        length_scale: 长度尺度参数
        variance: 方差参数

    返回:
        K: 核矩阵，形状为 [...]
    """
    # 计算欧几里得距离的平方
    dist_sq = torch.sum((x1 - x2) ** 2, dim=-1)

    # 计算RBF核
    K = variance * torch.exp(-dist_sq / (2 * length_scale ** 2))

    return K


def linear_kernel(
    x1: torch.Tensor,
    x2: torch.Tensor,
    variance: float = 1.0,
    bias: float = 0.0
) -> torch.Tensor:
    """
    线性核

    K(x, y) = variance * (x^T y + bias)

    参数:
        x1: 第一个输入，形状为 [..., n_features]
        x2: 第二个输入，形状为 [..., n_features]
        variance: 方差参数
        bias: 偏置参数

    返回:
        K: 核矩阵，形状为 [...]
    """
    # 计算内积
    inner_prod = torch.sum(x1 * x2, dim=-1)

    # 添加偏置和方差
    K = variance * (inner_prod + bias)

    return K


def polynomial_kernel(
    x1: torch.Tensor,
    x2: torch.Tensor,
    degree: int = 2,
    variance: float = 1.0,
    bias: float = 1.0
) -> torch.Tensor:
    """
    多项式核

    K(x, y) = variance * (x^T y + bias)^degree

    参数:
        x1: 第一个输入，形状为 [..., n_features]
        x2: 第二个输入，形状为 [..., n_features]
        degree: 多项式的阶数
        variance: 方差参数
        bias: 偏置参数

    返回:
        K: 核矩阵，形状为 [...]
    """
    # 计算内积
    inner_prod = torch.sum(x1 * x2, dim=-1)

    # 应用多项式变换
    K = variance * torch.pow(inner_prod + bias, degree)

    return K


class CachedKernel:
    """
    带缓存的核函数包装器

    缓存已计算的核矩阵以避免重复计算，提高效率。
    """

    def __init__(
        self,
        kernel_fn: Callable,
        use_cache: bool = True
    ):
        """
        参数:
            kernel_fn: 基础核函数
            use_cache: 是否使用缓存
        """
        self.kernel_fn = kernel_fn
        self.use_cache = use_cache
        self._cache = {}

    def __call__(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        计算核矩阵，使用缓存以避免重复计算

        参数:
            x1: 第一个输入
            x2: 第二个输入
            **kwargs: 传递给核函数的额外参数

        返回:
            K: 核矩阵
        """
        if not self.use_cache:
            return self.kernel_fn(x1, x2, **kwargs)

        # 创建缓存键
        cache_key = (
            id(x1),
            id(x2),
            tuple(sorted(kwargs.items()))
        )

        if cache_key in self._cache:
            return self._cache[cache_key]

        # 计算并缓存
        K = self.kernel_fn(x1, x2, **kwargs)
        self._cache[cache_key] = K

        return K

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()

    def get_cache_size(self) -> int:
        """获取缓存大小"""
        return len(self._cache)


def create_pytorch_kernel_fn(
    model: nn.Module,
    layer_name: Optional[str] = None,
    normalize: bool = True,
    temperature: float = 1.0
) -> Callable:
    """
    创建基于PyTorch模型输出的核函数

    返回的核函数计算两个输入通过模型（或特定层）后的内积。

    参数:
        model: PyTorch模型
        layer_name: 指定使用哪一层的输出（None表示使用最终输出）
        normalize: 是否对特征进行L2归一化
        temperature: 温度参数（用于缩放特征）

    返回:
        kernel_fn: 核函数，接受两个输入并返回核矩阵
    """
    # 如果需要提取中间层特征，创建钩子
    if layer_name is not None:
        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        # 注册前向钩子
        handle = None
        for name, layer in model.named_modules():
            if name == layer_name:
                handle = layer.register_forward_hook(get_activation(name))
                break

        if handle is None:
            raise ValueError(f"未找到层: {layer_name}")

    def kernel_fn(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        计算基于模型特征的核矩阵

        参数:
            x1: 第一个输入，形状为 [n1, ...]
            x2: 第二个输入，形状为 [n2, ...]

        返回:
            K: 核矩阵，形状为 [n1, n2]
        """
        device = next(model.parameters()).device

        x1 = x1.to(device)
        x2 = x2.to(device)

        model.eval()

        with torch.no_grad():
            # 获取x1的特征
            if layer_name is not None:
                _ = model(x1)
                features1 = activation[layer_name]
            else:
                features1 = model(x1)

            # 获取x2的特征
            if layer_name is not None:
                _ = model(x2)
                features2 = activation[layer_name]
            else:
                features2 = model(x2)

        # 展平特征（除了batch维度）
        features1 = features1.view(features1.size(0), -1)
        features2 = features2.view(features2.size(0), -1)

        # 应用温度缩放
        if temperature != 1.0:
            features1 = features1 / temperature
            features2 = features2 / temperature

        # L2归一化
        if normalize:
            features1 = nn.functional.normalize(features1, p=2, dim=1)
            features2 = nn.functional.normalize(features2, p=2, dim=1)

        # 计算内积
        K = torch.matmul(features1, features2.T)

        return K

    return kernel_fn


def compute_kernel_matrix(
    kernel_fn: Callable,
    X: torch.Tensor,
    Y: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    计算完整的核矩阵

    参数:
        kernel_fn: 核函数
        X: 输入数据，形状为 [n_samples_x, ...]
        Y: 另一个输入数据（可选），如果为None则计算K(X, X)

    返回:
        K: 核矩阵，形状为 [n_samples_x, n_samples_y]
    """
    if Y is None:
        Y = X

    n_x = X.size(0)
    n_y = Y.size(0)

    # 初始化核矩阵
    K = torch.zeros(n_x, n_y, device=X.device)

    # 逐对计算核函数
    for i in range(n_x):
        for j in range(n_y):
            K[i, j] = kernel_fn(X[i:i+1], Y[j:j+1])

    return K


def batch_compute_kernel_matrix(
    kernel_fn: Callable,
    X: torch.Tensor,
    Y: Optional[torch.Tensor] = None,
    batch_size: int = 32
) -> torch.Tensor:
    """
    批量计算核矩阵，提高内存效率

    参数:
        kernel_fn: 核函数
        X: 输入数据，形状为 [n_samples_x, ...]
        Y: 另一个输入数据（可选）
        batch_size: 批量大小

    返回:
        K: 核矩阵
    """
    if Y is None:
        Y = X

    n_x = X.size(0)
    n_y = Y.size(0)

    # 初始化核矩阵
    K = torch.zeros(n_x, n_y, device=X.device)

    # 分批计算
    for i in range(0, n_x, batch_size):
        end_i = min(i + batch_size, n_x)
        for j in range(0, n_y, batch_size):
            end_j = min(j + batch_size, n_y)

            # 计算该批次的核矩阵
            K_batch = kernel_fn(X[i:end_i], Y[j:end_j])
            K[i:end_i, j:end_j] = K_batch

    return K


# 预定义的核函数
KERNEL_FUNCTIONS = {
    'rbf': rbf_kernel,
    'linear': linear_kernel,
    'polynomial': polynomial_kernel,
}


def get_kernel_fn(kernel_name: str, **kernel_params) -> Callable:
    """
    获取预定义的核函数

    参数:
        kernel_name: 核函数名称 ('rbf', 'linear', 'polynomial')
        **kernel_params: 核函数参数

    返回:
        kernel_fn: 核函数
    """
    if kernel_name not in KERNEL_FUNCTIONS:
        raise ValueError(
            f"未知的核函数: {kernel_name}. "
            f"可选的核函数: {list(KERNEL_FUNCTIONS.keys())}"
        )

    base_kernel = KERNEL_FUNCTIONS[kernel_name]

    # 创建带有参数的核函数
    @functools.wraps(base_kernel)
    def kernel_fn(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return base_kernel(x1, x2, **kernel_params)

    return kernel_fn
