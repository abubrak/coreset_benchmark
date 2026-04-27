"""
NTK模块 - 神经正切核实现

此模块提供了计算神经网络神经正切核(NTK)的功能，替代neural-tangents库。
支持PyTorch模型，提供高效的NTK计算和核函数工具。
"""

from .empirical_ntk import empirical_ntk, compute_jacobian, ntk_features
from .kernel_utils import (
    rbf_kernel,
    linear_kernel,
    polynomial_kernel,
    CachedKernel,
    create_pytorch_kernel_fn
)
from .models import SimpleCNN, SimpleResNet, SimpleMLP

__all__ = [
    # 经验NTK函数
    'empirical_ntk',
    'compute_jacobian',
    'ntk_features',

    # 核函数工具
    'rbf_kernel',
    'linear_kernel',
    'polynomial_kernel',
    'CachedKernel',
    'create_pytorch_kernel_fn',

    # 预定义模型
    'SimpleCNN',
    'SimpleResNet',
    'SimpleMLP',
]

__version__ = '1.0.0'
