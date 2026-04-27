"""
经验神经正切核（Empirical NTK）实现

提供计算PyTorch模型的神经正切核的功能，支持分块计算和内存优化。
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Callable, Tuple
from functools import reduce


def empirical_ntk(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
    loss_function: Optional[Callable] = None,
    chunk_size: Optional[int] = None,
    diagonal_only: bool = False,
    show_progress: bool = False
) -> torch.Tensor:
    """
    计算经验神经正切核（Empirical NTK）

    参数:
        model: PyTorch神经网络模型
        inputs: 输入数据，形状为 [n_samples, ...]
        targets: 目标值（可选），用于特定损失函数的NTK计算
        loss_function: 损失函数（可选），如果为None则使用MSE
        chunk_size: 分块大小（可选），用于内存优化
        diagonal_only: 是否仅计算对角线元素
        show_progress: 是否显示进度条

    返回:
        K: NTK矩阵，形状为 [n_samples, n_samples] 或 [n_samples, n_outputs, n_samples, n_outputs]
    """
    model.eval()

    # 确定设备
    device = next(model.parameters()).device
    inputs = inputs.to(device)

    n_samples = inputs.size(0)

    # 如果没有指定损失函数，使用默认的MSE
    if loss_function is None:
        def loss_function(outputs, targets):
            if targets is None:
                # 使用输出的均方作为默认损失
                return 0.5 * torch.mean(outputs ** 2)
            return 0.5 * torch.mean((outputs - targets) ** 2)

    # 计算雅可比矩阵
    jacobians = compute_jacobian(
        model,
        inputs,
        targets=targets,
        loss_function=loss_function,
        chunk_size=chunk_size,
        show_progress=show_progress
    )

    # 计算NTK: K = J @ J^T
    if diagonal_only:
        # 仅计算对角线元素
        if jacobians.dim() == 3:  # [n_samples, n_params, n_outputs]
            K = torch.einsum('npo,npo->n', jacobians, jacobians)
        else:  # [n_samples, n_params]
            K = torch.einsum('np,np->n', jacobians, jacobians)
    else:
        # 计算完整的NTK矩阵
        if jacobians.dim() == 3:  # [n_samples, n_params, n_outputs]
            # 对每个输出分别计算
            n_outputs = jacobians.size(2)
            K = torch.zeros(n_samples, n_samples, n_outputs, n_outputs, device=device)
            for i in range(n_outputs):
                for j in range(n_outputs):
                    J_i = jacobians[:, :, i]  # [n_samples, n_params]
                    J_j = jacobians[:, :, j]  # [n_samples, n_params]
                    K[:, :, i, j] = torch.matmul(J_i, J_j.T)
        else:  # [n_samples, n_params]
            K = torch.matmul(jacobians, jacobians.T)

    return K


def compute_jacobian(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
    loss_function: Optional[Callable] = None,
    chunk_size: Optional[int] = None,
    show_progress: bool = False
) -> torch.Tensor:
    """
    计算模型输出相对于参数的雅可比矩阵

    参数:
        model: PyTorch神经网络模型
        inputs: 输入数据，形状为 [n_samples, ...]
        targets: 目标值（可选）
        loss_function: 损失函数（可选）
        chunk_size: 分块大小（可选）
        show_progress: 是否显示进度

    返回:
        J: 雅可比矩阵，形状为 [n_samples, n_params] 或 [n_samples, n_params, n_outputs]
    """
    model.eval()
    device = next(model.parameters()).device
    inputs = inputs.to(device)

    if targets is not None:
        targets = targets.to(device)

    n_samples = inputs.size(0)

    # 获取模型参数
    params = list(model.parameters())
    if not params:
        raise ValueError("模型没有可训练参数")

    # 计算总参数数量
    param_shapes = [p.shape for p in params]
    param_sizes = [p.numel() for p in params]
    total_params = sum(param_sizes)

    # 运行一次前向传播以确定输出维度
    with torch.no_grad():
        outputs = model(inputs[:1])  # 使用单个样本确定输出形状
        output_shape = outputs.shape[1:]  # 除了batch维度的所有维度
        n_outputs = outputs.numel() // outputs.size(0)  # 每个样本的输出元素数

    # 默认损失函数
    if loss_function is None:
        def loss_function(outputs, targets):
            if targets is None:
                return 0.5 * torch.mean(outputs ** 2)
            return 0.5 * torch.mean((outputs - targets) ** 2)

    # 存储雅可比矩阵
    if len(output_shape) == 1:  # 单维输出 [n_samples, n_features]
        jacobians = torch.zeros(n_samples, total_params, device=device)
    else:  # 多维输出 [n_samples, n_params, n_outputs]
        jacobians = torch.zeros(n_samples, total_params, n_outputs, device=device)

    # 分块计算以节省内存
    if chunk_size is None:
        chunk_size = n_samples

    iterator = range(0, n_samples, chunk_size)
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="计算NTK雅可比矩阵")
        except ImportError:
            pass

    for start_idx in iterator:
        end_idx = min(start_idx + chunk_size, n_samples)
        chunk_inputs = inputs[start_idx:end_idx]

        if targets is not None:
            chunk_targets = targets[start_idx:end_idx]
        else:
            chunk_targets = None

        # 计算该分块的雅可比矩阵
        chunk_jacobians = _compute_chunk_jacobian(
            model,
            chunk_inputs,
            chunk_targets,
            loss_function,
            param_sizes,
            output_shape
        )

        jacobians[start_idx:end_idx] = chunk_jacobians

    return jacobians


def _compute_chunk_jacobian(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: Optional[torch.Tensor],
    loss_function: Callable,
    param_sizes: list,
    output_shape: torch.Size
) -> torch.Tensor:
    """
    计算单个分块的雅可比矩阵

    参数:
        model: PyTorch模型
        inputs: 输入分块
        targets: 目标分块
        loss_function: 损失函数
        param_sizes: 各层参数大小列表
        output_shape: 输出形状

    返回:
        chunk_jacobians: 该分块的雅可比矩阵
    """
    chunk_size = inputs.size(0)
    total_params = sum(param_sizes)
    device = inputs.device

    # 确定输出维度
    if len(output_shape) == 1:  # [n_samples, n_features]
        n_outputs = output_shape[0]
    else:
        n_outputs = output_shape.numel()

    # 初始化分块雅可比矩阵
    if len(output_shape) == 1:
        chunk_jacobians = torch.zeros(chunk_size, total_params, device=device)
    else:
        chunk_jacobians = torch.zeros(chunk_size, total_params, n_outputs, device=device)

    # 对每个输出维度计算梯度
    for output_idx in range(n_outputs):
        # 创建一个one-hot向量
        if len(output_shape) == 1:
            # 单维输出
            one_hot = torch.zeros(chunk_size, n_outputs, device=device)
            one_hot[:, output_idx] = 1.0
        else:
            # 多维输出，需要重塑
            one_hot = torch.zeros(chunk_size, *output_shape, device=device)
            one_hot.view(chunk_size, -1)[:, output_idx] = 1.0

        # 前向传播
        outputs = model(inputs)

        # 计算损失（使用one-hot向量作为目标）
        loss = torch.mean(outputs * one_hot)

        # 反向传播计算梯度
        model.zero_grad()
        loss.backward()

        # 收集梯度
        grad_list = []
        for param in model.parameters():
            if param.grad is not None:
                # 展平梯度，保持batch维度一致
                grad_flat = param.grad.detach().view(-1)
                grad_list.append(grad_flat)

        if grad_list:
            # 拼接所有参数的梯度
            grads = torch.cat(grad_list, dim=0)  # [n_params]

            if len(output_shape) == 1:
                # 对单维输出，需要对每个样本重复梯度
                chunk_jacobians[:, :] += grads.unsqueeze(0).expand(chunk_size, -1)
            else:
                # 对多维输出，每个样本使用相同的梯度
                chunk_jacobians[:, :, output_idx] = grads.unsqueeze(0).expand(chunk_size, -1)

        # 清空梯度
        model.zero_grad()

    return chunk_jacobians


def ntk_features(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
    loss_function: Optional[Callable] = None,
    chunk_size: Optional[int] = None,
    feature_type: str = 'ntk'
) -> torch.Tensor:
    """
    计算基于NTK的特征表示

    参数:
        model: PyTorch模型
        inputs: 输入数据
        targets: 目标值（可选）
        loss_function: 损失函数（可选）
        chunk_size: 分块大小
        feature_type: 特征类型 ('ntk', 'jacobian', 'gradient')

    返回:
        features: 特征表示
    """
    if feature_type == 'ntk':
        # 计算完整的NTK矩阵
        K = empirical_ntk(
            model,
            inputs,
            targets,
            loss_function,
            chunk_size,
            diagonal_only=False
        )
        # 返回NTK特征（每个样本与所有其他样本的核值）
        features = K  # [n_samples, n_samples]

    elif feature_type == 'jacobian':
        # 返回雅可比矩阵作为特征
        features = compute_jacobian(
            model,
            inputs,
            targets,
            loss_function,
            chunk_size
        )

    elif feature_type == 'gradient':
        # 计算梯度特征
        jacobians = compute_jacobian(
            model,
            inputs,
            targets,
            loss_function,
            chunk_size
        )
        # 计算梯度的范数作为特征
        if jacobians.dim() == 3:
            features = torch.norm(jacobians, dim=1)  # [n_samples, n_outputs]
        else:
            features = torch.norm(jacobians, dim=1, keepdim=True)  # [n_samples, 1]

    else:
        raise ValueError(f"未知的特征类型: {feature_type}")

    return features
