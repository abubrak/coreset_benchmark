"""
CSReL 损失函数模块

该模块包含从原始 CSReL-Coreset-CL 项目移植的损失函数，
包括组合损失函数和知识蒸馏损失函数。
"""

import torch
import torch.nn as nn
from typing import Optional


class CompliedLoss(nn.Module):
    """
    组合损失函数，支持交叉熵损失和知识蒸馏损失

    该损失函数将标准的交叉熵损失与知识蒸馏损失（可以是 MSE 或 KL 散度）
    组合在一起，用于模型训练。

    参数
    ----
    ce_factor : float
        交叉熵损失的权重系数
    mse_factor : float
        知识蒸馏损失的权重系数
    reduction : str, default='mean'
        损失的归约方式，可选 'none', 'mean', 'sum'
    kd_mode : str, default='mse'
        知识蒸馏模式，可选 'mse'（MSE 损失）或 'ce'（KL 散度损失）

    示例
    ----
    >>> loss_fn = CompliedLoss(ce_factor=1.0, mse_factor=0.5, kd_mode='mse')
    >>> logits = model(images)
    >>> loss = loss_fn(logits, labels, ref_logits)
    """

    def __init__(
        self,
        ce_factor: float,
        mse_factor: float,
        reduction: str = 'mean',
        kd_mode: str = 'mse'
    ):
        super(CompliedLoss, self).__init__()
        self.reduction = reduction
        self.ce_factor = ce_factor
        self.mse_factor = mse_factor
        self.kd_mode = kd_mode

        # 交叉熵损失
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)

        # 根据模式选择知识蒸馏损失
        if self.kd_mode == 'mse':
            self.mse_loss = nn.MSELoss(reduction=reduction)
        elif self.kd_mode == 'ce':
            self.mse_loss = KDCrossEntropyLoss(reduction=reduction)
        else:
            raise ValueError('not a valid model')

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        logits: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算组合损失

        参数
        ----
        x : torch.Tensor
            模型输出的 logits，形状为 (batch_size, num_classes)
        y : torch.Tensor
            真实标签，形状为 (batch_size,)
        logits : Optional[torch.Tensor], default=None
            参考模型的 logits，用于知识蒸馏，形状为 (batch_size, num_classes)

        返回
        ----
        torch.Tensor
            计算得到的损失值
        """
        # 计算交叉熵损失
        loss_c = self.ce_factor * self.ce_loss(x, y)

        # 如果提供了参考 logits 且知识蒸馏因子大于 0，则计算知识蒸馏损失
        if self.mse_factor > 0 and logits is not None:
            loss_m = self.mse_loss(x, logits)

            # 如果 reduction 为 'none'，需要对最后一维求平均
            if self.reduction == 'none':
                loss_m = torch.mean(loss_m, dim=-1)

            # 组合两种损失
            loss = self.ce_factor * loss_c + self.mse_factor * loss_m
            return loss
        else:
            # 只返回交叉熵损失
            return self.ce_factor * loss_c


class KDCrossEntropyLoss(nn.Module):
    """
    知识蒸馏交叉熵损失（KL 散度损失）

    该损失函数计算两个概率分布之间的 KL 散度，用于知识蒸馏。
    使用 softmax 将 logits 转换为概率分布，然后计算 KL 散度。

    参数
    ----
    reduction : str
        损失的归约方式，可选 'none', 'mean', 'sum'

    注意事项
    --------
    - 该损失计算的是 KL 散度：KL(py || px) = sum(py * log(px))
    - 其中 py 是参考分布（teacher），px 是学生分布
    - 使用 softmax 将 logits 转换为概率分布

    示例
    ----
    >>> loss_fn = KDCrossEntropyLoss(reduction='mean')
    >>> student_logits = model(images)
    >>> teacher_logits = teacher_model(images)
    >>> loss = loss_fn(student_logits, teacher_logits)
    """

    def __init__(self, reduction: str):
        super(KDCrossEntropyLoss, self).__init__()
        assert reduction in ['none', 'mean', 'sum']
        self.reduction = reduction
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        计算 KL 散度损失

        参数
        ----
        x : torch.Tensor
            学生模型的 logits，形状为 (batch_size, num_classes)
        y : torch.Tensor
            教师模型的 logits，形状为 (batch_size, num_classes)

        返回
        ----
        torch.Tensor
            计算得到的 KL 散度损失
        """
        # 将 logits 转换为概率分布
        py = self.softmax(y)
        px = self.softmax(x)

        # 计算 KL 散度：sum(py * log(px))
        loss = torch.sum(py * torch.log(px), dim=-1)

        # 根据 reduction 参数进行归约
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return torch.mean(loss)
        else:  # 'sum'
            return torch.sum(loss)
