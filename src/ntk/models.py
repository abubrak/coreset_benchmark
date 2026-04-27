"""
预定义的神经网络模型

提供用于NTK计算的简单神经网络模型，包括CNN和ResNet变体。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Callable


class SimpleCNN(nn.Module):
    """
    简单的卷积神经网络

    适用于MNIST/CIFAR等图像数据集的NTK计算。

    架构:
        - Conv2d(1/3, 32, 3) + ReLU
        - Conv2d(32, 64, 3) + ReLU
        - MaxPool2d(2)
        - Flatten
        - Linear(64*12*12, 128) + ReLU
        - Linear(128, num_classes)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        input_size: int = 28
    ):
        """
        参数:
            in_channels: 输入通道数（1为灰度图，3为RGB）
            num_classes: 输出类别数
            input_size: 输入图像大小（假设为正方形）
        """
        super(SimpleCNN, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.input_size = input_size

        # 卷积层
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # 池化层
        self.pool = nn.MaxPool2d(2, 2)

        # 计算卷积层后的特征图大小
        # conv1: padding=1, kernel=3, 保持尺寸
        # conv2: padding=1, kernel=3, 保持尺寸
        # pool: kernel=2, stride=2, 尺寸减半
        # 所以最终特征图大小为 (input_size // 2) x (input_size // 2)
        conv_output_size = input_size // 2
        self.feature_size = 64 * conv_output_size * conv_output_size

        # 全连接层 - 使用LazyLinear自动计算输入维度
        self.fc1 = nn.Linear(self.feature_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # 初始化权重（对于NTK计算，使用合适的初始化）
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x: 输入张量，形状为 [batch_size, in_channels, height, width]

        返回:
            out: 输出张量，形状为 [batch_size, num_classes]
        """
        # 第一卷积块
        x = F.relu(self.conv1(x))

        # 第二卷积块
        x = F.relu(self.conv2(x))

        # 池化
        x = self.pool(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def get_feature_dim(self) -> int:
        """获取全连接层之前的特征维度"""
        return self.feature_size


class SimpleResNet(nn.Module):
    """
    简化的ResNet模型

    包含基本的残差连接，适用于NTK计算。

    架构:
        - 初始卷积层
        - 多个ResidualBlock
        - 全连接层
    """

    class ResidualBlock(nn.Module):
        """残差块"""

        def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
            super().__init__()

            self.conv1 = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, stride=stride,
                padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, stride=1,
                padding=1, bias=False
            )
            self.bn2 = nn.BatchNorm2d(out_channels)

            # 快捷连接
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_channels, out_channels,
                        kernel_size=1, stride=stride,
                        bias=False
                    ),
                    nn.BatchNorm2d(out_channels)
                )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        blocks: List[int] = [2, 2, 2, 2],
        base_channels: int = 64
    ):
        """
        参数:
            in_channels: 输入通道数
            num_classes: 输出类别数
            blocks: 每个阶段的残差块数量
            base_channels: 基础通道数
        """
        super(SimpleResNet, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # 初始卷积层
        self.conv1 = nn.Conv2d(
            in_channels, base_channels,
            kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(base_channels)

        # 残差块
        self.layer1 = self._make_layer(base_channels, blocks[0], stride=1)
        self.layer2 = self._make_layer(base_channels * 2, blocks[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 4, blocks[2], stride=2)
        self.layer4 = self._make_layer(base_channels * 8, blocks[3], stride=2)

        # 全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 8, num_classes)

        # 初始化权重
        self._initialize_weights()

    def _make_layer(
        self,
        out_channels: int,
        blocks: int,
        stride: int
    ) -> nn.Sequential:
        """创建残差层"""
        layers = []
        layers.append(self.ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(self.ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x: 输入张量，形状为 [batch_size, in_channels, height, width]

        返回:
            out: 输出张量，形状为 [batch_size, num_classes]
        """
        # 初始卷积
        x = F.relu(self.bn1(self.conv1(x)))

        # 残差块
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 全局平均池化和全连接层
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class SimpleMLP(nn.Module):
    """
    简单的多层感知机

    适用于简单数据集的NTK计算。

    架构:
        - Linear(input_dim, hidden_dim) + ReLU
        - Linear(hidden_dim, hidden_dim) + ReLU
        - Linear(hidden_dim, num_classes)
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        num_classes: int = 10,
        num_layers: int = 2
    ):
        """
        参数:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            num_classes: 输出类别数
            num_layers: 隐藏层数量
        """
        super(SimpleMLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # 构建网络层
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, num_classes))

        self.network = nn.Sequential(*layers)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x: 输入张量，形状为 [batch_size, input_dim] 或 [batch_size, 1, height, width]

        返回:
            out: 输出张量，形状为 [batch_size, num_classes]
        """
        # 如果输入是图像，展平
        if x.dim() == 4:
            x = x.view(x.size(0), -1)

        x = self.network(x)
        return x


# 模型工厂函数
def create_model(
    model_type: str,
    **kwargs
) -> nn.Module:
    """
    创建预定义模型的工厂函数

    参数:
        model_type: 模型类型 ('cnn', 'resnet', 'mlp')
        **kwargs: 模型特定参数

    返回:
        model: PyTorch模型
    """
    model_type = model_type.lower()

    if model_type == 'cnn':
        return SimpleCNN(**kwargs)
    elif model_type == 'resnet':
        return SimpleResNet(**kwargs)
    elif model_type == 'mlp':
        return SimpleMLP(**kwargs)
    else:
        raise ValueError(
            f"未知的模型类型: {model_type}. "
            f"可选类型: 'cnn', 'resnet', 'mlp'"
        )
