"""
简单的CNN模型用于MNIST和CIFAR数据集
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_MNIST(nn.Module):
    """
    用于MNIST数据集的简单CNN模型
    输入: 1x28x28的灰度图像
    """

    def __init__(self, num_classes=10):
        super(CNN_MNIST, self).__init__()

        # 第一个卷积块
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # 第二个卷积块
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # 第三个卷积块
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # 池化层
        self.pool = nn.MaxPool2d(2, 2)

        # 全连接层
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)

        # Dropout层
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 第一个卷积块
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 28x28x32 -> 14x14x32

        # 第二个卷积块
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 14x14x64 -> 7x7x64

        # 第三个卷积块
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 7x7x128 -> 3x3x128

        # 展平
        x = x.view(-1, 128 * 3 * 3)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class CNN_CIFAR(nn.Module):
    """
    用于CIFAR-10/CIFAR-100数据集的CNN模型
    输入: 3x32x32的RGB图像
    """

    def __init__(self, num_classes=10):
        super(CNN_CIFAR, self).__init__()

        # 第一个卷积块
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # 第二个卷积块
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # 第三个卷积块
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # 第四个卷积块
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        # 池化层
        self.pool = nn.MaxPool2d(2, 2)

        # 全连接层
        self.fc1 = nn.Linear(512 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, num_classes)

        # Dropout层
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 第一个卷积块
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32x32x64 -> 16x16x64

        # 第二个卷积块
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 16x16x128 -> 8x8x128

        # 第三个卷积块
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 8x8x256 -> 4x4x256

        # 第四个卷积块
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # 4x4x512 -> 2x2x512

        # 展平
        x = x.view(-1, 512 * 2 * 2)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
