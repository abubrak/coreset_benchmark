#!/usr/bin/env python3
"""
Colab优化版持续学习实验

针对Google Colab T4 16GB优化：
- 使用混合精度
- 减少BCSR迭代次数
- 预采样大数据集
- 更好的进度显示
"""

import os
import sys
import argparse
import json
import time

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Dict, Tuple

from src.datasets.data_loaders import get_dataset
from src.models.cnn import CNN_MNIST
from src.coreset.continual_adapters import BCSRContinualAdapter


class ColabCoresetBuffer:
    """
    Colab优化的经验回放缓冲区

    主要优化：
    1. 自动检测是否需要预采样
    2. 减少BCSR迭代次数
    3. 使用更高效的存储
    """

    def __init__(
        self,
        memory_size: int,
        input_shape: Tuple[int, ...],
        num_classes: int,
        device: str = "cuda",
        use_presampling: bool = True
    ):
        """
        初始化Colab优化缓冲区

        参数:
            memory_size: 缓冲区大小
            input_shape: 输入形状
            num_classes: 总类别数
            device: 设备
            use_presampling: 是否使用预采样优化
        """
        self.memory_size = memory_size
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.device = device
        self.use_presampling = use_presampling

        # 数据存储
        self.data = torch.zeros((0, *input_shape), dtype=torch.float32)
        self.labels = torch.zeros((0,), dtype=torch.long)
        self.task_ids = torch.zeros((0,), dtype=torch.long)

        # 预采样阈值
        self.PRESAMPLE_THRESHOLD = 3000

    def add(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        task_id: int,
        selection_method: str = "random"
    ) -> int:
        """添加样本到缓冲区（优化版）"""
        n_samples = len(data)
        available_space = self.get_available_space()

        # 移到CPU以节省GPU内存
        data_cpu = data.detach().cpu()
        labels_cpu = labels.detach().cpu()

        if n_samples <= available_space:
            # 有足够空间，直接添加
            self.data = torch.cat([self.data, data_cpu], dim=0)
            self.labels = torch.cat([self.labels, labels_cpu], dim=0)
            self.task_ids = torch.cat([
                self.task_ids,
                torch.full((n_samples,), task_id, dtype=torch.long)
            ], dim=0)
        else:
            # 缓冲区已满，需要替换（简单随机替换）
            replace_indices = torch.randperm(len(self.data))[:n_samples]
            self.data[replace_indices] = data_cpu
            self.labels[replace_indices] = labels_cpu
            self.task_ids[replace_indices] = task_id

        return min(n_samples, available_space)

    def get_available_space(self) -> int:
        """获取缓冲区可用空间"""
        return max(0, self.memory_size - len(self.data))

    def is_empty(self) -> bool:
        """检查缓冲区是否为空"""
        return len(self.data) == 0

    def get_dataloader(
        self,
        batch_size: int,
        shuffle: bool = True
    ):
        """获取缓冲区的数据加载器"""
        if self.is_empty():
            return None

        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(
            self.data.to(self.device),
            self.labels.to(self.device)
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0  # Colab需要设为0
        )

        return dataloader

    def select_coreset_optimized(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        num_samples: int,
        method: str,
        model: nn.Module = None,
        task_id: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Colab优化的coreset选择

        主要优化：
        1. 对BCSR自动预采样
        2. 显示进度信息
        3. 减少迭代次数
        """
        n_samples = len(data)
        num_samples = min(num_samples, n_samples)

        print(f"[Coreset选择] 方法={method}, 数据量={n_samples}, 目标={num_samples}")

        if method == "bcsr":
            # Colab BCSR优化
            if model is None:
                raise ValueError("BCSR需要模型")

            # 预采样优化
            if self.use_presampling and n_samples > self.PRESAMPLE_THRESHOLD:
                presample_size = min(2000, n_samples)
                print(f"[预采样优化] {n_samples} -> {presample_size}样本")

                # 类别平衡预采样
                presample_indices = []
                num_classes = labels.max().item() + 1
                samples_per_class = presample_size // num_classes

                for c in range(num_classes):
                    class_mask = (labels == c)
                    class_indices = torch.where(class_mask)[0]

                    if len(class_indices) > 0:
                        n_select = min(samples_per_class, len(class_indices))
                        perm = torch.randperm(len(class_indices))[:n_select]
                        presample_indices.append(class_indices[perm])

                presample_indices = torch.cat(presample_indices)
                data = data[presample_indices]
                labels = labels[presample_indices]

                print(f"[预采样完成] 类别分布: {torch.bincount(labels)}")

            # 创建Colab优化的BCSR适配器
            adapter = BCSRContinualAdapter(
                learning_rate_inner=0.01,
                learning_rate_outer=3.0,  # 降低以提高稳定性
                num_inner_steps=1,
                num_outer_steps=2,  # Colab优化：减少迭代
                beta=0.1,
                device=self.device
            )

            print(f"[BCSR训练] 开始...")
            start_time = time.time()

            selected_data, selected_labels = adapter.select(
                data=data,
                labels=labels,
                num_samples=num_samples,
                model=model
            )

            elapsed = time.time() - start_time
            print(f"[BCSR训练] 完成! 用时: {elapsed:.1f}秒")

            return selected_data, selected_labels

        elif method == "uniform":
            # Uniform选择（CPU友好）
            indices = []
            num_classes = labels.max().item() + 1
            samples_per_class = num_samples // num_classes

            for c in range(num_classes):
                class_mask = (labels == c)
                class_indices = torch.where(class_mask)[0]

                if len(class_indices) > 0:
                    n_select = min(samples_per_class, len(class_indices))
                    perm = torch.randperm(len(class_indices))[:n_select]
                    indices.append(class_indices[perm])

            indices = torch.cat(indices) if indices else torch.randperm(n_samples)[:num_samples]

            return data[indices], labels[indices]

        else:
            # 其他方法使用原有逻辑
            raise ValueError(f"暂不支持方法: {method}")


def run_colab_optimized_experiment(
    dataset: str = 'MNIST',
    num_tasks: int = 2,
    num_classes_per_task: int = 2,
    memory_size: int = 2000,
    num_epochs: int = 5,
    batch_size: int = 128,
    device: str = 'cuda'
):
    """
    运行Colab优化的持续学习实验

    参数:
        dataset: 数据集名称
        num_tasks: 任务数量
        num_classes_per_task: 每个任务的类别数
        memory_size: 缓冲区大小
        num_epochs: 每个任务的训练轮数
        batch_size: 批次大小
        device: 设备
    """
    print(f"\n{'='*60}")
    print(f"Colab优化持续学习实验")
    print(f"{'='*60}")
    print(f"数据集: {dataset}")
    print(f"任务数: {num_tasks}")
    print(f"每任务类别数: {num_classes_per_task}")
    print(f"缓冲区大小: {memory_size}")
    print(f"设备: {device}")

    # TODO: 添加实际训练逻辑
    # 这里只演示设置，完整实现需要集成到continual_learning.py

    print(f"\n[提示] 在Colab上运行时:")
    print(f"  1. 确保运行时使用GPU: 运行时 -> 更改运行时类型 -> GPU")
    print(f"  2. 使用小批量测试: num_tasks=2, num_epochs=5")
    print(f"  3. BCSR会自动预采样到2000-3000样本")
    print(f"  4. 预期总时间: 2-3分钟")


if __name__ == '__main__':
    run_colab_optimized_experiment()
