"""
CSReL v2 辅助数据集类模块

该模块包含用于 CSReL v2 算法的辅助数据集类，提供简化的数据集接口
和样本移除功能，用于增量选择过程。
"""

import pickle
import random
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class SimplePILDataset(Dataset):
    """
    简化的 PIL 图像数据集

    该数据集从 pickle 文件加载预保存的数据，支持数据增强。

    参数
    ----
    data_file : str
        数据文件路径（pickle 格式）
    transform : Optional[transforms.Compose], default=None
        数据增强变换
    shuffle : bool, default=True
        是否在加载时打乱数据顺序
    """

    def __init__(
        self,
        data_file: str,
        transform: Optional[transforms.Compose] = None,
        shuffle: bool = True
    ):
        self.data_file = data_file
        self.transform = transform
        self.data = []
        self.shuffle = shuffle

        # 从文件加载数据
        self._load_data()

        # 打乱数据
        if self.shuffle:
            random.shuffle(self.data)

    def _load_data(self) -> None:
        """从 pickle 文件加载数据"""
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data file not found: {self.data_file}")

        with open(self.data_file, 'rb') as fr:
            while True:
                try:
                    data_item = pickle.load(fr)
                    self.data.append(data_item)
                except EOFError:
                    break

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        获取单个样本

        返回
        ----
        Tuple[int, torch.Tensor, torch.Tensor]
            (id, sample, label)
        """
        data_item = self.data[idx]
        d_id = data_item[0]
        sample = data_item[1]
        label = data_item[2]

        # 应用数据增强
        if self.transform is not None:
            # 如果是 PIL Image，直接应用变换
            if hasattr(sample, 'convert'):
                sample = self.transform(sample)
            # 如果是 tensor，可能需要先转换
            elif isinstance(sample, torch.Tensor):
                # 这里可以根据需要添加变换逻辑
                pass

        return d_id, sample, label


class SimpleRandomDataset(Dataset):
    """
    简单随机采样数据集

    该数据集从给定的数据和标签中随机采样，用于 baseline 对比。

    参数
    ----
    data : List[torch.Tensor]
        数据列表
    labels : List[int]
        标签列表
    transform : Optional[transforms.Compose], default=None
        数据增强变换
    """

    def __init__(
        self,
        data: List[torch.Tensor],
        labels: List[int],
        transform: Optional[transforms.Compose] = None
    ):
        self.data = data
        self.labels = labels
        self.transform = transform

        # 创建索引列表
        self.indices = list(range(len(data)))

        # 打乱索引
        random.shuffle(self.indices)

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        获取单个样本

        返回
        ----
        Tuple[int, torch.Tensor, torch.Tensor]
            (id, sample, label)
        """
        real_idx = self.indices[idx]
        sample = self.data[real_idx]
        label = self.labels[real_idx]

        # 应用数据增强
        if self.transform is not None:
            sample = self.transform(sample)

        return real_idx, sample, label


class SimplePILDatasetWithRemoval(SimplePILDataset):
    """
    支持样本移除的 PIL 数据集

    该数据集继承自 SimplePILDataset，添加了样本移除功能，
    用于增量选择过程中移除已选样本。

    参数
    ----
    data_file : str
        数据文件路径（pickle 格式）
    transform : Optional[transforms.Compose], default=None
        数据增强变换
    shuffle : bool, default=True
        是否在加载时打乱数据顺序
    """

    def __init__(
        self,
        data_file: str,
        transform: Optional[transforms.Compose] = None,
        shuffle: bool = True
    ):
        super().__init__(data_file, transform, shuffle)
        # 维护一个有效索引集合
        self.valid_indices = set(range(len(self.data)))

    def remove_samples(self, indices_to_remove: List[int]) -> None:
        """
        从数据集中移除样本

        参数
        ----
        indices_to_remove : List[int]
            要移除的样本 ID 列表
        """
        for idx in indices_to_remove:
            if idx in self.valid_indices:
                self.valid_indices.remove(idx)

    def get_valid_data(self) -> List[tuple]:
        """
        获取所有有效样本

        返回
        ----
        List[tuple]
            有效样本列表
        """
        valid_data = []
        for i, data_item in enumerate(self.data):
            if i in self.valid_indices:
                valid_data.append(data_item)
        return valid_data

    def __len__(self) -> int:
        """返回有效数据集大小"""
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        获取单个有效样本

        注意：这里的 idx 是在有效样本中的索引，需要转换为原始索引

        返回
        ----
        Tuple[int, torch.Tensor, torch.Tensor]
            (id, sample, label)
        """
        # 获取所有有效索引并排序
        sorted_indices = sorted(self.valid_indices)
        real_idx = sorted_indices[idx]

        data_item = self.data[real_idx]
        d_id = data_item[0]
        sample = data_item[1]
        label = data_item[2]

        # 应用数据增强
        if self.transform is not None:
            if hasattr(sample, 'convert'):
                sample = self.transform(sample)
            elif isinstance(sample, torch.Tensor):
                pass

        return d_id, sample, label


# 需要导入 os 模块
import os
