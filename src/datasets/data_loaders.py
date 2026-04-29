"""
数据加载器模块
提供数据集获取和处理的函数
"""

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
import numpy as np
from typing import Tuple, Dict, List, Optional
import warnings


# 数据集归一化统计量
DATASET_STATS = {
    'MNIST': {
        'mean': (0.1307,),
        'std': (0.3081,),
        'num_classes': 10,
        'num_channels': 1,
        'img_size': 28
    },
    'CIFAR10': {
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2470, 0.2435, 0.2616),
        'num_classes': 10,
        'num_channels': 3,
        'img_size': 32
    },
    'CIFAR100': {
        'mean': (0.5071, 0.4867, 0.4408),
        'std': (0.2675, 0.2565, 0.2761),
        'num_classes': 100,
        'num_channels': 3,
        'img_size': 32
    }
}


def get_dataset(
    dataset_name: str,
    root: str = './data',
    train: bool = True,
    download: bool = True,
    transform: Optional[transforms.Compose] = None
) -> torch.utils.data.Dataset:
    """
    获取数据集

    参数:
        dataset_name: 数据集名称 ('MNIST', 'CIFAR10', 'CIFAR100')
        root: 数据集根目录
        train: 是否为训练集
        download: 是否下载数据集
        transform: 自定义数据变换

    返回:
        数据集对象
    """
    stats = DATASET_STATS.get(dataset_name)
    if stats is None:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    # 如果没有提供自定义变换，使用默认变换
    if transform is None:
        if train:
            # 训练集：数据增强 + 归一化
            if dataset_name == 'MNIST':
                transform = transforms.Compose([
                    transforms.RandomCrop(28, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(stats['mean'], stats['std'])
                ])
            else:  # CIFAR
                transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(stats['mean'], stats['std'])
                ])
        else:
            # 测试集：仅归一化
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(stats['mean'], stats['std'])
            ])

    # 获取数据集
    if dataset_name == 'MNIST':
        dataset = datasets.MNIST(root=root, train=train,
                                download=download, transform=transform)
    elif dataset_name == 'CIFAR10':
        dataset = datasets.CIFAR10(root=root, train=train,
                                  download=download, transform=transform)
    elif dataset_name == 'CIFAR100':
        dataset = datasets.CIFAR100(root=root, train=train,
                                   download=download, transform=transform)
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    return dataset


def get_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True
) -> DataLoader:
    """
    创建数据加载器

    参数:
        dataset: 数据集对象
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 数据加载的工作进程数
        pin_memory: 是否将数据固定在内存中

    返回:
        DataLoader对象
    """
    # Windows系统下num_workers设为0以避免多进程问题
    import platform
    if platform.system() == 'Windows':
        num_workers = 0

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return dataloader


def split_dataset_by_class(
    dataset: torch.utils.data.Dataset,
    num_classes: int,
    num_samples_per_class: int
) -> torch.utils.data.Dataset:
    """
    按类别均匀分割数据集，每个类别选择指定数量的样本

    参数:
        dataset: 原始数据集
        num_classes: 类别数
        num_samples_per_class: 每个类别的样本数

    返回:
        分割后的数据集
    """
    # 获取数据集的标签
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
    elif hasattr(dataset, 'labels'):
        targets = dataset.labels
    else:
        # 如果没有直接访问标签的方法，需要遍历数据集
        targets = [dataset[i][1] for i in range(len(dataset))]
        targets = torch.tensor(targets)

    # 为每个类别选择样本索引
    selected_indices = []
    for class_idx in range(num_classes):
        # 找到属于当前类别的所有样本索引
        class_indices = torch.where(targets == class_idx)[0]

        # 随机选择指定数量的样本
        if len(class_indices) < num_samples_per_class:
            warnings.warn(f"类别 {class_idx} 的样本数 ({len(class_indices)}) "
                         f"少于请求的数量 ({num_samples_per_class})")
            selected_indices.extend(class_indices.tolist())
        else:
            perm = torch.randperm(len(class_indices))[:num_samples_per_class]
            selected_indices.extend(class_indices[perm].tolist())

    # 创建Subset数据集
    subset_dataset = Subset(dataset, selected_indices)

    return subset_dataset


def get_split_dataset(
    dataset_name: str,
    root: str = './data',
    train: bool = True,
    num_samples_per_class: Optional[int] = None,
    download: bool = True
) -> Tuple[torch.utils.data.Dataset, Dict]:
    """
    获取分割后的数据集

    参数:
        dataset_name: 数据集名称
        root: 数据集根目录
        train: 是否为训练集
        num_samples_per_class: 每个类别的样本数，None表示使用全部数据
        download: 是否下载数据集

    返回:
        (数据集, 数据集信息字典)
    """
    # 获取数据集统计信息
    stats = DATASET_STATS.get(dataset_name)
    if stats is None:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    # 获取完整数据集
    full_dataset = get_dataset(dataset_name, root, train, download)

    # 如果指定了每个类别的样本数，进行分割
    if num_samples_per_class is not None:
        dataset = split_dataset_by_class(
            full_dataset,
            stats['num_classes'],
            num_samples_per_class
        )
    else:
        dataset = full_dataset

    # 返回数据集和信息
    info = {
        'name': dataset_name,
        'num_classes': stats['num_classes'],
        'num_channels': stats['num_channels'],
        'img_size': stats['img_size'],
        'num_samples': len(dataset),
        'mean': stats['mean'],
        'std': stats['std']
    }

    return dataset, info


# --- 全局参数：各数据集完整训练集大小，用于 tile 重复计算 ---
DATASET_TRAIN_SIZE = {
    'MNIST': 60000,
    'CIFAR10': 50000,
    'CIFAR100': 50000,
}


def get_coreset_train_loader(
    train_dataset: torch.utils.data.Dataset,
    indices: np.ndarray,
    coreset_size: int,
    dataset_name: str = 'CIFAR10',
    batch_size: int = 64,
    num_workers: int = 2,
) -> DataLoader:
    """
    为 coreset 子集构建训练 DataLoader，使用 tile 重复机制。

    关键机制：将 coreset 索引通过 np.tile 重复 N 次，
    使每个 epoch 的样本总数 ≈ 完整训练集大小（50,000 / 60,000）。
    配合数据增强（RandomCrop + RandomHorizontalFlip），
    同一张图片每次重复会产生不同的裁剪/翻转变体。

    等价于原始代码 data_summarization/resnet_cifar.py:93-102 的 get_train_loader。

    Args:
        train_dataset: 带数据增强的完整训练集
        indices: coreset 选出的样本索引
        coreset_size: coreset 大小（= len(indices)）
        dataset_name: 数据集名称，用于查询完整集大小
        batch_size: batch 大小
        num_workers: DataLoader 工作线程数

    Returns:
        配置好的 DataLoader
    """
    full_train_size = DATASET_TRAIN_SIZE.get(dataset_name, 50000)
    nr_repeats = full_train_size // coreset_size  # 210样本时 ≈ 238 (CIFAR)

    # tile 重复索引，使每 epoch 总样本 ≈ full_train_size
    inds_repeated = np.tile(indices, nr_repeats)
    # 截断至 batch_size 的整数倍
    truncate_len = len(inds_repeated) // batch_size * batch_size
    inds_repeated = inds_repeated[:truncate_len]

    print(f"[CoresetLoader] {coreset_size} unique samples x {nr_repeats} repeats "
          f"= {len(inds_repeated)} total samples/epoch "
          f"({len(inds_repeated) // batch_size} iterations)")

    subset = torch.utils.data.Subset(train_dataset, inds_repeated)

    # Windows系统下num_workers设为0
    import platform
    if platform.system() == 'Windows':
        num_workers = 0

    return DataLoader(subset, batch_size=batch_size, shuffle=True,
                      pin_memory=True, num_workers=num_workers)
