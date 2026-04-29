"""
持续学习实验脚本

实现基于coreset的持续学习，使用经验回放方法。
支持多种coreset选择策略和缓冲区管理方法。

主要功能：
1. CoresetBuffer: 持续学习的内存缓冲区
2. train_task: 在当前任务上训练模型
3. evaluate_all_tasks: 在所有已见任务上评估模型
4. run_continual_learning: 运行完整的持续学习实验
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm
import json
from typing import List, Dict, Tuple, Optional
from copy import deepcopy
import warnings

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.data_loaders import get_dataset, get_dataloader, DATASET_STATS
from src.models.cnn import CNN_MNIST
from src.models.resnet import ResNet18
from src.baselines.baseline_methods import get_baseline
from src.coreset.selection_functions import (
    select_by_loss_diff,
    select_by_margin,
    select_by_gradient_norm
)
# Import coreset adapters (CSReL and Bilevel adapters will be added in Tasks 3 and 5)
from src.coreset.continual_adapters import (
    BCSRContinualAdapter,
    CSReLContinualAdapter
)


class CoresetBuffer:
    """
    持续学习的内存缓冲区

    用于存储从过往任务中选择的代表性样本，
    在训练新任务时进行回放以减轻灾难性遗忘。
    """

    def __init__(
        self,
        memory_size: int,
        input_shape: Tuple[int, ...],
        num_classes: int,
        device: str = "cuda"
    ):
        """
        初始化缓冲区

        参数:
            memory_size: 缓冲区最大容量
            input_shape: 输入数据的形状 (C, H, W)
            num_classes: 总类别数
            device: 设备 ('cuda' or 'cpu')
        """
        self.memory_size = memory_size
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.device = device

        # 使用张量存储数据
        self.data = torch.zeros((0, *input_shape), dtype=torch.float32)
        self.labels = torch.zeros((0,), dtype=torch.long)
        self.task_ids = torch.zeros((0,), dtype=torch.long)

        # 每个任务的样本计数
        self.task_counts = {}

    def add(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        task_id: int,
        selection_method: str = "random"
    ) -> int:
        """
        添加样本到缓冲区

        如果缓冲区已满，会根据选择策略替换现有样本。

        参数:
            data: 要添加的数据 (n_samples, C, H, W)
            labels: 对应的标签 (n_samples,)
            task_id: 任务ID
            selection_method: 选择策略 ('random', 'loss', 'margin', 'gradient')

        返回:
            实际添加的样本数
        """
        n_samples = len(data)
        available_space = self.get_available_space()

        # 将数据移到CPU以节省GPU内存
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
            # 缓冲区已满，需要替换
            if selection_method == "random":
                # 随机替换
                replace_indices = torch.randperm(len(self.data))[:n_samples]
                self.data[replace_indices] = data_cpu
                self.labels[replace_indices] = labels_cpu
                self.task_ids[replace_indices] = task_id

            elif selection_method == "loss":
                # 基于损失替换（需要模型）
                warnings.warn("Loss-based selection requires model, using random instead")
                replace_indices = torch.randperm(len(self.data))[:n_samples]
                self.data[replace_indices] = data_cpu
                self.labels[replace_indices] = labels_cpu
                self.task_ids[replace_indices] = task_id

            elif selection_method == "reservoir":
                # Reservoir sampling
                for i in range(n_samples):
                    if len(self.data) < self.memory_size:
                        # 还没满，直接添加
                        self.data = torch.cat([self.data, data_cpu[i:i+1]], dim=0)
                        self.labels = torch.cat([self.labels, labels_cpu[i:i+1]], dim=0)
                        self.task_ids = torch.cat([
                            self.task_ids,
                            torch.full((1,), task_id, dtype=torch.long)
                        ], dim=0)
                    else:
                        # 随机替换
                        j = torch.randint(0, len(self.data), (1,)).item()
                        self.data[j] = data_cpu[i]
                        self.labels[j] = labels_cpu[i]
                        self.task_ids[j] = task_id
                return n_samples

        # 更新任务计数
        if task_id not in self.task_counts:
            self.task_counts[task_id] = 0
        self.task_counts[task_id] += min(n_samples, available_space)

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
        shuffle: bool = True,
        num_workers: int = 0
    ) -> DataLoader:
        """
        获取缓冲区的数据加载器

        参数:
            batch_size: 批次大小
            shuffle: 是否打乱数据
            num_workers: 数据加载的工作进程数

        返回:
            DataLoader对象
        """
        if self.is_empty():
            return None

        dataset = TensorDataset(
            self.data.to(self.device),
            self.labels.to(self.device)
        )

        # Windows系统下num_workers设为0
        import platform
        if platform.system() == 'Windows':
            num_workers = 0

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )

        return dataloader

    def get_class_balance(self) -> Dict[int, int]:
        """获取每个类别的样本数"""
        unique_labels, counts = torch.unique(self.labels, return_counts=True)
        return {int(label): int(count) for label, count in zip(unique_labels, counts)}

    def select_coreset(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        num_samples: int,
        method: str = "random",
        model: Optional[nn.Module] = None,
        task_id: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从数据中选择coreset样本

        参数:
            data: 输入数据
            labels: 标签
            num_samples: 要选择的样本数
            method: 选择方法 ('random', 'uniform', 'loss', 'margin', 'gradient',
                               'bcsr', 'csrel', 'bilevel')
            model: 用于基于模型的选择方法（BCSR/CSReL/Bilevel需要）
            task_id: 任务ID，用于配置不同方法的参数

        返回:
            选择的样本和标签
        """
        n_total = len(data)
        num_samples = min(num_samples, n_total)

        if method == "random":
            # 随机选择
            indices = torch.randperm(n_total)[:num_samples]

        elif method == "uniform":
            # 均匀选择（按类别）
            selected_indices = []
            num_classes = labels.max().item() + 1
            samples_per_class = num_samples // num_classes

            for c in range(num_classes):
                class_mask = (labels == c)
                class_indices = torch.where(class_mask)[0]

                if len(class_indices) > 0:
                    n_select = min(samples_per_class, len(class_indices))
                    perm = torch.randperm(len(class_indices))[:n_select]
                    selected_indices.append(class_indices[perm])

            indices = torch.cat(selected_indices) if selected_indices else torch.randperm(n_total)[:num_samples]

        elif method in ["loss", "margin", "gradient"]:
            # 基于模型的选择方法
            if model is None:
                warnings.warn(f"{method}-based selection requires model, using random instead")
                indices = torch.randperm(n_total)[:num_samples]
            else:
                model.eval()
                with torch.no_grad():
                    if method == "loss":
                        # 基于损失选择
                        outputs = model(data)
                        losses = nn.functional.cross_entropy(
                            outputs, labels, reduction='none'
                        )
                        _, indices = torch.topk(losses, num_samples)

                    elif method == "margin":
                        # 基于margin选择
                        outputs = model(data)
                        from src.coreset.selection_functions import select_by_margin
                        indices = select_by_margin(
                            outputs, labels, num_samples,
                            class_balance=True, num_classes=labels.max().item()+1
                        )

                    elif method == "gradient":
                        # 基于梯度范数选择
                        indices = select_by_gradient_norm(
                            model, data, labels, num_samples,
                            class_balance=True, num_classes=labels.max().item()+1
                        )

                indices = indices.to(data.device)

        elif method == "bcsr":
            # BCSR (Bilevel Coreset Selection with Reweighting)
            if model is None:
                raise ValueError("BCSR method requires a model")

            # Create adapter with default parameters
            adapter = BCSRContinualAdapter(
                learning_rate_inner=0.01,
                learning_rate_outer=5.0,
                num_inner_steps=1,
                num_outer_steps=5,
                beta=0.1,
                device=data.device
            )

            selected_data, selected_labels = adapter.select(
                data=data,
                labels=labels,
                num_samples=num_samples,
                model=model
            )

            # Return directly (already tensors)
            return selected_data, selected_labels

        elif method == "csrel":
            # CSReL (Classwise Spatial Representation Learning)
            if model is None:
                raise ValueError("CSReL method requires a model")

            # Create adapter with default parameters
            adapter = CSReLContinualAdapter(
                num_epochs=20,  # Reduced for faster training
                learning_rate=0.001,
                batch_size=128,
                selection_ratio=num_samples / len(data),
                class_balance=True,
                device=data.device
            )

            selected_data, selected_labels = adapter.select(
                data=data,
                labels=labels,
                num_samples=num_samples,
                model=model
            )

            # Return directly (already tensors)
            return selected_data, selected_labels

        else:
            raise ValueError(f"Unknown selection method: {method}")

        return data[indices], labels[indices]


def train_task(
    model: nn.Module,
    train_loader: DataLoader,
    buffer: CoresetBuffer,
    task_id: int,
    num_epochs: int,
    learning_rate: float,
    device: str,
    buffer_ratio: float = 0.5,
    selection_method: str = "random"
) -> Dict[str, float]:
    """
    在当前任务上训练模型，使用缓冲区回放

    参数:
        model: 要训练的模型
        train_loader: 当前任务的训练数据加载器
        buffer: 经验回放缓冲区
        task_id: 当前任务ID
        num_epochs: 训练轮数
        learning_rate: 学习率
        device: 设备
        buffer_ratio: 缓冲区数据与任务数据的比例
        selection_method: coreset选择方法

    返回:
        训练统计信息字典
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    # 获取缓冲区数据加载器
    buffer_loader = buffer.get_dataloader(
        batch_size=train_loader.batch_size,
        shuffle=True
    )

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        # 创建缓冲区数据的迭代器
        buffer_iter = iter(buffer_loader) if buffer_loader is not None else None

        pbar = tqdm(train_loader, desc=f"Task {task_id} Epoch {epoch+1}/{num_epochs}")

        for batch_idx, (data, labels) in enumerate(pbar):
            data, labels = data.to(device), labels.to(device)

            # 当前任务的损失
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)

            # 添加缓冲区回放损失
            if buffer_loader is not None and not buffer.is_empty():
                try:
                    buffer_data, buffer_labels = next(buffer_iter)
                except StopIteration:
                    buffer_iter = iter(buffer_loader)
                    buffer_data, buffer_labels = next(buffer_iter)

                buffer_data, buffer_labels = buffer_data.to(device), buffer_labels.to(device)

                # 缓冲区损失
                buffer_outputs = model(buffer_data)
                buffer_loss = criterion(buffer_outputs, buffer_labels)

                # 组合损失
                loss = (1 - buffer_ratio) * loss + buffer_ratio * buffer_loss

            loss.backward()
            optimizer.step()

            # 统计
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            epoch_total += labels.size(0)
            epoch_correct += predicted.eq(labels).sum().item()

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{epoch_loss/(batch_idx+1):.4f}",
                'acc': f"{100.*epoch_correct/epoch_total:.2f}%"
            })

        train_loss += epoch_loss
        train_correct += epoch_correct
        train_total += epoch_total

    # 计算平均准确率
    avg_loss = train_loss / (num_epochs * len(train_loader))
    avg_acc = 100. * train_correct / train_total

    return {
        'loss': avg_loss,
        'accuracy': avg_acc
    }


def evaluate_task(
    model: nn.Module,
    test_loader: DataLoader,
    device: str
) -> Dict[str, float]:
    """
    在单个任务上评估模型

    参数:
        model: 要评估的模型
        test_loader: 测试数据加载器
        device: 设备

    返回:
        评估结果字典
    """
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)

            outputs = model(data)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    avg_loss = test_loss / len(test_loader)
    avg_acc = 100. * test_correct / test_total

    return {
        'loss': avg_loss,
        'accuracy': avg_acc
    }


def evaluate_all_tasks(
    model: nn.Module,
    test_loaders: List[DataLoader],
    device: str
) -> List[float]:
    """
    在所有已见任务上评估模型

    参数:
        model: 要评估的模型
        test_loaders: 所有任务的测试数据加载器列表
        device: 设备

    返回:
        每个任务的准确率列表
    """
    accuracies = []

    for task_id, test_loader in enumerate(test_loaders):
        result = evaluate_task(model, test_loader, device)
        accuracies.append(result['accuracy'])
        print(f"  Task {task_id}: {result['accuracy']:.2f}%")

    return accuracies


def compute_forgetting_measure(
    accuracy_matrix: np.ndarray
) -> float:
    """
    计算遗忘度量

    遗忘度量定义为：在每个任务学习后，
    该任务的最终准确率与最高准确率之间的平均差异

    参数:
        accuracy_matrix: 准确率矩阵 (num_tasks, num_tasks)
                        accuracy_matrix[i, j] 表示学习任务i后在任务j上的准确率

    返回:
        遗忘度量
    """
    num_tasks = accuracy_matrix.shape[0]
    forgetting_measures = []

    for task_j in range(num_tasks):
        # 找到任务j在所有学习后的最高准确率
        max_acc = np.max(accuracy_matrix[:, task_j])

        # 最终准确率（学习所有任务后）
        final_acc = accuracy_matrix[-1, task_j]

        # 计算遗忘
        if max_acc > 0:
            forgetting = max_acc - final_acc
        else:
            forgetting = 0.0

        forgetting_measures.append(forgetting)

    # 平均遗忘
    avg_forgetting = np.mean(forgetting_measures)

    return avg_forgetting


def compute_average_accuracy(
    accuracy_matrix: np.ndarray
) -> float:
    """
    计算平均准确率

    参数:
        accuracy_matrix: 准确率矩阵 (num_tasks, num_tasks)

    返回:
        平均准确率
    """
    # 最后的平均准确率
    final_accuracies = accuracy_matrix[-1, :]
    return np.mean(final_accuracies)


def create_task_datasets(
    dataset_name: str,
    num_tasks: int,
    num_classes_per_task: int,
    batch_size: int,
    data_root: str = './data'
) -> Tuple[List[DataLoader], List[DataLoader], int, Tuple[int, ...]]:
    """
    创建任务增量学习的数据集

    将数据集的类别划分为多个任务，每个任务包含一组不重叠的类别。

    参数:
        dataset_name: 数据集名称
        num_tasks: 任务数量
        num_classes_per_task: 每个任务的类别数
        batch_size: 批次大小
        data_root: 数据根目录

    返回:
        (train_loaders, test_loaders, num_classes, input_shape)
    """
    # 获取数据集统计信息
    stats = DATASET_STATS.get(dataset_name)
    if stats is None:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    num_classes = stats['num_classes']
    input_shape = (stats['num_channels'], stats['img_size'], stats['img_size'])

    if num_classes < num_tasks * num_classes_per_task:
        raise ValueError(
            f"数据集只有{num_classes}个类别，"
            f"无法创建{num_tasks}个任务，每个任务{num_classes_per_task}个类别"
        )

    # 获取训练和测试数据集
    train_dataset = get_dataset(dataset_name, data_root, train=True, download=True)
    test_dataset = get_dataset(dataset_name, data_root, train=False, download=True)

    # 按类别划分数据
    train_loaders = []
    test_loaders = []

    import platform
    num_workers = 0 if platform.system() == 'Windows' else 2

    for task_id in range(num_tasks):
        # 计算当前任务的类别范围
        start_class = task_id * num_classes_per_task
        end_class = start_class + num_classes_per_task
        task_classes = list(range(start_class, end_class))

        print(f"创建任务 {task_id}: 类别 {task_classes}")

        # 筛选当前任务的样本
        train_indices = []
        test_indices = []

        # 获取标签
        if hasattr(train_dataset, 'targets'):
            train_targets = train_dataset.targets
            test_targets = test_dataset.targets
        else:
            train_targets = [train_dataset[i][1] for i in range(len(train_dataset))]
            test_targets = [test_dataset[i][1] for i in range(len(test_dataset))]

        # 筛选索引
        for idx, label in enumerate(train_targets):
            if label in task_classes:
                train_indices.append(idx)

        for idx, label in enumerate(test_targets):
            if label in task_classes:
                test_indices.append(idx)

        # 创建子集
        train_subset = Subset(train_dataset, train_indices)
        test_subset = Subset(test_dataset, test_indices)

        # 重新映射标签到0..num_classes_per_task-1
        class_mapping = {old_class: i for i, old_class in enumerate(task_classes)}

        # 创建数据加载器
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    return train_loaders, test_loaders, num_classes, input_shape


def run_continual_learning(args):
    """
    运行完整的持续学习实验

    参数:
        args: 命令行参数
    """
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 创建任务数据集
    print(f"\n创建任务增量学习数据集: {args.dataset}")
    print(f"任务数量: {args.num_tasks}")
    print(f"每个任务的类别数: {args.num_classes_per_task}")

    train_loaders, test_loaders, num_classes, input_shape = create_task_datasets(
        dataset_name=args.dataset,
        num_tasks=args.num_tasks,
        num_classes_per_task=args.num_classes_per_task,
        batch_size=args.batch_size,
        data_root=args.data_root
    )

    # 创建模型
    print(f"\n创建模型: {args.model}")
    if args.model == 'cnn':
        if args.dataset == 'MNIST':
            model = CNN_MNIST(num_classes=args.num_classes_per_task)
        else:
            model = ResNet18(num_classes=args.num_classes_per_task)
    else:
        raise ValueError(f"不支持的模型: {args.model}")

    model = model.to(device)

    # 创建缓冲区
    print(f"\n创建经验回放缓冲区")
    print(f"缓冲区大小: {args.memory_size}")
    print(f"选择方法: {args.selection_method}")

    buffer = CoresetBuffer(
        memory_size=args.memory_size,
        input_shape=input_shape,
        num_classes=num_classes,
        device=device
    )

    # 准确率矩阵
    accuracy_matrix = np.zeros((args.num_tasks, args.num_tasks))

    # 逐任务训练
    print(f"\n开始持续学习训练")
    print("=" * 60)

    for task_id in range(args.num_tasks):
        print(f"\n学习任务 {task_id}")
        print("-" * 60)

        # 在当前任务上训练
        train_stats = train_task(
            model=model,
            train_loader=train_loaders[task_id],
            buffer=buffer,
            task_id=task_id,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            device=device,
            buffer_ratio=args.buffer_ratio,
            selection_method=args.selection_method
        )

        print(f"训练完成 - 损失: {train_stats['loss']:.4f}, 准确率: {train_stats['accuracy']:.2f}%")

        # 从当前任务中选择样本添加到缓冲区
        # 收集所有数据
        all_data = []
        all_labels = []
        for data, labels in train_loaders[task_id]:
            all_data.append(data)
            all_labels.append(labels)

        all_data = torch.cat(all_data, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # 重新映射标签到全局空间
        global_labels = all_labels + task_id * args.num_classes_per_task

        # 选择coreset样本
        num_samples = min(args.memory_size // args.num_tasks, len(all_data))
        selected_data, selected_labels = buffer.select_coreset(
            data=all_data,
            labels=global_labels,
            num_samples=num_samples,
            method=args.selection_method,
            model=model
        )

        # 添加到缓冲区
        added = buffer.add(
            data=selected_data,
            labels=selected_labels,
            task_id=task_id,
            selection_method=args.selection_method
        )

        print(f"添加 {added} 个样本到缓冲区")
        print(f"缓冲区状态: {len(buffer.data)}/{buffer.memory_size}")
        print(f"类别分布: {buffer.get_class_balance()}")

        # 在所有已见任务上评估
        print(f"\n在所有任务上评估:")
        accuracies = evaluate_all_tasks(model, test_loaders, device)

        # 更新准确率矩阵
        accuracy_matrix[task_id, :task_id+1] = accuracies[:task_id+1]

    # 计算最终指标
    print("\n" + "=" * 60)
    print("实验结果")
    print("=" * 60)

    # 打印准确率矩阵
    print("\n准确率矩阵 (%):")
    print("任务 | ", end="")
    for i in range(args.num_tasks):
        print(f"  T{i}  |", end="")
    print()
    print("-" * 60)

    for i in range(args.num_tasks):
        print(f" T{i}  |", end="")
        for j in range(args.num_tasks):
            if j <= i:
                print(f" {accuracy_matrix[i, j]:5.2f}|", end="")
            else:
                print(f"  N/A  |", end="")
        print()

    # 计算指标
    avg_accuracy = compute_average_accuracy(accuracy_matrix)
    forgetting = compute_forgetting_measure(accuracy_matrix)

    print(f"\n最终指标:")
    print(f"  平均准确率: {avg_accuracy:.2f}%")
    print(f"  遗忘度量: {forgetting:.2f}%")

    # 保存结果
    if args.save_results:
        results = {
            'dataset': args.dataset,
            'model': args.model,
            'num_tasks': args.num_tasks,
            'num_classes_per_task': args.num_classes_per_task,
            'memory_size': args.memory_size,
            'selection_method': args.selection_method,
            'num_epochs': args.num_epochs,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'buffer_ratio': args.buffer_ratio,
            'seed': args.seed,
            'accuracy_matrix': accuracy_matrix.tolist(),
            'average_accuracy': float(avg_accuracy),
            'forgetting_measure': float(forgetting)
        }

        os.makedirs(args.results_dir, exist_ok=True)
        save_path = os.path.join(
            args.results_dir,
            f"{args.dataset}_{args.selection_method}_task{args.num_tasks}_mem{args.memory_size}_seed{args.seed}.json"
        )

        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"\n结果已保存到: {save_path}")

    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='持续学习实验 - 基于Coreset的经验回放',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 数据集参数
    parser.add_argument('--dataset', type=str, default='MNIST',
                       choices=['MNIST', 'CIFAR10', 'CIFAR100'],
                       help='数据集名称')
    parser.add_argument('--data_root', type=str, default='./data',
                       help='数据根目录')
    parser.add_argument('--num_tasks', type=int, default=5,
                       help='任务数量')
    parser.add_argument('--num_classes_per_task', type=int, default=2,
                       help='每个任务的类别数')

    # 模型参数
    parser.add_argument('--model', type=str, default='cnn',
                       choices=['cnn'],
                       help='模型类型')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=128,
                       help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='每个任务的训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--buffer_ratio', type=float, default=0.3,
                       help='缓冲区损失的权重')

    # 缓冲区参数
    parser.add_argument('--memory_size', type=int, default=2000,
                       help='经验回放缓冲区大小')
    parser.add_argument('--selection_method', type=str, default='random',
                       choices=['random', 'uniform', 'loss', 'margin', 'gradient',
                               'bcsr', 'csrel'],
                       help='Coreset选择方法')

    # 其他参数
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--save_results', action='store_true',
                       help='是否保存结果')
    parser.add_argument('--results_dir', type=str, default='./results/continual_learning',
                       help='结果保存目录')

    args = parser.parse_args()

    # 运行实验
    results = run_continual_learning(args)


if __name__ == '__main__':
    main()
