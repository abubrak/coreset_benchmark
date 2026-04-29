"""
Configuration classes for different coreset selection methods.
"""

from dataclasses import dataclass
from typing import Optional, List
from torchvision import transforms


@dataclass
class ExperimentConfig:
    """Base configuration for experiments."""
    dataset: str = "MNIST"
    num_classes: int = 10
    input_dim: int = 784
    hidden_dims: List[int] = None
    batch_size: int = 256
    learning_rate: float = 0.001
    num_epochs: int = 100
    device: str = "cuda"
    random_seed: int = 42

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]


@dataclass
class BilevelConfig(ExperimentConfig):
    """Configuration for Bilevel coreset selection."""
    gradient_steps: int = 100
    inner_lr: float = 0.01
    outer_lr: float = 0.001
    selection_ratio: float = 0.1
    val_ratio: float = 0.1
    init_strategy: str = "random"


@dataclass
class CSReLConfig(ExperimentConfig):
    """Configuration for CSReL (Classwise Spatial Representation Learning)."""
    num_components: int = 50
    num_neighbors: int = 5
    class_balance: bool = True
    selection_ratio: float = 0.1
    similarity_metric: str = "euclidean"


@dataclass
class DataSummarizationConfig(ExperimentConfig):
    """Configuration for Data Summarization (Herding/CRA)."""
    method: str = "herding"  # 'herding' or 'cra'
    num_prototypes: int = 10
    class_balance: bool = True
    selection_ratio: float = 0.1


@dataclass
class ContinualLearningConfig(ExperimentConfig):
    """Configuration for Continual Learning baselines."""
    memory_size: int = 2000
    num_tasks: int = 5
    selection_strategy: str = "random"  # 'random', 'greedy', 'gss'
    buffer_update: str = "reservoir"  # 'reservoir', 'ring', 'gradient'


@dataclass
class CSReLConfigV2(ExperimentConfig):
    """Configuration for CSReL v2 (Classwise Spatial Representation Learning v2)."""
    # Coreset 选择参数
    coreset_size: int = 1000  # 目标 coreset 大小
    incremental_size: int = 100  # 每轮增量选择的大小
    init_size: int = 100  # 初始随机采样大小

    # 参考模型训练参数
    ref_epochs: int = 100  # 参考模型训练轮数
    ref_lr: float = 0.01  # 参考模型学习率
    ref_opt_type: str = 'sgd'  # 参考模型优化器类型

    # 增量训练参数
    inc_epochs: int = 10  # 增量训练轮数
    inc_lr: float = 0.01  # 增量训练学习率
    inc_opt_type: str = 'sgd'  # 增量训练优化器类型

    # 损失函数参数
    ce_factor: float = 1.0  # 交叉熵损失权重
    mse_factor: float = 0.0  # 知识蒸馏损失权重
    kd_mode: str = 'mse'  # 知识蒸馏模式 ('mse' 或 'ce')

    # 其他参数
    use_cuda: bool = True  # 是否使用 CUDA
    batch_size: int = 128  # 批量大小
    shuffle: bool = True  # 是否打乱数据
    num_workers: int = 0  # 数据加载器工作进程数

    # 早停和正则化
    early_stop: int = 10  # 早停耐心值
    weight_decay: float = 5e-4  # 权重衰减
    grad_max_norm: float = None  # 梯度裁剪最大范数

    # 学习率调度
    scheduler_type: str = None  # 学习率调度器类型
    scheduler_param: dict = None  # 调度器参数

    # 数据增强
    train_transform = None  # 训练数据增强
    test_transform = None  # 测试数据增强

    # 临时文件路径
    temp_dir: str = "./temp_csrel_v2"  # 临时文件目录

    def __post_init__(self):
        """设置默认的数据增强"""
        if self.train_transform is None:
            # 为不同数据集设置默认变换
            if self.dataset == "MNIST":
                self.train_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            elif self.dataset in ["CIFAR10", "CIFAR100"]:
                self.train_transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                       (0.2023, 0.1994, 0.2010))
                ])
            else:
                self.train_transform = transforms.Compose([
                    transforms.ToTensor()
                ])

        if self.test_transform is None:
            # 为不同数据集设置默认变换
            if self.dataset == "MNIST":
                self.test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            elif self.dataset in ["CIFAR10", "CIFAR100"]:
                self.test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                       (0.2023, 0.1994, 0.2010))
                ])
            else:
                self.test_transform = transforms.Compose([
                    transforms.ToTensor()
                ])

        if self.scheduler_param is None:
            self.scheduler_param = {}
