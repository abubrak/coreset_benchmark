"""
Configuration classes for different coreset selection methods.
"""

from dataclasses import dataclass
from typing import Optional, List


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
