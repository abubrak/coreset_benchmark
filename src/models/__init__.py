"""
模型模块
包含各种神经网络模型定义
"""

from .cnn import CNN_MNIST, CNN_CIFAR
from .resnet import ResNet18, ResNet34

__all__ = [
    'CNN_MNIST',
    'CNN_CIFAR',
    'ResNet18',
    'ResNet34',
    'get_model'
]


def get_model(dataset: str, num_classes: int = 10):
    """
    根据数据集获取相应的模型

    Parameters
    ----------
    dataset : str
        数据集名称 ('MNIST', 'CIFAR10', 'CIFAR100')
    num_classes : int, optional
        类别数量, by default 10

    Returns
    -------
    nn.Module
        相应的神经网络模型

    Raises
    ------
    ValueError
        如果数据集名称未知
    """
    dataset = dataset.upper()

    if dataset == 'MNIST':
        return CNN_MNIST(num_classes=num_classes)
    elif dataset in ['CIFAR10', 'CIFAR-10', 'CIFAR100', 'CIFAR-100']:
        return CNN_CIFAR(num_classes=num_classes)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset}. "
            f"Supported datasets: MNIST, CIFAR10, CIFAR100"
        )
