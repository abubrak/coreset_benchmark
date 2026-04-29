"""
CSReL 输入验证测试
"""
import pytest
import torch
import numpy as np
from src.coreset.csrel_coreset import CSReLCoreset
from src.configs import CSReLConfig


def test_select_without_model_raises_error():
    """测试：未提供模型时应该抛出 ValueError"""
    # 创建配置
    config = CSReLConfig(
        dataset='MNIST',
        num_classes=10,
        input_dim=784,
        batch_size=64,
        learning_rate=0.001,
        num_epochs=5,
        device='cpu',
        random_seed=42
    )

    # 创建选择器
    selector = CSReLCoreset(config=config)

    # 创建模拟数据（MNIST 格式：batch, channels, height, width）
    train_data = torch.randn(100, 1, 28, 28)
    train_labels = torch.randint(0, 10, (100,))

    # 先训练参考模型
    selector.train_reference_model(
        train_data=train_data,
        train_labels=train_labels,
        verbose=False
    )

    # 尝试不提供模型进行选择 - 应该抛出 ValueError
    with pytest.raises(ValueError, match="model must be provided"):
        selector.select(
            train_data=train_data,
            train_labels=train_labels,
            model=None,  # 明确传入 None
            verbose=False
        )


def test_select_with_random_model_works():
    """测试：提供随机初始化模型时应该正常工作"""
    # 创建配置
    config = CSReLConfig(
        dataset='MNIST',
        num_classes=10,
        input_dim=784,
        batch_size=64,
        learning_rate=0.001,
        num_epochs=2,  # 减少训练时间
        device='cpu',
        random_seed=42
    )

    # 创建选择器
    selector = CSReLCoreset(config=config)

    # 创建模拟数据（MNIST 格式：batch, channels, height, width）
    train_data = torch.randn(100, 1, 28, 28)
    train_labels = torch.randint(0, 10, (100,))

    # 先训练参考模型
    ref_model, _ = selector.train_reference_model(
        train_data=train_data,
        train_labels=train_labels,
        verbose=False
    )

    # 创建一个随机初始化的模型（模拟未训练的模型）
    from src.models import get_model
    random_model = get_model(dataset='MNIST', num_classes=10)

    # 提供随机模型进行选择 - 应该正常工作
    selected_indices = selector.select(
        train_data=train_data,
        train_labels=train_labels,
        model=random_model,
        verbose=False
    )

    # 验证返回了选择结果
    assert isinstance(selected_indices, torch.Tensor)
    assert len(selected_indices) > 0
    assert len(selected_indices) <= len(train_data)


def test_select_error_message_is_helpful():
    """测试：错误消息应该提供有用的指导信息"""
    config = CSReLConfig(
        dataset='MNIST',
        num_classes=10,
        input_dim=784,
        batch_size=64,
        learning_rate=0.001,
        num_epochs=5,
        device='cpu',
        random_seed=42
    )

    selector = CSReLCoreset(config=config)

    train_data = torch.randn(50, 1, 28, 28)
    train_labels = torch.randint(0, 10, (50,))

    selector.train_reference_model(
        train_data=train_data,
        train_labels=train_labels,
        verbose=False
    )

    # 检查错误消息包含有用信息
    with pytest.raises(ValueError) as exc_info:
        selector.select(
            train_data=train_data,
            train_labels=train_labels,
            model=None,
            verbose=False
        )

    error_message = str(exc_info.value)
    assert "model" in error_message.lower()
    assert "provide" in error_message.lower() or "required" in error_message.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
