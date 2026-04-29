"""
CSReL 集成测试 - 验证端到端工作流程
"""
import torch
import pytest
from src.coreset.csrel_coreset import CSReLCoreset
from src.configs import CSReLConfig
from src.models import get_model


def test_csrel_full_workflow():
    """测试完整的 CSReL 工作流程"""
    # 创建配置
    config = CSReLConfig(
        dataset='MNIST',
        num_classes=10,
        input_dim=784,
        batch_size=32,
        learning_rate=0.001,
        num_epochs=3,  # 少量训练用于测试
        device='cpu',
        random_seed=42,
        class_balance=True,
        selection_ratio=0.2
    )

    # 创建选择器
    selector = CSReLCoreset(config=config)

    # 创建模拟数据
    n_samples = 200
    train_data = torch.randn(n_samples, 1, 28, 28)
    train_labels = torch.randint(0, 10, (n_samples,))

    # 步骤 1: 训练参考模型
    print("Training reference model...")
    ref_model, ref_losses = selector.train_reference_model(
        train_data=train_data,
        train_labels=train_labels,
        verbose=False
    )

    assert ref_model is not None
    assert len(ref_losses) == n_samples

    # 步骤 2: 创建一个不同的模型（模拟未训练/部分训练的模型）
    print("Creating current model...")
    current_model = get_model(dataset='MNIST', num_classes=10)

    # 步骤 3: 执行 coreset 选择
    print("Selecting coreset...")
    selected_indices = selector.select(
        train_data=train_data,
        train_labels=train_labels,
        model=current_model,
        verbose=False
    )

    # 验证选择结果
    expected_size = int(n_samples * config.selection_ratio)
    assert len(selected_indices) == expected_size, \
        f"Expected {expected_size} samples, got {len(selected_indices)}"

    # 验证选择的是不同的样本
    unique_indices = torch.unique(selected_indices)
    assert len(unique_indices) == len(selected_indices), "Selected indices contain duplicates"

    # 验证类别平衡
    selected_labels = train_labels[selected_indices]
    unique_labels = torch.unique(selected_labels)

    # 由于 class_balance=True，应该有多个类别被选中
    # （除非数据集太小或类别太少）
    assert len(unique_labels) >= 2, "Class balance failed - only one class selected"

    print(f"[OK] Successfully selected {len(selected_indices)} samples from {n_samples}")
    print(f"[OK] Selected samples cover {len(unique_labels)} classes")


def test_csrel_incremental_selection():
    """测试增量选择模式"""
    config = CSReLConfig(
        dataset='MNIST',
        num_classes=10,
        input_dim=784,
        batch_size=32,
        learning_rate=0.001,
        num_epochs=3,
        device='cpu',
        random_seed=42,
        selection_ratio=0.2
    )

    selector = CSReLCoreset(config=config)

    n_samples = 100
    train_data = torch.randn(n_samples, 1, 28, 28)
    train_labels = torch.randint(0, 10, (n_samples,))

    # 训练参考模型
    selector.train_reference_model(
        train_data=train_data,
        train_labels=train_labels,
        verbose=False
    )

    # 第一次选择：选择一部分
    current_model = get_model(dataset='MNIST', num_classes=10)
    selected_indices = selector.select(
        train_data=train_data,
        train_labels=train_labels,
        model=current_model,
        incremental=False,
        verbose=False
    )

    first_selection_size = len(selected_indices)
    assert first_selection_size == int(n_samples * config.selection_ratio)

    # 第二次选择：增量添加
    new_model = get_model(dataset='MNIST', num_classes=10)
    updated_indices = selector.select(
        train_data=train_data,
        train_labels=train_labels,
        model=new_model,
        incremental=True,
        current_indices=selected_indices,
        verbose=False
    )

    # 应该包含原始选择
    assert len(updated_indices) >= len(selected_indices)

    # 验证原始索引都在新选择中
    for idx in selected_indices:
        assert idx in updated_indices, "Original selection not preserved in incremental update"

    print(f"[OK] Incremental selection: {first_selection_size} -> {len(updated_indices)}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
