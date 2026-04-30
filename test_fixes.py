"""
快速验证修复后的持续学习框架
"""
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_simple():
    """简单测试验证修复"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    from experiments.continual_learning import create_task_datasets, CoresetBuffer
    from src.models.cnn import CNN_MNIST

    # 创建任务数据集（2个任务，每个2个类别）
    print("\n创建任务数据集...")
    train_loaders, test_loaders, num_classes, input_shape = create_task_datasets(
        dataset_name='MNIST',
        num_tasks=2,
        num_classes_per_task=2,
        batch_size=64,
        data_root='./data'
    )

    # 创建模型（2个输出类别）
    model = CNN_MNIST(num_classes=2).to(device)

    # 创建缓冲区
    buffer = CoresetBuffer(
        memory_size=200,
        input_shape=input_shape,
        num_classes=num_classes,
        device=device
    )

    # 从Task 0选择样本
    print("\n从Task 0选择样本...")
    all_data = []
    all_labels = []
    for i, (data, labels) in enumerate(train_loaders[0]):
        all_data.append(data)
        all_labels.append(labels)
        if i >= 9:  # 只取前10个batch
            break

    all_data = torch.cat(all_data, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    print(f"Task 0数据形状: {all_data.shape}, 标签范围: {all_labels.min()}-{all_labels.max()}")

    # 测试uniform方法（应该有类别平衡）
    selected_data, selected_labels = buffer.select_coreset(
        data=all_data,
        labels=all_labels,
        num_samples=20,
        method='uniform'
    )

    print(f"选择的样本形状: {selected_data.shape}")
    print(f"选择的标签: {selected_labels}")
    print(f"类别分布: {torch.bincount(selected_labels)}")

    # 添加到缓冲区
    buffer.add(selected_data, selected_labels, task_id=0)

    # 从Task 1选择样本
    print("\n从Task 1选择样本...")
    all_data2 = []
    all_labels2 = []
    for i, (data, labels) in enumerate(train_loaders[1]):
        all_data2.append(data)
        all_labels2.append(labels)
        if i >= 9:
            break

    all_data2 = torch.cat(all_data2, dim=0)
    all_labels2 = torch.cat(all_labels2, dim=0)

    print(f"Task 1数据形状: {all_data2.shape}, 标签范围: {all_labels2.min()}-{all_labels2.max()}")

    # 测试uniform方法
    selected_data2, selected_labels2 = buffer.select_coreset(
        data=all_data2,
        labels=all_labels2,
        num_samples=20,
        method='uniform'
    )

    print(f"选择的样本形状: {selected_data2.shape}")
    print(f"选择的标签: {selected_labels2}")
    print(f"类别分布: {torch.bincount(selected_labels2)}")

    # 添加到缓冲区
    buffer.add(selected_data2, selected_labels2, task_id=1)

    # 测试缓冲区回放
    print("\n测试缓冲区回放...")
    buffer_loader = buffer.get_dataloader(batch_size=32, shuffle=False)

    if buffer_loader is not None:
        print(f"缓冲区大小: {len(buffer_loader.dataset)}")
        print(f"缓冲区标签: {buffer_loader.dataset.tensors[1]}")

        # 测试前几个batch
        for i, (data, labels) in enumerate(buffer_loader):
            print(f"Batch {i}: 数据形状={data.shape}, 标签={labels}")
            if i >= 2:
                break

    print("\n[OK] 所有测试通过！")
    return True


if __name__ == '__main__':
    try:
        success = test_simple()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
