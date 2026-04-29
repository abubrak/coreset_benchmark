"""测试get_coreset_train_loader的tile重复逻辑"""
import torch
import numpy as np
from torchvision import datasets, transforms
from src.datasets.data_loaders import get_coreset_train_loader, DATASET_TRAIN_SIZE

def test_tile_repetition():
    # 创建小数据集
    transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # 选择100个样本索引
    indices = np.arange(100)
    coreset_size = 100

    # 创建loader
    loader = get_coreset_train_loader(
        train_dataset=full_dataset,
        indices=indices,
        coreset_size=coreset_size,
        dataset_name='MNIST',
        batch_size=32,
    )

    # 验证loader中的样本总数
    total_samples = sum(len(batch) for batch, _ in loader)
    expected_samples = (60000 // 100) * 100  # tile到接近60000

    print(f"Total samples in one epoch: {total_samples}")
    print(f"Expected: {expected_samples}")
    print(f"Dataset TRAIN_SIZE constant: {DATASET_TRAIN_SIZE['MNIST']}")

    # 验证重复次数
    assert total_samples == expected_samples, f"Expected {expected_samples}, got {total_samples}"
    assert DATASET_TRAIN_SIZE['MNIST'] == 60000, "MNIST size constant incorrect"

    print("[PASS] Tile repetition test passed!")

if __name__ == '__main__':
    test_tile_repetition()
