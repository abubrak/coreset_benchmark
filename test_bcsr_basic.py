"""
BCSR基本功能测试脚本
测试BCSRTraining和BCSRCoreset的基本功能
"""

import torch
import numpy as np
from src.training.bcsr_training import BCSRTraining
from src.coreset.bcsr_coreset import BCSRCoreset
from src.models.cnn import CNN_MNIST


def test_bcsr_coreset_basic():
    """测试BCSRCoreset的基本功能"""
    print("=" * 60)
    print("测试BCSRCoreset基本功能")
    print("=" * 60)

    # 创建简单的测试数据
    np.random.seed(42)
    torch.manual_seed(42)

    n_samples = 100
    n_features = 50
    n_classes = 5

    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, n_classes, (n_samples,))

    print(f"\n测试数据形状:")
    print(f"  X: {X.shape}")
    print(f"  y: {y.shape}")

    # 创建BCSR选择器（不使用模型，使用核方法）
    coreset_selector = BCSRCoreset(
        learning_rate_inner=0.01,
        learning_rate_outer=0.1,
        num_inner_steps=10,
        num_outer_steps=5,
        device='cpu',
        random_state=42
    )

    # 选择coreset
    coreset_size = 20
    selected_X, selected_y, info = coreset_selector.coreset_select(
        X=X,
        y=y,
        coreset_size=coreset_size,
        model=None  # 使用简化的核方法
    )

    print(f"\nCoreset选择结果:")
    print(f"  选择的样本数: {len(selected_X)}")
    print(f"  选择的样本形状: {selected_X.shape}")
    print(f"  权重统计:")
    print(f"    均值: {info['weights_mean']:.4f}")
    print(f"    标准差: {info['weights_std']:.4f}")
    print(f"    最小值: {info['weights_min']:.4f}")
    print(f"    最大值: {info['weights_max']:.4f}")

    # 验证结果
    assert len(selected_X) == coreset_size, "选择的样本数不正确"
    assert len(selected_y) == coreset_size, "选择的标签数不正确"
    assert info['n_samples'] == n_samples, "总样本数记录不正确"

    print("\n[OK] BCSR coreset基本功能测试通过")


def test_projection_onto_simplex():
    """测试单纯形投影功能"""
    print("\n" + "=" * 60)
    print("测试单纯形投影功能")
    print("=" * 60)

    coreset_selector = BCSRCoreset()

    # 测试用例1: 随机向量
    v = np.random.randn(10)
    w = coreset_selector.projection_onto_simplex(v)

    print(f"\n测试用例1: 随机向量")
    print(f"  输入向量: {v}")
    print(f"  投影后: {w}")
    print(f"  非负: {np.all(w >= 0)}")
    print(f"  和为1: {np.abs(w.sum() - 1.0) < 1e-6}")

    assert np.all(w >= 0), "投影结果包含负值"
    assert np.abs(w.sum() - 1.0) < 1e-6, "投影结果和不等于1"

    # 测试用例2: 全负向量
    v = -np.ones(5)
    w = coreset_selector.projection_onto_simplex(v)

    print(f"\n测试用例2: 全负向量")
    print(f"  输入向量: {v}")
    print(f"  投影后: {w}")
    print(f"  结果为均匀分布: {np.allclose(w, np.ones(5)/5)}")

    assert np.allclose(w, np.ones(5)/5), "全负向量应投影到均匀分布"

    print("\n[OK] 单纯形投影功能测试通过")


def test_bcsr_training_basic():
    """测试BCSRTraining的基本功能"""
    print("\n" + "=" * 60)
    print("测试BCSRTraining基本功能")
    print("=" * 60)

    # 创建简单的CNN模型
    model = CNN_MNIST(
        num_classes=5
    )

    # 创建测试数据
    n_samples = 50
    X = torch.randn(n_samples, 1, 28, 28)
    y = torch.randint(0, 5, (n_samples,))

    from torch.utils.data import TensorDataset, DataLoader

    train_dataset = TensorDataset(X[:40], y[:40])
    val_dataset = TensorDataset(X[40:], y[40:])

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10)

    # 创建训练器
    trainer = BCSRTraining(
        model=model,
        learning_rate_inner=0.01,
        learning_rate_outer=0.1,
        num_inner_steps=5,
        num_outer_steps=3,
        device='cpu'
    )

    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters())}")
    print(f"训练样本数: {40}")
    print(f"验证样本数: {10}")
    print(f"内层迭代步数: {trainer.num_inner_steps}")
    print(f"外层迭代步数: {trainer.num_outer_steps}")

    # 训练
    weights = trainer.train(train_loader, val_loader, n_samples=40)

    print(f"\n训练完成!")
    print(f"  权重形状: {weights.shape}")
    print(f"  权重范围: [{weights.min().item():.4f}, {weights.max().item():.4f}]")
    print(f"  权重和: {weights.sum().item():.4f}")

    assert weights.shape == (40,), "权重形状不正确"
    assert weights.sum().item() > 0, "权重和应该大于0"

    print("\n[OK] BSR训练基本功能测试通过")


if __name__ == '__main__':
    print("\n开始BCSR基本功能测试\n")

    try:
        test_projection_onto_simplex()
        test_bcsr_coreset_basic()
        test_bcsr_training_basic()

        print("\n" + "=" * 60)
        print("所有测试通过! [OK]")
        print("=" * 60)

    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
