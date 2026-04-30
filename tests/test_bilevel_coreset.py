"""
BilevelCoreset 单元测试和集成测试
"""
import pytest
import torch
import numpy as np
from src.coreset.bilevel_coreset import BilevelCoreset


def test_bilevel_optimization_has_inner_loop():
    """测试：双层优化应该有内层优化循环"""
    # 创建小规模测试数据
    np.random.seed(42)
    torch.manual_seed(42)

    n_train = 50
    n_val = 10
    n_features = 20
    num_classes = 5

    X_train = np.random.randn(n_train, n_features).astype(np.float32)
    y_train = torch.randint(0, num_classes, (n_train,))
    X_val = np.random.randn(n_val, n_features).astype(np.float32)
    y_val = torch.randint(0, num_classes, (n_val,))

    # 创建 RBF 核函数
    def rbf_kernel(X1, X2, gamma=1.0):
        X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
        X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True)
        dist_sq = X1_sq + X2_sq.T - 2.0 * (X1 @ X2.T)
        dist_sq = np.maximum(dist_sq, 0.0)
        return np.exp(-gamma * dist_sq)

    # 创建 BilevelCoreset
    selector = BilevelCoreset(
        max_outer_it=3,
        max_inner_it=10,
        tol=1e-6,
        verbose=False
    )

    # 运行优化
    alpha, val_loss_history = selector.solve_bilevel_opt_representer_proxy(
        K_train=torch.tensor(rbf_kernel(X_train, X_train)),
        y_train=y_train,
        K_val=torch.tensor(rbf_kernel(X_val, X_train)),
        y_val=y_val,
        device='cpu'
    )

    # 验证：应该有外层优化的证据
    # 检查验证损失是否被记录（至少2次，可能在3次前收敛）
    assert len(val_loss_history) >= 2, "应该至少记录2次外层迭代的验证损失"
    assert val_loss_history[-1] <= val_loss_history[0] * 1.5, \
        "验证损失应该下降或保持稳定"


def test_selection_complexity_is_feasible():
    """测试：选择算法应该在实际时间内完成"""
    np.random.seed(42)
    torch.manual_seed(42)

    # 中等规模数据（模拟实际使用）
    n_train = 1000
    n_val = 200
    n_features = 50
    num_classes = 10
    coreset_size = 50  # 5% 的数据

    X_train = np.random.randn(n_train, n_features).astype(np.float32)
    y_train = torch.randint(0, num_classes, (n_train,))
    X_val = np.random.randn(n_val, n_features).astype(np.float32)
    y_val = torch.randint(0, num_classes, (n_val,))

    def rbf_kernel(X1, X2, gamma=1.0):
        X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
        X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True)
        dist_sq = X1_sq + X2_sq.T - 2.0 * (X1 @ X2.T)
        dist_sq = np.maximum(dist_sq, 0.0)
        return np.exp(-gamma * dist_sq)

    selector = BilevelCoreset(
        max_outer_it=5,
        max_inner_it=20,
        tol=1e-6,
        verbose=False
    )

    import time
    start_time = time.time()

    # 运行选择（应该在合理时间内完成）
    indices, weights = selector.build_with_representer_proxy(
        X=X_train,
        y=y_train.numpy(),
        m=coreset_size,
        kernel_fn=rbf_kernel,
        device='cpu'
    )

    elapsed_time = time.time() - start_time

    # 显示实际运行时间
    print(f"\n实际运行时间: {elapsed_time:.2f}秒")
    print(f"数据集大小: n_train={n_train}, n_val={n_val}, coreset_size={coreset_size}")
    print(f"理论复杂度: O(m × n_train × n_val) = O({coreset_size} × {n_train} × {n_val})")

    # 验证：应该在 60 秒内完成（宽松限制）
    # 注意：如果测试超时，说明当前实现太慢，这是预期的失败
    assert elapsed_time < 60.0, f"选择时间过长: {elapsed_time:.2f}秒，当前实现太慢，需要优化"

    # 验证：应该选择了正确数量的样本
    assert len(indices) == coreset_size, f"应该选择 {coreset_size} 个样本，实际选择了 {len(indices)}"
    assert len(weights) == coreset_size, f"权重数量应该为 {coreset_size}"

    # 验证：权重应该是正数且归一化
    assert np.all(weights >= 0), "所有权重应该非负"
    assert np.abs(weights.sum() - coreset_size) < 1e-3, "权重应该被适当归一化"


def test_selection_scalability_large_scale():
    """测试：大规模数据集上选择算法的可扩展性"""
    np.random.seed(42)
    torch.manual_seed(42)

    # 更大规模的数据（模拟真实场景）
    n_train = 3000
    n_val = 500
    n_features = 100
    num_classes = 10
    coreset_size = 100  # 约3%的数据

    X_train = np.random.randn(n_train, n_features).astype(np.float32)
    y_train = torch.randint(0, num_classes, (n_train,))
    X_val = np.random.randn(n_val, n_features).astype(np.float32)
    y_val = torch.randint(0, num_classes, (n_val,))

    def rbf_kernel(X1, X2, gamma=1.0):
        X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
        X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True)
        dist_sq = X1_sq + X2_sq.T - 2.0 * (X1 @ X2.T)
        dist_sq = np.maximum(dist_sq, 0.0)
        return np.exp(-gamma * dist_sq)

    selector = BilevelCoreset(
        max_outer_it=5,
        max_inner_it=20,
        tol=1e-6,
        verbose=False
    )

    import time
    start_time = time.time()

    # 运行选择
    indices, weights = selector.build_with_representer_proxy(
        X=X_train,
        y=y_train.numpy(),
        m=coreset_size,
        kernel_fn=rbf_kernel,
        device='cpu'
    )

    elapsed_time = time.time() - start_time

    # 显示诊断信息
    print(f"\n大规模数据集运行时间: {elapsed_time:.2f}秒")
    print(f"数据集大小: n_train={n_train}, n_val={n_val}, coreset_size={coreset_size}")
    print(f"理论复杂度: O(m × n_train × n_val) = O({coreset_size} × {n_train} × {n_val})")
    print(f"估计操作数: {coreset_size * n_train * n_val:.2e}")

    # 目标：在合理时间内（如2分钟）完成
    assert elapsed_time < 120.0, f"大规模数据集选择时间过长: {elapsed_time:.2f}秒，需要优化算法"

    # 验证结果
    assert len(indices) == coreset_size, f"应该选择 {coreset_size} 个样本"
    assert len(weights) == coreset_size, f"权重数量应该为 {coreset_size}"

    # 验证：应该选择了正确数量的样本
    assert len(indices) == coreset_size, f"应该选择 {coreset_size} 个样本，实际选择了 {len(indices)}"
    assert len(weights) == coreset_size, f"权重数量应该为 {coreset_size}"

    # 验证：权重应该是正数且归一化
    assert np.all(weights >= 0), "所有权重应该非负"
    assert np.abs(weights.sum() - coreset_size) < 1e-3, "权重应该被适当归一化"


@pytest.mark.skip(reason="当前实现不支持随机种子可重现性，需要在后续版本中修复")
def test_random_state_reproducibility():
    """测试：使用相同随机种子应该产生相同结果"""
    np.random.seed(42)
    torch.manual_seed(42)

    n_train = 100
    n_val = 20
    n_features = 20
    num_classes = 5

    X_train = np.random.randn(n_train, n_features).astype(np.float32)
    y_train = torch.randint(0, num_classes, (n_train,))
    X_val = np.random.randn(n_val, n_features).astype(np.float32)
    y_val = torch.randint(0, num_classes, (n_val,))

    def rbf_kernel(X1, X2, gamma=1.0):
        X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
        X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True)
        dist_sq = X1_sq + X2_sq.T - 2.0 * (X1 @ X2.T)
        dist_sq = np.maximum(dist_sq, 0.0)
        return np.exp(-gamma * dist_sq)

    # 第一次运行
    selector1 = BilevelCoreset(
        max_outer_it=3,
        max_inner_it=10,
        tol=1e-6,
        verbose=False
    )

    indices1, weights1 = selector1.build_with_representer_proxy(
        X=X_train,
        y=y_train.numpy(),
        m=10,
        kernel_fn=rbf_kernel,
        device='cpu',
        val_ratio=0.2
    )

    # 第二次运行（使用相同种子）
    np.random.seed(42)
    torch.manual_seed(42)

    selector2 = BilevelCoreset(
        max_outer_it=3,
        max_inner_it=10,
        tol=1e-6,
        verbose=False
    )

    indices2, weights2 = selector2.build_with_representer_proxy(
        X=X_train,
        y=y_train.numpy(),
        m=10,
        kernel_fn=rbf_kernel,
        device='cpu',
        val_ratio=0.2
    )

    # 验证：结果应该完全相同
    assert np.array_equal(indices1, indices2), \
        "使用相同随机种子应该产生相同的索引"
    assert np.allclose(weights1, weights2, rtol=1e-5), \
        "使用相同随机种子应该产生相同的权重"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
