"""
测试BCSR性能优化效果
"""
import torch
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.coreset.bcsr_coreset import BCSRCoreset
from src.models.cnn import CNN_MNIST


def test_bcsr_performance():
    """测试BCSR在不同规模数据集上的性能"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 创建小规模测试数据
    torch.manual_seed(42)
    n_samples_list = [500, 2000, 5000, 12665]
    coreset_size = 200

    for n_samples in n_samples_list:
        print(f"\n{'='*60}")
        print(f"测试数据集大小: {n_samples}样本")
        print(f"{'='*60}")

        # 创建测试数据（MNIST风格的图像）
        data = torch.randn(n_samples, 1, 28, 28).to(device)
        labels = torch.randint(0, 2, (n_samples,)).to(device)

        # 创建简单模型
        model = CNN_MNIST(num_classes=2).to(device)

        # 创建BCSR选择器
        bcsr = BCSRCoreset(
            learning_rate_inner=0.01,
            learning_rate_outer=5.0,
            num_inner_steps=1,
            num_outer_steps=3,  # 减少到3以加快速度
            device=device
        )

        # 测试选择时间
        start_time = time.time()

        try:
            selected_X, selected_y, info = bcsr.coreset_select(
                X=data,
                y=labels,
                coreset_size=coreset_size,
                model=model
            )

            elapsed_time = time.time() - start_time

            print(f"[OK] BCSR选择完成")
            print(f"  - 选择样本数: {len(selected_X)}")
            print(f"  - 时间: {elapsed_time:.2f}秒")
            print(f"  - 方法: {info['method']}")
            print(f"  - 权重统计: 均值={info['weights_mean']:.4f}, 标准差={info['weights_std']:.4f}")

            # 验证类别分布
            unique, counts = torch.unique(torch.from_numpy(selected_y), return_counts=True)
            print(f"  - 类别分布: {dict(zip(unique.tolist(), counts.tolist()))}")

        except Exception as e:
            print(f"[FAIL] BCSR选择失败: {str(e)}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("性能测试完成")
    print(f"{'='*60}")


if __name__ == '__main__':
    test_bcsr_performance()
