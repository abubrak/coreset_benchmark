"""
快速集成测试 - 验证关键功能
"""
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_quick():
    """快速测试集成状态"""
    print("快速集成测试")
    print("="*50)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")

    # 测试导入
    print("\n1. 测试模块导入...")
    try:
        from experiments.continual_learning import CoresetBuffer, create_task_datasets
        from src.coreset.continual_adapters import (
            BCSRContinualAdapter,
            CSReLContinualAdapter,
            BilevelContinualAdapter
        )
        from src.models.cnn import CNN_MNIST
        print("   [OK] 所有模块导入成功")
    except Exception as e:
        print(f"   [FAIL] 模块导入失败: {e}")
        return False

    # 测试方法列表
    print("\n2. 检查支持的方法...")
    buffer = CoresetBuffer(
        memory_size=100,
        input_shape=(1, 28, 28),
        num_classes=2,
        device=device
    )

    # 创建小量测试数据
    torch.manual_seed(42)
    data = torch.randn(50, 1, 28, 28).to(device)
    labels = torch.randint(0, 2, (50,)).to(device)
    model = CNN_MNIST(num_classes=2).to(device)

    methods_to_test = ['random', 'uniform', 'bcsr']
    results = {}

    for method in methods_to_test:
        try:
            print(f"   测试 {method}...")
            selected_data, selected_labels = buffer.select_coreset(
                data=data,
                labels=labels,
                num_samples=10,
                method=method,
                model=model if method == 'bcsr' else None
            )
            results[method] = 'OK'
            print(f"     [OK] {method} - 选择了{len(selected_data)}个样本")
        except Exception as e:
            results[method] = f'FAIL: {str(e)[:50]}'
            print(f"     [FAIL] {method} - {e}")

    # 总结
    print("\n3. 测试总结:")
    print("   " + "="*40)
    for method, status in results.items():
        print(f"   {method:<10} {status}")

    all_passed = all(v == 'OK' for v in results.values())

    if all_passed:
        print("\n   [OK] 所有测试通过！")
    else:
        print("\n   [FAIL] 部分测试失败")

    return all_passed

if __name__ == '__main__':
    success = test_quick()
    sys.exit(0 if success else 1)
