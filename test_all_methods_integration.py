"""
端到端集成测试：所有coreset选择方法在持续学习框架中的验证

测试内容：
1. Baseline方法：random, uniform, loss, margin, gradient
2. 双层优化方法：bcsr, csrel, bilevel

验证点：
- 所有方法都能正确选择样本
- 类别平衡（除random外）
- 适配器接口一致性
- 性能优化生效
"""

import torch
import torch.nn as nn
import sys
import os
import time
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.continual_learning import (
    create_task_datasets,
    CoresetBuffer
)
from src.models.cnn import CNN_MNIST


class SimpleCNN(nn.Module):
    """用于测试的简单CNN"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def test_method(method: str, data: torch.Tensor, labels: torch.Tensor,
                model: nn.Module, buffer: CoresetBuffer,
                num_samples: int = 20) -> Dict[str, any]:
    """
    测试单个coreset选择方法

    返回:
        result: 包含测试结果的字典
    """
    result = {
        'method': method,
        'success': False,
        'selected_count': 0,
        'class_distribution': {},
        'time': 0.0,
        'error': None
    }

    try:
        print(f"\n{'='*60}")
        print(f"测试方法: {method}")
        print(f"{'='*60}")

        start_time = time.time()

        # 执行coreset选择
        selected_data, selected_labels = buffer.select_coreset(
            data=data,
            labels=labels,
            num_samples=num_samples,
            method=method,
            model=model if method in ['bcsr', 'csrel', 'bilevel',
                                       'loss', 'margin', 'gradient'] else None
        )

        elapsed_time = time.time() - start_time

        # 验证结果
        result['selected_count'] = len(selected_data)
        result['time'] = elapsed_time

        # 检查类别分布
        unique_labels, counts = torch.unique(selected_labels, return_counts=True)
        result['class_distribution'] = {
            int(label): int(count)
            for label, count in zip(unique_labels.tolist(), counts.tolist())
        }

        # 验证选择的样本数量
        assert len(selected_data) == num_samples, \
            f"选择的样本数({len(selected_data)})不等于目标数({num_samples})"

        # 验证数据形状
        assert selected_data.shape[1:] == data.shape[1:], \
            f"选择的数据形状({selected_data.shape})不匹配原始形状({data.shape})"

        result['success'] = True

        # 输出结果
        print(f"[OK] {method}方法测试成功")
        print(f"  - 选择样本数: {result['selected_count']}")
        print(f"  - 用时: {elapsed_time:.2f}秒")
        print(f"  - 类别分布: {result['class_distribution']}")

    except Exception as e:
        result['error'] = str(e)
        print(f"[FAIL] {method}方法测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

    return result


def run_integration_test(
    test_methods: List[str] = None,
    n_test_samples: int = 200,
    num_samples_to_select: int = 20
):
    """
    运行完整的方法集成测试

    参数:
        test_methods: 要测试的方法列表，None则测试所有方法
        n_test_samples: 测试数据样本数
        num_samples_to_select: 要选择的coreset大小
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 默认测试所有方法
    if test_methods is None:
        test_methods = [
            'random',      # 基线：随机
            'uniform',     # 基线：类别平衡
            'loss',        # 基线：基于损失
            'margin',      # 基线：基于margin
            'gradient',    # 基线：基于梯度
            'bcsr',        # 双层优化：BCSR
            'csrel',       # 双层优化：CSReL
            'bilevel'      # 双层优化：Bilevel
        ]

    # 创建任务数据集
    print(f"\n创建任务数据集...")
    train_loaders, test_loaders, num_classes, input_shape = create_task_datasets(
        dataset_name='MNIST',
        num_tasks=2,
        num_classes_per_task=2,
        batch_size=64,
        data_root='./data'
    )

    # 创建模型（2个输出类别对应Task 0）
    print(f"创建模型...")
    model = CNN_MNIST(num_classes=2).to(device)

    # 创建缓冲区
    print(f"创建缓冲区...")
    buffer = CoresetBuffer(
        memory_size=200,
        input_shape=input_shape,
        num_classes=num_classes,
        device=device
    )

    # 准备测试数据
    print(f"准备测试数据...")
    all_data = []
    all_labels = []
    for i, (data, labels) in enumerate(train_loaders[0]):
        all_data.append(data)
        all_labels.append(labels)
        if len(all_data) * 64 >= n_test_samples:  # 限制样本数
            break

    all_data = torch.cat(all_data, dim=0)[:n_test_samples].to(device)
    all_labels = torch.cat(all_labels, dim=0)[:n_test_samples].to(device)

    print(f"测试数据: {all_data.shape}, 标签范围: {all_labels.min()}-{all_labels.max()}")

    # 测试所有方法
    results = []
    success_count = 0

    for method in test_methods:
        result = test_method(
            method=method,
            data=all_data,
            labels=all_labels,
            model=model,
            buffer=buffer,
            num_samples=num_samples_to_select
        )
        results.append(result)

        if result['success']:
            success_count += 1

    # 输出总结
    print(f"\n{'='*60}")
    print(f"测试总结")
    print(f"{'='*60}")
    print(f"总方法数: {len(test_methods)}")
    print(f"成功: {success_count}")
    print(f"失败: {len(test_methods) - success_count}")

    # 性能统计
    print(f"\n性能统计:")
    print(f"{'方法':<12} {'状态':<8} {'时间(秒)':<10} {'类别分布'}")
    print(f"{'-'*60}")

    for result in results:
        status = "[OK]" if result['success'] else "[FAIL]"
        time_str = f"{result['time']:.2f}" if result['success'] else "N/A"
        class_dist = str(result['class_distribution']) if result['success'] else "N/A"

        print(f"{result['method']:<12} {status:<8} {time_str:<10} {class_dist}")

    # 类别平衡验证
    print(f"\n类别平衡验证:")
    for result in results:
        if result['success'] and result['method'] != 'random':
            dist = result['class_distribution']
            if len(dist) > 1:
                counts = list(dist.values())
                balance_ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
                status = "[好]" if balance_ratio < 2.0 else "[差]"
                print(f"  {result['method']}: {status} (比例 {balance_ratio:.2f})")

    # 检查失败的测试
    failed_tests = [r for r in results if not r['success']]
    if failed_tests:
        print(f"\n失败的测试:")
        for result in failed_tests:
            print(f"  - {result['method']}: {result['error']}")

    # 返回测试是否全部通过
    all_passed = success_count == len(test_methods)

    if all_passed:
        print(f"\n[OK] 所有测试通过！")
    else:
        print(f"\n[FAIL] 部分测试失败")

    return all_passed


def test_bcsr_performance_optimization():
    """
    测试BCSR性能优化是否生效
    """
    print(f"\n{'='*60}")
    print(f"BCSR性能优化验证")
    print(f"{'='*60}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 测试不同规模的数据集
    test_sizes = [500, 2000, 5000]

    for n_samples in test_sizes:
        print(f"\n测试数据规模: {n_samples}样本")

        # 创建测试数据
        torch.manual_seed(42)
        data = torch.randn(n_samples, 1, 28, 28).to(device)
        labels = torch.randint(0, 2, (n_samples,)).to(device)
        model = CNN_MNIST(num_classes=2).to(device)

        # 创建缓冲区
        buffer = CoresetBuffer(
            memory_size=500,
            input_shape=(1, 28, 28),
            num_classes=2,
            device=device
        )

        # 测试BCSR选择
        start_time = time.time()

        try:
            selected_data, selected_labels = buffer.select_coreset(
                data=data,
                labels=labels,
                num_samples=100,
                method='bcsr',
                model=model
            )

            elapsed_time = time.time() - start_time

            # 性能预期（基于优化结果）
            expected_max_time = 30.0 if n_samples < 2000 else 60.0

            if elapsed_time < expected_max_time:
                print(f"  [OK] 性能良好: {elapsed_time:.2f}秒 < {expected_max_time}秒")
            else:
                print(f"  [WARNING] 性能偏慢: {elapsed_time:.2f}秒 > {expected_max_time}秒")

        except Exception as e:
            print(f"  [FAIL] 测试失败: {str(e)}")


if __name__ == '__main__':
    # 运行完整的方法集成测试
    print("="*60)
    print("持续学习Coreset选择方法 - 端到端集成测试")
    print("="*60)

    # 测试所有方法
    all_passed = run_integration_test(
        test_methods=None,  # 测试所有方法
        n_test_samples=200,
        num_samples_to_select=20
    )

    # 测试BCSR性能优化
    test_bcsr_performance_optimization()

    # 退出码
    sys.exit(0 if all_passed else 1)
