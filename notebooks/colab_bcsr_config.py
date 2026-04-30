"""
Colab优化版BCSR配置

针对Google Colab T4 16GB环境优化：
1. 降低内存占用
2. 提高GPU利用率
3. 使用混合精度训练
4. 减少不必要的计算
"""

import torch


def get_colab_optimized_bcsr_config(n_samples: int, coreset_size: int) -> dict:
    """
    获取Colab优化的BCSR配置

    参数:
        n_samples: 数据集大小
        coreset_size: 目标coreset大小

    返回:
        配置字典
    """
    config = {
        'use_model_based': n_samples <= 2000,  # 小数据集才用模型方法
        'use_presampling': True,
        'presample_size': min(3000, n_samples),
        'num_outer_steps': 2,  # 从5减少到2
        'num_inner_steps': 1,
        'batch_size_kernel': 8192,  # GPU优化：8192对T4友好
        'use_mixed_precision': True,  # 使用FP16
    }

    if n_samples > 10000:
        config['num_outer_steps'] = 1  # 超大数据集只用1次外层迭代
        config['presample_size'] = 2000

    return config


def create_colab_bcsr_adapter(device='cuda'):
    """
    创建Colab优化的BCSR适配器
    """
    from src.coreset.continual_adapters import BCSRContinualAdapter

    # Colab T4优化参数
    adapter = BCSRContinualAdapter(
        learning_rate_inner=0.01,
        learning_rate_outer=3.0,  # 稍微降低以提高稳定性
        num_inner_steps=1,
        num_outer_steps=2,  # 关键：减少外层迭代
        beta=0.1,
        device=device
    )

    return adapter


# Colab环境检测和优化建议
def detect_colab_environment():
    """检测Colab环境并打印优化建议"""
    print("\n" + "="*60)
    print("Colab环境检测")
    print("="*60)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

        print(f"✓ GPU: {gpu_name}")
        print(f"✓ 显存: {gpu_memory:.2f} GB")

        # 根据GPU类型给出建议
        if 'T4' in gpu_name:
            print(f"\n针对T4 GPU的优化建议:")
            print(f"  - batch_size_kernel: 8192 (T4 Tensor Core友好)")
            print(f"  - 使用混合精度 (FP16)")
            print(f"  - 减少外层迭代次数")
            print(f"  - 预采样到2000-3000样本")

        elif 'V100' in gpu_name or 'A100' in gpu_name:
            print(f"\n针对{gpu_name}的优化建议:")
            print(f"  - 可以使用更大的batch_size")
            print(f"  - 外层迭代可以增加到3-4次")
    else:
        print("✗ 未检测到GPU，使用CPU模式")
        print(f"  警告: BCSR在CPU上会非常慢!")
        print(f"  建议: 减少样本数到<1000或使用uniform方法")


# 快速测试配置
QUICK_TEST_CONFIG = {
    'MNIST_2tasks': {
        'presample_size': 2000,
        'num_outer_steps': 2,
        'expected_time': '<1分钟'
    },
    'MNIST_5tasks': {
        'presample_size': 2000,
        'num_outer_steps': 1,
        'expected_time': '<2分钟'
    }
}


if __name__ == '__main__':
    detect_colab_environment()

    print(f"\n快速测试配置:")
    for name, config in QUICK_TEST_CONFIG.items():
        print(f"  {name}:")
        print(f"    预采样: {config['presample_size']}样本")
        print(f"    外层迭代: {config['num_outer_steps']}次")
        print(f"    预期时间: {config['expected_time']}")
