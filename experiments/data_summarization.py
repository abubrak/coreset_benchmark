"""
数据摘要实验脚本

实现各种数据摘要方法的实验，包括：
- Herding: 核牧群方法
- CRA: 类代表分配 (Class Representative Apportionment)
- BCSR: 双层优化的coreset选择
- 基线方法: Random, KCenter, KMeans
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.datasets.data_loaders import get_dataset, get_dataloader, DATASET_STATS
from src.baselines.baseline_methods import get_baseline
from src.coreset.bcsr_coreset import BCSRCoreset
from src.models.cnn import CNN_MNIST, CNN_CIFAR
from src.training.losses import cross_entropy_loss


def train_model(model, train_loader, val_loader, num_epochs, device, learning_rate=0.001):
    """
    在dataloader上训练模型

    参数:
        model: PyTorch模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 训练轮数
        device: 设备 ('cpu' 或 'cuda')
        learning_rate: 学习率

    返回:
        训练历史字典，包含训练和验证损失/准确率
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch_time': []
    }

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        train_acc = 100. * train_correct / train_total
        train_loss /= len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_acc = 100. * val_correct / val_total
        val_loss /= len(val_loader)

        epoch_time = time.time() - epoch_start

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_time'].append(epoch_time)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch: {epoch+1}/{num_epochs} | '
                  f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | '
                  f'Time: {epoch_time:.2f}s')

    history['best_val_acc'] = best_val_acc

    return history


def evaluate_model(model, test_loader, device):
    """
    评估模型准确率

    参数:
        model: PyTorch模型
        test_loader: 测试数据加载器
        device: 设备 ('cpu' 或 'cuda')

    返回:
        测试准确率 (%)
    """
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()

    test_acc = 100. * test_correct / test_total

    return test_acc


def run_experiment(args):
    """
    运行完整的数据摘要实验

    参数:
        args: 命令行参数

    返回:
        实验结果字典
    """
    print("=" * 80)
    print("数据摘要实验")
    print("=" * 80)

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"\n使用设备: {device}")

    # 获取数据集信息
    stats = DATASET_STATS.get(args.dataset)
    if stats is None:
        raise ValueError(f"不支持的数据集: {args.dataset}")

    num_classes = stats['num_classes']
    img_size = stats['img_size']
    num_channels = stats['num_channels']

    print(f"数据集: {args.dataset}")
    print(f"类别数: {num_classes}")
    print(f"图像大小: {img_size}x{img_size}")
    print(f"通道数: {num_channels}")

    # 加载数据集
    print("\n加载数据集...")
    train_dataset = get_dataset(args.dataset, train=True, download=True)
    test_dataset = get_dataset(args.dataset, train=False, download=True)

    # 限制数据集大小（用于快速实验）
    if args.num_samples is not None:
        print(f"限制训练集大小: {args.num_samples}")
        indices = torch.randperm(len(train_dataset))[:args.num_samples]
        train_dataset = Subset(train_dataset, indices)

    # 创建数据加载器
    train_loader = get_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = get_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 创建训练集和验证集的分割（用于BCSR）
    from torch.utils.data import random_split
    val_size = int(len(train_dataset) * 0.1)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    train_loader_full = get_dataloader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader_small = get_dataloader(val_subset, batch_size=args.batch_size, shuffle=False)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    # 计算coreset大小
    coreset_size = int(len(train_dataset) * args.selection_ratio)
    print(f"\nCoreset大小: {coreset_size} (选择比例: {args.selection_ratio})")

    # 选择方法
    print(f"\n方法: {args.method}")
    print(f"选择比例: {args.selection_ratio}")

    # 执行coreset选择
    print("\n执行coreset选择...")
    selection_start = time.time()

    if args.method == 'bcsr':
        # BCSR方法
        print("使用BCSR方法...")

        # 获取展平的特征用于BCSR（保持在 GPU 上）
        print("提取特征...")
        all_features = []
        all_labels = []

        for inputs, labels in train_loader_full:
            inputs = inputs.to(device)
            labels = labels.to(device)
            all_features.append(inputs.view(inputs.size(0), -1))
            all_labels.append(labels)

        X_train = torch.cat(all_features, dim=0)  # 保持在 GPU
        y_train = torch.cat(all_labels, dim=0)

        print(f"特征形状: {X_train.shape}, 设备: {X_train.device}")

        # 创建BCSR选择器
        coreset_selector = BCSRCoreset(
            learning_rate_inner=args.bcsr_inner_lr,
            learning_rate_outer=args.bcsr_outer_lr,
            num_inner_steps=args.bcsr_inner_steps,
            num_outer_steps=args.bcsr_outer_steps,
            device=device,
            random_state=args.seed
        )

        # 执行选择（直接传 GPU tensor）
        selected_X, selected_y, info = coreset_selector.coreset_select(
            X=X_train,
            y=y_train,
            coreset_size=coreset_size,
            model=None  # 使用简化的核方法
        )

        selected_indices = info['selected_indices']

    elif args.method in ['random', 'kcenter', 'kmeans', 'herding']:
        # 基线方法
        print(f"使用{args.method.upper()}方法...")

        # 获取展平的特征
        all_features = []
        all_labels = []

        for inputs, labels in train_loader:
            all_features.append(inputs.view(inputs.size(0), -1))
            all_labels.append(labels)

        X_train = torch.cat(all_features, dim=0).numpy()
        y_train = torch.cat(all_labels, dim=0).numpy()

        print(f"特征形状: {X_train.shape}")

        # 创建基线选择器
        baseline = get_baseline(args.method)

        # 执行选择
        if args.method == 'herding':
            # Herding使用RBF核
            selected_indices = baseline.select(
                X_train,
                y_train,
                size=coreset_size,
                kernel='rbf',
                gamma=0.1,
                random_state=args.seed
            )
        else:
            selected_indices = baseline.select(
                X_train,
                y_train,
                size=coreset_size,
                random_state=args.seed
            )
    else:
        raise ValueError(f"未知的方法: {args.method}")

    selection_time = time.time() - selection_start
    print(f"选择完成! 用时: {selection_time:.2f}秒")
    print(f"选择的样本数: {len(selected_indices)}")

    # 创建coreset数据集
    coreset_dataset = Subset(train_dataset, selected_indices)
    coreset_loader = get_dataloader(coreset_dataset, batch_size=args.batch_size, shuffle=True)

    print(f"Coreset数据集大小: {len(coreset_dataset)}")

    # 创建模型
    print("\n创建模型...")
    if args.dataset == 'MNIST':
        model = CNN_MNIST(num_classes=num_classes)
    elif args.dataset in ['CIFAR10', 'CIFAR100']:
        model = CNN_CIFAR(num_classes=num_classes)
    else:
        raise ValueError(f"不支持的数据集: {args.dataset}")

    model = model.to(device)

    # 计算模型参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")

    # 在完整训练集上训练
    print("\n" + "=" * 80)
    print("在完整训练集上训练")
    print("=" * 80)

    model_full = model.__class__(num_classes=num_classes).to(device)
    history_full = train_model(
        model_full,
        train_loader,
        val_loader_small,
        args.epochs,
        device,
        args.lr
    )

    test_acc_full = evaluate_model(model_full, test_loader, device)
    print(f"\n完整训练集测试准确率: {test_acc_full:.2f}%")

    # 在coreset上训练
    print("\n" + "=" * 80)
    print("在Coreset上训练")
    print("=" * 80)

    model_coreset = model.__class__(num_classes=num_classes).to(device)
    history_coreset = train_model(
        model_coreset,
        coreset_loader,
        val_loader_small,
        args.epochs,
        device,
        args.lr
    )

    test_acc_coreset = evaluate_model(model_coreset, test_loader, device)
    print(f"\nCoreset测试准确率: {test_acc_coreset:.2f}%")

    # 计算性能下降
    performance_drop = test_acc_full - test_acc_coreset

    # 汇总结果
    results = {
        'dataset': args.dataset,
        'method': args.method,
        'selection_ratio': args.selection_ratio,
        'coreset_size': coreset_size,
        'num_samples': len(train_dataset),
        'test_acc_full': test_acc_full,
        'test_acc_coreset': test_acc_coreset,
        'performance_drop': performance_drop,
        'selection_time': selection_time,
        'train_time_full': sum(history_full['epoch_time']),
        'train_time_coreset': sum(history_coreset['epoch_time']),
        'best_val_acc_full': history_full['best_val_acc'],
        'best_val_acc_coreset': history_coreset['best_val_acc'],
        'history_full': history_full,
        'history_coreset': history_coreset
    }

    print("\n" + "=" * 80)
    print("实验结果汇总")
    print("=" * 80)
    print(f"数据集: {results['dataset']}")
    print(f"方法: {results['method']}")
    print(f"Coreset大小: {results['coreset_size']} ({results['selection_ratio']*100:.1f}%)")
    print(f"完整训练集测试准确率: {results['test_acc_full']:.2f}%")
    print(f"Coreset测试准确率: {results['test_acc_coreset']:.2f}%")
    print(f"性能下降: {results['performance_drop']:.2f}%")
    print(f"选择时间: {results['selection_time']:.2f}秒")
    print(f"训练时间 (完整): {results['train_time_full']:.2f}秒")
    print(f"训练时间 (Coreset): {results['train_time_coreset']:.2f}秒")

    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='数据摘要实验')

    # 数据集参数
    parser.add_argument('--dataset', type=str, default='MNIST',
                       choices=['MNIST', 'CIFAR10', 'CIFAR100'],
                       help='数据集名称')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='限制训练集大小（用于快速实验）')

    # Coreset参数
    parser.add_argument('--method', type=str, default='herding',
                       choices=['random', 'kcenter', 'kmeans', 'herding', 'bcsr'],
                       help='Coreset选择方法')
    parser.add_argument('--selection_ratio', type=float, default=0.1,
                       help='Coreset选择比例')

    # BCSR特定参数
    parser.add_argument('--bcsr_inner_lr', type=float, default=0.01,
                       help='BCSR内层学习率')
    parser.add_argument('--bcsr_outer_lr', type=float, default=0.1,
                       help='BCSR外层学习率')
    parser.add_argument('--bcsr_inner_steps', type=int, default=50,
                       help='BCSR内层优化步数')
    parser.add_argument('--bcsr_outer_steps', type=int, default=20,
                       help='BCSR外层优化步数')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')

    # 其他参数
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cpu', 'cuda'],
                       help='设备')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='结果保存目录')

    args = parser.parse_args()

    # 创建结果目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 运行实验
    try:
        results = run_experiment(args)

        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = os.path.join(
            args.output_dir,
            f'{args.dataset}_{args.method}_{timestamp}.json'
        )

        # 转换numpy类型为Python原生类型
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        results_serializable = convert_to_serializable(results)

        with open(result_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        print(f"\n结果已保存到: {result_file}")

    except Exception as e:
        print(f"\n实验失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
