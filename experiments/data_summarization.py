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
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.datasets.data_loaders import get_dataset, get_dataloader, DATASET_STATS, get_coreset_train_loader
from src.baselines.baseline_methods import get_baseline
from src.coreset.bcsr_coreset import BCSRCoreset
from src.models.cnn import CNN_MNIST
from src.models.resnet import ResNet18


def train_model(model, train_loader, val_loader, num_epochs, device,
                learning_rate=0.1, optimizer_type='sgd', weight_decay=5e-4,
                label_smoothing=0.0, warmup_epochs=0, use_mixup=False,
                is_coreset=False):
    """
    在dataloader上训练模型

    参数:
        model: PyTorch模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 训练轮数
        device: 设备 ('cpu' 或 'cuda')
        learning_rate: 学习率
        optimizer_type: 优化器类型 ('sgd' 或 'adam')
        weight_decay: 权重衰减
        label_smoothing: 标签平滑系数 (0.0-0.2)
        warmup_epochs: 学习率预热轮数
        use_mixup: 是否使用 MixUp 数据增强
        is_coreset: 是否使用coreset优化器配置 (Adam lr=5e-5, 无scheduler)

    返回:
        训练历史字典，包含训练和验证损失/准确率
    """
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # is_coreset模式：使用Adam(5e-5)与小学习率，与原始resnet_cifar.py一致
    if is_coreset:
        optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=weight_decay)
        scheduler = None  # 固定学习率，不使用scheduler
    else:
        # 原有逻辑保持不变
        if optimizer_type == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                                  momentum=0.9, weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                                   weight_decay=weight_decay)

        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Mixup alpha 参数
    mixup_alpha = 1.0 if use_mixup else 0.0

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch_time': []
    }

    best_val_acc = 0.0
    best_state = None

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Warmup 学习率调整
        if warmup_epochs > 0 and epoch < warmup_epochs:
            warmup_lr = learning_rate * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            current_lr = warmup_lr
        else:
            current_lr = optimizer.param_groups[0]['lr']

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Mixup 数据增强（可选）
            if use_mixup and mixup_alpha > 0:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                batch_size = inputs.size(0)
                index = torch.randperm(batch_size).to(device)
                mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
                targets_a, targets_b = targets, targets[index]
                outputs = model(mixed_inputs)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
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

        # 更新学习率（warmup 后才开始 scheduler）
        if scheduler is not None:
            if warmup_epochs == 0 or epoch >= warmup_epochs:
                scheduler.step()

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

        # 保存最佳模型状态
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch: {epoch+1}/{num_epochs} | '
                  f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | '
                  f'LR: {current_lr:.6f} | '
                  f'Time: {epoch_time:.2f}s')

    # 恢复最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

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
    from torchvision import transforms

    stats = DATASET_STATS.get(args.dataset)

    # 创建无增强的 transform（用于验证和特征提取）
    transform_noaug = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(stats['mean'], stats['std'])
    ])

    # 训练集（带数据增强）
    train_dataset_aug = get_dataset(args.dataset, train=True, download=True)
    # 训练集（无数据增强，用于验证和特征提取）
    train_dataset_noaug = get_dataset(args.dataset, train=True, download=True, transform=transform_noaug)
    # 测试集
    test_dataset = get_dataset(args.dataset, train=False, download=True)

    # 限制数据集大小（用于快速实验）
    if args.num_samples is not None:
        print(f"限制训练集大小: {args.num_samples}")
        all_indices = torch.randperm(len(train_dataset_aug))[:args.num_samples]
        train_dataset_aug = Subset(train_dataset_aug, all_indices.tolist())
        train_dataset_noaug = Subset(train_dataset_noaug, all_indices.tolist())

    # 创建独立的验证集索引：从训练集中随机抽取10%
    num_train = len(train_dataset_aug)
    val_ratio = 0.1
    val_size = int(num_train * val_ratio)
    perm = torch.randperm(num_train)
    val_indices = perm[:val_size].tolist()
    train_indices = perm[val_size:].tolist()

    # 训练子集（带增强，排除验证集）
    train_subset_aug = Subset(train_dataset_aug, train_indices)
    # 训练子集（无增强，用于特征提取，与 train_subset_aug 索引一致）
    train_subset_noaug = Subset(train_dataset_noaug, train_indices)
    # 验证子集（无增强）
    val_subset = Subset(train_dataset_noaug, val_indices)

    # 数据加载器
    train_loader = get_dataloader(train_subset_aug, batch_size=args.batch_size, shuffle=True)
    train_loader_noshuffle = get_dataloader(train_subset_noaug, batch_size=args.batch_size, shuffle=False)
    val_loader = get_dataloader(val_subset, batch_size=args.batch_size, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"训练集大小: {len(train_subset_aug)}")
    print(f"验证集大小: {val_size}")
    print(f"测试集大小: {len(test_dataset)}")

    # 计算coreset大小（基于训练子集，不包括验证集）
    coreset_size = int(len(train_subset_aug) * args.selection_ratio)
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

        # 先在训练集上预训练特征提取器（少量 epoch 即可获得有区分力的特征）
        print("预训练特征提取器...")
        # 为MNIST使用CNN_MNIST，为CIFAR使用ResNet18
        if args.dataset == 'MNIST':
            feat_model = CNN_MNIST(num_classes=num_classes).to(device)
        else:
            feat_model = ResNet18(num_classes=num_classes).to(device)
        feat_optimizer = optim.SGD(feat_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        feat_criterion = nn.CrossEntropyLoss()
        feat_model.train()
        pretrain_epochs = 10  # 10 epoch 足以学到有意义的特征表示
        for ep in range(pretrain_epochs):
            for inputs, labels in train_loader:
                inputs, targets = inputs.to(device), labels.to(device)
                feat_optimizer.zero_grad()
                outputs = feat_model(inputs)
                loss = feat_criterion(outputs, targets)
                loss.backward()
                feat_optimizer.step()
            if (ep + 1) % 5 == 0:
                print(f"  特征预训练 epoch {ep+1}/{pretrain_epochs}")

        # 提取深度特征
        print("提取深度特征...")
        # 定义特征提取函数
        if args.dataset == 'MNIST':
            def extract_features(model, x):
                # 第一个卷积块
                x = F.relu(model.bn1(model.conv1(x)))
                x = model.pool(x)
                # 第二个卷积块
                x = F.relu(model.bn2(model.conv2(x)))
                x = model.pool(x)
                # 第三个卷积块
                x = F.relu(model.bn3(model.conv3(x)))
                x = model.pool(x)
                # 展平
                x = x.view(x.size(0), -1)
                return x
        else:
            def extract_features(model, x):
                model.fc = nn.Identity()
                return model(x)

        feat_model.eval()

        all_features = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in train_loader_noshuffle:
                inputs = inputs.to(device)
                labels = labels.to(device)
                features = extract_features(feat_model, inputs)
                all_features.append(features)
                all_labels.append(labels)

        X_train = torch.cat(all_features, dim=0)
        y_train = torch.cat(all_labels, dim=0)

        print(f"特征形状: {X_train.shape}, 设备: {X_train.device}")

        # 创建BCSR选择器
        coreset_selector = BCSRCoreset(
            learning_rate_inner=args.bcsr_inner_lr,
            learning_rate_outer=args.bcsr_outer_lr,
            num_inner_steps=args.bcsr_inner_steps,
            num_outer_steps=args.bcsr_outer_steps,
            beta=args.bcsr_beta,
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

    elif args.method in ['uniform', 'kcenter', 'kmeans', 'herding']:
        # 基线方法
        print(f"使用{args.method.upper()}方法...")

        # 使用深度特征（与BCSR一致）
        print("提取深度特征...")
        # 为MNIST使用CNN_MNIST，为CIFAR使用ResNet18
        if args.dataset == 'MNIST':
            feature_extractor = CNN_MNIST(num_classes=num_classes).to(device)
            # 对于MNIST，使用fc1之前的特征（在ReLU之前）
            def extract_features(model, x):
                # 第一个卷积块
                x = F.relu(model.bn1(model.conv1(x)))
                x = model.pool(x)
                # 第二个卷积块
                x = F.relu(model.bn2(model.conv2(x)))
                x = model.pool(x)
                # 第三个卷积块
                x = F.relu(model.bn3(model.conv3(x)))
                x = model.pool(x)
                # 展平
                x = x.view(x.size(0), -1)
                return x
        else:
            feature_extractor = ResNet18(num_classes=num_classes).to(device)
            feature_extractor.fc = nn.Identity()
            def extract_features(model, x):
                return model(x)
        feature_extractor.eval()

        all_features = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in train_loader_noshuffle:
                inputs = inputs.to(device)
                features = extract_features(feature_extractor, inputs)
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        X_train = np.concatenate(all_features, axis=0)
        y_train = np.concatenate(all_labels, axis=0)

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

    # 创建coreset训练DataLoader（使用tile重复机制）
    # selected_indices 是 train_subset_noaug 的局部索引，映射到 train_subset_aug
    coreset_loader = get_coreset_train_loader(
        train_dataset=train_subset_aug,
        indices=selected_indices,
        coreset_size=coreset_size,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=2,
    )

    print(f"Coreset数据集大小: {coreset_size}")

    # 创建模型
    print("\n创建模型...")
    if args.dataset == 'MNIST':
        model_factory = lambda num_classes: CNN_MNIST(num_classes=num_classes)
        optimizer_type = 'adam'
        lr = args.lr
    elif args.dataset in ['CIFAR10', 'CIFAR100']:
        model_factory = lambda num_classes: ResNet18(num_classes=num_classes)
        optimizer_type = 'sgd'
        lr = 0.1
    else:
        raise ValueError(f"不支持的数据集: {args.dataset}")

    model = model_factory(num_classes=num_classes).to(device)

    # 计算模型参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")

    # 在完整训练集上训练
    print("\n" + "=" * 80)
    print("在完整训练集上训练")
    print("=" * 80)

    model_full = model_factory(num_classes=num_classes).to(device)
    history_full = train_model(
        model_full,
        train_loader,
        val_loader,
        args.epochs,
        device,
        learning_rate=lr,
        optimizer_type=optimizer_type
    )

    test_acc_full = evaluate_model(model_full, test_loader, device)
    print(f"\n完整训练集测试准确率: {test_acc_full:.2f}%")

    # 在coreset上训练
    print("\n" + "=" * 80)
    print("在Coreset上训练")
    print("=" * 80)

    model_coreset = model_factory(num_classes=num_classes).to(device)

    # Coreset 训练配置（与原始resnet_cifar.py一致）
    # tile重复+数据增强+Adam(5e-5)+6 epochs
    coreset_epochs = 6
    coreset_weight_decay = 1e-4
    coreset_label_smoothing = 0.0
    coreset_warmup_epochs = 0

    print(f"Coreset 训练配置：{coreset_epochs} epochs, Adam(5e-5), "
          f"weight_decay={coreset_weight_decay}, tile重复机制")

    history_coreset = train_model(
        model_coreset,
        coreset_loader,
        val_loader,
        coreset_epochs,
        device,
        is_coreset=True  # 使用Adam(5e-5)，tile重复模式下不使用自适应lr
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
        'num_samples': len(train_dataset_aug),
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
                       choices=['uniform', 'kcenter', 'kmeans', 'herding', 'bcsr'],
                       help='Coreset选择方法')
    parser.add_argument('--selection_ratio', type=float, default=0.1,
                       help='Coreset选择比例')

    # BCSR特定参数
    parser.add_argument('--bcsr_inner_lr', type=float, default=5.0,
                       help='BCSR内层学习率 (默认: 5.0)')
    parser.add_argument('--bcsr_outer_lr', type=float, default=5.0,
                       help='BCSR外层学习率 (默认: 5.0)')
    parser.add_argument('--bcsr_inner_steps', type=int, default=1,
                       help='BCSR内层优化步数 (默认: 1)')
    parser.add_argument('--bcsr_outer_steps', type=int, default=5,
                       help='BCSR外层优化步数 (默认: 5)')
    parser.add_argument('--bcsr_beta', type=float, default=0.1,
                       help='BCSR正则化系数beta (默认: 0.1)')

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
