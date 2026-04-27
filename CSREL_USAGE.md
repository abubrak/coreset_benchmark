# CSReL方法使用指南

## 概述

CSReL (Classwise Spatial Representation Learning) 是一种基于可约损失的coreset选择方法。该方法通过训练参考模型并计算可约损失来选择最具信息量的样本。

## 主要特性

- **基于可约损失**: 选择当前模型损失与参考损失差异最大的样本
- **类别平衡**: 支持按类别平衡选择,确保每个类别都有代表样本
- **增量选择**: 支持增量添加新样本到已有选择
- **灵活配置**: 通过配置对象轻松调整参数

## 快速开始

### 1. 基本使用

```python
import torch
from src.coreset import CSReLCoreset
from src.configs import CSReLConfig
from src.datasets import get_mnist_loader

# 创建配置
config = CSReLConfig(
    dataset="MNIST",
    num_classes=10,
    num_epochs=50,
    batch_size=256,
    learning_rate=0.001,
    selection_ratio=0.1,  # 选择10%的样本
    class_balance=True,    # 启用类别平衡
    device="cuda"
)

# 准备数据
train_loader = get_mnist_loader(train=True)
train_data = []  # 收集所有训练数据
train_labels = []

for batch_x, batch_y in train_loader:
    train_data.append(batch_x)
    train_labels.append(batch_y)

train_data = torch.cat(train_data)
train_labels = torch.cat(train_labels)

# 初始化CSReL选择器
selector = CSReLCoreset(config=config)

# 训练参考模型
print("训练参考模型...")
reference_model, reference_losses = selector.train_reference_model(
    train_data=train_data,
    train_labels=train_labels,
    verbose=True
)

# 选择coreset
print("选择coreset...")
selected_indices = selector.select(
    train_data=train_data,
    train_labels=train_labels,
    verbose=True
)

print(f"选择了 {len(selected_indices)} 个样本")

# 获取选择统计信息
stats = selector.get_selection_stats(train_labels)
print(f"选择比例: {stats['selection_ratio']:.2%}")
print(f"类别分布: {stats['class_distribution']}")
```

### 2. 增量选择

```python
# 第一轮选择
selector = CSReLCoreset(config=config)
selector.train_reference_model(train_data, train_labels)
selected_indices = selector.select(train_data, train_labels)

# 第二轮增量选择(添加新样本)
new_indices = selector.select(
    train_data=train_data,
    train_labels=train_labels,
    incremental=True,
    current_indices=selected_indices
)
```

### 3. 使用自定义模型

```python
from src.models import CNN_MNIST

# 创建自定义模型
model = CNN_MNIST(num_classes=10)

# 作为参考模型
selector = CSReLCoreset(config=config, model=model)
selector.train_reference_model(train_data, train_labels)

# 使用训练中的模型进行选择
trained_model = ...  # 您训练的模型
selected_indices = selector.select(
    train_data=train_data,
    train_labels=train_labels,
    model=trained_model  # 使用训练中的模型
)
```

### 4. 保存和加载

```python
# 保存选择器状态
selector.save("csrel_checkpoint.pth")

# 加载选择器状态
new_selector = CSReLCoreset(config=config)
model = CNN_MNIST(num_classes=10)
new_selector.load("csrel_checkpoint.pth", model)
```

## 选择函数

除了CSReLCoreset类,还可以直接使用选择函数:

```python
from src.coreset import select_by_loss_diff, select_by_margin

# 基于可约损失选择
selected = select_by_loss_diff(
    losses=current_losses,
    reference_losses=reference_losses,
    num_samples=1000,
    class_balance=True,
    labels=train_labels,
    num_classes=10
)

# 基于分类边界选择
selected = select_by_margin(
    logits=model_outputs,
    labels=train_labels,
    num_samples=1000,
    class_balance=True,
    num_classes=10
)
```

## 配置参数说明

### CSReLConfig主要参数

- `dataset`: 数据集名称 ("MNIST", "CIFAR10", "CIFAR100")
- `num_classes`: 类别数量
- `num_epochs`: 参考模型训练轮数
- `batch_size`: 批次大小
- `learning_rate`: 学习率
- `selection_ratio`: 选择比例 (0.1表示选择10%)
- `class_balance`: 是否类别平衡
- `device`: 设备 ("cuda" 或 "cpu")

## 实现细节

### 可约损失计算

可约损失定义为:
```
reducible_loss = current_loss - reference_loss
```

其中:
- `current_loss`: 当前模型在样本上的损失
- `reference_loss`: 参考模型在样本上的损失

高可约损失的样本意味着当前模型在该样本上表现较差,而这些样本是参考模型已经学会的,因此更具训练价值。

### 类别平衡策略

当启用类别平衡时,样本按以下方式分配:

1. 计算每个类别应选择的样本数: `n_per_class = total_samples // num_classes`
2. 对每个类别,选择可约损失最高的样本
3. 余数样本分配给前面的类别

这确保了即使在类别不平衡的数据集中,coreset也能公平地代表所有类别。

## 性能优化建议

1. **参考模型训练**: 使用较少的epoch训练参考模型(20-50即可)
2. **批次计算**: 损失计算使用批处理以提高内存效率
3. **GPU加速**: 确保数据和模型在相同设备上
4. **缓存参考损失**: 对于相同数据集,可以保存并重用参考损失

## 常见问题

### Q: 如何确定选择比例?

A: 通常根据计算资源和任务需求:
- 资源受限: 0.05-0.1 (5%-10%)
- 标准: 0.1-0.2 (10%-20%)
- 资源充足: 0.2-0.3 (20%-30%)

### Q: 参考模型需要训练多久?

A: 通常20-50个epoch足够。参考模型只需达到合理性能,不需要完全收敛。

### Q: 何时使用增量选择?

A: 在持续学习场景或分批训练时有用。可以逐步扩充coreset而不需要重新选择所有样本。

## 引用

如果您在研究中使用CSReL方法,请引用相关论文。

## 许可证

本项目遵循相应许可证。
