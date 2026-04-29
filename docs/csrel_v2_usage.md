# CSReL v2 使用指南

## 概述

CSReL v2 (Classwise Spatial Representation Learning v2) 是 CSReL 方法的改进版本，通过**增量选择机制**构建高质量的 coreset。相比 v1 版本，v2 版本采用迭代优化策略，逐步选择最具信息量的样本。

### 主要特性

- **增量选择机制**：通过迭代优化逐步构建 coreset
- **可约损失计算**：基于参考模型和当前模型的损失差异选择样本
- **类别平衡**：确保每个类别都有足够的代表样本
- **自动停止**：达到目标 coreset 大小后自动停止
- **内存优化**：支持临时文件自动清理

### 与 CSReL v1 的主要区别

| 特性 | CSReL v1 | CSReL v2 |
|------|----------|----------|
| 选择策略 | 一次性选择所有样本 | 增量迭代选择 |
| 计算效率 | 较快 | 较慢（需多轮训练） |
| Coreset 质量 | 良好 | 更优 |
| 内存使用 | 较高 | 较低（增量训练） |
| 适用场景 | 快速选择 | 追求最优性能 |

## 快速开始

### 1. 基本使用

```python
import torch
from torch.utils.data import DataLoader
from src.coreset import CSReLCoresetV2
from src.configs import CSReLConfigV2
from src.models import SimpleCNN
from src.datasets import get_mnist

# 准备数据
train_dataset = get_mnist(train=True)

# 创建配置
config = CSReLConfigV2(
    dataset="MNIST",
    num_classes=10,
    coreset_size=1000,        # 目标 coreset 大小
    incremental_size=100,     # 每轮增量选择的大小
    init_size=100,            # 初始随机采样大小
    ref_epochs=100,           # 参考模型训练轮数
    inc_epochs=10,            # 增量训练轮数
    batch_size=128,
    device="cuda"
)

# 创建模型
model = SimpleCNN(num_classes=10)

# 创建选择器
selector = CSReLCoresetV2(
    model=model,
    full_dataset=train_dataset,
    config=config
)

# 执行选择
print("开始 CSReL v2 选择...")
coreset_indices = selector.select()

print(f"选择了 {len(coreset_indices)} 个样本")

# 获取选择的样本
selected_data = []
selected_labels = []
for idx in coreset_indices:
    data, label = train_dataset[idx]
    selected_data.append(data)
    selected_labels.append(label)

# 创建 Coreset 数据集
coreset_dataset = torch.utils.data.TensorDataset(
    torch.stack(selected_data),
    torch.tensor(selected_labels)
)
```

### 2. 使用验证集

```python
# 准备验证集（用于早停）
val_dataset = get_mnist(train=False)

selector = CSReLCoresetV2(
    model=model,
    full_dataset=train_dataset,
    config=config,
    eval_dataset=val_dataset  # 添加验证集
)

coreset_indices = selector.select()
```

### 3. 自定义数据增强

```python
from torchvision import transforms

# 自定义训练变换
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 自定义测试变换
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

config = CSReLConfigV2(
    dataset="MNIST",
    num_classes=10,
    coreset_size=1000,
    train_transform=train_transform,
    test_transform=test_transform,
    # ... 其他参数
)
```

## 命令行使用

CSReL v2 已集成到实验框架中，可以通过命令行使用：

### 基本命令

```bash
python experiments/data_summarization.py \
    --dataset MNIST \
    --method csrel_v2 \
    --selection_ratio 0.1 \
    --batch_size 128 \
    --epochs 50
```

### 自定义参数

```bash
python experiments/data_summarization.py \
    --dataset CIFAR10 \
    --method csrel_v2 \
    --selection_ratio 0.1 \
    --csrel_init_size 100 \
    --csrel_selection_steps 100 \
    --csrel_cur_lr 0.01 \
    --csrel_cur_steps 10 \
    --csrel_ref_epochs 100 \
    --csrel_ref_lr 0.01 \
    --batch_size 128 \
    --epochs 50
```

### 快速测试

```bash
python experiments/data_summarization.py \
    --dataset MNIST \
    --method csrel_v2 \
    --selection_ratio 0.1 \
    --num_samples 1000 \
    --epochs 1 \
    --csrel_init_size 10 \
    --csrel_selection_steps 10 \
    --csrel_ref_epochs 2 \
    --csrel_cur_steps 1
```

## 配置参数详解

### CSReLConfigV2 参数说明

#### 核心选择参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `coreset_size` | int | 1000 | 目标 coreset 大小 |
| `incremental_size` | int | 100 | 每轮增量选择的大小 |
| `init_size` | int | 100 | 初始随机采样大小 |

#### 参考模型训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ref_epochs` | int | 100 | 参考模型训练轮数 |
| `ref_lr` | float | 0.01 | 参考模型学习率 |
| `ref_opt_type` | str | 'sgd' | 优化器类型 ('sgd' 或 'adam') |
| `batch_size` | int | 128 | 批次大小 |
| `weight_decay` | float | 5e-4 | 权重衰减 |
| `early_stop` | int | 10 | 早停耐心值 |

#### 增量训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `inc_epochs` | int | 10 | 增量训练轮数 |
| `inc_lr` | float | 0.01 | 增量训练学习率 |
| `inc_opt_type` | str | 'sgd' | 优化器类型 |

#### 损失函数参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ce_factor` | float | 1.0 | 交叉熵损失权重 |
| `mse_factor` | float | 0.0 | 知识蒸馏损失权重 |
| `kd_mode` | str | 'mse' | 知识蒸馏模式 ('mse' 或 'ce') |

#### 其他参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dataset` | str | - | 数据集名称 ('MNIST', 'CIFAR10', 'CIFAR100') |
| `num_classes` | int | - | 类别数量 |
| `use_cuda` | bool | True | 是否使用 CUDA |
| `temp_dir` | str | './temp_csrel_v2' | 临时文件目录 |
| `train_transform` | transform | - | 训练数据增强 |
| `test_transform` | transform | - | 测试数据增强 |

## 算法流程

CSReL v2 的增量选择算法包含以下步骤：

```
1. 训练参考模型
   ├─ 在完整数据集上训练模型
   └─ 计算所有样本的参考损失

2. 初始化 Coreset
   ├─ 随机选择 init_size 个样本
   └─ 确保类别平衡

3. 增量选择循环
   while coreset_size < target_size:
       ├─ 在当前 coreset 上训练模型
       ├─ 计算剩余样本的可约损失
       ├─ 选择可约损失最高的样本
       └─ 更新 coreset
```

### 可约损失计算

可约损失定义为：
```
reducible_loss = reference_loss - current_loss
```

- `reference_loss`：参考模型在样本上的损失
- `current_loss`：当前模型在样本上的损失
- 高可约损失的样本更具训练价值

### 类别平衡策略

- **初始化阶段**：确保每个类别至少有 `init_size // num_classes` 个样本
- **增量选择阶段**：按类别配额选择，确保类别分布平衡

## 推荐配置

### MNIST 数据集

```python
config = CSReLConfigV2(
    dataset="MNIST",
    num_classes=10,
    coreset_size=1000,      # 10% of 10k samples
    incremental_size=100,
    init_size=100,
    ref_epochs=50,          # MNIST 收敛快
    inc_epochs=5,
    ref_lr=0.01,
    inc_lr=0.01
)
```

### CIFAR-10 数据集

```python
config = CSReLConfigV2(
    dataset="CIFAR10",
    num_classes=10,
    coreset_size=5000,      # 10% of 50k samples
    incremental_size=500,
    init_size=500,
    ref_epochs=100,
    inc_epochs=10,
    ref_lr=0.01,
    inc_lr=0.01
)
```

### CIFAR-100 数据集

```python
config = CSReLConfigV2(
    dataset="CIFAR100",
    num_classes=100,
    coreset_size=10000,     # 20% of 50k samples
    incremental_size=1000,
    init_size=1000,
    ref_epochs=150,         # 更多类别需要更长时间
    inc_epochs=15,
    ref_lr=0.01,
    inc_lr=0.01
)
```

## 性能优化建议

### 1. 参考模型训练

- 使用较少的 epoch 训练参考模型（50-100 即可）
- 参考模型只需达到合理性能，不需要完全收敛
- 使用早停机制避免过拟合

### 2. 增量训练

- 每轮增量训练使用较少的 epoch（5-10）
- 学习率可以略高于参考模型训练
- 使用较小的批次大小以提高效率

### 3. 内存优化

- 启用 `use_cuda` 以利用 GPU 加速
- 设置合理的 `batch_size` 避免内存溢出
- 定期清理临时文件

### 4. 类别平衡

- 对于不平衡数据集，启用类别平衡
- 增加 `init_size` 确保初始覆盖所有类别
- 调整 `incremental_size` 以平衡选择速度和质量

## 常见问题

### Q: CSReL v2 比 v1 慢这么多，值得使用吗？

A: CSReL v2 通过增量选择机制获得更高质量的 coreset，虽然计算时间更长，但在 downstream 任务中通常表现更好。如果时间紧迫，可以使用较小的 `incremental_size` 或减少 `ref_epochs` 和 `inc_epochs`。

### Q: 如何选择合适的 coreset_size？

A: 通常根据计算资源和任务需求：
- 资源受限：5%-10% 的原始数据集大小
- 标准：10%-20%
- 追求性能：20%-30%

### Q: 增量选择需要多少轮？

A: 计算公式：`rounds = (coreset_size - init_size) / incremental_size`
- coreset_size=1000, init_size=100, incremental_size=100 → 9 轮

### Q: 如何处理不平衡数据集？

A: CSReL v2 内置类别平衡机制，会自动确保每个类别都有足够的代表样本。对于严重不平衡的数据集，可以增加 `init_size`。

### Q: 临时文件在哪里？

A: 默认在 `./temp_csrel_v2` 目录，可以通过 `temp_dir` 参数自定义。选择完成后会自动清理。

### Q: 如何从检查点恢复？

A: 使用 `save()` 和 `load()` 方法：

```python
# 保存
selector.save("csrel_v2_checkpoint.pth")

# 加载
new_selector = CSReLCoresetV2(model, dataset, config)
new_selector.load("csrel_v2_checkpoint.pth")
```

## 实现细节

### 数据格式要求

CSReL v2 要求数据集返回 `(id, sample, label)` 格式的数据。如果使用标准 PyTorch 数据集，会自动进行包装转换。

### 临时文件管理

CSReL v2 在训练过程中会创建临时文件：
- 参考模型检查点
- 每轮增量训练的模型检查点
- 这些文件会在选择完成后自动清理

### 多进程支持

CSReL v2 支持多进程数据加载，可以通过 DataLoader 的 `num_workers` 参数设置。

## 示例：完整的训练流程

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from src.coreset import CSReLCoresetV2
from src.configs import CSReLConfigV2
from src.models import SimpleCNN
from src.datasets import get_mnist

# 1. 准备数据
train_dataset = get_mnist(train=True)
test_dataset = get_mnist(train=False)

# 2. 创建配置
config = CSReLConfigV2(
    dataset="MNIST",
    num_classes=10,
    coreset_size=1000,
    incremental_size=100,
    init_size=100,
    ref_epochs=50,
    inc_epochs=5
)

# 3. 创建模型
model = SimpleCNN(num_classes=10)

# 4. 执行 coreset 选择
selector = CSReLCoresetV2(model, train_dataset, config)
coreset_indices = selector.select()

# 5. 创建 coreset 数据集
coreset_dataset = Subset(train_dataset, coreset_indices)
coreset_loader = DataLoader(coreset_dataset, batch_size=128, shuffle=True)

# 6. 在 coreset 上训练最终模型
final_model = SimpleCNN(num_classes=10)
optimizer = optim.SGD(final_model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

for epoch in range(50):
    for batch_x, batch_y in coreset_loader:
        optimizer.zero_grad()
        outputs = final_model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

# 7. 评估
test_loader = DataLoader(test_dataset, batch_size=128)
correct = 0
total = 0

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs = final_model(batch_x)
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

accuracy = 100 * correct / total
print(f"测试准确率: {accuracy:.2f}%")
```

## 引用

如果您在研究中使用 CSReL v2 方法，请引用相关论文。

## 许可证

本项目遵循相应许可证。
