# CSReL 版本对比文档

## 概述

本文档详细对比了 CSReL 的三个版本：
- **CSReL v1**：原始实现，一次性选择所有样本
- **CSReL Original**：论文原始算法的另一种实现
- **CSReL v2**：改进版本，使用增量选择机制

## 版本对比总览

| 特性 | CSReL v1 | CSReL Original | CSReL v2 |
|------|----------|----------------|----------|
| **选择策略** | 一次性选择 | 一次性选择 | 增量迭代选择 |
| **计算效率** | 快 | 中等 | 较慢 |
| **Coreset 质量** | 良好 | 良好 | 最优 |
| **内存使用** | 较高 | 中等 | 较低 |
| **适用场景** | 快速选择 | 标准实现 | 追求最优性能 |
| **实现文件** | `csrel_coreset.py` | - | `csrel_coreset_v2.py` |
| **代码行数** | ~400 行 | - | ~574 行 |

## 算法流程对比

### CSReL v1 流程

```
1. 训练参考模型
   └─ 在完整数据集上训练模型
   └─ 计算所有样本的参考损失

2. 选择 Coreset
   ├─ 计算当前模型损失（如果提供）
   ├─ 计算可约损失 = 当前损失 - 参考损失
   └─ 一次性选择所有目标样本
```

**特点**：
- 简单直接
- 计算快速
- 只需训练一个参考模型

### CSReL v2 流程

```
1. 训练参考模型
   └─ 在完整数据集上训练模型
   └─ 计算所有样本的参考损失

2. 初始化 Coreset
   └─ 随机选择少量样本（类别平衡）

3. 增量选择循环
   while coreset_size < target_size:
       ├─ 在当前 coreset 上训练模型
       ├─ 计算剩余样本的可约损失
       ├─ 选择可约损失最高的样本
       └─ 更新 coreset
```

**特点**：
- 迭代优化
- 逐步改进
- 需要多次训练模型

## 核心差异

### 1. 选择机制

#### CSReL v1
```python
def select(self, train_data, train_labels, model=None):
    # 一次性计算所有样本的可约损失
    current_losses = self._compute_losses(model, train_data, train_labels)
    reducible_loss = current_losses - self.reference_losses

    # 一次性选择所有样本
    selected = select_by_loss_diff(
        reducible_loss,
        num_samples=target_size,
        class_balance=True
    )
    return selected
```

#### CSReL v2
```python
def incremental_selection(self):
    # 初始化
    self._initialize_coreset()

    # 增量选择循环
    while len(self.coreset_ids) < self.config.coreset_size:
        # 在当前 coreset 上训练
        model = self._train_on_coreset()

        # 计算可约损失
        reducible_loss = self._compute_reducible_loss(model)

        # 选择一批样本
        new_ids = self._select_by_loss_diff(reducible_loss)

        # 更新 coreset
        self.coreset_ids.update(new_ids)
```

### 2. 可约损失计算

#### CSReL v1
- **参考模型**：在完整数据集上训练一次
- **当前模型**：可选，使用提供的模型或参考模型
- **可约损失**：一次性计算所有样本

#### CSReL v2
- **参考模型**：在完整数据集上训练一次
- **当前模型**：每轮都在当前 coreset 上重新训练
- **可约损失**：每轮重新计算剩余样本

### 3. 类别平衡

#### CSReL v1
```python
def select_by_loss_diff(losses, num_samples, class_balance, labels):
    if class_balance:
        # 按类别分配配额
        n_per_class = num_samples // num_classes
        for class_id in range(num_classes):
            # 选择该类别的 top-k 样本
            class_samples = [i for i in range(len(labels)) if labels[i] == class_id]
            top_samples = sorted(class_samples, key=lambda i: losses[i])[:n_per_class]
            selected.extend(top_samples)
```

#### CSReL v2
```python
# 初始化阶段：确保类别平衡
def _initialize_coreset(self):
    class_sizes = make_class_sizes(
        self.config.init_size,
        self.num_classes,
        self.full_y
    )
    for class_id, size in class_sizes.items():
        # 随机选择该类别的样本
        class_samples = np.where(self.full_y == class_id)[0]
        selected = np.random.choice(class_samples, size, replace=False)
        self.coreset_ids.update(selected)

# 增量选择阶段：按类别配额选择
def _select_by_loss_diff(self, reducible_loss):
    class_sizes = make_class_sizes(
        self.config.incremental_size,
        self.num_classes,
        self.full_y,
        exclude_ids=self.coreset_ids
    )
    # 类似的类别平衡选择逻辑
```

### 4. 内存管理

#### CSReL v1
- 需要存储所有样本的参考损失
- 需要存储当前模型损失（如果提供）
- 内存使用与数据集大小成正比

#### CSReL v2
- 存储参考损失
- 每轮只计算剩余样本的损失
- 使用临时文件存储中间结果
- 自动清理临时文件

## API 对比

### 初始化

#### CSReL v1
```python
selector = CSReLCoreset(
    config=CSReLConfig(
        dataset="MNIST",
        num_classes=10,
        selection_ratio=0.1
    )
)
```

#### CSReL v2
```python
selector = CSReLCoresetV2(
    model=model,
    full_dataset=train_dataset,
    config=CSReLConfigV2(
        dataset="MNIST",
        num_classes=10,
        coreset_size=1000
    )
)
```

### 训练参考模型

#### CSReL v1
```python
ref_model, ref_losses = selector.train_reference_model(
    train_data=train_data,
    train_labels=train_labels
)
```

#### CSReL v2
```python
# 参考模型训练在 select() 内部自动完成
coreset_indices = selector.select()
```

### 选择 Coreset

#### CSReL v1
```python
selected_indices = selector.select(
    train_data=train_data,
    train_labels=train_labels,
    model=current_model  # 可选
)
```

#### CSReL v2
```python
selected_indices = selector.select()
```

## 配置参数对比

### CSReLConfig (v1)

```python
config = CSReLConfig(
    dataset="MNIST",
    num_classes=10,
    num_epochs=50,
    batch_size=256,
    learning_rate=0.001,
    selection_ratio=0.1,      # 选择比例
    class_balance=True,
    device="cuda"
)
```

### CSReLConfigV2 (v2)

```python
config = CSReLConfigV2(
    dataset="MNIST",
    num_classes=10,
    coreset_size=1000,         # 目标大小
    incremental_size=100,      # 增量步长
    init_size=100,             # 初始大小
    ref_epochs=100,            # 参考模型轮数
    inc_epochs=10,             # 增量训练轮数
    ref_lr=0.01,               # 参考模型学习率
    inc_lr=0.01,               # 增量训练学习率
    batch_size=128,
    use_cuda=True
)
```

## 性能对比

### 计算复杂度

#### CSReL v1
- **时间复杂度**：O(N × E_ref + N × log N)
  - N：数据集大小
  - E_ref：参考模型训练轮数
- **空间复杂度**：O(N)

#### CSReL v2
- **时间复杂度**：O(N × E_ref + M × E_inc × R)
  - M：coreset 大小
  - E_inc：增量训练轮数
  - R：增量轮数 = (M - init_size) / incremental_size
- **空间复杂度**：O(M)

### 实验性能（估算）

| 数据集 | 方法 | 选择时间 | Coreset 大小 | 下游准确率 |
|--------|------|----------|--------------|------------|
| MNIST | v1 | ~5 分钟 | 1000 | 98.5% |
| MNIST | v2 | ~15 分钟 | 1000 | 98.8% |
| CIFAR-10 | v1 | ~30 分钟 | 5000 | 85.2% |
| CIFAR-10 | v2 | ~90 分钟 | 5000 | 86.1% |

*注：实际性能取决于硬件配置和超参数设置*

## 使用场景推荐

### 使用 CSReL v1 的场景

1. **快速原型开发**
   - 需要快速验证想法
   - 计算资源有限

2. **大规模数据集**
   - 数据集非常大（>100k 样本）
   - 计算时间敏感

3. **基准对比**
   - 作为 baseline 方法
   - 需要可重复的结果

### 使用 CSReL v2 的场景

1. **追求最优性能**
   - 下游任务性能至关重要
   - 有足够的计算资源

2. **中小规模数据集**
   - 数据集适中（<50k 样本）
   - 可以接受较长的选择时间

3. **高质量 Coreset**
   - 需要高质量的样本子集
   - 愿意投入更多计算资源

## 代码迁移指南

### 从 v1 迁移到 v2

#### 1. 导入更新

```python
# v1
from src.coreset import CSReLCoreset
from src.configs import CSReLConfig

# v2
from src.coreset import CSReLCoresetV2
from src.configs import CSReLConfigV2
```

#### 2. 配置更新

```python
# v1
config = CSReLConfig(
    dataset="MNIST",
    num_classes=10,
    selection_ratio=0.1,  # 使用比例
    num_epochs=50
)

# v2
config = CSReLConfigV2(
    dataset="MNIST",
    num_classes=10,
    coreset_size=1000,    # 使用绝对数量
    ref_epochs=50,
    inc_epochs=10
)
```

#### 3. 选择器初始化

```python
# v1
selector = CSReLCoreset(config=config)

# v2
model = SimpleCNN(num_classes=10)
selector = CSReLCoresetV2(
    model=model,
    full_dataset=train_dataset,
    config=config
)
```

#### 4. 执行选择

```python
# v1
selector.train_reference_model(train_data, train_labels)
selected = selector.select(train_data, train_labels)

# v2
selected = selector.select()  # 自动训练参考模型
```

## 最佳实践

### 1. 数据准备

#### CSReL v1
```python
# 需要手动准备数据
train_loader = get_data_loader(train=True)
train_data = []
train_labels = []
for batch_x, batch_y in train_loader:
    train_data.append(batch_x)
    train_labels.append(batch_y)
train_data = torch.cat(train_data)
train_labels = torch.cat(train_labels)
```

#### CSReL v2
```python
# 直接使用数据集
train_dataset = get_dataset(train=True)
selector = CSReLCoresetV2(model, train_dataset, config)
```

### 2. 模型管理

#### CSReL v1
```python
# 模型由选择器管理
selector.train_reference_model(train_data, train_labels)
selected = selector.select(train_data, train_labels)
```

#### CSReL v2
```python
# 需要提供模型
model = SimpleCNN(num_classes=10)
selector = CSReLCoresetV2(model, train_dataset, config)
selected = selector.select()
```

### 3. 验证集使用

#### CSReL v1
```python
# 不支持验证集
selector.train_reference_model(train_data, train_labels)
```

#### CSReL v2
```python
# 支持验证集用于早停
selector = CSReLCoresetV2(
    model,
    train_dataset,
    config,
    eval_dataset=val_dataset  # 添加验证集
)
```

## 关键差异总结

| 方面 | CSReL v1 | CSReL v2 |
|------|----------|----------|
| **设计哲学** | 快速简单 | 高质量优化 |
| **选择方式** | 一次性 | 增量迭代 |
| **模型训练** | 一次参考模型 | 参考模型 + 多轮增量训练 |
| **数据格式** | Numpy 数组 | PyTorch 数据集 |
| **配置方式** | 比例 | 绝对数量 |
| **验证集** | 不支持 | 支持 |
| **临时文件** | 无 | 自动清理 |
| **内存效率** | 中等 | 较高 |
| **计算时间** | 短 | 长 |
| **Coreset 质量** | 良好 | 最优 |

## 建议

### 选择建议

1. **首次使用**：从 CSReL v1 开始，理解基本概念
2. **生产环境**：根据需求选择
   - 快速部署 → v1
   - 最佳性能 → v2
3. **科研实验**：使用 v2 获得更强的 baseline

### 参数调优建议

#### CSReL v1
- 重点关注 `selection_ratio` 和 `num_epochs`
- 类别不平衡数据集必须启用 `class_balance`

#### CSReL v2
- 重点关注 `coreset_size`、`incremental_size` 和 `init_size`
- 根据数据集大小调整 `ref_epochs` 和 `inc_epochs`
- 不平衡数据集增加 `init_size`

### 性能优化建议

1. **CSReL v1**
   - 使用 GPU 加速
   - 批量计算损失
   - 缓存参考损失

2. **CSReL v2**
   - 合理设置 `incremental_size`（太大降低质量，太慢增加时间）
   - 使用验证集早停
   - 定期清理临时文件

## 结论

CSReL v1 和 v2 都是有效的 coreset 选择方法，各有优势：

- **CSReL v1**：简单快速，适合快速原型和大规模数据
- **CSReL v2**：质量更高，适合追求最优性能的场景

选择哪个版本取决于具体需求、计算资源和性能要求。

## 引用

如果您在研究中使用 CSReL 方法，请引用相关论文。

## 许可证

本项目遵循相应许可证。
