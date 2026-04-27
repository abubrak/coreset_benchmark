# BCSR方法使用说明

## 概述

BCSR (Bilevel Coreset Selection with Reweighting) 是一种基于双层优化的coreset选择方法。

## 核心组件

### 1. BCSRTraining 类

负责训练阶段的双层优化：

- **内层优化**: 在加权数据上训练代理模型
- **外层优化**: 基于验证损失更新样本权重

#### 主要参数

```python
BCSRTraining(
    model: nn.Module,              # PyTorch模型
    kernel_fn: Optional[Callable], # 核函数（可选）
    learning_rate_inner: float = 0.01,    # 内层学习率
    learning_rate_outer: float = 0.1,     # 外层学习率
    num_inner_steps: int = 50,           # 内层优化步数
    num_outer_steps: int = 20,           # 外层优化步数
    lmbda: float = 0.0,                  # L2正则化系数
    device: str = 'cpu'
)
```

#### 使用示例

```python
from src.training.bcsr_training import BCSRTraining
from src.models.cnn import CNN_MNIST
from torch.utils.data import DataLoader

# 创建模型
model = CNN_MNIST(num_classes=10)

# 创建训练器
trainer = BCSRTraining(
    model=model,
    learning_rate_inner=0.01,
    learning_rate_outer=0.1,
    num_inner_steps=50,
    num_outer_steps=20,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# 训练并获取权重
weights = trainer.train(train_loader, val_loader, n_samples=len(train_dataset))

# 访问训练历史
print(f"训练损失: {trainer.history['train_loss']}")
print(f"训练准确率: {trainer.history['train_acc']}")
```

### 2. BCSRCoreset 类

负责coreset选择：

- **有模型模式**: 使用完整的双层优化框架
- **无模型模式**: 使用简化的核方法

#### 主要参数

```python
BCSRCoreset(
    kernel_fn: Optional[Callable] = None,
    learning_rate_inner: float = 0.01,
    learning_rate_outer: float = 0.1,
    num_inner_steps: int = 50,
    num_outer_steps: int = 20,
    lmbda: float = 0.0,
    device: str = 'cpu',
    random_state: Optional[int] = None
)
```

#### 使用示例

##### 方式1: 使用模型进行coreset选择（推荐）

```python
from src.coreset.bcsr_coreset import BCSRCoreset
from src.models.cnn import CNN_MNIST
import torch

# 准备数据
X = torch.randn(1000, 1, 28, 28)  # MNIST图像
y = torch.randint(0, 10, (1000,))  # 标签

# 创建模型
model = CNN_MNIST(num_classes=10)

# 创建选择器
selector = BCSRCoreset(
    learning_rate_inner=0.01,
    learning_rate_outer=0.1,
    num_inner_steps=50,
    num_outer_steps=20,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    random_state=42
)

# 选择coreset
selected_X, selected_y, info = selector.coreset_select(
    X=X,
    y=y,
    coreset_size=100,  # 选择100个样本
    model=model,       # 使用模型
    validation_split=0.2,
    batch_size=128
)

print(f"选择了 {len(selected_X)} 个样本")
print(f"权重统计: 均值={info['weights_mean']:.4f}")
```

##### 方式2: 不使用模型（简化版本）

```python
# 对于扁平化特征数据
X = torch.randn(1000, 50)  # 特征向量
y = torch.randint(0, 5, (1000,))

selector = BCSRCoreset(random_state=42)

selected_X, selected_y, info = selector.coreset_select(
    X=X,
    y=y,
    coreset_size=100,
    model=None  # 不使用模型，使用核方法
)
```

## 输出说明

### coreset_select 返回值

```python
selected_X, selected_y, info = selector.coreset_select(...)
```

- **selected_X**: 选择的coreset数据 (numpy array)
- **selected_y**: 选择的coreset标签 (numpy array)
- **info**: 包含详细信息的字典
  - `method`: 使用的方法名称
  - `n_samples`: 原始样本数
  - `coreset_size`: coreset大小
  - `weights_mean/std/min/max`: 权重统计信息
  - `selected_indices`: 选择的样本索引
  - `all_weights`: 所有样本的权重

## 工具方法

### projection_onto_simplex

将向量投影到单纯形（非负且和为1）：

```python
selector = BCSRCoreset()
v = np.random.randn(10)
w = selector.projection_onto_simplex(v)

# 验证
assert np.all(w >= 0)        # 非负
assert np.abs(w.sum() - 1.0) < 1e-6  # 和为1
```

## 注意事项

1. **计算复杂度**: BCSR方法涉及多次模型训练，计算成本较高
2. **内存使用**: 需要存储完整的核矩阵（如果使用核方法）
3. **参数调优**: 学习率和迭代次数需要根据具体任务调整
4. **设备选择**: 建议使用GPU加速计算

## 与其他方法的比较

- **vs 随机采样**: BCSR能选择更有代表性的样本
- **vs 基于梯度的方法**: BCSR考虑了验证集性能
- **vs 基于聚类的方法**: BCSR考虑了样本对模型训练的影响

## 参考文献

BCSR方法基于双层优化框架，用于同时优化样本权重和模型参数。
