# CSReL 使用指南

## 正确使用方式

CSReL (Classwise Spatial Representation Learning) 需要两个不同的模型：

1. **Reference Model**: 在完整训练集上训练的参考模型
2. **Current Model**: 当前的、未充分训练的模型（用于计算可约损失）

## 错误示例（会抛出异常）

```python
# ❌ 错误：model=None
selector = CSReLCoreset(config)
selector.train_reference_model(train_data, train_labels)
selected = selector.select(train_data, train_labels, model=None)  # ValueError!
```

## 正确示例

```python
# ✅ 正确：提供不同的模型
selector = CSReLCoreset(config)

# 训练参考模型
ref_model, _ = selector.train_reference_model(train_data, train_labels)

# 创建未训练的模型
from src.models import get_model
current_model = get_model(dataset='MNIST', num_classes=10)

# 使用当前模型进行选择
selected = selector.select(train_data, train_labels, model=current_model)
```

## 为什么需要两个模型？

CSReL 通过**可约损失 (reducible loss)** 选择样本：
- `reducible_loss = current_loss - reference_loss`
- Reference model 代表"理想"训练后的损失
- Current model 代表当前的训练状态
- 可约损失大的样本更值得学习

如果使用同一个模型，`current_loss ≈ reference_loss`，所有样本的 `reducible_loss ≈ 0`，选择失去意义。

## API 参考

### train_reference_model()

```python
ref_model, ref_losses = selector.train_reference_model(
    train_data,      # 训练数据
    train_labels,    # 训练标签
    val_data=None,   # 可选验证数据
    val_labels=None, # 可选验证标签
    verbose=True     # 是否打印进度
)
```

### select()

```python
selected_indices = selector.select(
    train_data,      # 训练数据
    train_labels,    # 训练标签
    model,           # ⚠️ 必需：当前模型（不能为 None）
    incremental=False,           # 是否增量选择
    current_indices=None,        # 当前选择的索引（incremental=True 时必需）
    verbose=True     # 是否打印进度
)
```

## 常见问题

### Q: 我应该使用什么样的 current_model？

A: `current_model` 应该是与 `reference_model` 不同的模型：
- **随机初始化的模型**：选择有代表性的样本
- **部分训练的模型**：选择对当前模型最有价值的样本
- **不同架构的模型**：选择跨架构有用的样本

### Q: 如何从检查点恢复？

A: 使用 `save()` 和 `load()` 方法：

```python
# 保存
selector.save('csrel_checkpoint.pth')

# 加载
new_selector = CSReLCoreset(config)
model = get_model('MNIST', num_classes=10)
new_selector.load('csrel_checkpoint.pth', model)
```

### Q: 增量选择如何工作？

A: 增量选择允许逐步添加样本：

```python
# 第一次选择
selected = selector.select(train_data, train_labels, model=model1)

# 第二次选择（添加更多样本）
updated = selector.select(
    train_data, train_labels,
    model=model2,
    incremental=True,
    current_indices=selected  # 保留已选择的样本
)
```
