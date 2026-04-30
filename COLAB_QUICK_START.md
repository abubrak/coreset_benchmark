# Colab持续学习实验 - 快速开始指南

## 🚀 快速开始

### 1. 环境准备

```python
# 在Colab中运行
!pip install torch torchvision numpy scipy scikit-learn matplotlib seaborn tqdm pandas -q

# 挂载Google Drive（可选）
from google.colab import drive
drive.mount('/content/drive')

# 切换到项目目录
import os
os.chdir('/content/f--paper-code/coreset_benchmark')
```

### 2. 验证GPU环境

```python
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

预期输出（T4 GPU）:
```
PyTorch版本: 2.x.x
CUDA可用: True
GPU: Tesla T4
显存: 15.75 GB
```

## 📊 快速测试

### 测试1: 验证集成状态

```python
# 运行快速验证测试
!python quick_integration_test.py
```

预期输出:
```
[OK] 所有测试通过！
```

### 测试2: BCSR性能测试

```python
# 在BCSR_Colab_Guide.ipynb中运行性能测试
# 见notebooks/BCSR_Colab_Guide.ipynb
```

## 🎯 运行实验

### 小规模快速测试（推荐首先运行）

```bash
python experiments/continual_learning.py \
    --dataset MNIST \
    --num_tasks 2 \
    --num_classes_per_task 2 \
    --selection_method bcsr \
    --memory_size 1000 \
    --num_epochs 5 \
    --save_results
```

**预期时间**: 2-3分钟
**预期准确率**: Task 0: >95%, Task 1: >90%

### 中等规模实验

```bash
python experiments/continual_learning.py \
    --dataset MNIST \
    --num_tasks 3 \
    --num_classes_per_task 2 \
    --selection_method bcsr \
    --memory_size 2000 \
    --num_epochs 10 \
    --save_results
```

**预期时间**: 5-8分钟
**内存使用**: <8GB

### 方法对比实验

```bash
# 创建并运行对比脚本
cat > run_comparison.sh << 'EOF'
#!/bin/bash

methods=("random" "uniform" "loss" "margin" "gradient" "bcsr")

for method in "${methods[@]}"; do
    echo "Running $method..."
    python experiments/continual_learning.py \
        --dataset MNIST \
        --num_tasks 3 \
        --num_classes_per_task 2 \
        --selection_method $method \
        --memory_size 2000 \
        --num_epochs 10 \
        --save_results \
        --output_dir "./results/$method"
done
EOF

chmod +x run_comparison.sh
./run_comparison.sh
```

## 🔧 不同方法的配置建议

### BCSR（推荐用于中小规模）

```bash
python experiments/continual_learning.py \
    --dataset MNIST \
    --num_tasks 3 \
    --selection_method bcsr \
    --memory_size 2000
```

**优势**:
- ✅ 理论上最优（双层优化）
- ✅ 性能优化后速度可接受
- ✅ 自动类别平衡

**注意**:
- 大数据集(>10000)会自动预采样
- 首次选择可能需要30-60秒

### Uniform（最快，适合大规模）

```bash
python experiments/continual_learning.py \
    --dataset MNIST \
    --num_tasks 5 \
    --selection_method uniform \
    --memory_size 2000
```

**优势**:
- ✅ 速度最快（<1秒）
- ✅ 类别平衡
- ✅ 内存占用小

**适用场景**:
- 快速原型验证
- 大规模实验
- 作为baseline

### CSReL（适合需要参考模型的场景）

```bash
python experiments/continual_learning.py \
    --dataset MNIST \
    --num_tasks 3 \
    --selection_method csrel \
    --memory_size 2000
```

**优势**:
- ✅ 基于reducible loss
- ✅ 考虑空间表示

**注意**:
- 首次需要训练参考模型（额外1-2分钟）
- 适合有充足时间的情况

### Bilevel（简化版，适合快速实验）

```bash
python experiments/continual_learning.py \
    --dataset MNIST \
    --num_tasks 3 \
    --selection_method bilevel \
    --memory_size 2000
```

**优势**:
- ✅ kernel herding近似
- ✅ 类别平衡
- ✅ 速度适中

**适用场景**:
- 中等规模实验
- 需要多样性选择

## 📈 结果分析

### 查看结果

```python
import json
import pandas as pd

# 读取结果文件
with open('./results/results.json', 'r') as f:
    results = json.load(f)

# 转换为DataFrame
df = pd.DataFrame(results)
print(df)
```

### 绘制对比图

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 假设你有多个方法的结果
methods = ['random', 'uniform', 'bcsr', 'csrel']
accuracies = {
    'Task 0': [0.95, 0.96, 0.97, 0.96],
    'Task 1': [0.85, 0.88, 0.91, 0.89],
    'Task 2': [0.75, 0.80, 0.85, 0.82]
}

# 绘制条形图
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(methods))
width = 0.2

for i, (task, accs) in enumerate(accuracies.items()):
    ax.bar(x + i*width, accs, width, label=task)

ax.set_xlabel('Method')
ax.set_ylabel('Accuracy')
ax.set_title('Coreset Selection Methods Comparison')
ax.set_xticks(x + width)
ax.set_xticklabels(methods)
ax.legend()
plt.tight_layout()
plt.show()
```

## ⚠️ 常见问题解决

### OOM (内存不足)

**问题**: `RuntimeError: CUDA out of memory`

**解决方案**:
```bash
# 减少缓冲区大小
--memory_size 1000  # 从2000减少到1000

# 或使用uniform方法
--selection_method uniform

# 或减少任务数
--num_tasks 2  # 从5减少到2
```

### BCSR太慢

**问题**: BCSR选择时间超过预期

**检查**:
```python
# 查看是否使用了预采样优化
# 应该看到: "[性能优化] 数据集过大(X样本)，先随机预采样到3000个样本"
```

**解决方案**:
```bash
# 减少数据集大小
# 或使用uniform方法
--selection_method uniform
```

### 类别不平衡

**问题**: 某些类别样本过多

**检查**:
```python
# 查看类别分布
# 应该看到: "类别分布: {0: 500, 1: 500}"
```

**解决方案**:
```bash
# 使用uniform或bilevel方法（自动类别平衡）
--selection_method uniform
```

## 💡 性能优化技巧

### 1. 使用混合精度（实验性）

```python
# 在代码中添加（需要修改代码）
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

### 2. 调整批大小

```bash
# 在continual_learning.py中调整
batch_size=256  # 从64增加到256
```

### 3. 使用更小的模型

```python
# 修改模型架构
# 使用更少的卷积核或层数
```

## 📚 参考文档

- [完整集成文档](INTEGRATION_COMPLETE.md)
- [BCSR性能分析](BCSR_PERFORMANCE_ANALYSIS.md)
- [Colab交互式指南](notebooks/BCSR_Colab_Guide.ipynb)

## 🎯 推荐实验流程

### 第一步：快速验证
```bash
# 运行小规模测试验证环境
python quick_integration_test.py
```

### 第二步：Baseline对比
```bash
# 对比baseline方法
for method in random uniform loss margin; do
    python experiments/continual_learning.py \
        --dataset MNIST --num_tasks 2 \
        --selection_method $method --memory_size 1000
done
```

### 第三步：BCSR测试
```bash
# 测试BCSR方法
python experiments/continual_learning.py \
    --dataset MNIST --num_tasks 2 \
    --selection_method bcsr --memory_size 1000
```

### 第四步：完整实验
```bash
# 运行完整实验对比所有方法
# (如果有足够时间)
```

---

**最后更新**: 2026-04-30
**Colab环境**: T4 GPU (16GB)
**Python版本**: 3.8+
**PyTorch版本**: 2.0+
