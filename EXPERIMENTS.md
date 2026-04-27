# 实验指南

本文档提供了Coreset选择方法实验的完整指南。

## 目录

1. [环境设置](#环境设置)
2. [数据准备](#数据准备)
3. [实验类型](#实验类型)
4. [运行实验](#运行实验)
5. [结果分析](#结果分析)
6. [常见问题](#常见问题)

## 环境设置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖包括：
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Jupyter Notebook
- scikit-learn
- tqdm

### 2. 创建目录结构

```bash
# 创建必要的目录
mkdir -p data
mkdir -p results/data_summarization
mkdir -p results/continual_learning
mkdir -p figures
mkdir -p tables
mkdir -p logs
```

## 数据准备

### 支持的数据集

实验支持以下数据集：

1. **MNIST** (60,000训练样本, 10类别)
2. **CIFAR10** (50,000训练样本, 10类别)
3. **CIFAR100** (50,000训练样本, 100类别)

数据集会在首次运行时自动下载。

### 数据集统计

| 数据集 | 训练集大小 | 测试集大小 | 类别数 | 图像大小 | 通道数 |
|--------|-----------|-----------|--------|---------|--------|
| MNIST | 60,000 | 10,000 | 10 | 28 | 1 |
| CIFAR10 | 50,000 | 10,000 | 10 | 32 | 3 |
| CIFAR100 | 50,000 | 10,000 | 100 | 32 | 3 |

## 实验类型

本项目支持两种类型的实验：

### 1. 数据摘要实验

评估coreset选择方法在数据压缩任务中的性能。

**目标**：使用少量代表性样本训练模型，达到接近完整数据集的性能。

**评估指标**：
- 测试准确率
- 性能下降（与完整训练集的差距）
- 选择时间
- 训练时间

**支持的方法**：
- `random`: 随机选择
- `kcenter`: K-Center
- `kmeans`: K-Means聚类
- `herding`: 核牧群方法
- `bcsr`: 双层优化的coreset选择

### 2. 持续学习实验

评估coreset在经验回放中的效果。

**目标**：在任务增量学习场景中减轻灾难性遗忘。

**评估指标**：
- 平均准确率
- 遗忘度量
- 任务间准确率矩阵

**支持的选择策略**：
- `random`: 随机选择
- `uniform`: 均衡类别选择
- `loss`: 基于损失选择
- `margin`: 基于margin选择
- `gradient`: 基于梯度范数选择

## 运行实验

### 数据摘要实验

#### 基础用法

```bash
# 运行单个实验
python experiments/data_summarization.py \
    --dataset MNIST \
    --method herding \
    --selection_ratio 0.1 \
    --epochs 50 \
    --batch_size 128 \
    --seed 42
```

#### 参数说明

**数据集参数**：
- `--dataset`: 数据集名称 (MNIST, CIFAR10, CIFAR100)
- `--num_samples`: 限制训练集大小（用于快速测试）

**Coreset参数**：
- `--method`: 选择方法 (random, kcenter, kmeans, herding, bcsr)
- `--selection_ratio`: coreset选择比例 (0.01-0.5)

**BCSR特定参数**：
- `--bcsr_inner_lr`: 内层学习率 (默认: 0.01)
- `--bcsr_outer_lr`: 外层学习率 (默认: 0.1)
- `--bcsr_inner_steps`: 内层优化步数 (默认: 50)
- `--bcsr_outer_steps`: 外层优化步数 (默认: 20)

**训练参数**：
- `--epochs`: 训练轮数 (默认: 50)
- `--batch_size`: 批次大小 (默认: 128)
- `--lr`: 学习率 (默认: 0.001)

**其他参数**：
- `--device`: 设备 (cpu, cuda)
- `--seed`: 随机种子
- `--output_dir`: 结果保存目录

#### 实验示例

```bash
# MNIST数据集，10% coreset，不同方法
for method in random kcenter kmeans herding bcsr; do
    python experiments/data_summarization.py \
        --dataset MNIST \
        --method $method \
        --selection_ratio 0.1 \
        --epochs 50 \
        --seed 42
done

# CIFAR10数据集，不同选择比例
for ratio in 0.05 0.1 0.2 0.3; do
    python experiments/data_summarization.py \
        --dataset CIFAR10 \
        --method herding \
        --selection_ratio $ratio \
        --epochs 100 \
        --seed 42
done

# 快速测试（限制样本数）
python experiments/data_summarization.py \
    --dataset MNIST \
    --method herding \
    --selection_ratio 0.1 \
    --num_samples 5000 \
    --epochs 20 \
    --seed 42
```

### 持续学习实验

#### 基础用法

```bash
# 运行持续学习实验
python experiments/continual_learning.py \
    --dataset MNIST \
    --num_tasks 5 \
    --num_classes_per_task 2 \
    --memory_size 2000 \
    --selection_method random \
    --num_epochs 10 \
    --seed 42 \
    --save_results
```

#### 参数说明

**数据集参数**：
- `--dataset`: 数据集名称 (MNIST, CIFAR10, CIFAR100)
- `--data_root`: 数据根目录 (默认: ./data)
- `--num_tasks`: 任务数量 (默认: 5)
- `--num_classes_per_task`: 每个任务的类别数 (默认: 2)

**模型参数**：
- `--model`: 模型类型 (cnn)

**训练参数**：
- `--batch_size`: 批次大小 (默认: 128)
- `--num_epochs`: 每个任务的训练轮数 (默认: 10)
- `--learning_rate`: 学习率 (默认: 0.001)
- `--buffer_ratio`: 缓冲区损失权重 (默认: 0.3)

**缓冲区参数**：
- `--memory_size`: 经验回放缓冲区大小 (默认: 2000)
- `--selection_method`: coreset选择方法 (random, uniform, loss, margin, gradient)

**其他参数**：
- `--device`: 设备 (cuda or cpu)
- `--seed`: 随机种子
- `--save_results`: 是否保存结果
- `--results_dir`: 结果保存目录

#### 实验示例

```bash
# MNIST，5个任务，不同选择方法
for method in random uniform loss margin gradient; do
    python experiments/continual_learning.py \
        --dataset MNIST \
        --num_tasks 5 \
        --num_classes_per_task 2 \
        --memory_size 2000 \
        --selection_method $method \
        --num_epochs 10 \
        --seed 42 \
        --save_results
done

# 不同缓冲区大小
for mem_size in 500 1000 2000 5000; do
    python experiments/continual_learning.py \
        --dataset MNIST \
        --num_tasks 5 \
        --num_classes_per_task 2 \
        --memory_size $mem_size \
        --selection_method herding \
        --num_epochs 10 \
        --seed 42 \
        --save_results
done

# CIFAR10，更多任务
python experiments/continual_learning.py \
    --dataset CIFAR10 \
    --num_tasks 10 \
    --num_classes_per_task 1 \
    --memory_size 5000 \
    --selection_method loss \
    --num_epochs 20 \
    --seed 42 \
    --save_results
```

## 结果分析

### 使用Python脚本

```bash
# 绘制数据摘要实验结果
python scripts/plot_results.py \
    --log_dir ./results/data_summarization \
    --experiment_type data_summarization \
    --output_dir ./figures \
    --create_tables

# 绘制持续学习实验结果
python scripts/plot_results.py \
    --log_dir ./results/continual_learning \
    --experiment_type continual_learning \
    --output_dir ./figures \
    --create_tables

# 仅打印统计摘要
python scripts/plot_results.py \
    --log_dir ./results \
    --summary_only

# 过滤特定数据集
python scripts/plot_results.py \
    --log_dir ./results \
    --dataset MNIST \
    --experiment_type both
```

### 使用Jupyter Notebook

```bash
# 启动Jupyter Notebook
jupyter notebook notebooks/Results_Analysis.ipynb
```

Notebook提供交互式分析界面，包括：
1. 加载实验结果
2. 查看统计信息
3. 创建可视化图表
4. 生成比较表格
5. 统计检验
6. 相关性分析
7. 导出分析报告

### 结果文件格式

实验结果保存为JSON格式：

**数据摘要实验结果示例**：
```json
{
  "dataset": "MNIST",
  "method": "herding",
  "selection_ratio": 0.1,
  "coreset_size": 6000,
  "num_samples": 60000,
  "test_acc_full": 98.5,
  "test_acc_coreset": 97.8,
  "performance_drop": 0.7,
  "selection_time": 45.2,
  "train_time_full": 320.5,
  "train_time_coreset": 32.1,
  "best_val_acc_full": 98.7,
  "best_val_acc_coreset": 98.0,
  "history_full": {...},
  "history_coreset": {...}
}
```

**持续学习实验结果示例**：
```json
{
  "dataset": "MNIST",
  "model": "cnn",
  "num_tasks": 5,
  "num_classes_per_task": 2,
  "memory_size": 2000,
  "selection_method": "random",
  "num_epochs": 10,
  "learning_rate": 0.001,
  "batch_size": 128,
  "buffer_ratio": 0.3,
  "seed": 42,
  "accuracy_matrix": [[...], [...], ...],
  "average_accuracy": 85.3,
  "forgetting_measure": 12.5
}
```

## 批量实验

### 使用Shell脚本

创建 `run_experiments.sh`：

```bash
#!/bin/bash

# 数据摘要实验批量运行
DATASETS=("MNIST" "CIFAR10")
METHODS=("random" "kcenter" "kmeans" "herding" "bcsr")
RATIOS=(0.05 0.1 0.2)

for dataset in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
        for ratio in "${RATIOS[@]}"; do
            echo "Running: $dataset, $method, $ratio"
            python experiments/data_summarization.py \
                --dataset $dataset \
                --method $method \
                --selection_ratio $ratio \
                --epochs 50 \
                --seed 42
        done
    done
done
```

运行：
```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

### 使用Python脚本

创建 `run_batch_experiments.py`：

```python
import subprocess
import itertools

datasets = ['MNIST', 'CIFAR10']
methods = ['random', 'kcenter', 'kmeans', 'herding', 'bcsr']
ratios = [0.05, 0.1, 0.2]

for dataset, method, ratio in itertools.product(datasets, methods, ratios):
    print(f"Running: {dataset}, {method}, {ratio}")
    cmd = [
        'python', 'experiments/data_summarization.py',
        '--dataset', dataset,
        '--method', method,
        '--selection_ratio', str(ratio),
        '--epochs', '50',
        '--seed', '42'
    ]
    subprocess.run(cmd)
```

## 常见问题

### 1. 内存不足

**问题**：CUDA out of memory

**解决方案**：
- 减小批次大小：`--batch_size 64`
- 减小模型或数据集规模
- 使用CPU：`--device cpu`

### 2. 训练太慢

**解决方案**：
- 使用更少的训练样本：`--num_samples 10000`
- 减少训练轮数：`--epochs 20`
- 使用更小的数据集（MNIST而非CIFAR）

### 3. 结果不稳定

**解决方案**：
- 使用不同的随机种子运行多次
- 增加训练轮数
- 调整学习率

### 4. BCSR收敛问题

**解决方案**：
- 调整学习率：`--bcsr_inner_lr 0.001 --bcsr_outer_lr 0.01`
- 增加优化步数：`--bcsr_inner_steps 100 --bcsr_outer_steps 50`
- 检查数据预处理

### 5. 结果文件未保存

**检查**：
- 确保指定了 `--output_dir` 或 `--save_results`
- 检查目录权限
- 查看错误日志

## 性能优化建议

### 1. 数据加载

```python
# 使用更多工作进程
DataLoader(dataset, batch_size=128, num_workers=4)

# 启用pin_memory（GPU训练）
DataLoader(dataset, batch_size=128, pin_memory=True)
```

### 2. 模型训练

```python
# 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)
```

### 3. Coreset选择

- 对大规模数据集使用近似算法
- 考虑使用GPU加速的距离计算
- 缓存中间结果

## 扩展实验

### 添加新的数据集

1. 在 `src/datasets/data_loaders.py` 中添加数据集
2. 更新 `DATASET_STATS` 字典
3. 实现相应的模型（如果需要）

### 添加新的选择方法

1. 在 `src/baselines/baseline_methods.py` 或 `src/coreset/` 中实现方法
2. 在实验脚本中注册新方法
3. 更新参数解析器

### 添加新的评估指标

1. 在实验脚本中计算新指标
2. 将指标添加到结果字典
3. 更新可视化脚本以显示新指标

## 引用

如果您在研究中使用了本代码，请引用：

```bibtex
@software{coreset_benchmark,
  title={Coreset Selection Methods Benchmark},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/coreset_benchmark}
}
```

## 联系方式

如有问题或建议，请：
- 提交Issue
- 发送邮件至：your.email@example.com
- 查看项目文档：README.md

## 更新日志

- **2025-04-28**: 初始版本
  - 实现数据摘要实验
  - 实现持续学习实验
  - 添加结果可视化工具
  - 创建实验指南

---

**祝实验顺利！**
