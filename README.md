# Coreset Selection Benchmark

A comprehensive benchmark for evaluating coreset selection methods in neural network training.

## Project Structure

```
coreset_benchmark/
├── src/
│   ├── configs.py            # Configuration classes
│   ├── coreset/              # Core coreset selection algorithms
│   │   ├── bcsr_coreset.py       # BCSR implementation
│   │   ├── bilevel_coreset.py     # Bilevel optimization
│   │   ├── csrel_coreset.py       # CSReL v1
│   │   ├── csrel_coreset_v2.py    # CSReL v2 (optimized)
│   │   ├── continual_adapters.py  # Continual learning adapters
│   │   └── selection_functions.py # Selection utilities
│   ├── models/               # Neural network architectures (CNN, ResNet)
│   ├── datasets/             # Dataset loaders (MNIST, CIFAR-10/100)
│   ├── baselines/            # Baseline methods (6 methods)
│   ├── ntk/                  # Neural Tangent Kernel
│   ├── training/             # Training loops and losses
│   └── utils/                # Checkpointing and memory utilities
├── experiments/              # Experiment scripts
│   ├── data_summarization.py     # Data summarization experiments
│   └── continual_learning.py     # Continual learning experiments
├── tests/                    # Test suite (pytest)
├── scripts/                  # Visualization utilities
├── docs/                     # Documentation
├── notebooks/                # Jupyter notebooks and Colab helpers
├── results/                  # Experiment results (gitignored)
├── data/                     # Datasets (gitignored)
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from src.configs import BilevelConfig
from src.coreset.bilevel import BilevelCoreset

# Configure experiment
config = BilevelConfig(
    dataset="MNIST",
    selection_ratio=0.1,
    gradient_steps=100
)

# Run coreset selection
selector = BilevelCoreset(config)
coreset_indices = selector.select(train_data, train_labels)
```

## Methods

| Category | Method | Description |
|----------|--------|-------------|
| **Bilevel** | Bilevel Coreset | Gradient-based selection via bilevel optimization |
| **Bilevel** | BCSR | Bilevel Coreset Selection with Reweighting |
| **CSReL** | CSReL v1 | Classwise Spatial Representation Learning |
| **CSReL** | CSReL v2 | CSReL with optimized incremental selection |
| **Baseline** | Random / Uniform | Random sampling baselines |
| **Baseline** | K-Center / K-Means | Geometry-based selection |
| **Baseline** | Herding | Herding-based coreset |
| **Baseline** | Entropy / Loss | Uncertainty-based selection |

## Coreset训练机制

本框架复现了原始论文的**tile重复机制**：

1. **样本重复**: 小coreset（如200样本）通过 `np.tile` 重复 ~250 次，使每epoch样本数 ≈ 50,000
2. **数据增强**: RandomCrop + RandomHorizontalFlip 使每次重复产生不同变体
3. **优化器配置**: Adam(lr=5e-5, weight_decay=1e-4) + 6 epochs

这确保了小coreset的训练迭代次数与完整数据集相当，同时通过数据增强防止过拟合。

## Citation

If you use this code, please cite:
```bibtex
@article{your_paper,
  title={Your Paper Title},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## 持续学习实验

### 支持的Coreset选择方法

实验框架支持以下coreset选择方法：

#### Baseline方法
- `random`: 随机采样
- `uniform`: 按类别均匀随机采样
- `loss`: 基于交叉熵损失选择
- `margin`: 基于决策margin选择
- `gradient`: 基于梯度范数选择

#### 双层优化方法
- `bcsr`: Bilevel Coreset Selection with Reweighting
  - 使用双层优化学习样本权重
  - 需要提供模型参数
  - 参数：`--learning_rate_inner`, `--learning_rate_outer`, `--num_outer_steps`

- `csrel`: Classwise Spatial Representation Learning
  - 基于reducible loss选择样本
  - 需要先训练参考模型
  - 参数：`--csrel_epochs`, `--csrel_lr`

- `bilevel`: Bilevel Coreset (简化版)
  - 使用kernel herding近似双层优化
  - 无需额外参数

### 运行示例

```bash
# 使用BCSR方法进行持续学习
python experiments/continual_learning.py \
    --dataset MNIST \
    --num_tasks 5 \
    --num_classes_per_task 2 \
    --selection_method bcsr \
    --memory_size 2000 \
    --save_results

# 使用CSReL方法
python experiments/continual_learning.py \
    --dataset MNIST \
    --selection_method csrel \
    --memory_size 2000 \
    --save_results

# 使用Bilevel方法
python experiments/continual_learning.py \
    --dataset MNIST \
    --selection_method bilevel \
    --memory_size 2000 \
    --save_results

# 对比所有方法
for method in random uniform loss bcsr csrel bilevel; do
    python experiments/continual_learning.py \
        --dataset MNIST \
        --num_tasks 5 \
        --selection_method $method \
        --save_results
done
```

### 实验结果

结果将保存到 `results/continual_learning/` 目录，包含：
- 准确率矩阵
- 平均准确率
- 遗忘度量

### 端到端测试

运行测试套件验证所有方法：

```bash
# 运行正式测试
python -m pytest tests/ -v

# 快速冒烟测试
python test_all_methods.py
```
