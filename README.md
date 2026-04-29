# Coreset Selection Benchmark

A comprehensive benchmark for evaluating coreset selection methods in neural network training.

## Project Structure

```
coreset_benchmark/
├── src/
│   ├── ntk/              # Neural Tangent Kernel implementations
│   ├── coreset/          # Core coreset selection algorithms
│   ├── models/           # Neural network architectures
│   ├── datasets/         # Dataset loaders and preprocessing
│   ├── baselines/        # Baseline methods
│   ├── utils/            # Utility functions
│   └── training/         # Training loops and evaluation
├── requirements.txt
└── README.md
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

- **Bilevel Optimization**: Gradient-based coreset selection using bilevel optimization
- **CSReL**: Classwise Spatial Representation Learning for coreset selection
- **Data Summarization**: Herding and Class Representative Apportionment (CRA)
- **Continual Learning**: GSS (Gradient-based Sample Selection) and other baselines

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
