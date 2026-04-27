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
