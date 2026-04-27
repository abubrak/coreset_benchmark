# Colab使用指南

本指南说明如何在Google Colab中使用Coreset Benchmark项目。

## 快速开始

### 1. 克隆项目

在Colab中运行以下单元格来克隆项目：

```python
# 克隆项目到Colab
!git clone https://github.com/yourusername/coreset_benchmark.git /content/coreset_benchmark

# 进入项目目录
%cd /content/coreset_benchmark
```

### 2. 运行环境设置

```python
# 运行设置脚本
!python notebooks/setup_colab.py
```

或者使用辅助脚本：

```python
# 导入辅助脚本
import sys
sys.path.insert(0, '/content/coreset_benchmark')

from notebooks import colab_helper
colab_helper.print_environment_info()
colab_helper.verify_project_structure()
```

### 3. 运行实验

现在可以运行任何实验notebook：

1. **数据摘要实验**: `Data_Summarization_Experiment.ipynb`
2. **持续学习实验**: `Continual_Learning_Experiment.ipynb`
3. **结果分析**: `Results_Analysis.ipynb`

## 路径说明

所有notebook都包含自动路径检测功能：

- **Colab环境**: 自动使用 `/content/coreset_benchmark`
- **本地环境**: 自动使用项目根目录

路径检测代码示例：

```python
from pathlib import Path
import sys

def get_project_path():
    """自动检测项目路径"""
    if 'google.colab' in sys.modules:
        # Colab环境
        return Path('/content/coreset_benchmark')
    else:
        # 本地环境
        return Path().absolute().parent

project_root = get_project_path()
sys.path.insert(0, str(project_root))
```

## GPU加速

在Colab中启用GPU：

1. 点击菜单：`运行时` -> `更改运行时类型`
2. 硬件加速器选择：`GPU`
3. 点击`保存`

验证GPU是否可用：

```python
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## 文件结构

```
/content/coreset_benchmark/
├── src/                    # 源代码
│   ├── __init__.py
│   ├── data.py            # 数据加载
│   ├── coreset.py         # Coreset方法
│   ├── models.py          # 模型定义
│   └── utils.py           # 工具函数
├── notebooks/             # Jupyter notebooks
│   ├── setup_colab.py     # 环境设置
│   ├── colab_helper.py    # 辅助函数
│   ├── Data_Summarization_Experiment.ipynb
│   ├── Continual_Learning_Experiment.ipynb
│   └── Results_Analysis.ipynb
├── data/                  # 数据目录
│   ├── raw/              # 原始数据
│   └── processed/        # 处理后的数据
└── results/              # 实验结果
    ├── data_summarization/
    └── continual_learning/
```

## 故障排除

### 导入错误

如果遇到模块导入错误：

```python
# 检查项目路径
import sys
from pathlib import Path
project_path = Path('/content/coreset_benchmark')
print(f"项目存在: {project_path.exists()}")
print(f"在sys.path中: {str(project_path) in sys.path}")

# 添加到路径
if str(project_path) not in sys.path:
    sys.path.insert(0, str(project_path))
```

### 文件未找到

如果遇到文件未找到错误，使用相对路径：

```python
from pathlib import Path
import sys

# 获取项目根目录
if 'google.colab' in sys.modules:
    project_root = Path('/content/coreset_benchmark')
else:
    project_root = Path().absolute().parent

# 使用相对于项目根目录的路径
data_file = project_root / 'data' / 'processed' / 'data.pkl'
```

### 内存不足

Colab的内存有限，如果遇到OOM错误：

1. 减小批大小 (`BATCH_SIZE`)
2. 减小数据集大小
3. 使用更小的模型

## 最佳实践

1. **保存进度**: 定期保存模型和结果到Google Drive
2. **使用检查点**: 训练时保存检查点
3. **清理内存**: 在长时间运行中定期清理GPU缓存

```python
# 清理GPU缓存
import torch
torch.cuda.empty_cache()

# 保存到Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 复制结果到Drive
!cp -r /content/coreset_benchmark/results /content/drive/MyDrive/coreset_results
```

## 性能优化

1. **使用GPU**: 确保在GPU运行时上运行
2. **批处理**: 调整`BATCH_SIZE`以充分利用GPU
3. **数据预加载**: 使用`DataLoader`的`num_workers`参数

```python
# 多线程数据加载
train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,  # 使用多个工作线程
    pin_memory=True  # 加速GPU传输
)
```

## 联系方式

如有问题，请：
- 提交GitHub Issue
- 查看项目文档
- 参考示例notebooks
