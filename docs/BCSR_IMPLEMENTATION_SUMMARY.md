# BCSR方法实现总结

## 完成时间
2026-04-28

## 实现内容

### 1. 创建的文件

#### F:/paper_code/coreset_benchmark/src/training/bcsr_training.py
- **BCSRTraining类**: 实现基于双层优化的训练框架
  - `__init__`: 初始化训练器
  - `train_inner`: 内层优化（训练代理模型）
  - `train_outer`: 外层优化（更新样本权重）
  - `train`: 完整训练流程
  - `_projection_onto_simplex`: 权重投影到单纯形
  - `_evaluate`: 模型评估

#### F:/paper_code/coreset_benchmark/src/coreset/bcsr_coreset.py
- **BCSRCoreset类**: 实现coreset选择
  - `__init__`: 初始化选择器
  - `projection_onto_simplex`: 向量投影到单纯形
  - `coreset_select`: 主选择方法
  - `_optimize_weights_with_model`: 使用模型优化权重
  - `_optimize_weights_kernel`: 使用核方法优化权重
  - `_compute_rbf_kernel`: 计算RBF核矩阵
  - `get_weights`: 获取学习到的权重
  - `get_selected_indices`: 获取选择的索引

### 2. 修改的文件

#### F:/paper_code/coreset_benchmark/src/training/__init__.py
- 添加了BCSRTraining的导入和导出

### 3. 测试文件

#### F:/paper_code/coreset_benchmark/test_bcsr_basic.py
- 测试单纯形投影功能
- 测试BCSR coreset基本功能
- 测试BCSR训练基本功能

### 4. 文档

#### F:/paper_code/coreset_benchmark/docs/BCSR_USAGE.md
- 使用说明和示例代码

## 核心算法

### 双层优化框架

```
外层循环（优化权重）:
    初始化权重 w = uniform(1/n)

    内层循环（训练模型）:
        在加权训练集上训练模型
        计算验证集损失

    更新权重 w = w - lr_outer * gradient
    投影到单纯形: w = projection_onto_simplex(w)

选择top-k样本作为coreset
```

### 单纯形投影算法

使用(Duchi et al., 2008)的算法，确保权重非负且和为1：

```python
def projection_onto_simplex(v):
    # 1. 截断到非负
    v = max(v, 0)

    # 2. 排序并找到阈值
    u = sort(v, descending=True)
    rho = max{j : u[j] - (sum(u[:j]) - 1) / j > 0}
    theta = (sum(u[:rho]) - 1) / rho

    # 3. 投影
    w = max(v - theta, 0)

    return w
```

## 测试结果

### 所有测试通过 ✓

1. **单纯形投影测试**: 通过
   - 随机向量投影正确
   - 全负向量投影到均匀分布

2. **BCSR coreset选择测试**: 通过
   - 从100个样本中选择20个
   - 权重计算正确
   - 索引选择正确

3. **BCSR训练测试**: 通过
   - 双层优化框架运行正常
   - 权重更新正确
   - 训练历史记录完整

## 技术特点

### 1. 灵活性
- 支持有模型和无模型两种模式
- 可配置的核函数
- 可调节的优化参数

### 2. 鲁棒性
- 数值稳定性处理（防止除零）
- 边界情况处理（空权重、全负向量）
- 类型转换（numpy/torch互操作）

### 3. 可扩展性
- 模块化设计
- 清晰的接口
- 详细的文档

## 使用示例

```python
from src.coreset.bcsr_coreset import BCSRCoreset
from src.models.cnn import CNN_MNIST
import torch

# 准备数据
X = torch.randn(1000, 1, 28, 28)
y = torch.randint(0, 10, (1000,))

# 创建模型和选择器
model = CNN_MNIST(num_classes=10)
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
    X=X, y=y, coreset_size=100, model=model
)

print(f"选择了 {len(selected_X)} 个样本")
```

## 代码统计

- **总行数**: 约700行（含注释和文档）
- **核心功能**: 5个主要类/方法
- **测试覆盖**: 3个测试函数
- **文档**: 2个Markdown文件

## 后续工作

1. 性能优化
   - 批量核矩阵计算
   - GPU加速
   - 内存优化

2. 功能扩展
   - 支持更多核函数
   - 自适应学习率
   - 早停机制

3. 集成测试
   - 与其他coreset方法对比
   - 在真实数据集上验证
   - 性能基准测试

## 实现质量

- ✓ 代码语法正确
- ✓ 导入测试通过
- ✓ 基本功能测试通过
- ✓ 文档完整
- ✓ 符合项目代码风格
