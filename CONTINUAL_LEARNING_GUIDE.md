# 持续学习实验运行指南

## 已修复的关键问题

### 1. Bilevel方法类别不平衡问题
**问题**: BilevelContinualAdapter只选择类别0的样本
**原因**: kernel herding实现没有考虑labels
**修复**: 添加类别平衡的kernel herding，从每个类均匀选择样本

### 2. CUDA类别数不匹配错误
**问题**: `Assertion 't >= 0 && t < n_classes' failed`
**原因**: 模型输出2个类别（本地标签0,1），但收到全局标签（0-9）
**修复**: 
- 添加`RemappedDataset`类，将任务标签映射到本地空间
- Task 0 (原始类别0,1) → 本地标签(0,1)
- Task 1 (原始类别2,3) → 本地标签(0,1)
- 缓冲区存储本地标签，通过task_id区分来源任务

### 3. Gradient方法无法运行
**问题**: RuntimeError: element 0 of tensors does not require grad
**原因**: gradient方法在`torch.no_grad()`上下文中被调用
**修复**: 将gradient分支移出no_grad上下文，单独处理

## 如何运行实验

### Baseline方法（已验证可用）

```bash
# Random
python experiments/continual_learning.py --dataset MNIST --num_tasks 5 --num_classes_per_task 2 --selection_method random --save_results

# Uniform (类别平衡)
python experiments/continual_learning.py --dataset MNIST --num_tasks 5 --num_classes_per_task 2 --selection_method uniform --save_results

# Loss
python experiments/continual_learning.py --dataset MNIST --num_tasks 5 --num_classes_per_task 2 --selection_method loss --save_results

# Margin
python experiments/continual_learning.py --dataset MNIST --num_tasks 5 --num_classes_per_task 2 --selection_method margin --save_results

# Gradient
python experiments/continual_learning.py --dataset MNIST --num_tasks 5 --num_classes_per_task 2 --selection_method gradient --save_results
```

### 双层优化方法（已修复，待测试）

```bash
# BCSR (注意：大数据集可能较慢)
python experiments/continual_learning.py --dataset MNIST --num_tasks 5 --num_classes_per_task 2 --selection_method bcsr --memory_size 2000 --save_results

# CSReL
python experiments/continual_learning.py --dataset MNIST --num_tasks 5 --num_classes_per_task 2 --selection_method csrel --memory_size 2000 --save_results

# Bilevel
python experiments/continual_learning.py --dataset MNIST --num_tasks 5 --num_classes_per_task 2 --selection_method bilevel --memory_size 2000 --save_results
```

### 快速测试（验证修复）

```bash
python test_fixes.py
```

## 实验参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | MNIST | 数据集：MNIST/CIFAR10/CIFAR100 |
| `--num_tasks` | 5 | 任务数量 |
| `--num_classes_per_task` | 2 | 每个任务的类别数 |
| `--memory_size` | 2000 | 经验回放缓冲区大小 |
| `--selection_method` | random | Coreset选择方法 |
| `--buffer_ratio` | 0.3 | 缓冲区损失权重 |
| `--num_epochs` | 10 | 每个任务的训练轮数 |
| `--batch_size` | 128 | 批次大小 |
| `--learning_rate` | 0.001 | 学习率 |
| `--save_results` | False | 是否保存结果到JSON |

## 结果文件

结果保存在 `results/continual_learning/` 目录，文件名格式：
```
{dataset}_{method}_task{num_tasks}_mem{memory_size}_seed{seed}.json
```

结果包含：
- 准确率矩阵 (accuracy_matrix)
- 平均准确率 (average_accuracy)
- 遗忘度量 (forgetting_measure)
- 实验配置

## 性能建议

1. **BCSR方法**: 大数据集（>10000样本）选择coreset时较慢，建议：
   - 减少`num_outer_steps`（默认5→3）
   - 减小`memory_size`
   - 先在小规模数据上测试

2. **CSReL方法**: 需要训练参考模型，首次运行较慢

3. **Bilevel方法**: 使用简化kernel herding，速度较快

## 故障排除

### CUDA OOM错误
- 减小`batch_size`（128→64）
- 减小`memory_size`
- 使用CPU: `--device cpu`

### 类别数不匹配
如果仍然看到`Assertion 't >= 0 && t < n_classes' failed`：
- 确保使用最新的修复（commit 02fe906）
- 检查数据集是否正确下载
- 尝试`python test_fixes.py`验证修复

### BCSR运行太慢
```bash
# 在continual_adapters.py中修改默认参数
BCSRContinualAdapter(
    num_outer_steps=3,  # 从5减少到3
    num_inner_steps=1   # 保持1
)
```

## 下一步

1. ✅ Baseline方法（random/uniform/loss/margin/gradient）已验证可用
2. ⏳ 需要测试BCSR在完整实验中的表现
3. ⏳ 需要测试CSReL在完整实验中的表现
4. ⏳ 需要测试Bilevel在完整实验中的表现
5. ⏳ 对比所有方法的性能

## 相关文件

- `experiments/continual_learning.py` - 主实验脚本
- `src/coreset/continual_adapters.py` - 双层优化方法适配器
- `src/coreset/selection_functions.py` - Baseline选择函数
- `test_fixes.py` - 快速验证测试
