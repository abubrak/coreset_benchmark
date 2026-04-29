# CSReL v2 集成说明

## 任务完成状态: DONE

## 实施内容

### 1. 修改 `experiments/data_summarization.py`
- ✅ 添加了 CSReL v2 导入语句
- ✅ 创建了 `CSReLWrapperDataset` 类来适配数据格式
- ✅ 在方法选择部分添加了 `csrel_v2` 分支
- ✅ 实现了完整的 CSReL v2 选择逻辑

### 2. 新增 CSReL v2 命令行参数
- `--csrel_init_size`: 初始随机采样大小 (默认: 100)
- `--csrel_selection_steps`: 每轮增量选择的大小 (默认: 100)
- `--csrel_cur_lr`: 增量训练学习率 (默认: 0.01)
- `--csrel_cur_steps`: 增量训练轮数 (默认: 10)
- `--csrel_ref_epochs`: 参考模型训练轮数 (默认: 100)
- `--csrel_ref_lr`: 参考模型学习率 (默认: 0.01)

### 3. 更新方法选择列表
- ✅ 在 `--method` 参数中添加了 `'csrel_v2'` 选项

### 4. 集成验证
- ✅ 所有导入测试通过
- ✅ 数据格式转换测试通过
- ✅ 配置类测试通过
- ✅ 功能集成测试通过

## 使用方法

### 基本用法
```bash
python experiments/data_summarization.py \
    --dataset MNIST \
    --method csrel_v2 \
    --selection_ratio 0.1 \
    --batch_size 128 \
    --epochs 50
```

### 自定义参数
```bash
python experiments/data_summarization.py \
    --dataset CIFAR10 \
    --method csrel_v2 \
    --selection_ratio 0.1 \
    --csrel_init_size 100 \
    --csrel_selection_steps 100 \
    --csrel_cur_lr 0.01 \
    --csrel_cur_steps 10 \
    --csrel_ref_epochs 100 \
    --csrel_ref_lr 0.01 \
    --batch_size 128 \
    --epochs 50
```

### 快速测试 (小规模)
```bash
python experiments/data_summarization.py \
    --dataset MNIST \
    --method csrel_v2 \
    --selection_ratio 0.1 \
    --num_samples 1000 \
    --epochs 1 \
    --csrel_init_size 10 \
    --csrel_selection_steps 10 \
    --csrel_ref_epochs 2 \
    --csrel_cur_steps 1
```

## 技术细节

### 数据格式适配
CSReL v2 期望的数据格式是 `(id, sample, label)`，但标准 PyTorch 数据集返回 `(sample, label)`。我们创建了 `CSReLWrapperDataset` 类来解决这个不匹配问题：

```python
class CSReLWrapperDataset(torch.utils.data.Dataset):
    """将标准 PyTorch 数据集包装为 CSReL v2 期望的格式"""
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample, label = self.base_dataset[idx]
        return idx, sample, label  # 转换为 (id, sample, label)
```

### 集成逻辑
1. 创建适合数据集的模型 (CNN_MNIST 或 ResNet18)
2. 准备训练数据并转换为 numpy 数组
3. 创建 CSReLConfigV2 配置对象
4. 包装数据集为 CSReL v2 格式
5. 创建 CSReLCoresetV2 选择器
6. 执行选择并获取索引
7. 验证索引有效性

## 关键文件

- `experiments/data_summarization.py`: 主要实验脚本 (已修改)
- `src/coreset/csrel_coreset_v2.py`: CSReL v2 核心算法
- `src/coreset/csrel_utils.py`: CSReL 工具函数
- `src/coreset/csrel_train.py`: CSReL 训练方法
- `src/coreset/csrel_loss.py`: CSReL 损失函数
- `src/configs.py`: 配置类 (包含 CSReLConfigV2)

## 注意事项

1. **数据格式**: CSReL v2 需要特定的数据格式，已通过包装器解决
2. **计算资源**: CSReL v2 需要训练参考模型，计算量较大
3. **内存使用**: 需要将数据转换为 numpy 数组，内存使用较高
4. **临时文件**: CSReL v2 会在 `./temp_csrel_v2` 目录创建临时文件

## 测试结果

所有集成测试均通过：
- ✅ 导入测试
- ✅ 数据格式测试
- ✅ 配置类测试
- ✅ 功能集成测试

## 下一步

集成已完成，可以开始运行 CSReL v2 实验。建议：
1. 从小规模实验开始验证功能
2. 逐步增加数据集大小
3. 调整超参数以获得最佳性能
4. 与其他方法进行比较实验

## Git 提交建议

```
Add CSReL v2 integration to data summarization experiments

- Add CSReLWrapperDataset for data format adaptation
- Implement csrel_v2 method branch in data_summarization.py
- Add 6 CSReL v2 specific command line arguments
- Update method choices to include 'csrel_v2'
- All integration tests passed successfully
```