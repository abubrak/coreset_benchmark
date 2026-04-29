"""
CSReL 训练方法单元测试

测试 csrel_loss.py 和 csrel_train.py 中的损失函数和训练方法。
"""

import os
import tempfile
import shutil
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


class SimpleDataset(Dataset):
    """简单的测试数据集"""

    def __init__(self, num_samples=100, num_features=10, num_classes=5):
        self.num_samples = num_samples
        self.features = torch.randn(num_samples, num_features)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return idx, self.features[idx], self.labels[idx]


class SimpleModel(nn.Module):
    """简单的测试模型"""

    def __init__(self, input_dim=10, num_classes=5):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class TestCompliedLoss:
    """测试 CompliedLoss 损失函数"""

    def test_init_default(self):
        """测试默认初始化"""
        from src.coreset.csrel_loss import CompliedLoss

        loss_fn = CompliedLoss(ce_factor=1.0, mse_factor=0.5)
        assert loss_fn.ce_factor == 1.0
        assert loss_fn.mse_factor == 0.5
        assert loss_fn.reduction == 'mean'
        assert loss_fn.kd_mode == 'mse'

    def test_init_with_ce_mode(self):
        """测试使用 CE 模式初始化"""
        from src.coreset.csrel_loss import CompliedLoss

        loss_fn = CompliedLoss(ce_factor=1.0, mse_factor=0.5, kd_mode='ce')
        assert loss_fn.kd_mode == 'ce'

    def test_init_invalid_mode(self):
        """测试无效的 kd_mode"""
        from src.coreset.csrel_loss import CompliedLoss

        with pytest.raises(ValueError, match='not a valid model'):
            CompliedLoss(ce_factor=1.0, mse_factor=0.5, kd_mode='invalid')

    def test_forward_ce_only(self):
        """测试只有交叉熵损失的向前传播"""
        from src.coreset.csrel_loss import CompliedLoss

        loss_fn = CompliedLoss(ce_factor=1.0, mse_factor=0.0)
        logits = torch.randn(10, 5)
        labels = torch.randint(0, 5, (10,))

        loss = loss_fn(logits, labels)
        assert loss.item() > 0

    def test_forward_with_kd_mse(self):
        """测试带 MSE 知识蒸馏的向前传播"""
        from src.coreset.csrel_loss import CompliedLoss

        loss_fn = CompliedLoss(ce_factor=1.0, mse_factor=0.5, kd_mode='mse')
        logits = torch.randn(10, 5)
        labels = torch.randint(0, 5, (10,))
        ref_logits = torch.randn(10, 5)

        loss = loss_fn(logits, labels, ref_logits)
        assert loss.item() > 0

    def test_forward_with_kd_ce(self):
        """测试带 CE 知识蒸馏的向前传播"""
        from src.coreset.csrel_loss import CompliedLoss

        loss_fn = CompliedLoss(ce_factor=1.0, mse_factor=0.5, kd_mode='ce')
        logits = torch.randn(10, 5)
        labels = torch.randint(0, 5, (10,))
        ref_logits = torch.randn(10, 5)

        loss = loss_fn(logits, labels, ref_logits)
        assert loss.item() > 0

    def test_forward_no_logits(self):
        """测试不提供 ref_logits 时的向前传播"""
        from src.coreset.csrel_loss import CompliedLoss

        loss_fn = CompliedLoss(ce_factor=1.0, mse_factor=0.5)
        logits = torch.randn(10, 5)
        labels = torch.randint(0, 5, (10,))

        loss = loss_fn(logits, labels)
        assert loss.item() > 0


class TestKDCrossEntropyLoss:
    """测试 KDCrossEntropyLoss 损失函数"""

    def test_init_mean(self):
        """测试 mean reduction 初始化"""
        from src.coreset.csrel_loss import KDCrossEntropyLoss

        loss_fn = KDCrossEntropyLoss(reduction='mean')
        assert loss_fn.reduction == 'mean'

    def test_init_none(self):
        """测试 none reduction 初始化"""
        from src.coreset.csrel_loss import KDCrossEntropyLoss

        loss_fn = KDCrossEntropyLoss(reduction='none')
        assert loss_fn.reduction == 'none'

    def test_init_sum(self):
        """测试 sum reduction 初始化"""
        from src.coreset.csrel_loss import KDCrossEntropyLoss

        loss_fn = KDCrossEntropyLoss(reduction='sum')
        assert loss_fn.reduction == 'sum'

    def test_forward_mean(self):
        """测试 mean reduction 向前传播"""
        from src.coreset.csrel_loss import KDCrossEntropyLoss

        loss_fn = KDCrossEntropyLoss(reduction='mean')
        x = torch.randn(10, 5)
        y = torch.randn(10, 5)

        loss = loss_fn(x, y)
        # KL 散度可以是负数（实际是负的 KL 散度）
        assert loss.dim() == 0  # 标量

    def test_forward_none(self):
        """测试 none reduction 向前传播"""
        from src.coreset.csrel_loss import KDCrossEntropyLoss

        loss_fn = KDCrossEntropyLoss(reduction='none')
        x = torch.randn(10, 5)
        y = torch.randn(10, 5)

        loss = loss_fn(x, y)
        assert loss.shape[0] == 10  # 每个样本一个损失值

    def test_forward_sum(self):
        """测试 sum reduction 向前传播"""
        from src.coreset.csrel_loss import KDCrossEntropyLoss

        loss_fn = KDCrossEntropyLoss(reduction='sum')
        x = torch.randn(10, 5)
        y = torch.randn(10, 5)

        loss = loss_fn(x, y)
        # KL 散度可以是负数（实际是负的 KL 散度）
        assert loss.dim() == 0  # 标量


class TestEvalModel:
    """测试 eval_model 函数"""

    def test_eval_model_basic(self):
        """测试基本评估功能"""
        from src.coreset.csrel_train import eval_model

        model = SimpleModel()
        dataset = SimpleDataset(num_samples=50)
        dataloader = DataLoader(dataset, batch_size=10)

        acc = eval_model(model, dataloader, on_cuda=False)
        assert 0.0 <= acc <= 1.0

    def test_eval_model_with_loss(self):
        """测试带损失返回的评估"""
        from src.coreset.csrel_train import eval_model

        model = SimpleModel()
        dataset = SimpleDataset(num_samples=50)
        dataloader = DataLoader(dataset, batch_size=10)

        acc, avg_loss = eval_model(model, dataloader, on_cuda=False, return_loss=True)
        assert 0.0 <= acc <= 1.0
        assert avg_loss > 0


class TestComputeAccuracy:
    """测试 compute_accuracy 函数"""

    def test_compute_accuracy_basic(self):
        """测试基本准确率计算"""
        from src.coreset.csrel_train import compute_accuracy

        logits = torch.tensor([
            [1.0, 2.0, 0.5],
            [0.5, 1.0, 2.0],
            [2.0, 0.5, 1.0]
        ])
        labels = torch.tensor([1, 2, 0])

        acc = compute_accuracy(logits, labels)
        assert acc == 1.0  # 全部正确

    def test_compute_accuracy_partial(self):
        """测试部分正确的准确率"""
        from src.coreset.csrel_train import compute_accuracy

        logits = torch.tensor([
            [1.0, 2.0, 0.5],
            [0.5, 1.0, 2.0],
            [2.0, 0.5, 1.0]
        ])
        labels = torch.tensor([0, 2, 0])  # 第一个错误

        acc = compute_accuracy(logits, labels)
        assert acc == 2.0 / 3.0


class TestComputeAvgLoss:
    """测试 compute_avg_loss 函数"""

    def test_compute_avg_loss_basic(self):
        """测试平均损失计算"""
        from src.coreset.csrel_train import compute_avg_loss

        logits = torch.randn(10, 5)
        labels = torch.randint(0, 5, (10,))
        loss_fn = nn.CrossEntropyLoss(reduction='none')

        avg_loss = compute_avg_loss(logits, labels, loss_fn)
        assert avg_loss > 0


class TestComputeLossVar:
    """测试 compute_loss_var 函数"""

    def test_compute_loss_var_basic(self):
        """测试损失方差计算"""
        from src.coreset.csrel_train import compute_loss_var

        logits = torch.randn(10, 5)
        labels = torch.randint(0, 5, (10,))
        loss_fn = nn.CrossEntropyLoss(reduction='none')

        loss_var = compute_loss_var(logits, labels, loss_fn)
        assert loss_var >= 0  # 方差非负


class TestSaveLoadModel:
    """测试模型保存和加载"""

    def test_save_and_load_model(self):
        """测试保存和加载模型"""
        from src.coreset.csrel_train import save_model, load_model

        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleModel()

            # 保存模型
            save_model(tmpdir, model, on_cuda=False)

            # 检查文件是否存在
            model_path = os.path.join(tmpdir, 'best_model.pkl')
            assert os.path.exists(model_path)

            # 加载模型
            loaded_model = load_model(tmpdir)

            # 验证参数相同
            for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
                assert torch.equal(p1, p2)

    def test_save_model_with_name(self):
        """测试使用自定义名称保存模型"""
        from src.coreset.csrel_train import save_model

        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleModel()

            save_model(tmpdir, model, on_cuda=False, save_name='test_model.pkl')

            model_path = os.path.join(tmpdir, 'test_model.pkl')
            assert os.path.exists(model_path)

    def test_load_model_with_name(self):
        """测试使用自定义名称加载模型"""
        from src.coreset.csrel_train import save_model, load_model

        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleModel()

            save_model(tmpdir, model, on_cuda=False, save_name='custom.pkl')
            loaded_model = load_model(tmpdir, save_name='custom.pkl')

            # 验证参数相同
            for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
                assert torch.equal(p1, p2)


class TestClearTempModel:
    """测试临时模型清理"""

    def test_clear_temp_model(self):
        """测试清理临时模型文件"""
        from src.coreset.csrel_train import save_model, clear_temp_model

        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleModel()
            save_model(tmpdir, model, on_cuda=False)

            model_path = os.path.join(tmpdir, 'best_model.pkl')
            assert os.path.exists(model_path)

            clear_temp_model(tmpdir)
            assert not os.path.exists(model_path)

    def test_clear_temp_model_no_file(self):
        """测试清理不存在的文件（不应报错）"""
        from src.coreset.csrel_train import clear_temp_model

        with tempfile.TemporaryDirectory() as tmpdir:
            # 文件不存在，不应报错
            clear_temp_model(tmpdir)


class TestTrainModel:
    """测试 train_model 函数"""

    def test_train_model_basic(self):
        """测试基本训练流程"""
        from src.coreset.csrel_train import train_model

        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleModel()
            train_dataset = SimpleDataset(num_samples=100)
            train_loader = DataLoader(train_dataset, batch_size=10)

            eval_dataset = SimpleDataset(num_samples=50)
            eval_loader = DataLoader(eval_dataset, batch_size=10)

            train_params = {
                'lr': 0.01,
                'use_cuda': False,
                'epochs': 2,
                'opt_type': 'adam'
            }

            trained_model = train_model(
                tmpdir, model, train_loader, eval_loader,
                epochs=2, train_params=train_params,
                verbose=False, save_ckpt=False
            )

            assert trained_model is not None

    def test_train_model_with_early_stop(self):
        """测试带早停的训练"""
        from src.coreset.csrel_train import train_model

        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleModel()
            train_dataset = SimpleDataset(num_samples=100)
            train_loader = DataLoader(train_dataset, batch_size=10)

            eval_dataset = SimpleDataset(num_samples=50)
            eval_loader = DataLoader(eval_dataset, batch_size=10)

            train_params = {
                'lr': 0.01,
                'use_cuda': False,
                'epochs': 10,
                'early_stop': 2,
                'opt_type': 'adam'
            }

            trained_model = train_model(
                tmpdir, model, train_loader, eval_loader,
                epochs=10, train_params=train_params,
                verbose=False, save_ckpt=False
            )

            assert trained_model is not None

    def test_train_model_with_scheduler(self):
        """测试带学习率调度器的训练"""
        from src.coreset.csrel_train import train_model

        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleModel()
            train_dataset = SimpleDataset(num_samples=100)
            train_loader = DataLoader(train_dataset, batch_size=10)

            eval_dataset = SimpleDataset(num_samples=50)
            eval_loader = DataLoader(eval_dataset, batch_size=10)

            train_params = {
                'lr': 0.01,
                'use_cuda': False,
                'epochs': 2,
                'scheduler_type': 'CosineAnnealingLR',
                'opt_type': 'adam'
            }

            trained_model = train_model(
                tmpdir, model, train_loader, eval_loader,
                epochs=2, train_params=train_params,
                verbose=False, save_ckpt=False
            )

            assert trained_model is not None

    def test_train_model_with_kd_loss(self):
        """测试带知识蒸馏损失的训练"""
        from src.coreset.csrel_train import train_model

        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleModel()

            # 创建包含 ref_logits 的数据集
            class KD_dataset(Dataset):
                def __init__(self, num_samples=100):
                    self.num_samples = num_samples
                    self.features = torch.randn(num_samples, 10)
                    self.labels = torch.randint(0, 5, (num_samples,))
                    self.ref_logits = torch.randn(num_samples, 5)

                def __len__(self):
                    return self.num_samples

                def __getitem__(self, idx):
                    return idx, self.features[idx], self.labels[idx], self.ref_logits[idx]

            train_dataset = KD_dataset(num_samples=100)
            train_loader = DataLoader(train_dataset, batch_size=10)

            eval_dataset = SimpleDataset(num_samples=50)
            eval_loader = DataLoader(eval_dataset, batch_size=10)

            train_params = {
                'lr': 0.01,
                'use_cuda': False,
                'epochs': 2,
                'loss_params': {
                    'ce_factor': 1.0,
                    'mse_factor': 0.5
                },
                'opt_type': 'adam'
            }

            trained_model = train_model(
                tmpdir, model, train_loader, eval_loader,
                epochs=2, train_params=train_params,
                verbose=False, save_ckpt=False
            )

            assert trained_model is not None
