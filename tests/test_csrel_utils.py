"""
CSReL 工具函数测试

测试从原始 CSReL 项目移植的核心工具函数
"""
import pytest
import numpy as np
import torch
import pickle
import tempfile
import os
from typing import Dict, List, Set

from src.coreset.csrel_utils import (
    get_class_dic,
    make_class_sizes,
    get_subset_by_id,
    add_new_data,
    compute_loss_dic
)


class TestGetClassDic:
    """测试 get_class_dic 函数"""

    def test_simple_classification(self):
        """测试简单的分类任务"""
        y = np.array([0, 1, 0, 2, 1, 2])
        result = get_class_dic(y)

        expected = {
            0: [0, 2],
            1: [1, 4],
            2: [3, 5]
        }

        assert result == expected

    def test_single_class(self):
        """测试只有一个类别的情况"""
        y = np.array([0, 0, 0, 0])
        result = get_class_dic(y)

        expected = {0: [0, 1, 2, 3]}
        assert result == expected

    def test_empty_array(self):
        """测试空数组"""
        y = np.array([])
        result = get_class_dic(y)

        assert result == {}

    def test_non_consecutive_classes(self):
        """测试非连续的类别标签"""
        y = np.array([0, 2, 5, 0, 2, 5])
        result = get_class_dic(y)

        expected = {
            0: [0, 3],
            2: [1, 4],
            5: [2, 5]
        }

        assert result == expected


class TestMakeClassSizes:
    """测试 make_class_sizes 函数"""

    def test_balanced_distribution(self):
        """测试均衡的类别分布"""
        class_ids = {
            0: {0, 1, 2},
            1: {3, 4, 5},
            2: {6, 7, 8}
        }
        incremental_size = 6

        result = make_class_sizes(class_ids, incremental_size)

        # 所有类别大小相同，应该平均分配
        assert sum(result.values()) == 6
        # 每个类别应该得到相同的大小（2）
        assert result[0] == 2
        assert result[1] == 2
        assert result[2] == 2

    def test_imbalanced_distribution(self):
        """测试不平衡的类别分布"""
        class_ids = {
            0: {0, 1},           # 2 个样本
            1: {2, 3, 4},        # 3 个样本
            2: {5}               # 1 个样本
        }
        incremental_size = 6

        result = make_class_sizes(class_ids, incremental_size)

        assert sum(result.values()) == 6
        # 样本最少的类别应该获得最多的新样本
        assert result[2] >= result[0]  # 类别 2 样本最少
        assert result[2] >= result[1]  # 类别 2 样本最少

    def test_empty_class_ids(self):
        """测试空的 class_ids"""
        class_ids = {}
        incremental_size = 5

        result = make_class_sizes(class_ids, incremental_size)

        assert result == {}

    def test_incremental_size_zero(self):
        """测试增量为 0 的情况"""
        class_ids = {
            0: {0, 1},
            1: {2, 3}
        }
        incremental_size = 0

        result = make_class_sizes(class_ids, incremental_size)

        assert result[0] == 0
        assert result[1] == 0


class TestGetSubsetById:
    """测试 get_subset_by_id 函数"""

    def test_basic_subset(self):
        """测试基本的子集选择"""
        x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        ids = {0, 2}

        result = get_subset_by_id(x, y, ids)

        assert len(result) == 2
        assert result[0][0] == 0  # ID
        assert result[0][2] == 0  # label
        assert result[1][0] == 2  # ID
        assert result[1][2] == 0  # label

    def test_with_id_list(self):
        """测试使用自定义 ID 列表"""
        x = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 2])
        ids = {10, 30}
        id_list = [10, 20, 30]

        result = get_subset_by_id(x, y, ids, id_list=id_list)

        assert len(result) == 2
        assert result[0][0] == 10
        assert result[1][0] == 30

    def test_with_id2logit(self):
        """测试包含 logit 的情况"""
        x = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 2])
        ids = {0, 1}
        id2logit = {
            0: np.array([0.1, 0.9]),
            1: np.array([0.8, 0.2])
        }

        result = get_subset_by_id(x, y, ids, id2logit=id2logit)

        assert len(result) == 2
        assert len(result[0]) == 4  # 包含 logit
        assert len(result[1]) == 4
        np.testing.assert_array_almost_equal(result[0][3], id2logit[0])
        np.testing.assert_array_almost_equal(result[1][3], id2logit[1])

    def test_empty_ids(self):
        """测试空的 ID 集合"""
        x = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        ids = set()

        result = get_subset_by_id(x, y, ids)

        assert len(result) == 0


class TestAddNewData:
    """测试 add_new_data 函数"""

    def test_add_to_new_file(self):
        """测试添加到新文件"""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, 'test_data.pkl')
            new_data = [
                [0, torch.tensor([1, 2]), 0],
                [1, torch.tensor([3, 4]), 1]
            ]

            add_new_data(data_file, new_data, shuffle=False)

            # 验证数据被正确写入
            with open(data_file, 'rb') as f:
                data1 = pickle.load(f)
                data2 = pickle.load(f)

                assert data1[0] == 0
                assert data2[0] == 1

    def test_add_to_existing_file(self):
        """测试添加到已存在的文件"""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, 'test_data.pkl')

            # 首先写入一些数据
            original_data = [[0, torch.tensor([1, 2]), 0]]
            with open(data_file, 'wb') as f:
                pickle.dump(original_data[0], f)

            # 添加新数据
            new_data = [[1, torch.tensor([3, 4]), 1]]
            add_new_data(data_file, new_data, shuffle=False)

            # 验证原始数据和新数据都存在
            with open(data_file, 'rb') as f:
                data1 = pickle.load(f)
                data2 = pickle.load(f)

                assert data1[0] == 0
                assert data2[0] == 1

    def test_prevent_duplicates(self):
        """测试防止重复添加相同 ID 的数据"""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, 'test_data.pkl')

            # 添加第一批数据
            data1 = [[0, torch.tensor([1, 2]), 0], [1, torch.tensor([3, 4]), 1]]
            add_new_data(data_file, data1, shuffle=False)

            # 尝试添加包含重复 ID 的数据
            data2 = [[1, torch.tensor([5, 6]), 1], [2, torch.tensor([7, 8]), 2]]
            add_new_data(data_file, data2, shuffle=False)

            # 验证只有 3 个数据点（ID 1 只出现一次）
            with open(data_file, 'rb') as f:
                count = 0
                ids = set()
                while True:
                    try:
                        data = pickle.load(f)
                        ids.add(data[0])
                        count += 1
                    except EOFError:
                        break

                assert count == 3
                assert ids == {0, 1, 2}

    def test_shuffle(self):
        """测试数据打乱"""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, 'test_data.pkl')

            new_data = [
                [i, torch.tensor([i, i+1]), i % 3]
                for i in range(10)
            ]

            add_new_data(data_file, new_data, shuffle=True)

            # 读取所有数据
            with open(data_file, 'rb') as f:
                loaded_data = []
                while True:
                    try:
                        data = pickle.load(f)
                        loaded_data.append(data[0])
                    except EOFError:
                        break

            # 验证所有数据都存在
            assert len(loaded_data) == 10
            assert set(loaded_data) == set(range(10))


class TestComputeLossDic:
    """测试 compute_loss_dic 函数"""

    @pytest.fixture
    def simple_model(self):
        """创建简单的测试模型"""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 3)
        )
        return model

    @pytest.fixture
    def simple_dataloader(self):
        """创建简单的数据加载器"""
        # 创建简单的数据集
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self):
                self.data = torch.randn(20, 10)
                self.labels = torch.randint(0, 3, (20,))
                self.ids = torch.arange(20)

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return (self.ids[idx], self.data[idx], self.labels[idx])

        dataset = SimpleDataset()
        return torch.utils.data.DataLoader(dataset, batch_size=4)

    def test_compute_loss_basic(self, simple_model, simple_dataloader):
        """测试基本的损失计算"""
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

        result = compute_loss_dic(
            model=simple_model,
            data_loader=simple_dataloader,
            use_cuda=False,
            loss_fn=loss_fn,
            aug_iters=1
        )

        # 验证返回类型
        assert isinstance(result, dict)

        # 验证包含所有样本
        assert len(result) == 20

        # 验证损失值为正数
        for loss_val in result.values():
            assert loss_val >= 0

    def test_compute_loss_with_aug_iters(self, simple_model, simple_dataloader):
        """测试多次增强迭代的损失计算"""
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

        result = compute_loss_dic(
            model=simple_model,
            data_loader=simple_dataloader,
            use_cuda=False,
            loss_fn=loss_fn,
            aug_iters=3
        )

        # 验证包含所有样本
        assert len(result) == 20

        # 验证损失值（多次迭代应该得到平均值）
        for loss_val in result.values():
            assert isinstance(loss_val, float)
            assert loss_val >= 0

    def test_compute_loss_mse_factor(self, simple_model, simple_dataloader):
        """测试包含 MSE 因子的损失计算"""
        # 创建包含 logits 的数据集
        class DatasetWithLogits(torch.utils.data.Dataset):
            def __init__(self):
                self.data = torch.randn(20, 10)
                self.labels = torch.randint(0, 3, (20,))
                self.logits = torch.randn(20, 3)
                self.ids = torch.arange(20)

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return (self.ids[idx], self.data[idx], self.labels[idx], self.logits[idx])

        dataset = DatasetWithLogits()
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)

        # 创建复合损失函数
        class CompliedLoss:
            def __init__(self, ce_factor=1.0, mse_factor=0.5, reduction='mean'):
                self.ce_factor = ce_factor
                self.mse_factor = mse_factor
                self.reduction = reduction

            def __call__(self, x, y, logits=None):
                ce_loss = torch.nn.functional.cross_entropy(x, y, reduction='none')
                if logits is not None and self.mse_factor > 0:
                    mse_loss = torch.nn.functional.mse_loss(x, logits, reduction='none')
                    mse_loss = torch.mean(mse_loss, dim=-1)
                    return self.ce_factor * ce_loss + self.mse_factor * mse_loss
                return self.ce_factor * ce_loss

        loss_fn = CompliedLoss(ce_factor=1.0, mse_factor=0.5)

        result = compute_loss_dic(
            model=simple_model,
            data_loader=dataloader,
            use_cuda=False,
            loss_fn=loss_fn,
            aug_iters=1
        )

        # 验证返回结果
        assert len(result) == 20
        for loss_val in result.values():
            assert isinstance(loss_val, float)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
