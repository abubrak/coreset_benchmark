"""
CSReL v2 核心算法实现

该模块实现了 CSReL v2 (Classwise Spatial Representation Learning v2) 的完整增量选择算法。
CSReL v2 通过训练参考模型、计算可约损失、增量选择高损失样本来构建高质量的 coreset。

核心算法流程：
1. 训练参考模型并计算参考损失
2. 初始化：随机选择少量样本作为初始 coreset
3. 增量选择循环：
   a. 在当前 coreset 上训练模型
   b. 计算剩余样本的可约损失
   c. 选择可约损失最高的样本加入 coreset
   d. 重复直到达到目标大小
"""

import os
import copy
import pickle
from typing import Dict, List, Set, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .csrel_utils import (
    get_class_dic,
    make_class_sizes,
    get_subset_by_id,
    add_new_data,
    compute_loss_dic
)
from .csrel_train import train_model, eval_model
from .csrel_loss import CompliedLoss


class CSReLCoresetV2:
    """
    CSReL v2 增量选择算法

    该类实现了完整的 CSReL v2 算法，包括参考模型训练、可约损失计算
    和增量选择循环。

    参数
    ----
    model : nn.Module
        神经网络模型
    full_dataset : Dataset
        完整训练数据集
    config : CSReLConfigV2
        CSReL v2 配置对象
    full_x : np.ndarray
        完整训练数据特征数组（可选，用于快速访问）
    full_y : np.ndarray
        完整训练数据标签数组（可选，用于快速访问）
    eval_dataset : Dataset
        评估数据集（可选）
    logger : object
        日志记录器（可选）

    示例
    ----
    >>> model = SimpleCNN(num_classes=10)
    >>> config = CSReLConfigV2(coreset_size=1000, incremental_size=100)
    >>> selector = CSReLCoresetV2(model, train_dataset, config)
    >>> coreset_indices = selector.select()
    >>> print(f"Selected {len(coreset_indices)} samples")
    """

    def __init__(
        self,
        model: nn.Module,
        full_dataset: Dataset,
        config,
        full_x: Optional[np.ndarray] = None,
        full_y: Optional[np.ndarray] = None,
        eval_dataset: Optional[Dataset] = None,
        logger: Optional[object] = None
    ):
        self.model = model
        self.full_dataset = full_dataset
        self.config = config
        self.eval_dataset = eval_dataset
        self.logger = logger

        # 提取完整数据
        if full_x is not None and full_y is not None:
            self.full_x = full_x
            self.full_y = full_y
        else:
            # 从数据集提取数据
            self.full_x, self.full_y = self._extract_data_from_dataset(full_dataset)

        # 获取数据集大小
        self.full_size = self.full_y.shape[0]

        # 初始化变量
        self.coreset_ids: Set[int] = set()
        self.ref_loss_dic: Dict[int, float] = {}
        self.id2logit: Dict[int, np.ndarray] = {}

        # 创建临时目录
        os.makedirs(self.config.temp_dir, exist_ok=True)

        # 日志函数
        if logger is None:
            self.logger = self._default_logger
        else:
            self.logger = logger

    def _default_logger(self, message: str) -> None:
        """默认日志函数"""
        print(f"[CSReL v2] {message}")

    def _extract_data_from_dataset(self, dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        从数据集提取数据和标签

        参数
        ----
        dataset : Dataset
            数据集对象

        返回
        ----
        Tuple[np.ndarray, np.ndarray]
            (数据数组, 标签数组)
        """
        data_list = []
        label_list = []

        for i in range(len(dataset)):
            _, sample, label = dataset[i]
            data_list.append(sample.numpy() if isinstance(sample, torch.Tensor) else sample)
            label_list.append(label)

        # 转换为 numpy 数组
        full_x = np.array(data_list)
        full_y = np.array(label_list)

        return full_x, full_y

    def _build_model(self) -> nn.Module:
        """
        构建新的模型实例

        返回
        ----
        nn.Module
            新的模型实例
        """
        return copy.deepcopy(self.model)

    def train_reference_model(self) -> nn.Module:
        """
        训练参考模型并计算参考损失

        该方法在完整数据集上训练一个参考模型，然后计算所有样本的参考损失，
        用于后续的可约损失计算。

        返回
        ----
        nn.Module
            训练好的参考模型
        """
        self.logger("Training reference model...")

        # 构建新模型
        ref_model = self._build_model()

        # 准备训练参数
        train_params = self._get_ref_train_params()

        # 创建完整数据集的数据加载器
        full_loader = DataLoader(
            self.full_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers
        )

        # 训练参考模型
        ref_model = train_model(
            local_path=self.config.temp_dir,
            model=ref_model,
            train_loader=full_loader,
            eval_loader=None,  # 参考模型不需要评估
            epochs=self.config.ref_epochs,
            train_params=train_params,
            verbose=False,
            save_ckpt=False,
            load_best=False,
            weight_decay=self.config.weight_decay,
            log_file=None
        )

        # 计算参考损失字典
        self.logger("Computing reference losses...")
        self.ref_loss_dic = self._compute_ref_loss_dic(ref_model)

        self.logger(f"Reference model trained. Avg loss: {np.mean(list(self.ref_loss_dic.values())):.4f}")

        return ref_model

    def _compute_ref_loss_dic(self, model: nn.Module) -> Dict[int, float]:
        """
        计算参考损失字典

        参数
        ----
        model : nn.Module
            参考模型

        返回
        ----
        Dict[int, float]
            ID 到参考损失的映射
        """
        # 创建损失函数
        if self.config.mse_factor > 0:
            loss_fn = CompliedLoss(
                ce_factor=self.config.ce_factor,
                mse_factor=self.config.mse_factor,
                reduction='none',
                kd_mode=self.config.kd_mode
            )
        else:
            loss_fn = nn.CrossEntropyLoss(reduction='none')

        # 创建数据加载器
        full_loader = DataLoader(
            self.full_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )

        # 计算损失字典
        loss_dic = compute_loss_dic(
            model=model,
            data_loader=full_loader,
            use_cuda=self.config.use_cuda,
            loss_fn=loss_fn,
            aug_iters=1
        )

        return loss_dic

    def incremental_selection(self) -> List[int]:
        """
        执行增量选择算法（核心算法）

        这是 CSReL v2 的核心算法，通过增量选择的方式构建 coreset：
        1. 随机初始化少量样本
        2. 循环执行：
           a. 在当前 coreset 上训练模型
           b. 计算剩余样本的可约损失（当前损失 - 参考损失）
           c. 选择可约损失最高的样本加入 coreset
           d. 直到达到目标大小

        返回
        ----
        List[int]
            选择的 coreset 样本 ID 列表
        """
        self.logger("Starting incremental selection...")

        # 第一步：训练参考模型
        ref_model = self.train_reference_model()

        # 第二步：初始化（随机选择少量样本）
        self._initialize_coreset()

        # 第三步：增量选择循环
        iteration = 0
        while len(self.coreset_ids) < self.config.coreset_size:
            iteration += 1
            self.logger(f"Iteration {iteration}: Current coreset size = {len(self.coreset_ids)}")

            # 计算本轮需要选择的样本数
            remaining_size = self.config.coreset_size - len(self.coreset_ids)
            current_incremental_size = min(self.config.incremental_size, remaining_size)

            # 在当前 coreset 上训练模型
            cur_model = self._train_on_coreset()

            # 基于可约损失选择样本
            new_ids = self._select_by_loss_diff(
                model=cur_model,
                ref_model=ref_model,
                num_samples=current_incremental_size
            )

            # 将新选择的样本加入 coreset
            self.coreset_ids.update(new_ids)

            self.logger(f"Selected {len(new_ids)} new samples")

        # 转换为列表并排序
        coreset_id_list = sorted(list(self.coreset_ids))

        self.logger(f"Incremental selection completed. Final coreset size: {len(coreset_id_list)}")

        return coreset_id_list

    def _initialize_coreset(self) -> None:
        """初始化 coerset（随机选择少量样本）"""
        self.logger(f"Initializing coreset with {self.config.init_size} random samples...")

        # 获取类别字典
        class_dic = get_class_dic(self.full_y)

        # 计算每个类别的初始样本数
        class_sizes = make_class_sizes(class_dic, self.config.init_size)

        # 从每个类别随机选择样本
        for class_label, num_samples in class_sizes.items():
            if num_samples == 0:
                continue

            # 获取该类别的所有样本
            class_samples = class_dic[class_label]

            # 随机选择
            selected = np.random.choice(class_samples, size=min(num_samples, len(class_samples)), replace=False)

            # 加入 coreset
            self.coreset_ids.update(selected)

        self.logger(f"Initialization completed. Coreset size: {len(self.coreset_ids)}")

    def _train_on_coreset(self) -> nn.Module:
        """
        在当前 coreset 上训练模型

        返回
        ----
        nn.Module
            训练好的模型
        """
        # 构建新模型
        cur_model = self._build_model()

        # 准备训练参数
        train_params = self._get_cur_train_params()

        # 获取 coreset 数据
        coreset_data = get_subset_by_id(
            x=self.full_x,
            y=self.full_y,
            ids=self.coreset_ids,
            id2logit=self.id2logit if len(self.id2logit) > 0 else None
        )

        # 保存到临时文件
        coreset_file = os.path.join(self.config.temp_dir, 'coreset_data.pkl')
        with open(coreset_file, 'wb') as fw:
            for data_item in coreset_data:
                pickle.dump(data_item, fw)

        # 创建数据加载器
        try:
            from .csrel_dataset import SimplePILDataset
            coreset_dataset = SimplePILDataset(
                data_file=coreset_file,
                transform=self.config.train_transform,
                shuffle=True
            )
        except ImportError:
            # 如果导入失败，使用简单的数据集
            coreset_dataset = [(d[0], d[1], d[2]) for d in coreset_data]

        coreset_loader = DataLoader(
            coreset_dataset,
            batch_size=min(self.config.batch_size, len(coreset_data)),
            shuffle=True,
            num_workers=self.config.num_workers
        )

        # 训练模型
        cur_model = train_model(
            local_path=self.config.temp_dir,
            model=cur_model,
            train_loader=coreset_loader,
            eval_loader=self.eval_dataset,
            epochs=self.config.inc_epochs,
            train_params=train_params,
            verbose=False,
            save_ckpt=False,
            load_best=False,
            weight_decay=self.config.weight_decay,
            log_file=None
        )

        # 清理临时文件
        if os.path.exists(coreset_file):
            os.remove(coreset_file)

        return cur_model

    def _select_by_loss_diff(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        num_samples: int
    ) -> List[int]:
        """
        基于可约损失选择样本

        可约损失 = 参考模型的损失 - 当前模型的损失
        选择可约损失最大的样本，即那些在当前模型下表现最差
        但在参考模型下表现较好的样本。

        参数
        ----
        model : nn.Module
            当前训练的模型
        ref_model : nn.Module
            参考模型
        num_samples : int
            要选择的样本数量

        返回
        ----
        List[int]
            选择的样本 ID 列表
        """
        # 计算当前模型的损失
        cur_loss_dic = self._compute_ref_loss_dic(model)

        # 计算可约损失
        reducible_loss_dic = {}
        for d_id in self.ref_loss_dic.keys():
            if d_id not in self.coreset_ids:  # 只考虑未选择的样本
                ref_loss = self.ref_loss_dic[d_id]
                cur_loss = cur_loss_dic.get(d_id, ref_loss)  # 如果没有当前损失，使用参考损失
                reducible_loss = ref_loss - cur_loss
                reducible_loss_dic[d_id] = reducible_loss

        # 获取类别字典（用于类别平衡）
        class_dic = get_class_dic(self.full_y)

        # 构建当前未选择样本的类别字典
        remaining_class_dic = {}
        for class_label, samples in class_dic.items():
            remaining_samples = [s for s in samples if s not in self.coreset_ids]
            if remaining_samples:
                remaining_class_dic[class_label] = set(remaining_samples)

        # 计算每个类别的配额
        class_sizes = make_class_sizes(remaining_class_dic, num_samples)

        # 从每个类别选择可约损失最高的样本
        selected_ids = []
        for class_label, quota in class_sizes.items():
            if quota == 0:
                continue

            # 获取该类别的所有样本及其可约损失
            class_samples = remaining_class_dic[class_label]
            class_losses = [
                (d_id, reducible_loss_dic.get(d_id, -1e9))
                for d_id in class_samples
            ]

            # 按可约损失排序（从高到低）
            class_losses.sort(key=lambda x: x[1], reverse=True)

            # 选择前 quota 个
            for i in range(min(quota, len(class_losses))):
                selected_ids.append(class_losses[i][0])

        return selected_ids

    def _get_ref_train_params(self) -> dict:
        """
        获取参考模型的训练参数

        返回
        ----
        dict
            训练参数字典
        """
        params = {
            'lr': self.config.ref_lr,
            'opt_type': self.config.ref_opt_type,
            'use_cuda': self.config.use_cuda,
            'epochs': self.config.ref_epochs
        }

        # 添加损失函数参数
        if self.config.mse_factor > 0:
            params['loss_params'] = {
                'ce_factor': self.config.ce_factor,
                'mse_factor': self.config.mse_factor
            }

        # 添加早停参数
        if self.config.early_stop > 0:
            params['early_stop'] = self.config.early_stop

        # 添加梯度裁剪参数
        if self.config.grad_max_norm is not None:
            params['grad_max_norm'] = self.config.grad_max_norm

        # 添加学习率调度器参数
        if self.config.scheduler_type is not None:
            params['scheduler_type'] = self.config.scheduler_type
            params['scheduler_param'] = self.config.scheduler_param

        return params

    def _get_cur_train_params(self) -> dict:
        """
        获取当前增量训练的参数

        返回
        ----
        dict
            训练参数字典
        """
        params = {
            'lr': self.config.inc_lr,
            'opt_type': self.config.inc_opt_type,
            'use_cuda': self.config.use_cuda,
            'epochs': self.config.inc_epochs
        }

        # 添加损失函数参数
        if self.config.mse_factor > 0 and len(self.id2logit) > 0:
            params['loss_params'] = {
                'ce_factor': self.config.ce_factor,
                'mse_factor': self.config.mse_factor
            }

        # 添加早停参数
        if self.config.early_stop > 0:
            params['early_stop'] = self.config.early_stop

        # 添加梯度裁剪参数
        if self.config.grad_max_norm is not None:
            params['grad_max_norm'] = self.config.grad_max_norm

        # 添加学习率调度器参数
        if self.config.scheduler_type is not None:
            params['scheduler_type'] = self.config.scheduler_type
            params['scheduler_param'] = self.config.scheduler_param

        return params

    def clear(self) -> None:
        """清理临时文件"""
        import shutil
        if os.path.exists(self.config.temp_dir):
            shutil.rmtree(self.config.temp_dir)
        self.logger("Temporary files cleared")

    def select(self) -> List[int]:
        """
        执行 coreset 选择（对外接口）

        返回
        ----
        List[int]
            选择的 coreset 样本 ID 列表
        """
        try:
            # 执行增量选择
            coreset_ids = self.incremental_selection()

            return coreset_ids
        finally:
            # 清理临时文件
            self.clear()
