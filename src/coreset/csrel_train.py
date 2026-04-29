"""
CSReL 训练方法模块

该模块包含从原始 CSReL-Coreset-CL 项目移植的训练和评估方法，
用于 CSReL coreset 选择过程中的模型训练和评估。
"""

import os
import copy
import pickle
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .csrel_loss import CompliedLoss


def train_model(
    local_path: str,
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: Optional[DataLoader],
    epochs: int,
    train_params: dict,
    verbose: bool = True,
    save_ckpt: bool = False,
    load_best: bool = False,
    weight_decay: float = 0,
    log_file: Optional[str] = None
) -> nn.Module:
    """
    训练模型用于 CSReL 选择过程

    该函数实现了完整的训练流程，包括损失计算、优化器更新、学习率调度、
    早停机制和模型保存等功能。

    参数
    ----
    local_path : str
        用于保存模型的本地路径
    model : nn.Module
        要训练的模型
    train_loader : DataLoader
        训练数据加载器，每个 batch 应该包含：
        - (id, sample, label) 或
        - (id, sample, label, ref_logits)
    eval_loader : Optional[DataLoader]
        评估数据加载器，如果为 None 则跳过评估
    epochs : int
        训练轮数
    train_params : dict
        训练参数字典，包含：
        - lr: 学习率
        - use_cuda: 是否使用 CUDA
        - opt_type: 优化器类型 ('adam' 或 'sgd')
        - scheduler_type: 学习率调度器类型 (可选)
        - scheduler_param: 调度器参数 (可选)
        - loss_params: 损失函数参数 (可选)
          * ce_factor: 交叉熵权重
          * mse_factor: 知识蒸馏权重
        - early_stop: 早停耐心值 (可选)
        - grad_max_norm: 梯度裁剪最大范数 (可选)
        - log_steps: 日志输出步数 (可选)
    verbose : bool, default=True
        是否打印训练信息
    save_ckpt : bool, default=False
        是否保存每个 epoch 的检查点
    load_best : bool, default=False
        是否在训练后加载最佳模型
    weight_decay : float, default=0
        权重衰减（L2 正则化）系数
    log_file : Optional[str], default=None
        损失日志文件路径，如果为 None 则不保存

    返回
    ----
    nn.Module
        训练完成的模型（在 CPU 上）
    """
    # 获取日志步数
    log_steps = train_params.get('log_steps', 100)

    # 初始化损失函数
    if 'loss_params' in train_params:
        loss_fn = CompliedLoss(
            ce_factor=train_params['loss_params']['ce_factor'],
            mse_factor=train_params['loss_params']['mse_factor']
        )
    else:
        loss_fn = nn.CrossEntropyLoss()

    # 移动模型到 GPU
    if train_params['use_cuda']:
        model = model.cuda()

    # 初始化优化器
    opt_type = train_params.get('opt_type', 'sgd')
    if opt_type == 'adam':
        opt = torch.optim.Adam(
            lr=train_params['lr'],
            weight_decay=weight_decay,
            params=model.parameters()
        )
    else:
        opt = torch.optim.SGD(
            lr=train_params['lr'],
            weight_decay=weight_decay,
            params=model.parameters()
        )

    # 初始化学习率调度器
    scheduler = None
    if 'scheduler_type' in train_params:
        scheduler_type = train_params['scheduler_type']
        if scheduler_type == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=train_params['epochs']
            )
        elif scheduler_type == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                factor=train_params['scheduler_param']['factor'],
                patience=train_params['scheduler_param']['patience'],
                min_lr=train_params['scheduler_param']['min_lr'],
                verbose=verbose
            )

    # 训练循环
    best_acc = None
    bad_cnt = 0
    losses = []

    for i in range(epochs):
        # 训练一个 epoch
        step = 0
        total_loss = 0

        for data in train_loader:
            # 解包数据
            if len(data) == 2:
                _, sp, lab = data
                ref_logits = None
            elif len(data) == 4:
                _, sp, lab, ref_logits = data
            else:
                _, sp, lab = data
                ref_logits = None

            # 移动到 GPU
            if train_params['use_cuda']:
                sp = sp.cuda()
                lab = lab.cuda()
                if ref_logits is not None:
                    ref_logits = ref_logits.cuda()

            # 前向传播
            out = model(sp)

            # 计算损失
            if 'loss_params' in train_params:
                loss = loss_fn(x=out, y=lab, logits=ref_logits)
            else:
                loss = loss_fn(out, lab)

            # 反向传播
            opt.zero_grad()
            loss.backward()

            # 梯度裁剪
            if 'grad_max_norm' in train_params and train_params['grad_max_norm'] is not None:
                torch.nn.utils.clip_grad_norm_(
                    parameters=model.parameters(),
                    max_norm=train_params['grad_max_norm'],
                    norm_type=2
                )

            # 更新参数
            opt.step()

            total_loss += loss.item()

            # 日志输出
            if step % log_steps == 0:
                if verbose:
                    print('loss at step:', step, 'is', loss.item())
                if log_file is not None:
                    losses.append(float(loss.item()))

            step += 1

        # 计算平均损失
        avg_loss = total_loss / step

        # 每 10 个 epoch 或最后一个 epoch 进行评估和保存
        if i % 10 == 0 or i == epochs - 1:
            # 保存检查点
            if save_ckpt:
                save_name = 'model_' + str(i) + '.pkl'
                save_model(
                    local_path=local_path,
                    model=model,
                    on_cuda=train_params['use_cuda'],
                    save_name=save_name
                )

            # 评估
            if eval_loader is not None:
                acc = eval_model(
                    model=model,
                    on_cuda=train_params['use_cuda'],
                    eval_loader=eval_loader
                )
            else:
                acc = None

            if verbose:
                print('accuracy in epoch', i, 'is:', acc)

            # 保存最佳模型
            if best_acc is None or acc > best_acc:
                best_acc = acc
                bad_cnt = 0
                save_model(
                    local_path=local_path,
                    model=model,
                    on_cuda=train_params['use_cuda']
                )
            else:
                bad_cnt += 1

            # 早停
            if 'early_stop' in train_params and bad_cnt == train_params['early_stop']:
                if verbose:
                    print('\tearly stop at epoch:', i)
                break

        # 更新学习率
        if scheduler is not None:
            if train_params['scheduler_type'] == 'CosineAnnealingLR':
                scheduler.step()
            elif train_params['scheduler_type'] == 'ReduceLROnPlateau':
                scheduler.step(avg_loss)

        # 数据集 shuffle（如果支持）
        if i < epochs - 1 and hasattr(train_loader.dataset, 'shuffle_dataset'):
            train_loader.dataset.shuffle_dataset()

    # 保存最终检查点
    if save_ckpt:
        save_name = 'model_' + str(epochs) + '.pkl'
        save_model(
            local_path=local_path,
            model=model,
            on_cuda=train_params['use_cuda'],
            save_name=save_name
        )

    # 加载最佳模型
    if eval_loader is not None and load_best and train_params.get('early_stop', 0) > 0:
        model = load_model(local_path=local_path)

    if verbose:
        print('\tbest accuracy is:', best_acc)

    # 移回 CPU
    if train_params['use_cuda']:
        model = model.cpu()

    # 清理临时文件
    clear_temp_model(local_path=local_path)

    # 清理 shuffle 文件（如果支持）
    if hasattr(train_loader.dataset, 'remove_shuffle_file'):
        train_loader.dataset.remove_shuffle_file()

    # 保存损失日志
    if log_file is not None:
        with open(log_file, 'wb') as fw:
            pickle.dump(losses, fw)

    return model


def eval_model(
    model: nn.Module,
    eval_loader: DataLoader,
    on_cuda: bool = False,
    return_loss: bool = False
) -> Union[float, Tuple[float, float]]:
    """
    评估模型性能

    参数
    ----
    model : nn.Module
        要评估的模型
    eval_loader : DataLoader
        评估数据加载器
    on_cuda : bool, default=False
        是否使用 CUDA
    return_loss : bool, default=False
        是否同时返回损失

    返回
    ----
    Union[float, Tuple[float, float]]
        如果 return_loss=False，返回准确率
        如果 return_loss=True，返回 (准确率, 平均损失)
    """
    status = model.training
    model.eval()

    loss_fn = nn.CrossEntropyLoss(reduction='sum')

    with torch.no_grad():
        total_num = 0
        total_loss = 0
        right_num = 0

        for e_data in eval_loader:
            # 解包数据
            if len(e_data) == 2:
                _, sp, lab = e_data
            else:
                _, sp, lab = e_data

            # 移动到 GPU
            if on_cuda:
                sp = sp.cuda()
                lab = lab.cuda()

            # 前向传播
            out = model(sp).clone().detach()
            loss = loss_fn(out, lab)

            # 移回 CPU
            if on_cuda:
                out = out.cpu()
                lab = lab.cpu()
                loss = loss.cpu()

            out = out.numpy()
            lab = lab.numpy()
            loss = loss.clone().detach().numpy()

            total_loss += loss

            # 计算准确率
            for j in range(out.shape[0]):
                pred = np.argmax(out[j, :])
                if int(lab[j]) == int(pred):
                    right_num += 1
                total_num += 1

        acc = right_num / total_num
        avg_loss = total_loss / total_num

    model.train(status)

    if return_loss:
        return acc, avg_loss
    else:
        return acc


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    计算准确率

    参数
    ----
    logits : torch.Tensor
        模型输出的 logits，形状为 (batch_size, num_classes)
    labels : torch.Tensor
        真实标签，形状为 (batch_size,)

    返回
    ----
    float
        准确率（0 到 1 之间）
    """
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy


def compute_avg_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: nn.Module
) -> float:
    """
    计算平均损失

    参数
    ----
    logits : torch.Tensor
        模型输出的 logits，形状为 (batch_size, num_classes)
    labels : torch.Tensor
        真实标签，形状为 (batch_size,)
    loss_fn : nn.Module
        损失函数（reduction 应为 'none'）

    返回
    ----
    float
        平均损失值
    """
    losses = loss_fn(logits, labels)
    avg_loss = torch.mean(losses).item()
    return avg_loss


def compute_loss_var(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: nn.Module
) -> float:
    """
    计算损失方差

    参数
    ----
    logits : torch.Tensor
        模型输出的 logits，形状为 (batch_size, num_classes)
    labels : torch.Tensor
        真实标签，形状为 (batch_size,)
    loss_fn : nn.Module
        损失函数（reduction 应为 'none'）

    返回
    ----
    float
        损失方差
    """
    losses = loss_fn(logits, labels)
    loss_var = torch.var(losses).item()
    return loss_var


def save_model(
    local_path: str,
    model: nn.Module,
    on_cuda: bool,
    save_name: Optional[str] = None
) -> None:
    """
    保存模型到文件

    参数
    ----
    local_path : str
        保存路径
    model : nn.Module
        要保存的模型
    on_cuda : bool
        模型是否在 CUDA 上
    save_name : Optional[str], default=None
        保存的文件名，如果为 None 则使用 'best_model.pkl'
    """
    if save_name is None:
        out_model_file = os.path.join(local_path, 'best_model.pkl')
    else:
        out_model_file = os.path.join(local_path, save_name)

    saved_model = copy.deepcopy(model)
    if on_cuda:
        saved_model.cpu()

    torch.save(saved_model, out_model_file)


def load_model(
    local_path: str,
    save_name: Optional[str] = None
) -> nn.Module:
    """
    从文件加载模型

    参数
    ----
    local_path : str
        模型文件所在路径
    save_name : Optional[str], default=None
        模型文件名，如果为 None 则使用 'best_model.pkl'

    返回
    ----
    nn.Module
        加载的模型
    """
    if save_name is None:
        out_model_file = os.path.join(local_path, 'best_model.pkl')
    else:
        out_model_file = os.path.join(local_path, save_name)

    model = torch.load(out_model_file, map_location='cpu', weights_only=False)
    return model


def clear_temp_model(local_path: str) -> None:
    """
    清理临时模型文件

    参数
    ----
    local_path : str
        临时模型文件所在路径
    """
    temp_model_file = os.path.join(local_path, 'best_model.pkl')
    if os.path.exists(temp_model_file):
        os.remove(temp_model_file)
