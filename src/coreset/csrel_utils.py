"""
CSReL 核心工具函数模块

该模块包含从原始 CSReL-Coreset-CL 项目移植的核心工具函数，
主要用于 coreset 选择和数据管理。
"""
import os
import random
import pickle
from typing import Dict, List, Set, Optional, Union

import numpy as np
import torch


def get_class_dic(y: np.ndarray) -> Dict[int, List[int]]:
    """
    根据标签数组构建类别字典

    将标签数组转换为字典，其中键是类别标签，值是属于该类别的样本索引列表。

    参数
    ----
    y : np.ndarray
        标签数组，形状为 (n_samples,)

    返回
    ----
    Dict[int, List[int]]
        类别字典，键为类别标签，值为样本索引列表

    示例
    ----
    >>> y = np.array([0, 1, 0, 2, 1, 2])
    >>> get_class_dic(y)
    {0: [0, 2], 1: [1, 4], 2: [3, 5]}
    """
    class_dic: Dict[int, List[int]] = {}
    for i in range(y.shape[0]):
        lab = int(y[i])
        if lab not in class_dic:
            class_dic[lab] = [i]
        else:
            class_dic[lab].append(i)
    return class_dic


def make_class_sizes(class_ids: Dict[int, Set[int]], incremental_size: int) -> Dict[int, int]:
    """
    根据当前类别中的样本数量分配新样本的配额

    该函数实现了一种平衡策略：样本数较少的类别会获得更多的新样本配额，
    以促进类别间的平衡。

    参数
    ----
    class_ids : Dict[int, Set[int]]
        类别到样本 ID 集合的字典
    incremental_size : int
        需要新增的样本总数

    返回
    ----
    Dict[int, int]
        每个类别应该新增的样本数量

    策略说明
    --------
    1. 首先计算每个类别当前的样本数量
    2. 找到样本数最多的类别作为基准
    3. 对于每个类别，分配 (max_count - current_count) 的样本数
    4. 如果还有剩余配额，平均分配给所有类别

    示例
    ----
    >>> class_ids = {0: {0, 1}, 1: {2}, 2: {3, 4, 5}}
    >>> make_class_sizes(class_ids, 6)
    {0: 2, 1: 3, 2: 1}
    """
    if not class_ids:
        return {}

    # 计算每个类别的样本数量
    class_cnts: Dict[int, int] = {}
    max_cnt = -1
    for ci in class_ids.keys():
        class_cnts[ci] = len(class_ids[ci])
        if len(class_ids[ci]) > max_cnt:
            max_cnt = len(class_ids[ci])

    # 按样本数量排序（从少到多）
    sorted_cnt = sorted(class_cnts.items(), key=lambda x: x[1])

    # 初始化每个类别的配额
    class_sizes: Dict[int, int] = {}
    for ci in class_ids.keys():
        class_sizes[ci] = 0

    # 第一阶段：根据差异分配配额
    rest_size = incremental_size
    for i in range(len(sorted_cnt)):
        ci, cnt = sorted_cnt[i]
        # 分配的数量 = 最大数量 - 当前数量
        to_select = max_cnt - cnt
        to_select = min(to_select, rest_size)
        class_sizes[ci] = to_select
        rest_size -= to_select
        if rest_size == 0:
            break

    # 第二阶段：如果还有剩余配额，平均分配
    if rest_size > 0:
        base_size = int(rest_size // len(class_ids))
        for ci in class_sizes.keys():
            class_sizes[ci] = class_sizes[ci] + base_size
        rest_size = rest_size - base_size * len(class_ids)

        # 处理余数
        for ci in class_sizes.keys():
            if rest_size == 0:
                break
            class_sizes[ci] += 1
            rest_size -= 1

    return class_sizes


def get_subset_by_id(
    x: np.ndarray,
    y: np.ndarray,
    ids: Set[int],
    id_list: Optional[List[int]] = None,
    id2logit: Optional[Dict[int, np.ndarray]] = None
) -> List[tuple]:
    """
    根据 ID 集合从数据中提取子集

    参数
    ----
    x : np.ndarray
        特征数组
    y : np.ndarray
        标签数组
    ids : Set[int]
        要提取的样本 ID 集合
    id_list : Optional[List[int]], default=None
        自定义 ID 列表。如果为 None，则使用数组索引作为 ID
    id2logit : Optional[Dict[int, np.ndarray]], default=None
        ID 到 logit 的映射。如果提供，返回的数据将包含 logit

    返回
    ----
    List[tuple]
        提取的数据列表，每个元素是一个元组：
        - 如果 id2logit 为 None: (id, sample, label)
        - 如果 id2logit 不为 None: (id, sample, label, logit)

    示例
    ----
    >>> x = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y = np.array([0, 1, 0])
    >>> get_subset_by_id(x, y, {0, 2})
    [(0, tensor([1, 2]), 0), (2, tensor([5, 6]), 0)]
    """
    selected_data: List[tuple] = []

    # 确定要提取的位置索引
    if id_list is None:
        d_pos = ids
    else:
        d_pos = []
        id_pool = set(ids)
        for i, d_id in enumerate(id_list):
            if d_id in id_pool:
                d_pos.append(i)

    # 提取数据
    for pi in d_pos:
        # 转换样本为 tensor
        sp = torch.tensor(x[pi], dtype=torch.float32).clone().detach()

        # 确定 ID
        if id_list is None:
            d_id = pi
        else:
            d_id = id_list[pi]

        # 构建数据元组
        data: tuple
        if id2logit is not None and d_id in id2logit:
            data = (d_id, sp, int(y[pi]), id2logit[d_id])
        else:
            data = (d_id, sp, int(y[pi]))

        selected_data.append(data)

    return selected_data


def add_new_data(data_file: str, new_data: list, shuffle: bool = True) -> None:
    """
    将新数据添加到 pickle 文件中，自动防止重复

    该函数会读取现有数据（如果文件存在），合并新数据（去重），
    然后写回文件。

    参数
    ----
    data_file : str
        数据文件路径
    new_data : list
        要添加的新数据列表，每个元素应该是一个元组，第一个元素是 ID
    shuffle : bool, default=True
        是否在保存前打乱数据顺序

    注意事项
    --------
    - 数据的第一个元素必须是 ID，用于去重
    - 如果 ID 已存在，则跳过该数据
    - 文件以二进制 pickle 格式保存
    """
    # 读取现有数据
    ori_data: list = []
    ori_ids: Set[int] = set()

    if os.path.exists(data_file):
        with open(data_file, 'rb') as fr:
            while True:
                try:
                    di = pickle.load(fr)
                    d_id = int(di[0])
                    ori_ids.add(d_id)
                    ori_data.append(di)
                except EOFError:
                    break

    # 合并新数据（去重）
    all_data = ori_data.copy()
    for di in new_data:
        d_id = int(di[0])
        if d_id not in ori_ids:
            all_data.append(di)

    # 可选：打乱数据
    if shuffle:
        random.shuffle(all_data)

    # 写回文件
    with open(data_file, 'wb') as fw:
        for di in all_data:
            pickle.dump(di, fw)


def compute_loss_dic(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    use_cuda: bool,
    loss_fn: Union[torch.nn.Module, callable],
    aug_iters: int = 1
) -> Dict[int, float]:
    """
    计算数据集中每个样本的平均损失

    该函数遍历数据加载器多次（由 aug_iters 指定），计算每个样本的
    平均损失值，用于后续的 coreset 选择。

    参数
    ----
    model : torch.nn.Module
        用于计算损失的模型
    data_loader : torch.utils.data.DataLoader
        数据加载器，每个 batch 应该包含 (id, sample, label) 或
        (id, sample, label, logit)
    use_cuda : bool
        是否使用 CUDA
    loss_fn : Union[torch.nn.Module, callable]
        损失函数，应该接受 (logits, labels) 或 (logits, labels, logits) 作为输入
    aug_iters : int, default=1
        计算损失的迭代次数（用于数据增强），最终结果是多次迭代的平均值

    返回
    ----
    Dict[int, float]
        ID 到平均损失的映射

    注意事项
    --------
    - 模型在评估模式下运行（model.eval()）
    - 不会进行梯度计算（torch.no_grad()）
    - 如果使用 CUDA，模型会在函数内部移动到 GPU，计算完成后移回 CPU
    - 数据加载器的每个 batch 应该包含 ID 作为第一个元素

    示例
    ----
    >>> model = MyModel()
    >>> dataloader = get_dataloader()
    >>> loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    >>> loss_dic = compute_loss_dic(model, dataloader, use_cuda=False, loss_fn=loss_fn)
    >>> print(loss_dic)
    {0: 0.523, 1: 0.341, 2: 0.789, ...}
    """
    model.eval()

    if use_cuda:
        model.cuda()

    loss_dic: Dict[int, List[float]] = {}

    with torch.no_grad():
        for _ in range(aug_iters):
            for data in data_loader:
                # 解包数据
                if len(data) == 4:
                    d_ids, sps, labs, logits = data
                else:
                    d_ids, sps, labs = data
                    logits = None

                # 移动到 GPU
                if use_cuda:
                    sps = sps.cuda()
                    labs = labs.cuda()
                    if logits is not None:
                        logits = logits.cuda()

                # 计算损失
                if logits is not None:
                    loss = loss_fn(model(sps), labs, logits)
                else:
                    loss = loss_fn(model(sps), labs)

                # 移回 CPU
                if use_cuda:
                    loss = loss.cpu()

                loss = loss.clone().detach().numpy()
                batch_size = sps.shape[0]

                # 累积每个样本的损失
                for j in range(batch_size):
                    d_id = int(d_ids[j].numpy())
                    if d_id not in loss_dic:
                        loss_dic[d_id] = [float(loss[j])]
                    else:
                        loss_dic[d_id].append(float(loss[j]))

    # 计算平均损失
    avg_loss_dic: Dict[int, float] = {}
    for d_id, loss_list in loss_dic.items():
        avg_loss_dic[d_id] = float(np.mean(loss_list))

    # 不要移动模型设备！模型应该保持在调用者指定的设备上
    # 移除 model.cpu() 调用以避免设备不匹配问题

    return avg_loss_dic
