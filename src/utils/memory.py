"""
内存优化工具模块
提供GPU内存管理、梯度计算优化和检查点功能
"""

import torch
import gc
from contextlib import contextmanager
from typing import Optional, Callable, Any, Dict, List
import logging

logger = logging.getLogger(__name__)


@contextmanager
def torch_memory_saver():
    """
    PyTorch内存节省上下文管理器

    在进入时清理内存,在退出时再次清理并重置内存统计
    确保在关键操作前后释放不必要的内存

    使用示例:
        ```python
        with torch_memory_saver():
            model = LargeModel()
            output = model(input)
        ```
    """
    # 进入时清理内存
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    try:
        yield
    finally:
        # 退出时再次清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def chunked_hessian_vector_product(
    f: Callable,
    params: List[torch.Tensor],
    v: List[torch.Tensor],
    chunk_size: int = 10
) -> List[torch.Tensor]:
    """
    分块计算Hessian-vector product以节省内存

    对于大型模型,一次性计算所有参数的HVP可能导致OOM
    通过将参数分成小块分别计算来降低内存峰值

    参数:
        f: 损失函数,返回标量张量
        params: 模型参数列表
        v: 向量,与params形状相同
        chunk_size: 每块包含的参数数量

    返回:
        Hessian-vector product结果,与v形状相同
    """
    hvp_results = []

    # 将参数分块
    for i in range(0, len(params), chunk_size):
        chunk_params = params[i:i+chunk_size]
        chunk_v = v[i:i+chunk_size]

        # 计算当前块的HVP
        # 使用自动微分: Hv = grad(<grad f, v>, params)
        grad_f = torch.autograd.grad(
            f, chunk_params, create_graph=True, retain_graph=True
        )

        # 计算内积
        grad_v_dot = sum([torch.sum(g * cv) for g, cv in zip(grad_f, chunk_v)])

        # 对内积求梯度得到HVP
        hvp_chunk = torch.autograd.grad(
            grad_v_dot, chunk_params, retain_graph=True
        )

        hvp_results.extend(hvp_chunk)

        # 清理中间结果
        del grad_f, grad_v_dot

    return hvp_results


def conjugate_gradient(
    A: Callable,
    b: torch.Tensor,
    max_iter: int = 100,
    tol: float = 1e-10,
    x0: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    共轭梯度法求解线性系统 Ax = b

    使用CG法避免显式构造矩阵A,节省内存
    A通过函数接口提供,可以是隐式的矩阵-向量乘积

    参数:
        A: 矩阵-向量乘积函数 A(x)
        b: 右侧向量
        max_iter: 最大迭代次数
        tol: 收敛容忍度
        x0: 初始解,若为None则使用零向量

    返回:
        解向量 x
    """
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone()

    r = b - A(x)
    p = r.clone()
    rs_old = torch.sum(r * r)

    for i in range(max_iter):
        Ap = A(p)
        alpha = rs_old / torch.sum(p * Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.sum(r * r)

        if torch.sqrt(rs_new) < tol:
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x


def implicit_gradient(
    loss: torch.Tensor,
    params: List[torch.Tensor],
    v: List[torch.Tensor],
    hvp_fn: Optional[Callable] = None,
    cg_max_iter: int = 100,
    cg_tol: float = 1e-10
) -> List[torch.Tensor]:
    """
    隐式计算梯度,避免存储完整的Hessian矩阵

    通过共轭梯度法隐式求解 H^{-1}v,其中H是Hessian矩阵
    这在二阶优化和影响函数计算中非常有用

    参数:
        loss: 损失张量(标量)
        params: 模型参数列表
        v: 向量,与params形状相同
        hvp_fn: Hessian-vector product函数,若为None则自动构造
        cg_max_iter: CG法最大迭代次数
        cg_tol: CG法收敛容忍度

    返回:
        隐式梯度结果,与params形状相同
    """
    if hvp_fn is None:
        # 默认使用分块HVP
        def hvp_fn(v_flat):
            # 重塑v为参数列表
            v_list = []
            idx = 0
            for p in params:
                size = p.numel()
                v_list.append(v_flat[idx:idx+size].view_as(p))
                idx += size

            # 计算HVP
            hvp_list = chunked_hessian_vector_product(loss, params, v_list)

            # 展平结果
            return torch.cat([h.view(-1) for h in hvp_list])

    # 展平v
    v_flat = torch.cat([v_i.view(-1) for v_i in v])

    # 使用CG求解 Hx = v
    x_flat = conjugate_gradient(
        hvp_fn, v_flat,
        max_iter=cg_max_iter,
        tol=cg_tol
    )

    # 重塑结果为参数列表
    implicit_grad = []
    idx = 0
    for p in params:
        size = p.numel()
        implicit_grad.append(x_flat[idx:idx+size].view_as(p))
        idx += size

    return implicit_grad


class GradientCheckpoint:
    """
    梯度检查点管理器

    在前向传播中选择性地保存中间激活值,
    在反向传播时重新计算未保存的激活值,
    以计算换取内存节省
    """

    def __init__(
        self,
        model: torch.nn.Module,
        checkpoint_interval: int = 2
    ):
        """
        参数:
            model: PyTorch模型
            checkpoint_interval: 每隔多少层检查点一次
                                较小的值节省更多内存但增加计算
        """
        self.model = model
        self.checkpoint_interval = checkpoint_interval
        self._original_forward = None
        self._checkpointed_layers = []

    def enable(self):
        """启用梯度检查点"""
        self._original_forward = self.model.forward

        def checkpointed_forward(*args, **kwargs):
            # 应用torch.utils.checkpoint
            return torch.utils.checkpoint.checkpoint(
                self._original_forward, *args, **kwargs
            )

        self.model.forward = checkpointed_forward
        logger.info("Gradient checkpointing enabled")

    def disable(self):
        """禁用梯度检查点"""
        if self._original_forward is not None:
            self.model.forward = self._original_forward
            self._original_forward = None
            logger.info("Gradient checkpointing disabled")

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disable()


def get_memory_usage(device: Optional[torch.device] = None) -> Dict[str, float]:
    """
    获取当前内存使用情况

    参数:
        device: 指定设备,若为None则使用当前CUDA设备

    返回:
        包含内存使用信息的字典:
        - 'allocated': 已分配内存(GB)
        - 'reserved': 已保留内存(GB)
        - 'max_allocated': 历史最大分配(GB)
        - 'max_reserved': 历史最大保留(GB)
    """
    if not torch.cuda.is_available():
        return {
            'allocated': 0.0,
            'reserved': 0.0,
            'max_allocated': 0.0,
            'max_reserved': 0.0
        }

    if device is None:
        device = torch.cuda.current_device()

    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    max_reserved = torch.cuda.max_memory_reserved(device) / 1024**3

    return {
        'allocated': allocated,
        'reserved': reserved,
        'max_allocated': max_allocated,
        'max_reserved': max_reserved
    }


def clear_model_memory(model: torch.nn.Module):
    """
    清理模型占用的内存

    删除模型梯度、清空缓存并运行垃圾回收

    参数:
        model: PyTorch模型
    """
    # 清空模型梯度
    for p in model.parameters():
        if p.grad is not None:
            del p.grad
        p.grad = None

    # 运行垃圾回收
    gc.collect()

    # 清空CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("CUDA memory cleared")
