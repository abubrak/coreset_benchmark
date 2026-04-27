"""
实验检查点和结果记录工具
提供实验状态保存、恢复和结果记录功能
"""

import os
import json
import torch
import pickle
import logging
from pathlib import Path
from typing import Any, Dict, Optional, List
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ExperimentCheckpoint:
    """
    实验检查点管理器

    自动保存实验状态,包括模型、优化器、训练进度等
    支持定期检查点和手动保存
    """

    def __init__(
        self,
        checkpoint_dir: str,
        experiment_name: str,
        max_keep: int = 5,
        save_interval: int = 100
    ):
        """
        参数:
            checkpoint_dir: 检查点根目录
            experiment_name: 实验名称
            max_keep: 保留的最大检查点数量
            save_interval: 自动保存间隔(步数)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_name = experiment_name
        self.max_keep = max_keep
        self.save_interval = save_interval
        self.step_count = 0

        # 创建检查点目录
        self.exp_dir = self.checkpoint_dir / experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Checkpoint manager initialized: {self.exp_dir}")

    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        additional_data: Optional[Dict] = None,
        filename: Optional[str] = None
    ):
        """
        保存检查点

        参数:
            model: PyTorch模型
            optimizer: 优化器(可选)
            scheduler: 学习率调度器(可选)
            additional_data: 额外数据字典(可选)
            filename: 检查点文件名,若为None则自动生成
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"checkpoint_step_{self.step_count}_{timestamp}.pt"

        checkpoint_path = self.exp_dir / filename

        # 准备检查点数据
        checkpoint = {
            'step': self.step_count,
            'model_state_dict': model.state_dict(),
            'timestamp': datetime.now().isoformat()
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        if additional_data is not None:
            checkpoint['additional_data'] = additional_data

        # 保存检查点
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

        # 清理旧检查点
        self._cleanup_old_checkpoints()

    def load(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        checkpoint_path: Optional[str] = None,
        filename: Optional[str] = None
    ) -> Dict:
        """
        加载检查点

        参数:
            model: PyTorch模型
            optimizer: 优化器(可选)
            scheduler: 学习率调度器(可选)
            checkpoint_path: 完整检查点路径
            filename: 检查点文件名

        返回:
            包含加载信息的字典
        """
        if checkpoint_path is None:
            if filename is None:
                # 查找最新的检查点
                checkpoints = list(self.exp_dir.glob("checkpoint_*.pt"))
                if not checkpoints:
                    raise FileNotFoundError("No checkpoints found")
                checkpoint_path = max(checkpoints, key=os.path.getctime)
            else:
                checkpoint_path = self.exp_dir / filename

        # 加载检查点
        checkpoint = torch.load(checkpoint_path)

        # 恢复模型状态
        model.load_state_dict(checkpoint['model_state_dict'])

        # 恢复优化器状态
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 恢复调度器状态
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # 更新步数
        self.step_count = checkpoint['step']

        logger.info(f"Checkpoint loaded: {checkpoint_path}, step: {self.step_count}")

        return checkpoint

    def _cleanup_old_checkpoints(self):
        """清理旧的检查点,保留最近的max_keep个"""
        checkpoints = list(self.exp_dir.glob("checkpoint_*.pt"))

        if len(checkpoints) > self.max_keep:
            # 按创建时间排序
            checkpoints.sort(key=os.path.getctime, reverse=True)

            # 删除旧检查点
            for old_checkpoint in checkpoints[self.max_keep:]:
                old_checkpoint.unlink()
                logger.debug(f"Old checkpoint removed: {old_checkpoint}")

    @contextmanager
    def auto_save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        additional_data: Optional[Dict] = None
    ):
        """
        自动保存上下文管理器

        每隔save_interval步自动保存检查点

        使用示例:
            ```python
            with checkpoint_manager.auto_save(model, optimizer):
                for epoch in range(epochs):
                    train_step()
                    checkpoint_manager.step_count += 1
            ```
        """
        self.step_count = 0
        last_save_step = 0

        try:
            yield self
        finally:
            # 保存最终检查点
            self.save(model, optimizer, scheduler, additional_data)

    def step(self):
        """增加步数计数"""
        self.step_count += 1


class ResultLogger:
    """
    结果记录器

    将实验指标记录到文件,支持多种格式
    """

    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        log_format: str = "json"
    ):
        """
        参数:
            log_dir: 日志目录
            experiment_name: 实验名称
            log_format: 日志格式("json", "csv", "txt")
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.log_format = log_format

        # 创建日志目录
        self.exp_dir = self.log_dir / experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # 初始化日志文件
        self._init_log_files()

        logger.info(f"Result logger initialized: {self.exp_dir}")

    def _init_log_files(self):
        """初始化日志文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.log_format == "json":
            self.log_file = self.exp_dir / f"results_{timestamp}.jsonl"
        elif self.log_format == "csv":
            self.log_file = self.exp_dir / f"results_{timestamp}.csv"
            # 写入CSV头部
            with open(self.log_file, 'w') as f:
                f.write("timestamp,step,metric,value\n")
        elif self.log_format == "txt":
            self.log_file = self.exp_dir / f"results_{timestamp}.txt"
        else:
            raise ValueError(f"Unsupported log format: {self.log_format}")

    def log(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        timestamp: Optional[str] = None
    ):
        """
        记录指标

        参数:
            metrics: 指标字典 {metric_name: value}
            step: 当前步数
            timestamp: 时间戳,若为None则自动生成
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        if step is None:
            step = 0

        if self.log_format == "json":
            self._log_json(metrics, step, timestamp)
        elif self.log_format == "csv":
            self._log_csv(metrics, step, timestamp)
        elif self.log_format == "txt":
            self._log_txt(metrics, step, timestamp)

    def _log_json(self, metrics: Dict[str, float], step: int, timestamp: str):
        """以JSON格式记录"""
        log_entry = {
            'timestamp': timestamp,
            'step': step,
            **metrics
        }

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def _log_csv(self, metrics: Dict[str, float], step: int, timestamp: str):
        """以CSV格式记录"""
        with open(self.log_file, 'a') as f:
            for metric_name, value in metrics.items():
                f.write(f"{timestamp},{step},{metric_name},{value}\n")

    def _log_txt(self, metrics: Dict[str, float], step: int, timestamp: str):
        """以文本格式记录"""
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] Step {step}\n")
            for metric_name, value in metrics.items():
                f.write(f"  {metric_name}: {value}\n")
            f.write("\n")

    def save_summary(self, summary_data: Dict[str, Any]):
        """
        保存实验摘要

        参数:
            summary_data: 摘要数据字典
        """
        summary_file = self.exp_dir / "summary.json"

        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)

        logger.info(f"Summary saved: {summary_file}")

    def load_results(self) -> List[Dict]:
        """
        加载已记录的结果

        返回:
            结果列表
        """
        results = []

        if self.log_format == "json" and self.log_file.exists():
            with open(self.log_file, 'r') as f:
                for line in f:
                    results.append(json.loads(line))
        elif self.log_format == "csv" and self.log_file.exists():
            import pandas as pd
            df = pd.read_csv(self.log_file)
            results = df.to_dict('records')

        return results
