"""
Coreset selection adapters for continual learning.

This module provides adapters to integrate various coreset selection methods
(BCSR, CSReL, Bilevel) into the continual learning framework.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from .bcsr_coreset import BCSRCoreset
from .csrel_coreset import CSReLCoreset
from .bilevel_coreset import BilevelCoreset
from ..configs import BilevelConfig, CSReLConfig


class BCSRContinualAdapter:
    """
    Adapter for BCSR method in continual learning scenarios.

    Wraps the BCSRCoreset class to provide a simple interface compatible
    with CoresetBuffer.select_coreset().
    """

    def __init__(
        self,
        learning_rate_inner: float = 0.01,
        learning_rate_outer: float = 5.0,
        num_inner_steps: int = 1,
        num_outer_steps: int = 5,
        beta: float = 0.1,
        device: str = 'cuda'
    ):
        """
        Initialize BCSR adapter.

        Parameters
        ----------
        learning_rate_inner : float
            Inner optimization learning rate for model training.
        learning_rate_outer : float
            Outer optimization learning rate for weight updates.
        num_inner_steps : int
            Number of inner optimization steps.
        num_outer_steps : int
            Number of outer optimization steps.
        beta : float
            Smooth top-K regularization coefficient.
        device : str
            Computation device ('cuda' or 'cpu').
        """
        self.device = device
        self.bcsr_selector = BCSRCoreset(
            learning_rate_inner=learning_rate_inner,
            learning_rate_outer=learning_rate_outer,
            num_inner_steps=num_inner_steps,
            num_outer_steps=num_outer_steps,
            beta=beta,
            device=device
        )

    def select(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        num_samples: int,
        model: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select coreset samples using BCSR method.

        Parameters
        ----------
        data : torch.Tensor
            Input data of shape (n_samples, C, H, W) or (n_samples, n_features).
        labels : torch.Tensor
            Labels of shape (n_samples,).
        num_samples : int
            Number of samples to select.
        model : nn.Module
            PyTorch model for bilevel optimization.

        Returns
        -------
        selected_data : torch.Tensor
            Selected samples of shape (num_samples, ...).
        selected_labels : torch.Tensor
            Selected labels of shape (num_samples,).
        """
        # Ensure data and labels are on correct device
        data = data.to(self.device)
        labels = labels.to(self.device)

        # Use BCSR to select samples
        selected_X, selected_y, info = self.bcsr_selector.coreset_select(
            X=data,
            y=labels,
            coreset_size=num_samples,
            model=model
        )

        # Convert back to torch tensors
        selected_data = torch.from_numpy(selected_X).to(self.device)
        selected_labels = torch.from_numpy(selected_y).long().to(self.device)

        return selected_data, selected_labels
