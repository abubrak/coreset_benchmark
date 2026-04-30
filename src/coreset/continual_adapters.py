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
from ..models.resnet import ResNet18
from ..models.cnn import CNN_MNIST


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

        # 确保模型在正确的设备上并记录原始设备
        original_device = next(model.parameters()).device
        model = model.to(self.device)

        # Use BCSR to select samples
        selected_X, selected_y, info = self.bcsr_selector.coreset_select(
            X=data,
            y=labels,
            coreset_size=num_samples,
            model=model
        )

        # 恢复模型到原始设备
        model = model.to(original_device)

        # Convert back to torch tensors
        selected_data = torch.from_numpy(selected_X).to(self.device)
        selected_labels = torch.from_numpy(selected_y).long().to(self.device)

        return selected_data, selected_labels


class CSReLContinualAdapter:
    """
    Adapter for CSReL method in continual learning scenarios.

    Wraps the CSReLCoreset class to provide a simple interface compatible
    with CoresetBuffer.select_coreset().
    """

    def __init__(
        self,
        num_epochs: int = 50,
        learning_rate: float = 0.001,
        batch_size: int = 128,
        selection_ratio: float = 0.1,
        class_balance: bool = True,
        device: str = 'cuda'
    ):
        """
        Initialize CSReL adapter.

        Parameters
        ----------
        num_epochs : int
            Reference model training epochs.
        learning_rate : float
            Learning rate for reference model training.
        batch_size : int
            Batch size for training.
        selection_ratio : float
            Ratio of samples to select.
        class_balance : bool
            Whether to balance classes in selection.
        device : str
            Computation device ('cuda' or 'cpu').
        """
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.selection_ratio = selection_ratio
        self.class_balance = class_balance
        self.csrel_selector = None  # Will be created on first use

    def _get_selector(self, num_classes: int, dataset: str = 'MNIST') -> CSReLCoreset:
        """Get or create CSReL selector."""
        if self.csrel_selector is None:
            config = CSReLConfig(
                dataset=dataset,
                num_classes=num_classes,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                num_epochs=self.num_epochs,
                selection_ratio=self.selection_ratio,
                class_balance=self.class_balance,
                device=self.device
            )
            self.csrel_selector = CSReLCoreset(config=config)
        return self.csrel_selector

    def select(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        num_samples: int,
        model: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select coreset samples using CSReL method.

        Parameters
        ----------
        data : torch.Tensor
            Input data of shape (n_samples, C, H, W) or (n_samples, n_features).
        labels : torch.Tensor
            Labels of shape (n_samples,).
        num_samples : int
            Number of samples to select.
        model : nn.Module
            Current model for computing reducible loss.

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

        # 保护原始模型设备
        original_device = next(model.parameters()).device
        model = model.to(self.device)

        num_classes = labels.max().item() + 1

        # Get or create selector
        selector = self._get_selector(
            num_classes=num_classes,
            dataset='MNIST' if data.shape[1] == 1 else 'CIFAR10'
        )

        # Train reference model (first time only)
        if selector.reference_model is None:
            print("Training CSReL reference model...")
            selector.train_reference_model(
                train_data=data,
                train_labels=labels,
                verbose=False
            )

        # Select samples using CSReL
        # CSReL needs a different model for reducible loss computation
        # Use a randomly initialized model as "current" model
        # Get num_classes from the reference model's last layer
        if hasattr(selector.reference_model, 'fc'):
            ref_num_classes = selector.reference_model.fc.out_features
        elif hasattr(selector.reference_model, 'fc2'):
            ref_num_classes = selector.reference_model.fc2.out_features
        else:
            ref_num_classes = num_classes

        # Use factory functions instead of type() reflection to avoid
        # constructor signature mismatch (e.g., ResNet needs block, layers)
        model_cls = type(selector.reference_model)
        from ..models.resnet import ResNet
        if isinstance(selector.reference_model, ResNet):
            current_model = ResNet18(num_classes=ref_num_classes).to(self.device)
        else:
            current_model = model_cls(num_classes=ref_num_classes).to(self.device)

        selected_indices = selector.select(
            train_data=data,
            train_labels=labels,
            model=current_model,
            incremental=False,
            verbose=False
        )

        # 恢复模型到原始设备
        model = model.to(original_device)

        # Extract selected samples
        selected_data = data[selected_indices]
        selected_labels = labels[selected_indices]

        return selected_data, selected_labels


class BilevelContinualAdapter:
    """
    Adapter for Bilevel Coreset method in continual learning scenarios.

    Simplified version that uses the kernel-based approach without
    explicit bilevel optimization (which requires validation set split).
    """

    def __init__(
        self,
        val_ratio: float = 0.2,
        max_outer_it: int = 5,  # Reduced for speed
        device: str = 'cuda'
    ):
        """
        Initialize Bilevel adapter.

        Parameters
        ----------
        val_ratio : float
            Validation set ratio for bilevel optimization.
        max_outer_it : int
            Maximum outer optimization iterations.
        device : str
            Computation device ('cuda' or 'cpu').
        """
        self.device = device
        self.val_ratio = val_ratio
        self.max_outer_it = max_outer_it

    def _rbf_kernel(self, X1: torch.Tensor, X2: torch.Tensor, gamma: float = 1.0):
        """Compute RBF kernel matrix."""
        X1_flat = X1.view(X1.size(0), -1)
        X2_flat = X2.view(X2.size(0), -1)

        # Compute squared Euclidean distances
        X1_norm = (X1_flat ** 2).sum(dim=1)
        X2_norm = (X2_flat ** 2).sum(dim=1)
        dist_sq = X1_norm.unsqueeze(1) + X2_norm.unsqueeze(0) - 2 * X1_flat @ X2_flat.T
        dist_sq = torch.clamp(dist_sq, min=0.0)

        # RBF kernel
        K = torch.exp(-gamma * dist_sq)
        return K

    def select(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        num_samples: int,
        model: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select coreset samples using simplified Bilevel method with class balancing.

        Uses kernel herding as a proxy for bilevel optimization.
        Ensures class-balanced selection by sampling proportionally from each class.

        Parameters
        ----------
        data : torch.Tensor
            Input data of shape (n_samples, C, H, W).
        labels : torch.Tensor
            Labels of shape (n_samples,).
        num_samples : int
            Number of samples to select.
        model : nn.Module
            Model (not used in simplified version, kept for interface compatibility).

        Returns
        -------
        selected_data : torch.Tensor
            Selected samples of shape (num_samples, ...).
        selected_labels : torch.Tensor
            Selected labels of shape (num_samples,).
        """
        # Ensure data is on correct device
        data = data.to(self.device)
        labels = labels.to(self.device)

        n_samples = data.shape[0]
        num_samples = min(num_samples, n_samples)

        # Get unique labels and their counts
        unique_labels = torch.unique(labels)
        num_classes = len(unique_labels)

        # Class-balanced selection
        selected_indices = []
        samples_per_class = num_samples // num_classes
        remaining_samples = num_samples % num_classes

        # Select samples from each class using kernel herding
        for class_idx, label in enumerate(unique_labels):
            # Get indices of samples in this class
            class_mask = (labels == label)
            class_indices = torch.where(class_mask)[0]

            # Determine how many to select from this class
            n_select = samples_per_class
            if class_idx < remaining_samples:
                n_select += 1

            n_select = min(n_select, len(class_indices))

            if n_select == 0:
                continue

            # Get data for this class
            class_data = data[class_indices]

            # Kernel herding within this class
            class_selected = self._kernel_herding(class_data, n_select)

            # Map back to original indices
            original_indices = class_indices[class_selected]
            selected_indices.append(original_indices)

        # Concatenate all selected indices
        if len(selected_indices) > 0:
            selected_indices = torch.cat(selected_indices)
        else:
            # Fallback to random selection
            selected_indices = torch.randperm(n_samples)[:num_samples]

        selected_data = data[selected_indices]
        selected_labels = labels[selected_indices]

        return selected_data, selected_labels

    def _kernel_herding(
        self,
        data: torch.Tensor,
        num_samples: int
    ) -> torch.Tensor:
        """
        Kernel herding for a single class.

        Parameters
        ----------
        data : torch.Tensor
            Data samples of shape (n_samples, C, H, W).
        num_samples : int
            Number of samples to select.

        Returns
        -------
        torch.Tensor
            Indices of selected samples within the input data.
        """
        n = data.shape[0]
        num_samples = min(num_samples, n)

        # Flatten data for kernel computation
        data_flat = data.view(n, -1)

        # Compute kernel matrix for this class
        K = self._rbf_kernel(data, data)

        # Mean in kernel space
        kernel_mean = K.mean(dim=0)

        # Kernel herding
        selected = []
        remaining = list(range(n))
        current_sum = torch.zeros(n, device=self.device)

        for _ in range(num_samples):
            best_idx = None
            best_dist = float('inf')

            for idx in remaining:
                new_sum = current_sum + K[idx]
                new_mean = new_sum / (len(selected) + 1)
                dist = torch.norm(kernel_mean - new_mean)

                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
                current_sum += K[best_idx]

        return torch.tensor(selected, dtype=torch.long)
