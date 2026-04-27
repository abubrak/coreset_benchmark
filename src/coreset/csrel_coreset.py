"""
CSReL (Classwise Spatial Representation Learning) for coreset selection.

This module implements the CSReL method which selects coreset samples based on
reducible loss computed from a reference model. The method supports class-
balanced selection and incremental selection strategies.

Reference:
    [Paper Title and citation to be added]
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple, Dict, Any
from copy import deepcopy

from .selection_functions import select_by_loss_diff
from ..configs import CSReLConfig


class CSReLCoreset:
    """
    Classwise Spatial Representation Learning for coreset selection.

    This method selects samples by:
    1. Training a reference model on the full dataset
    2. Computing reference losses for all samples
    3. Selecting samples with highest reducible loss (difference between
       current and reference loss)

    The method supports class-balanced selection to ensure fair representation
    of all classes in the coreset.

    Attributes
    ----------
    config : CSReLConfig
        Configuration object with experiment parameters.
    device : torch.device
        Device to use for computation (CPU or GPU).
    model : nn.Module, optional
        Current model being trained.
    reference_model : nn.Module, optional
        Reference model trained on full dataset.
    reference_losses : torch.Tensor, optional
        Reference losses computed on full dataset.
    selected_indices : torch.Tensor
        Indices of selected coreset samples.
    """

    def __init__(
        self,
        config: CSReLConfig,
        model: Optional[nn.Module] = None
    ):
        """
        Initialize CSReL coreset selector.

        Parameters
        ----------
        config : CSReLConfig
            Configuration object with experiment parameters.
        model : nn.Module, optional
            Initial model to use as reference, by default None.
            If None, a new model will be created based on config.
        """
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # Initialize model
        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = None  # Will be created based on dataset

        # Reference model and losses
        self.reference_model = None
        self.reference_losses = None

        # Selected indices
        self.selected_indices = None

    def _create_model(self) -> nn.Module:
        """
        Create a model based on the dataset configuration.

        Returns
        -------
        nn.Module
            Neural network model appropriate for the dataset.
        """
        from ..models import get_model

        model = get_model(
            dataset=self.config.dataset,
            num_classes=self.config.num_classes
        )
        return model.to(self.device)

    def train_reference_model(
        self,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        val_data: Optional[torch.Tensor] = None,
        val_labels: Optional[torch.Tensor] = None,
        verbose: bool = True
    ) -> Tuple[nn.Module, torch.Tensor]:
        """
        Train reference model and compute reference losses.

        This method trains a model on the full training set (or a subset)
        and computes the reference losses that will be used for coreset
        selection.

        Parameters
        ----------
        train_data : torch.Tensor
            Training data of shape (n_samples, ...).
        train_labels : torch.Tensor
            Training labels of shape (n_samples,).
        val_data : torch.Tensor, optional
            Validation data for early stopping, by default None.
        val_labels : torch.Tensor, optional
            Validation labels, by default None.
        verbose : bool, optional
            Whether to print training progress, by default True.

        Returns
        -------
        reference_model : nn.Module
            Trained reference model.
        reference_losses : torch.Tensor
            Loss values for each training sample.
        """
        # Create model if not exists
        if self.reference_model is None:
            self.reference_model = self._create_model()

        # Setup optimizer
        optimizer = optim.Adam(
            self.reference_model.parameters(),
            lr=self.config.learning_rate
        )

        # Move data to device
        train_data = train_data.to(self.device)
        train_labels = train_labels.to(self.device)

        n_samples = len(train_data)
        num_epochs = self.config.num_epochs
        batch_size = self.config.batch_size

        # Training loop
        self.reference_model.train()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            n_batches = 0

            # Shuffle data each epoch
            perm = torch.randperm(n_samples)

            for i in range(0, n_samples, batch_size):
                # Get batch
                indices = perm[i:i+batch_size]
                batch_data = train_data[indices]
                batch_labels = train_labels[indices]

                # Forward pass
                optimizer.zero_grad()
                outputs = self.reference_model(batch_data)
                loss = nn.functional.cross_entropy(outputs, batch_labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)

            if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

            # Validation
            if val_data is not None and val_labels is not None:
                with torch.no_grad():
                    self.reference_model.eval()
                    val_outputs = self.reference_model(val_data.to(self.device))
                    val_loss = nn.functional.cross_entropy(
                        val_outputs,
                        val_labels.to(self.device)
                    )
                    if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
                        print(f"  Val Loss: {val_loss.item():.4f}")
                    self.reference_model.train()

        # Compute reference losses
        self.reference_model.eval()
        reference_losses = []

        with torch.no_grad():
            # Compute in batches for memory efficiency
            for i in range(0, n_samples, batch_size):
                batch_data = train_data[i:i+batch_size]
                batch_labels = train_labels[i:i+batch_size]

                outputs = self.reference_model(batch_data)
                batch_losses = nn.functional.cross_entropy(
                    outputs,
                    batch_labels,
                    reduction='none'
                )
                reference_losses.append(batch_losses)

        self.reference_losses = torch.cat(reference_losses)

        return self.reference_model, self.reference_losses

    def select(
        self,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        model: Optional[nn.Module] = None,
        incremental: bool = False,
        current_indices: Optional[torch.Tensor] = None,
        verbose: bool = True
    ) -> torch.Tensor:
        """
        Select coreset using CSReL method.

        This method selects samples based on reducible loss, which is the
        difference between the current model's loss and the reference loss.
        Samples with higher reducible loss are more informative and are
        prioritized for selection.

        Parameters
        ----------
        train_data : torch.Tensor
            Training data of shape (n_samples, ...).
        train_labels : torch.Tensor
            Training labels of shape (n_samples,).
        model : nn.Module, optional
            Current model to compute current losses, by default None.
            If None, uses the reference model for both.
        incremental : bool, optional
            Whether to perform incremental selection, by default False.
            If True, adds new samples to existing selection.
        current_indices : torch.Tensor, optional
            Currently selected indices (required if incremental=True).
        verbose : bool, optional
            Whether to print selection progress, by default True.

        Returns
        -------
        selected_indices : torch.Tensor
            Indices of selected coreset samples.

        Raises
        ------
        ValueError
            If reference model has not been trained.
            If incremental=True but current_indices is None.
        """
        if self.reference_model is None or self.reference_losses is None:
            raise ValueError(
                "Reference model must be trained first. "
                "Call train_reference_model() before select()."
            )

        # Move data to device
        train_data = train_data.to(self.device)
        train_labels = train_labels.to(self.device)

        # Determine which model to use for current losses
        if model is not None:
            current_model = model.to(self.device)
        else:
            current_model = self.reference_model

        # Compute current losses
        current_model.eval()
        n_samples = len(train_data)
        batch_size = self.config.batch_size
        current_losses = []

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch_data = train_data[i:i+batch_size]
                batch_labels = train_labels[i:i+batch_size]

                outputs = current_model(batch_data)
                batch_losses = nn.functional.cross_entropy(
                    outputs,
                    batch_labels,
                    reduction='none'
                )
                current_losses.append(batch_losses)

        current_losses = torch.cat(current_losses)

        # Compute selection size
        if incremental:
            if current_indices is None:
                raise ValueError(
                    "current_indices must be provided for incremental selection"
                )

            # For incremental: select new samples to add
            n_current = len(current_indices)
            n_total_to_select = int(n_samples * self.config.selection_ratio)
            n_new_to_select = n_total_to_select - n_current

            if n_new_to_select <= 0:
                if verbose:
                    print("Already have enough samples, returning current selection")
                return current_indices

            if verbose:
                print(f"Incremental selection: adding {n_new_to_select} new samples")

            # Exclude already selected samples
            mask = torch.ones(n_samples, dtype=torch.bool)
            mask[current_indices] = False
            available_indices = torch.where(mask)[0]

            # Select from available samples only
            available_losses = current_losses[available_indices]
            available_ref_losses = self.reference_losses[available_indices]
            available_labels = train_labels[available_indices]

            new_selected = select_by_loss_diff(
                losses=available_losses,
                reference_losses=available_ref_losses,
                num_samples=n_new_to_select,
                class_balance=self.config.class_balance,
                labels=available_labels,
                num_classes=self.config.num_classes
            )

            # Map back to original indices
            new_selected = available_indices[new_selected]

            # Combine with current selection
            selected_indices = torch.cat([current_indices, new_selected])

        else:
            # Standard selection: select all samples at once
            n_to_select = int(n_samples * self.config.selection_ratio)

            if verbose:
                print(f"Selecting {n_to_select} samples from {n_samples} total samples")

            selected_indices = select_by_loss_diff(
                losses=current_losses,
                reference_losses=self.reference_losses,
                num_samples=n_to_select,
                class_balance=self.config.class_balance,
                labels=train_labels,
                num_classes=self.config.num_classes
            )

        self.selected_indices = selected_indices

        if verbose:
            # Print class distribution
            unique, counts = torch.unique(
                train_labels[selected_indices],
                return_counts=True
            )
            print("\nSelected samples distribution:")
            for cls, count in zip(unique.tolist(), counts.tolist()):
                print(f"  Class {cls}: {count} samples")

        return selected_indices

    def update_reference(
        self,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        new_model: nn.Module,
        verbose: bool = True
    ):
        """
        Update reference model with a new model.

        This is useful for continual learning scenarios where the reference
        model should be updated periodically.

        Parameters
        ----------
        train_data : torch.Tensor
            Training data of shape (n_samples, ...).
        train_labels : torch.Tensor
            Training labels of shape (n_samples,).
        new_model : nn.Module
            New model to use as reference.
        verbose : bool, optional
            Whether to print update progress, by default True.
        """
        if verbose:
            print("Updating reference model...")

        # Clone the new model
        self.reference_model = deepcopy(new_model)
        self.reference_model.eval()

        # Recompute reference losses
        train_data = train_data.to(self.device)
        train_labels = train_labels.to(self.device)

        n_samples = len(train_data)
        batch_size = self.config.batch_size
        reference_losses = []

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch_data = train_data[i:i+batch_size]
                batch_labels = train_labels[i:i+batch_size]

                outputs = self.reference_model(batch_data)
                batch_losses = nn.functional.cross_entropy(
                    outputs,
                    batch_labels,
                    reduction='none'
                )
                reference_losses.append(batch_losses)

        self.reference_losses = torch.cat(reference_losses)

        if verbose:
            print("Reference model updated")

    def get_selection_stats(
        self,
        train_labels: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Get statistics about the current selection.

        Parameters
        ----------
        train_labels : torch.Tensor
            All training labels.

        Returns
        -------
        stats : Dict[str, Any]
            Dictionary containing selection statistics including:
            - n_selected: Number of selected samples
            - selection_ratio: Actual selection ratio
            - class_distribution: Per-class sample counts
            - class_ratios: Per-class selection ratios
        """
        if self.selected_indices is None:
            return {
                "n_selected": 0,
                "selection_ratio": 0.0,
                "class_distribution": {},
                "class_ratios": {}
            }

        selected_labels = train_labels[self.selected_indices]
        unique, counts = torch.unique(selected_labels, return_counts=True)

        n_total = len(train_labels)
        n_selected = len(self.selected_indices)

        # Compute class distribution
        class_dist = {}
        class_ratios = {}

        for cls in range(self.config.num_classes):
            cls_mask = (train_labels == cls)
            n_cls_total = cls_mask.sum().item()

            if cls in unique:
                cls_idx = (unique == cls).nonzero(as_tuple=True)[0].item()
                n_cls_selected = counts[cls_idx].item()
            else:
                n_cls_selected = 0

            class_dist[int(cls)] = n_cls_selected

            if n_cls_total > 0:
                class_ratios[int(cls)] = n_cls_selected / n_cls_total
            else:
                class_ratios[int(cls)] = 0.0

        return {
            "n_selected": n_selected,
            "selection_ratio": n_selected / n_total,
            "class_distribution": class_dist,
            "class_ratios": class_ratios
        }

    def save(self, filepath: str):
        """
        Save the selector state to disk.

        Parameters
        ----------
        filepath : str
            Path to save the state.
        """
        state = {
            'selected_indices': self.selected_indices,
            'reference_losses': self.reference_losses
        }

        # Save reference model state dict
        if self.reference_model is not None:
            state['reference_model_state'] = self.reference_model.state_dict()

        # Save config separately as dict to avoid pickle issues
        state['config_dict'] = {
            'dataset': self.config.dataset,
            'num_classes': self.config.num_classes,
            'input_dim': self.config.input_dim,
            'hidden_dims': self.config.hidden_dims,
            'batch_size': self.config.batch_size,
            'learning_rate': self.config.learning_rate,
            'num_epochs': self.config.num_epochs,
            'device': self.config.device,
            'random_seed': self.config.random_seed,
            'num_components': self.config.num_components,
            'num_neighbors': self.config.num_neighbors,
            'class_balance': self.config.class_balance,
            'selection_ratio': self.config.selection_ratio,
            'similarity_metric': self.config.similarity_metric
        }

        torch.save(state, filepath)

    def load(self, filepath: str, model: nn.Module):
        """
        Load the selector state from disk.

        Parameters
        ----------
        filepath : str
            Path to load the state from.
        model : nn.Module
            Model architecture to load reference model into.
        """
        state = torch.load(filepath, map_location=self.device, weights_only=False)

        # Reconstruct config from dict
        from ..configs import CSReLConfig
        config_dict = state.get('config_dict', {})
        self.config = CSReLConfig(**config_dict)

        self.selected_indices = state['selected_indices']
        self.reference_losses = state['reference_losses']

        if 'reference_model_state' in state:
            self.reference_model = model.to(self.device)
            self.reference_model.load_state_dict(state['reference_model_state'])
            self.reference_model.eval()
