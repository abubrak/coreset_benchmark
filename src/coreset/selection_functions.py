"""
Selection functions for coreset selection methods.

This module provides utility functions for sample selection based on
various criteria including reducible loss.
"""

import torch
import numpy as np
from typing import Tuple, Optional


def select_by_loss_diff(
    losses: torch.Tensor,
    reference_losses: torch.Tensor,
    num_samples: int,
    class_balance: bool = True,
    labels: Optional[torch.Tensor] = None,
    num_classes: int = 10
) -> torch.Tensor:
    """
    Select samples based on reducible loss (loss difference).

    The reducible loss is defined as the difference between current loss
    and reference loss. Samples with higher reducible loss are more
    informative and should be prioritized.

    Parameters
    ----------
    losses : torch.Tensor
        Current losses of shape (n_samples,).
    reference_losses : torch.Tensor
        Reference losses of shape (n_samples,).
    num_samples : int
        Number of samples to select.
    class_balance : bool, optional
        Whether to maintain class balance in selection, by default True.
    labels : torch.Tensor, optional
        Class labels of shape (n_samples,), required if class_balance=True.
    num_classes : int, optional
        Number of classes, by default 10.

    Returns
    -------
    torch.Tensor
        Indices of selected samples.

    Raises
    ------
    ValueError
        If class_balance=True but labels are not provided.
        If num_samples is larger than the total number of samples.
    """
    n_total = len(losses)

    if num_samples > n_total:
        raise ValueError(
            f"Cannot select {num_samples} samples from {n_total} samples"
        )

    # Compute reducible loss
    reducible_loss = losses - reference_losses

    # Ensure non-negative (numerical stability)
    reducible_loss = torch.clamp(reducible_loss, min=0.0)

    if class_balance:
        if labels is None:
            raise ValueError("labels must be provided when class_balance=True")

        # Select samples per class
        selected_indices = []

        # Compute samples per class
        samples_per_class = num_samples // num_classes
        remainder = num_samples % num_classes

        for c in range(num_classes):
            # Get indices of class c
            class_mask = (labels == c)
            class_indices = torch.where(class_mask)[0]

            if len(class_indices) == 0:
                continue

            # Number of samples to select from this class
            n_select = samples_per_class
            if c < remainder:
                n_select += 1

            # Limit to available samples
            n_select = min(n_select, len(class_indices))

            # Get reducible losses for this class
            class_losses = reducible_loss[class_indices]

            # Select top samples
            _, top_idx = torch.topk(class_losses, n_select)
            selected = class_indices[top_idx]
            selected_indices.append(selected)

        # Concatenate all selections
        selected_indices = torch.cat(selected_indices)

    else:
        # Select samples with highest reducible loss
        _, top_idx = torch.topk(reducible_loss, num_samples)
        selected_indices = top_idx

    return selected_indices


def select_by_margin(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_samples: int,
    class_balance: bool = True,
    num_classes: int = 10
) -> torch.Tensor:
    """
    Select samples based on classification margin.

    Margin is defined as the difference between the logit of the correct
    class and the maximum logit among incorrect classes. Smaller margins
    indicate more uncertain/samples near decision boundary.

    Parameters
    ----------
    logits : torch.Tensor
        Model predictions of shape (n_samples, num_classes).
    labels : torch.Tensor
        Class labels of shape (n_samples,).
    num_samples : int
        Number of samples to select.
    class_balance : bool, optional
        Whether to maintain class balance in selection, by default True.
    num_classes : int, optional
        Number of classes, by default 10.

    Returns
    -------
    torch.Tensor
        Indices of selected samples.
    """
    n_total = len(logits)

    if num_samples > n_total:
        raise ValueError(
            f"Cannot select {num_samples} samples from {n_total} samples"
        )

    # Compute margin for each sample
    # Get correct class logits
    correct_logits = logits[torch.arange(n_total), labels]

    # Get maximum incorrect class logits
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[torch.arange(n_total), labels] = False
    incorrect_logits = logits.masked_select(mask).view(n_total, -1)
    max_incorrect = incorrect_logits.max(dim=1)[0]

    # Margin: higher values = more confident
    margins = correct_logits - max_incorrect

    # For selection, we want low margin (high uncertainty)
    # So we negate to use with topk
    scores = -margins

    if class_balance:
        selected_indices = []
        samples_per_class = num_samples // num_classes
        remainder = num_samples % num_classes

        for c in range(num_classes):
            class_mask = (labels == c)
            class_indices = torch.where(class_mask)[0]

            if len(class_indices) == 0:
                continue

            n_select = samples_per_class
            if c < remainder:
                n_select += 1
            n_select = min(n_select, len(class_indices))

            class_scores = scores[class_indices]
            _, top_idx = torch.topk(class_scores, n_select)
            selected = class_indices[top_idx]
            selected_indices.append(selected)

        selected_indices = torch.cat(selected_indices)
    else:
        _, top_idx = torch.topk(scores, num_samples)
        selected_indices = top_idx

    return selected_indices


def select_by_gradient_norm(
    model: torch.nn.Module,
    data: torch.Tensor,
    labels: torch.Tensor,
    num_samples: int,
    class_balance: bool = True,
    num_classes: int = 10
) -> torch.Tensor:
    """
    Select samples based on gradient norm.

    Computes the norm of the gradient of the loss with respect to the
    model parameters for each sample. Samples with larger gradient
    norms are more informative.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model.
    data : torch.Tensor
        Input data of shape (n_samples, ...).
    labels : torch.Tensor
        Class labels of shape (n_samples,).
    num_samples : int
        Number of samples to select.
    class_balance : bool, optional
        Whether to maintain class balance in selection, by default True.
    num_classes : int, optional
        Number of classes, by default 10.

    Returns
    -------
    torch.Tensor
        Indices of selected samples.
    """
    n_total = len(data)

    if num_samples > n_total:
        raise ValueError(
            f"Cannot select {num_samples} samples from {n_total} samples"
        )

    model.eval()
    gradient_norms = []

    # Compute gradient norm for each sample
    for i in range(n_total):
        model.zero_grad()

        x = data[i:i+1]
        y = labels[i:i+1]

        # Forward pass
        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, y)

        # Backward pass
        loss.backward()

        # Compute gradient norm
        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2) ** 2
        grad_norm = grad_norm ** 0.5

        gradient_norms.append(grad_norm)

    gradient_norms = torch.stack(gradient_norms)

    if class_balance:
        selected_indices = []
        samples_per_class = num_samples // num_classes
        remainder = num_samples % num_classes

        for c in range(num_classes):
            class_mask = (labels == c)
            class_indices = torch.where(class_mask)[0]

            if len(class_indices) == 0:
                continue

            n_select = samples_per_class
            if c < remainder:
                n_select += 1
            n_select = min(n_select, len(class_indices))

            class_norms = gradient_norms[class_indices]
            _, top_idx = torch.topk(class_norms, n_select)
            selected = class_indices[top_idx]
            selected_indices.append(selected)

        selected_indices = torch.cat(selected_indices)
    else:
        _, top_idx = torch.topk(gradient_norms, num_samples)
        selected_indices = top_idx

    return selected_indices
