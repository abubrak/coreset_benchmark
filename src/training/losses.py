# src/training/losses.py
import torch
import torch.nn.functional as F


def cross_entropy_loss(K: torch.Tensor, alpha: torch.Tensor, y: torch.Tensor,
                       weights: torch.Tensor, lmbda: float = 0.0) -> torch.Tensor:
    """
    Weighted cross-entropy loss with kernel representation.

    Args:
        K: Kernel matrix of shape (n, m)
        alpha: Coefficients of shape (m, num_classes)
        y: Labels of shape (n,)
        weights: Sample weights of shape (n,)
        lmbda: L2 regularization coefficient

    Returns:
        Scalar loss
    """
    logits = K @ alpha  # (n, num_classes)
    loss = F.cross_entropy(logits, y.long(), reduction='none')
    weighted_loss = (loss * weights).mean()

    if lmbda > 0:
        # L2 regularization on alpha
        reg = lmbda * torch.trace(alpha.T @ K @ alpha)
        weighted_loss += reg

    return weighted_loss


def accuracy(K: torch.Tensor, alpha: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute accuracy given kernel representation"""
    logits = K @ alpha
    preds = logits.argmax(dim=1)
    acc = (preds == y.long()).float().mean()
    return acc
