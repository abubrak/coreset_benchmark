"""
Baseline methods for coreset selection.

This module implements various baseline selection strategies including:
- Uniform random sampling
- K-center clustering
- K-means clustering
- Herding (kernel herding)
- Entropy-based selection
- Loss-based selection
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import pairwise_distances


class BaselineSelector(ABC):
    """Abstract base class for baseline coreset selection methods."""

    @abstractmethod
    def select(self, X: np.ndarray, y: Optional[np.ndarray] = None,
               size: int = 100, **kwargs) -> np.ndarray:
        """
        Select a coreset from the data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray, optional
            Labels of shape (n_samples,). Some methods may use labels.
        size : int
            Number of samples to select.
        **kwargs : dict
            Additional method-specific parameters.

        Returns
        -------
        np.ndarray
            Indices of selected samples.
        """
        pass


class UniformSelector(BaselineSelector):
    """Uniform random sampling selector."""

    def select(self, X: np.ndarray, y: Optional[np.ndarray] = None,
               size: int = 100, **kwargs) -> np.ndarray:
        """
        Select samples uniformly at random.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray, optional
            Labels (not used).
        size : int
            Number of samples to select.
        **kwargs : dict
            Additional parameters (not used).

        Returns
        -------
        np.ndarray
            Indices of selected samples.
        """
        n_samples = X.shape[0]
        if size > n_samples:
            raise ValueError(f"Cannot select {size} samples from {n_samples} samples")

        indices = np.random.choice(n_samples, size=size, replace=False)
        return indices


class KCenterSelector(BaselineSelector):
    """K-center clustering selector using greedy farthest-point algorithm."""

    def select(self, X: np.ndarray, y: Optional[np.ndarray] = None,
               size: int = 100, **kwargs) -> np.ndarray:
        """
        Select samples using k-center clustering.

        Uses greedy farthest-point traversal to find centers that maximize
        minimum distance to existing centers.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray, optional
            Labels (not used).
        size : int
            Number of samples to select.
        **kwargs : dict
            Additional parameters:
            - metric : str, distance metric (default: 'euclidean')
            - random_state : int, random seed (default: None)

        Returns
        -------
        np.ndarray
            Indices of selected samples.
        """
        n_samples = X.shape[0]
        if size > n_samples:
            raise ValueError(f"Cannot select {size} samples from {n_samples} samples")

        metric = kwargs.get('metric', 'euclidean')
        random_state = kwargs.get('random_state', None)

        if random_state is not None:
            np.random.seed(random_state)

        # Initialize with random point
        indices = [np.random.randint(0, n_samples)]

        # Greedy farthest-point traversal
        for _ in range(1, size):
            # Compute distances to nearest selected point
            selected_points = X[indices]
            dists = pairwise_distances(X, selected_points, metric=metric)
            min_dists = np.min(dists, axis=1)

            # Select point farthest from current centers
            new_idx = np.argmax(min_dists)
            indices.append(new_idx)

        return np.array(indices)


class KMeansSelector(BaselineSelector):
    """K-means clustering selector."""

    def select(self, X: np.ndarray, y: Optional[np.ndarray] = None,
               size: int = 100, **kwargs) -> np.ndarray:
        """
        Select samples using k-means clustering.

        Selects samples closest to cluster centers.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray, optional
            Labels (not used).
        size : int
            Number of samples to select.
        **kwargs : dict
            Additional parameters:
            - random_state : int, random seed (default: 42)
            - max_samples : int, max samples for MiniBatchKMeans (default: 10000)

        Returns
        -------
        np.ndarray
            Indices of selected samples.
        """
        n_samples = X.shape[0]
        if size > n_samples:
            raise ValueError(f"Cannot select {size} samples from {n_samples} samples")

        random_state = kwargs.get('random_state', 42)
        max_samples = kwargs.get('max_samples', 10000)

        # Use MiniBatchKMeans for large datasets
        if X.shape[0] > max_samples:
            kmeans = MiniBatchKMeans(
                n_clusters=size,
                random_state=random_state,
                n_init=10
            )
        else:
            kmeans = KMeans(
                n_clusters=size,
                random_state=random_state,
                n_init=10
            )

        kmeans.fit(X)
        centers = kmeans.cluster_centers_

        # Find closest samples to centers
        indices = []
        for center in centers:
            dists = np.linalg.norm(X - center, axis=1)
            # Avoid selecting same sample twice
            for idx in indices:
                dists[idx] = np.inf
            closest_idx = np.argmin(dists)
            indices.append(closest_idx)

        return np.array(indices)


class HerdingSelector(BaselineSelector):
    """Kernel herding selector with memory-efficient batched computation."""

    def select(self, X: np.ndarray, y: Optional[np.ndarray] = None,
               size: int = 100, **kwargs) -> np.ndarray:
        """
        Select samples using kernel herding.

        Uses batched distance computation to avoid O(n^2) memory allocation.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray, optional
            Labels (not used).
        size : int
            Number of samples to select.
        **kwargs : dict
            Additional parameters:
            - kernel : str, kernel type ('rbf' or 'linear', default: 'rbf')
            - gamma : float, RBF kernel parameter (default: 1.0)
            - random_state : int, random seed (default: None)

        Returns
        -------
        np.ndarray
            Indices of selected samples.
        """
        n_samples = X.shape[0]
        if size > n_samples:
            raise ValueError(f"Cannot select {size} samples from {n_samples} samples")

        kernel = kwargs.get('kernel', 'rbf')
        gamma = kwargs.get('gamma', 1.0)
        random_state = kwargs.get('random_state', None)

        if random_state is not None:
            np.random.seed(random_state)

        def _rbf_kernel_batch(X_a, X_b, gamma):
            """Compute RBF kernel between two sets in batches."""
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a^T*b
            X_a_sq = np.sum(X_a ** 2, axis=1, keepdims=True)
            X_b_sq = np.sum(X_b ** 2, axis=1, keepdims=True)
            dist_sq = X_a_sq + X_b_sq.T - 2.0 * (X_a @ X_b.T)
            dist_sq = np.maximum(dist_sq, 0.0)
            return np.exp(-gamma * dist_sq)

        # Compute mean kernel vector: mu_i = (1/n) * sum_j K(x_i, x_j)
        if kernel == 'rbf':
            X_sq = np.sum(X ** 2, axis=1)
            mean_K = np.zeros(n_samples)
            batch_size = 1024
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                # ||X[start:end] - X||^2 = X_sq[start:end, None] + X_sq[None, :] - 2*X_batch @ X.T
                dist_sq = X_sq[start:end, None] + X_sq[None, :] - 2.0 * (X[start:end] @ X.T)
                dist_sq = np.maximum(dist_sq, 0.0)
                K_batch = np.exp(-gamma * dist_sq)
                mean_K[start:end] = K_batch.mean(axis=1)
        elif kernel == 'linear':
            KX = X @ X.T  # (n, n) - can be large but float64 ~ 28GB for 60k
            # For large datasets, compute mean per-row incrementally
            mean_K = np.zeros(n_samples)
            batch_size = 1024
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                K_batch = X[start:end] @ X.T
                mean_K[start:end] = K_batch.mean(axis=1)
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

        # Herding algorithm - compute kernel columns on demand
        indices = []
        selected = np.zeros(n_samples, dtype=bool)
        cumulative_K = np.zeros(n_samples)

        for t in range(size):
            if t == 0:
                scores = mean_K
            else:
                # empirical_mean = mean of K[:, indices] columns
                scores = mean_K - cumulative_K / t

            scores[selected] = -np.inf
            idx = np.argmax(scores)
            indices.append(idx)
            selected[idx] = True

            # Accumulate kernel column for the selected sample
            if kernel == 'rbf':
                diff = X - X[idx]
                col = np.exp(-gamma * np.sum(diff ** 2, axis=1))
            else:
                col = X @ X[idx]
            cumulative_K += col

        return np.array(indices)


class EntropySelector(BaselineSelector):
    """Entropy-based selector using prediction uncertainty."""

    def select(self, X: np.ndarray, y: Optional[np.ndarray] = None,
               size: int = 100, **kwargs) -> np.ndarray:
        """
        Select samples with highest prediction entropy.

        Requires a model with predict_proba method or pre-computed probabilities.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray, optional
            Labels (not used directly).
        size : int
            Number of samples to select.
        **kwargs : dict
            Additional parameters:
            - model : object, model with predict_proba method (optional)
            - probas : np.ndarray, pre-computed probabilities (optional)

        Returns
        -------
        np.ndarray
            Indices of selected samples.
        """
        n_samples = X.shape[0]
        if size > n_samples:
            raise ValueError(f"Cannot select {size} samples from {n_samples} samples")

        model = kwargs.get('model', None)
        probas = kwargs.get('probas', None)

        # Get probabilities
        if probas is not None:
            if probas.shape[0] != n_samples:
                raise ValueError("probabilities must have same number of samples as X")
        elif model is not None:
            if not hasattr(model, 'predict_proba'):
                raise ValueError("model must have predict_proba method")
            probas = model.predict_proba(X)
        else:
            raise ValueError("Must provide either 'model' or 'probas'")

        # Compute entropy
        eps = 1e-10
        entropy = -np.sum(probas * np.log(probas + eps), axis=1)

        # Select samples with highest entropy
        indices = np.argsort(entropy)[-size:][::-1]

        return indices


class LossSelector(BaselineSelector):
    """Loss-based selector using training loss."""

    def select(self, X: np.ndarray, y: Optional[np.ndarray] = None,
               size: int = 100, **kwargs) -> np.ndarray:
        """
        Select samples with highest training loss.

        Requires a model with predict method or pre-computed losses.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Labels (required).
        size : int
            Number of samples to select.
        **kwargs : dict
            Additional parameters:
            - model : object, model with predict method (optional)
            - losses : np.ndarray, pre-computed losses (optional)
            - loss_fn : callable, loss function (default: None, uses 0-1 loss)

        Returns
        -------
        np.ndarray
            Indices of selected samples.
        """
        n_samples = X.shape[0]
        if size > n_samples:
            raise ValueError(f"Cannot select {size} samples from {n_samples} samples")

        if y is None:
            raise ValueError("y must be provided for loss-based selection")

        model = kwargs.get('model', None)
        losses = kwargs.get('losses', None)
        loss_fn = kwargs.get('loss_fn', None)

        # Get losses
        if losses is not None:
            if losses.shape[0] != n_samples:
                raise ValueError("losses must have same number of samples as X")
        elif model is not None:
            if not hasattr(model, 'predict'):
                raise ValueError("model must have predict method")

            preds = model.predict(X)

            if loss_fn is not None:
                losses = loss_fn(y, preds)
            else:
                # Default: use 0-1 loss (misclassification)
                losses = (preds != y).astype(float)
        else:
            raise ValueError("Must provide either 'model' or 'losses'")

        # Select samples with highest loss
        indices = np.argsort(losses)[-size:][::-1]

        return indices


def get_baseline(method: str) -> BaselineSelector:
    """
    Factory function to get baseline selector.

    Parameters
    ----------
    method : str
        Name of the baseline method. Options:
        - 'uniform' : Uniform random sampling
        - 'kcenter' : K-center clustering
        - 'kmeans' : K-means clustering
        - 'herding' : Kernel herding
        - 'entropy' : Entropy-based selection
        - 'loss' : Loss-based selection

    Returns
    -------
    BaselineSelector
        Instance of the specified selector.

    Raises
    ------
    ValueError
        If method name is unknown.
    """
    method = method.lower()

    if method == 'uniform':
        return UniformSelector()
    elif method == 'kcenter':
        return KCenterSelector()
    elif method == 'kmeans':
        return KMeansSelector()
    elif method == 'herding':
        return HerdingSelector()
    elif method == 'entropy':
        return EntropySelector()
    elif method == 'loss':
        return LossSelector()
    else:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Available methods: 'uniform', 'kcenter', 'kmeans', "
            f"'herding', 'entropy', 'loss'"
        )
