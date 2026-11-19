"""Loss functions for training the neural parts of the Sogitech Codec."""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def spectral_loss(pred: np.ndarray, target: np.ndarray, n_fft: int = 2048) -> float:
    """Simple multi-resolution spectral magnitude loss (numpy version)."""
    def mag(x):
        return np.abs(np.fft.rfft(x, n=n_fft))

    loss = 0.0
    for factor in (1, 2, 4):
        n = n_fft // factor
        loss += np.mean(np.abs(mag(pred) - mag(target)))
    return float(loss)


def parameter_loss(pred: dict, target: dict, weights: dict | None = None) -> float:
    """L1 loss over the parameter vector."""
    if weights is None:
        weights = {k: 1.0 for k in pred}
    total = 0.0
    for k in pred:
        w = weights.get(k, 1.0)
        total += w * abs(pred[k] - target.get(k, pred[k]))
    return total


if HAS_TORCH:

    def spectral_loss_torch(pred: torch.Tensor, target: torch.Tensor, n_fft: int = 2048) -> torch.Tensor:
        def mag(x):
            return torch.abs(torch.fft.rfft(x, n=n_fft))

        loss = 0.0
        for factor in (1, 2, 4):
            n = n_fft // factor
            loss = loss + F.l1_loss(mag(pred), mag(target))
        return loss
