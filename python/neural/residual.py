"""Optional residual network for the hybrid neural codec path."""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:

    class ResidualNet(nn.Module):
        """
        Tiny residual predictor.
        Input: mel or waveform frame → residual waveform frame.
        In practice you would train this against the difference between
        original audio and the pure DSP resynthesis.
        """

        def __init__(self, in_dim: int = 64, hidden: int = 128, out_len: int = 512):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, out_len),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

else:

    class ResidualNet:
        def __init__(self, in_dim: int = 64, hidden: int = 128, out_len: int = 512):
            self.out_len = out_len
            rng = np.random.default_rng(7)
            self.W = rng.normal(0, 0.05, (in_dim, out_len))

        def forward(self, x: np.ndarray) -> np.ndarray:
            return (x @ self.W).astype(np.float32)
