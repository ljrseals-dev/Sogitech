"""Neural parameter controller with bounded, serialisable outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

PARAM_NAMES = [
    "osc1Wave", "osc2Wave", "osc1Level", "osc2Level", "subLevel", "noiseLevel",
    "additiveBlend", "powerSagDepth", "filterLP", "filterHP", "filterRes",
    "keyTrack", "velMod", "glideTime", "midiNote", "velocity", "pressure",
    "residualGain",
]
PARAM_RANGES = {
    "osc1Wave": (0.0, 3.0), "osc2Wave": (0.0, 3.0),
    "osc1Level": (0.0, 1.0), "osc2Level": (0.0, 1.0),
    "subLevel": (0.0, 1.0), "noiseLevel": (0.0, 1.0),
    "additiveBlend": (0.0, 1.0), "powerSagDepth": (0.0, 1.0),
    "filterLP": (0.0, 1.0), "filterHP": (0.0, 1.0),
    "filterRes": (0.0, 1.0), "keyTrack": (0.0, 1.0),
    "velMod": (0.0, 1.0), "glideTime": (0.0, 0.5),
    "midiNote": (0.0, 127.0), "velocity": (0.0, 1.0),
    "pressure": (0.0, 1.0), "residualGain": (0.0, 1.0),
}
_LO = np.asarray([PARAM_RANGES[k][0] for k in PARAM_NAMES], dtype=np.float32)
_SPAN = np.asarray([PARAM_RANGES[k][1] - PARAM_RANGES[k][0] for k in PARAM_NAMES],
                   dtype=np.float32)

if HAS_TORCH:
    class ParameterController(nn.Module):
        """64-feature MLP producing 18 normalised parameter values."""
        def __init__(self, in_dim: int = 64, hidden: int = 128, out_dim: int = 18):
            super().__init__()
            if out_dim != len(PARAM_NAMES):
                raise ValueError(f"out_dim must be {len(PARAM_NAMES)}")
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, out_dim), nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

        def predict_dict(self, features: np.ndarray) -> Dict[str, float]:
            self.eval()
            with torch.no_grad():
                t = torch.as_tensor(np.asarray(features, dtype=np.float32)).reshape(1, -1)
                if t.shape[1] != 64:
                    raise ValueError("controller expects exactly 64 features")
                normalised = self(t).squeeze(0).cpu().numpy()
            values = _LO + normalised * _SPAN
            return {name: float(values[i]) for i, name in enumerate(PARAM_NAMES)}
else:
    class ParameterController:
        """Numpy inference fallback; cannot load or train PyTorch checkpoints."""
        def __init__(self, in_dim: int = 64, hidden: int = 128, out_dim: int = 18):
            self.in_dim, self.out_dim = in_dim, out_dim
            rng = np.random.default_rng(42)
            self.W1 = rng.normal(0, 0.05, (in_dim, hidden))
            self.b1 = np.zeros(hidden)
            self.W2 = rng.normal(0, 0.05, (hidden, hidden))
            self.b2 = np.zeros(hidden)
            self.W3 = rng.normal(0, 0.02, (hidden, out_dim))
            self.b3 = np.zeros(out_dim)

        def forward(self, x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float32)
            h = np.maximum(0, x @ self.W1 + self.b1)
            h = np.maximum(0, h @ self.W2 + self.b2)
            raw = h @ self.W3 + self.b3
            return 1.0 / (1.0 + np.exp(-np.clip(raw, -30, 30)))

        def predict_dict(self, features: np.ndarray) -> Dict[str, float]:
            normalised = self.forward(np.asarray(features).reshape(1, -1))[0]
            values = _LO + normalised * _SPAN
            return {name: float(values[i]) for i, name in enumerate(PARAM_NAMES)}

def load_controller(path: Optional[str] = None) -> ParameterController:
    """Load a trained PyTorch state dict; never silently substitute random weights."""
    if path is None:
        raise ValueError("trained weights are required; train with scripts/train_controller.py")
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"controller checkpoint not found: {checkpoint_path}")
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to load controller checkpoints; install .[neural]")
    if checkpoint_path.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError("only .pt/.pth state-dict checkpoints are supported")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise ValueError("checkpoint must contain a state_dict")
    model = ParameterController()
    model.load_state_dict(state, strict=True)
    model.eval()
    return model
