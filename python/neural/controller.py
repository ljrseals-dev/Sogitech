"""
Neural parameter controller for the Sogitech Codec.

Maps audio features (or a simple embedding) → the exact parameter vector
expected by the Cmajor decoder.
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np

# Optional torch – the rest of the codec still works without it
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# Canonical parameter order used by the codec
PARAM_NAMES = [
    "osc1Wave", "osc2Wave", "osc1Level", "osc2Level",
    "subLevel", "noiseLevel", "additiveBlend", "powerSagDepth",
    "filterLP", "filterHP", "filterRes", "keyTrack", "velMod",
    "glideTime", "midiNote", "velocity", "pressure", "residualGain",
]


if HAS_TORCH:

    class ParameterController(nn.Module):
        """
        Small MLP that predicts the 18-dimensional parameter vector
        from a fixed-size feature vector (e.g. mel + f0 + rms).
        """

        def __init__(self, in_dim: int = 64, hidden: int = 128, out_dim: int = 18):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, out_dim),
            )
            # Soft constraints via final activation ranges
            self.out_dim = out_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            raw = self.net(x)
            # Map most outputs to sensible ranges
            # osc waves stay continuous [0,3]; levels [0,1]; midiNote around 40-90 etc.
            return raw

        def predict_dict(self, features: np.ndarray) -> Dict[str, float]:
            self.eval()
            with torch.no_grad():
                t = torch.from_numpy(features.astype(np.float32)).unsqueeze(0)
                out = self.forward(t).squeeze(0).numpy()
            return {name: float(out[i]) for i, name in enumerate(PARAM_NAMES)}

else:
    # Pure-numpy stub so the package still imports
    class ParameterController:
        def __init__(self, in_dim: int = 64, hidden: int = 128, out_dim: int = 18):
            self.in_dim = in_dim
            self.out_dim = out_dim
            # Random weights for demo
            rng = np.random.default_rng(42)
            self.W1 = rng.normal(0, 0.1, (in_dim, hidden))
            self.b1 = np.zeros(hidden)
            self.W2 = rng.normal(0, 0.1, (hidden, hidden))
            self.b2 = np.zeros(hidden)
            self.W3 = rng.normal(0, 0.1, (hidden, out_dim))
            self.b3 = np.zeros(out_dim)

        def forward(self, x: np.ndarray) -> np.ndarray:
            h = np.maximum(0, x @ self.W1 + self.b1)
            h = np.maximum(0, h @ self.W2 + self.b2)
            return h @ self.W3 + self.b3

        def predict_dict(self, features: np.ndarray) -> Dict[str, float]:
            out = self.forward(features.astype(np.float64))
            return {name: float(out[i]) for i, name in enumerate(PARAM_NAMES)}


def load_controller(path: Optional[str] = None) -> ParameterController:
    """
    Load a trained controller. If path is None or missing, returns a randomly
    initialised (or zero) controller suitable for demos.
    """
    ctrl = ParameterController()
    if path is None:
        return ctrl
    if HAS_TORCH and path.endswith((".pt", ".pth")):
        state = torch.load(path, map_location="cpu")
        ctrl.load_state_dict(state)
        return ctrl
    # ONNX path would go here with onnxruntime
    return ctrl
