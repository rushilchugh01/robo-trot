from __future__ import annotations

from typing import Any, Sequence

import numpy as np

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None


_BaseModule = nn.Module if nn is not None else object


class MLPPolicy(_BaseModule):
    """Feed-forward behavior-cloning policy for normalized A1 actions.

    The network maps `[B, obs_dim]` observations to tanh-bounded `[B, 12]` labels.
    """

    def __init__(
        self,
        obs_dim: int = 56,
        action_dim: int = 12,
        hidden_sizes: Sequence[int] = (256, 256, 128),
    ) -> None:
        """Create the MLP layers and tanh action head.

        Torch is imported lazily so repository tests can run without it installed.
        """
        if torch is None or nn is None:
            raise ModuleNotFoundError("torch is required to construct MLPPolicy")
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        layers: list[Any] = []
        last_dim = self.obs_dim
        for hidden_dim in self.hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, self.action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, obs: Any) -> Any:
        """Return tanh-bounded normalized action labels.

        Inputs must be float tensors shaped `[batch, obs_dim]`.
        """
        return torch.tanh(self.network(obs))

    def reset(self, rng: np.random.Generator | None = None) -> None:
        """Reset rollout state for protocol compatibility.

        Feed-forward MLP policies do not keep recurrent state.
        """
        del rng

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Return one normalized action for a single observation.

        The returned NumPy vector is clipped by the model tanh to `[-1, 1]`.
        """
        if torch is None:
            raise ModuleNotFoundError("torch is required to run MLPPolicy")
        self.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(np.asarray(obs, dtype=np.float32).reshape(1, self.obs_dim))
            action = self.forward(tensor).cpu().numpy()[0]
        return action.astype(np.float32)

    def config_dict(self) -> dict[str, Any]:
        """Return constructor metadata for checkpoint manifests.

        Checkpoint loaders use this dictionary to rebuild the model shape.
        """
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_sizes": list(self.hidden_sizes),
        }
