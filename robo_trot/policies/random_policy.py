from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robo_trot.policies.action_adapter import validate_action_label


@dataclass
class RandomPolicy:
    """Policy that samples bounded random normalized joint-offset actions.

    Instances expose a documented contract used by rollout, data, or policy code.
    """

    action_dim: int = 12
    action_limit: float = 0.25

    def __post_init__(self) -> None:
        """Validate policy dimensions and sampling range.

        It validates dataclass parameters before the instance is used.
        """
        if int(self.action_dim) <= 0:
            raise ValueError("action_dim must be positive")
        if not 0.0 <= float(self.action_limit) <= 1.0:
            raise ValueError("action_limit must be in [0, 1]")
        self._rng = np.random.default_rng()

    def reset(self, rng: np.random.Generator) -> None:
        """Attach the rollout RNG used for future random actions.

        It prepares per-episode state before rollout or simulation resumes.
        """
        self._rng = rng

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Sample one bounded normalized action for the observation.

        The returned action follows the normalized policy action contract.
        """
        del obs
        action = self._rng.uniform(-float(self.action_limit), float(self.action_limit), size=int(self.action_dim))
        return validate_action_label(action.astype(np.float32), action_dim=int(self.action_dim))
