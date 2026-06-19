from __future__ import annotations

from typing import Protocol

import numpy as np


class Policy(Protocol):
    """Protocol for policies that map actor observations to normalized actions."""

    def reset(self, rng: np.random.Generator) -> None:
        """Reset policy state for a new rollout."""
        ...

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Return a normalized 12D action label for one observation."""
        ...
