from __future__ import annotations

from typing import Protocol

import numpy as np


class Policy(Protocol):
    """Protocol for policies that map actor observations to normalized actions.

    Instances expose a documented contract used by rollout, data, or policy code.
    """

    def reset(self, rng: np.random.Generator) -> None:
        """Reset policy state for a new rollout.

        It prepares per-episode state before rollout or simulation resumes.
        """
        ...

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Return a normalized 12D action label for one observation.

        The returned action follows the normalized policy action contract.
        """
        ...
