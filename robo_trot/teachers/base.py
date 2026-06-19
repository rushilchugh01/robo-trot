from __future__ import annotations

from typing import Protocol, TypedDict

import numpy as np


class TeacherOutput(TypedDict):
    """Typed return payload produced by teacher controllers."""

    q_teacher: np.ndarray
    phase: float
    extra: dict


class Teacher(Protocol):
    """Protocol implemented by teacher controllers used for data collection."""

    def reset(self, rng: np.random.Generator) -> None:
        """Reset teacher internal state using the provided RNG."""
        ...

    def compute(self, state: dict, command: np.ndarray) -> TeacherOutput:
        """Return a teacher joint target for the current simulator state and command."""
        ...
