from __future__ import annotations

from typing import Protocol, TypedDict

import numpy as np


class TeacherOutput(TypedDict):
    """Typed return payload produced by teacher controllers.

    Instances expose a documented contract used by rollout, data, or policy code.
    """

    q_teacher: np.ndarray
    phase: float
    extra: dict


class Teacher(Protocol):
    """Protocol implemented by teacher controllers used for data collection.

    Instances expose a documented contract used by rollout, data, or policy code.
    """

    def reset(self, rng: np.random.Generator) -> None:
        """Reset teacher internal state using the provided RNG.

        It prepares per-episode state before rollout or simulation resumes.
        """
        ...

    def compute(self, state: dict, command: np.ndarray) -> TeacherOutput:
        """Return a teacher joint target for the current simulator state and command.

        The returned payload follows the teacher-controller rollout contract.
        """
        ...
