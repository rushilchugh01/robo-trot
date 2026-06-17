from __future__ import annotations

from typing import Protocol, TypedDict

import numpy as np


class TeacherOutput(TypedDict):
    q_teacher: np.ndarray
    phase: float
    extra: dict


class Teacher(Protocol):
    def reset(self, rng: np.random.Generator) -> None:
        ...

    def compute(self, state: dict, command: np.ndarray) -> TeacherOutput:
        ...
