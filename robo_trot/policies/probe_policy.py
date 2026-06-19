from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from robo_trot.policies.action_adapter import validate_action_label


@dataclass
class SineJointProbePolicy:
    """Deterministic sine-wave policy for visibly probing joint order and motion."""

    action_dim: int = 12
    amplitude: float = 0.35
    frequency_hz: float = 0.5
    policy_dt: float = 0.02
    joint_index: int | None = None

    def __post_init__(self) -> None:
        """Validate probe dimensions, amplitude, frequency, and joint selection."""
        if int(self.action_dim) <= 0:
            raise ValueError("action_dim must be positive")
        if not 0.0 <= float(self.amplitude) <= 1.0:
            raise ValueError("amplitude must be in [0, 1]")
        if float(self.frequency_hz) <= 0.0:
            raise ValueError("frequency_hz must be positive")
        if float(self.policy_dt) <= 0.0:
            raise ValueError("policy_dt must be positive")
        if self.joint_index is not None and not 0 <= int(self.joint_index) < int(self.action_dim):
            raise ValueError(f"joint_index must be in [0, {int(self.action_dim) - 1}]")
        self._step = 0

    def reset(self, rng: np.random.Generator) -> None:
        """Reset the probe phase at rollout start."""
        del rng
        self._step = 0

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Return a deterministic sinusoidal action for one or all joints."""
        del obs
        phase = 2.0 * math.pi * float(self.frequency_hz) * float(self.policy_dt) * float(self._step)
        self._step += 1
        action = np.zeros(int(self.action_dim), dtype=np.float32)
        if self.joint_index is None:
            offsets = np.arange(int(self.action_dim), dtype=np.float32) * (2.0 * math.pi / float(self.action_dim))
            action[:] = float(self.amplitude) * np.sin(phase + offsets)
        else:
            action[int(self.joint_index)] = float(self.amplitude) * math.sin(phase)
        return validate_action_label(action, action_dim=int(self.action_dim))
