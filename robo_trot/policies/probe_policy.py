from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from robo_trot.policies.action_adapter import validate_action_label


@dataclass
class SineJointProbePolicy:
    """Deterministic sine-wave policy for visibly probing joint order and motion.

    Math: angles are expressed in radians unless the caller documents otherwise.
    Frame conventions and equations are made explicit for quaternion, yaw, or IK paths.
    Outputs preserve the repository joint/contact ordering contract.
    """

    action_dim: int = 12
    amplitude: float = 0.35
    frequency_hz: float = 0.5
    policy_dt: float = 0.02
    joint_index: int | None = None

    def __post_init__(self) -> None:
        """Validate probe dimensions, amplitude, frequency, and joint selection.

        It validates dataclass parameters before the instance is used.
        """
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
        """Reset the probe phase at rollout start.

        It prepares per-episode state before rollout or simulation resumes.
        """
        del rng
        self._step = 0

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Return a deterministic sinusoidal action for one or all joints.

        Math: angles are expressed in radians unless the caller documents otherwise.
        Frame conventions and equations are made explicit for quaternion, yaw, or IK paths.
        Outputs preserve the repository joint/contact ordering contract.
        """
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


@dataclass
class SineFlailPolicy:
    """Deterministic multi-joint sine policy for obvious full-leg motion.

    Math: angles are expressed in radians unless the caller documents otherwise.
    Frame conventions and equations are made explicit for quaternion, yaw, or IK paths.
    Outputs preserve the repository joint/contact ordering contract.
    """

    action_dim: int = 12
    amplitude: float = 0.8
    frequency_hz: float = 0.7
    policy_dt: float = 0.02
    randomize_phases: bool = True

    def __post_init__(self) -> None:
        """Validate flail dimensions and wave parameters.

        It validates dataclass parameters before the instance is used.
        """
        if int(self.action_dim) <= 0:
            raise ValueError("action_dim must be positive")
        if not 0.0 <= float(self.amplitude) <= 1.0:
            raise ValueError("amplitude must be in [0, 1]")
        if float(self.frequency_hz) <= 0.0:
            raise ValueError("frequency_hz must be positive")
        if float(self.policy_dt) <= 0.0:
            raise ValueError("policy_dt must be positive")
        self._step = 0
        self._phases = np.linspace(0.0, 2.0 * math.pi, int(self.action_dim), endpoint=False, dtype=np.float32)
        self._scales = np.ones(int(self.action_dim), dtype=np.float32)

    def reset(self, rng: np.random.Generator) -> None:
        """Reset the flail wave and optionally sample per-joint phases.

        It prepares per-episode state before rollout or simulation resumes.
        """
        self._step = 0
        if self.randomize_phases:
            self._phases = rng.uniform(0.0, 2.0 * math.pi, size=int(self.action_dim)).astype(np.float32)
            self._scales = rng.uniform(0.65, 1.0, size=int(self.action_dim)).astype(np.float32)

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Return a coherent high-amplitude sinusoidal action for all joints.

        Math: angles are expressed in radians unless the caller documents otherwise.
        Frame conventions and equations are made explicit for quaternion, yaw, or IK paths.
        Outputs preserve the repository joint/contact ordering contract.
        """
        del obs
        phase = 2.0 * math.pi * float(self.frequency_hz) * float(self.policy_dt) * float(self._step)
        self._step += 1
        action = float(self.amplitude) * self._scales * np.sin(phase + self._phases)
        return validate_action_label(action.astype(np.float32), action_dim=int(self.action_dim))


@dataclass
class SineJointScanPolicy:
    """Sequential sine probe that sweeps one joint at a time through all joints.

    Math: angles are expressed in radians unless the caller documents otherwise.
    Frame conventions and equations are made explicit for quaternion, yaw, or IK paths.
    Outputs preserve the repository joint/contact ordering contract.
    """

    action_dim: int = 12
    amplitude: float = 0.6
    frequency_hz: float = 0.5
    policy_dt: float = 0.02
    steps_per_joint: int = 100

    def __post_init__(self) -> None:
        """Validate scan dimensions and timing parameters.

        It validates dataclass parameters before the instance is used.
        """
        if int(self.action_dim) <= 0:
            raise ValueError("action_dim must be positive")
        if not 0.0 <= float(self.amplitude) <= 1.0:
            raise ValueError("amplitude must be in [0, 1]")
        if float(self.frequency_hz) <= 0.0:
            raise ValueError("frequency_hz must be positive")
        if float(self.policy_dt) <= 0.0:
            raise ValueError("policy_dt must be positive")
        if int(self.steps_per_joint) <= 0:
            raise ValueError("steps_per_joint must be positive")
        self._step = 0

    def reset(self, rng: np.random.Generator) -> None:
        """Reset the sequential scan to the first joint.

        It prepares per-episode state before rollout or simulation resumes.
        """
        del rng
        self._step = 0

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Return a sine action for the currently active scan joint.

        Math: angles are expressed in radians unless the caller documents otherwise.
        Frame conventions and equations are made explicit for quaternion, yaw, or IK paths.
        Outputs preserve the repository joint/contact ordering contract.
        """
        del obs
        active_joint = (self._step // int(self.steps_per_joint)) % int(self.action_dim)
        local_step = self._step % int(self.steps_per_joint)
        phase = 2.0 * math.pi * float(self.frequency_hz) * float(self.policy_dt) * float(local_step)
        self._step += 1
        action = np.zeros(int(self.action_dim), dtype=np.float32)
        action[active_joint] = float(self.amplitude) * math.sin(phase)
        return validate_action_label(action, action_dim=int(self.action_dim))
