from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from robo_trot.policies.action_adapter import action_label_to_q_des
from robo_trot.robot.a1 import ACTION_SCALE, Q_HOME


@dataclass(frozen=True)
class ActionMappingAuditResult:
    """Per-action result from a normalized-action to joint-control audit.

    Instances expose a documented contract used by rollout, data, or policy code.
    """

    index: int
    joint_name: str
    actuator_name: str
    expected_q_delta: float
    ctrl_delta: float
    observed_q_delta: float
    dominant_joint_index: int
    dominant_joint_name: str
    passed: bool
    reason: str


def audit_action_mapping(
    env: Any,
    action_value: float = 0.5,
    settle_steps: int = 10,
    min_observed_delta: float = 1e-3,
) -> list[ActionMappingAuditResult]:
    """Probe every action index and report whether it drives the matching joint.

    This documents the callable contract used by the surrounding pipeline.
    """
    if not -1.0 <= float(action_value) <= 1.0 or float(action_value) == 0.0:
        raise ValueError("action_value must be nonzero and in [-1, 1]")
    if int(settle_steps) <= 0:
        raise ValueError("settle_steps must be positive")
    if float(min_observed_delta) < 0.0:
        raise ValueError("min_observed_delta must be nonnegative")

    results: list[ActionMappingAuditResult] = []
    joint_names = [str(name) for name in env.joint_names]
    actuator_names = [str(name) for name in env.actuator_names]
    if len(joint_names) != 12 or len(actuator_names) != 12:
        raise ValueError("env must expose 12 joint_names and 12 actuator_names")

    for index in range(12):
        results.append(_audit_one_action_index(env, index, float(action_value), int(settle_steps), float(min_observed_delta)))
    return results


def _audit_one_action_index(
    env: Any,
    index: int,
    action_value: float,
    settle_steps: int,
    min_observed_delta: float,
) -> ActionMappingAuditResult:
    """Audit a single normalized action index against one environment joint.

    This documents the callable contract used by the surrounding pipeline.
    """
    env.reset(seed=index)
    q_before, _ = env.get_q_qdot()
    action = np.zeros(12, dtype=np.float32)
    action[index] = np.float32(action_value)
    q_des = action_label_to_q_des(action)
    expected_q_delta = float(ACTION_SCALE[index] * action_value)
    reasons: list[str] = []

    if not np.isclose(float(q_des[index] - Q_HOME[index]), expected_q_delta, rtol=1e-5, atol=1e-6):
        reasons.append("q_des_delta")

    done_reason = ""
    for _ in range(settle_steps):
        _, done, info = env.step_q_des(q_des)
        if done:
            done_reason = str(info.get("done_reason", "done"))
            break
    if done_reason:
        reasons.append(f"terminated:{done_reason}")

    q_after, _ = env.get_q_qdot()
    observed_delta = np.asarray(q_after - q_before, dtype=np.float32)
    dominant_joint_index = int(np.argmax(np.abs(observed_delta)))
    observed_q_delta = float(observed_delta[index])
    if dominant_joint_index != int(index):
        reasons.append("dominant_joint_mismatch")
    if abs(observed_q_delta) < min_observed_delta:
        reasons.append("insufficient_observed_motion")
    if np.sign(observed_q_delta) != np.sign(expected_q_delta):
        reasons.append("motion_sign_mismatch")

    ctrl_delta = _read_ctrl_delta(env, index)
    if ctrl_delta is None:
        ctrl_delta_value = float("nan")
    else:
        ctrl_delta_value = float(ctrl_delta)
        if not np.isclose(ctrl_delta_value, expected_q_delta, rtol=1e-4, atol=1e-5):
            reasons.append("ctrl_delta_mismatch")

    return ActionMappingAuditResult(
        index=int(index),
        joint_name=str(env.joint_names[index]),
        actuator_name=str(env.actuator_names[index]),
        expected_q_delta=expected_q_delta,
        ctrl_delta=ctrl_delta_value,
        observed_q_delta=observed_q_delta,
        dominant_joint_index=dominant_joint_index,
        dominant_joint_name=str(env.joint_names[dominant_joint_index]),
        passed=not reasons,
        reason="ok" if not reasons else ",".join(reasons),
    )


def _read_ctrl_delta(env: Any, index: int) -> float | None:
    """Return the control-slot delta from home when the environment exposes it.

    Callers rely on the returned value shape and semantics described here.
    """
    data = getattr(env, "data", None)
    ctrl = getattr(data, "ctrl", None)
    if ctrl is None:
        return None
    ctrl_array = np.asarray(ctrl, dtype=np.float32)
    if ctrl_array.shape[0] <= int(index):
        return None
    return float(ctrl_array[index] - Q_HOME[index])
