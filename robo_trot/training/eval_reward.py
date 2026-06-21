from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EvalReward:
    """Scalar evaluation reward plus named term contributions.

    Reward is used for logging and checkpoint comparison, not BC optimization.
    """

    total: float
    terms: dict[str, float]


def compute_eval_reward(
    state: dict,
    command: np.ndarray,
    action: np.ndarray,
    prev_action: np.ndarray,
    prev_foot_pos: np.ndarray | None = None,
    done: bool = False,
) -> EvalReward:
    """Compute logging reward terms for one simulated policy step.

    Math uses body-frame x velocity in m/s and body-frame yaw rate in rad/s.
    Uprightness is `clip(-projected_gravity_z, 0, 1)` in the base frame.
    Penalties are negative additive terms so the total is the sum of `terms`.
    Foot slip is measured as contacted-foot displacement between policy steps.
    """
    command = np.asarray(command, dtype=np.float32).reshape(3)
    action = np.asarray(action, dtype=np.float32).reshape(12)
    prev_action = np.asarray(prev_action, dtype=np.float32).reshape(12)
    base_lin_vel = np.asarray(state.get("base_lin_vel_body", np.zeros(3)), dtype=np.float32).reshape(3)
    base_ang_vel = np.asarray(state.get("base_ang_vel_body", np.zeros(3)), dtype=np.float32).reshape(3)
    projected_gravity = np.asarray(state.get("projected_gravity", np.array([0.0, 0.0, -1.0])), dtype=np.float32).reshape(3)
    base_pos = np.asarray(state.get("base_pos", np.array([0.0, 0.0, 0.32])), dtype=np.float32).reshape(3)
    torque = np.asarray(state.get("torque", np.zeros(12)), dtype=np.float32).reshape(12)
    roll = float(state.get("roll", 0.0))
    pitch = float(state.get("pitch", 0.0))

    vx_actual = float(base_lin_vel[0])
    yaw_actual = float(base_ang_vel[2])
    upright_raw = float(np.clip(-projected_gravity[2], 0.0, 1.0))
    height_error = abs(float(base_pos[2]) - 0.32)
    roll_pitch = abs(roll) + abs(pitch)
    fall = bool(done) or float(base_pos[2]) < 0.18 or upright_raw < 0.45 or abs(roll) > 0.9 or abs(pitch) > 0.9

    forward_reward = 1.0 * float(np.exp(-2.0 * abs(vx_actual - float(command[0]))))
    yaw_reward = 0.5 * float(np.exp(-2.0 * abs(yaw_actual - float(command[2]))))
    upright_reward = 0.5 * upright_raw
    height_reward = 0.2 * float(np.exp(-10.0 * height_error))
    alive_reward = 0.2 if not fall else 0.0
    action_penalty = -0.02 * float(np.mean(np.square(action)))
    smoothness_penalty = -0.02 * float(np.mean(np.square(action - prev_action)))
    torque_penalty = -0.0001 * float(np.mean(np.square(torque)))
    roll_pitch_penalty = -0.2 * float(roll_pitch)
    slip_penalty = -_contacted_foot_slip(state, prev_foot_pos)
    fall_penalty = -2.0 if fall else 0.0

    terms = {
        "forward_velocity": forward_reward,
        "yaw_tracking": yaw_reward,
        "upright": upright_reward,
        "base_height_stability": height_reward,
        "alive": alive_reward,
        "action_magnitude_penalty": action_penalty,
        "action_smoothness_penalty": smoothness_penalty,
        "torque_penalty": torque_penalty,
        "roll_pitch_penalty": roll_pitch_penalty,
        "foot_slip_penalty": slip_penalty,
        "fall_penalty": fall_penalty,
    }
    return EvalReward(total=float(sum(terms.values())), terms={key: float(value) for key, value in terms.items()})


def summarize_reward_terms(rewards: list[EvalReward]) -> dict[str, float]:
    """Average reward totals and term values across a rollout.

    Empty inputs return zeros so dashboards can render pending evaluations.
    """
    if not rewards:
        return {"reward_mean": 0.0}
    keys = sorted({key for reward in rewards for key in reward.terms})
    summary = {"reward_mean": float(np.mean([reward.total for reward in rewards]))}
    for key in keys:
        summary[f"{key}_mean"] = float(np.mean([reward.terms.get(key, 0.0) for reward in rewards]))
    return summary


def _contacted_foot_slip(state: dict, prev_foot_pos: np.ndarray | None) -> float:
    """Return the contacted-foot slip penalty for one policy interval.

    Distances are Euclidean world-frame foot displacements in meters.
    Equation: penalty = 0.05 * mean(||foot_pos_t - foot_pos_t-1||_2).
    Only feet with current contact flags above 0.5 contribute to the mean.
    The foot position frame is the MuJoCo world frame from `geom_xpos`.
    """
    if prev_foot_pos is None or "foot_pos" not in state or "foot_contacts" not in state:
        return 0.0
    foot_pos = np.asarray(state["foot_pos"], dtype=np.float32)
    previous = np.asarray(prev_foot_pos, dtype=np.float32)
    contacts = np.asarray(state["foot_contacts"], dtype=np.float32).reshape(4) > 0.5
    if foot_pos.shape != (4, 3) or previous.shape != (4, 3) or not np.any(contacts):
        return 0.0
    displacement = np.linalg.norm(foot_pos[contacts] - previous[contacts], axis=1)
    return 0.05 * float(np.mean(displacement))
