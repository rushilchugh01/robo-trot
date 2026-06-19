from __future__ import annotations

import numpy as np

from robo_trot.robot.a1 import ACTION_SCALE, Q_HOME


def validate_action_label(action_label: np.ndarray, action_dim: int = 12) -> np.ndarray:
    """Return a validated normalized action label as float32.

    Callers rely on the returned value shape and semantics described here.
    """
    action = np.asarray(action_label, dtype=np.float32)
    if action.shape != (int(action_dim),):
        raise ValueError(f"action_label shape must be {(int(action_dim),)}, got {action.shape}")
    if not np.all(np.isfinite(action)):
        raise ValueError("action_label must contain only finite values")
    if np.any(action < -1.0) or np.any(action > 1.0):
        raise ValueError("action_label values must be in [-1, 1]")
    return action


def action_label_to_q_des(action_label: np.ndarray) -> np.ndarray:
    """Convert a normalized policy action label into raw joint targets.

    This documents the callable contract used by the surrounding pipeline.
    """
    action = validate_action_label(action_label, action_dim=12)
    return (Q_HOME + ACTION_SCALE * action).astype(np.float32)
