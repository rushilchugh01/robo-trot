from __future__ import annotations

import numpy as np


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    """Return a unit-length quaternion, rejecting zero-norm inputs."""
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm <= 0.0:
        raise ValueError("Quaternion norm must be positive")
    return quat / norm


def quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """Convert a wxyz quaternion into a body-to-world rotation matrix."""
    w, x, y, z = normalize_quat(quat)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def rotate_world_to_body(quat_wxyz: np.ndarray, vec_world: np.ndarray) -> np.ndarray:
    """Rotate a world-frame vector into the body frame."""
    rot_body_to_world = quat_to_rotmat(quat_wxyz)
    return (rot_body_to_world.T @ np.asarray(vec_world, dtype=np.float32)).astype(np.float32)


def roll_pitch_from_quat(quat_wxyz: np.ndarray) -> tuple[float, float]:
    """Extract roll and pitch angles from a wxyz quaternion."""
    rot = quat_to_rotmat(quat_wxyz)
    roll = float(np.arctan2(rot[2, 1], rot[2, 2]))
    pitch = float(np.arctan2(-rot[2, 0], np.sqrt(rot[2, 1] ** 2 + rot[2, 2] ** 2)))
    return roll, pitch
