from __future__ import annotations

import numpy as np


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    """Return a unit-length quaternion, rejecting zero-norm inputs.

    Formula: q_unit = q / ||q|| for MuJoCo wxyz quaternions.
    This keeps downstream rotation matrices on SO(3) instead of scaling vectors.
    A zero norm is invalid because the normalization equation is undefined.
    """
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm <= 0.0:
        raise ValueError("Quaternion norm must be positive")
    return quat / norm


def quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """Convert a wxyz quaternion into a body-to-world rotation matrix.

    Formula: the returned matrix is R(q) for normalized MuJoCo q=[w,x,y,z].
    It maps v_body to v_world with the standard Hamilton quaternion convention.
    Callers that need world-to-body use R(q).T, not a separate inverse path.
    """
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
    """Rotate a world-frame vector into the body frame.

    Formula: v_body = R(q)^T * v_world where R(q) maps body to world.
    The quaternion is MuJoCo wxyz order and the vector is a 3D Cartesian value.
    This is used for base velocities and projected gravity actor observations.
    """
    rot_body_to_world = quat_to_rotmat(quat_wxyz)
    return (rot_body_to_world.T @ np.asarray(vec_world, dtype=np.float32)).astype(np.float32)


def roll_pitch_from_quat(quat_wxyz: np.ndarray) -> tuple[float, float]:
    """Extract roll and pitch angles from a wxyz quaternion.

    Formula: roll = atan2(R32, R33), pitch = atan2(-R31, sqrt(R32^2 + R33^2)).
    Returned angles are radians and follow the MuJoCo body-to-world rotation frame.
    Yaw is intentionally omitted because termination checks only need roll/pitch.
    """
    rot = quat_to_rotmat(quat_wxyz)
    roll = float(np.arctan2(rot[2, 1], rot[2, 2]))
    pitch = float(np.arctan2(-rot[2, 0], np.sqrt(rot[2, 1] ** 2 + rot[2, 2] ** 2)))
    return roll, pitch
