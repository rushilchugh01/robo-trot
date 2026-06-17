from __future__ import annotations

import numpy as np

Q_HOME = np.array(
    [
        0.0, 0.9, -1.8,
        0.0, 0.9, -1.8,
        0.0, 0.9, -1.8,
        0.0, 0.9, -1.8,
    ],
    dtype=np.float32,
)

ACTION_SCALE = np.array(
    [
        0.25, 0.60, 0.60,
        0.25, 0.60, 0.60,
        0.25, 0.60, 0.60,
        0.25, 0.60, 0.60,
    ],
    dtype=np.float32,
)

KP = np.array([30.0, 40.0, 40.0] * 4, dtype=np.float32)
KD = np.array([1.0, 1.0, 1.0] * 4, dtype=np.float32)
TAU_LIMIT = np.array([25.0, 35.0, 35.0] * 4, dtype=np.float32)

OBS_DIM_WITH_CONTACTS = 56
OBS_DIM_NO_CONTACTS = 52
GRAVITY_WORLD = np.array([0.0, 0.0, -1.0], dtype=np.float32)
