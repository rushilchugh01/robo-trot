from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from robo_trot.robot.a1 import ACTION_SCALE, Q_HOME
from robo_trot.robot.model_info import actuator_joint_maps
from robo_trot.teachers.base import TeacherOutput


@dataclass
class FootspaceCPGIKTeacher:
    xml_path: str | Path
    policy_dt: float = 0.02
    base_freq: float = 1.6
    max_freq: float = 2.8
    frequency_speed_ref: float = 1.1
    swing_duty: float = 0.42
    min_gait_speed: float = 0.08
    step_length_min: float = 0.06
    step_length_max: float = 0.18
    clearance_min: float = 0.055
    clearance_max: float = 0.095
    stance_depth: float = 0.01
    smoothing_alpha: float = 0.65
    ik_iters: int = 8
    ik_damping: float = 2e-3
    yaw_cmd_limit: float = 0.8
    yaw_stride_gain: float = 0.0
    yaw_stride_min_scale: float = 0.45
    yaw_stride_max_scale: float = 1.65
    yaw_step_bias_gain: float = 0.05
    yaw_lateral_bias_gain: float = 0.0
    yaw_stance_bias_fraction: float = 0.25

    def __post_init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        maps = actuator_joint_maps(self.model)
        if len(maps) != 12:
            raise ValueError(f"Expected 12 A1 actuators, got {len(maps)}")
        self.qposadr = np.array([item.qposadr for item in maps], dtype=np.int32)
        self.dofadr = np.array([item.dofadr for item in maps], dtype=np.int32)
        self.leg_names = ["FR", "FL", "RR", "RL"]
        self.foot_geom_ids = self._detect_foot_geom_ids()
        if set(self.foot_geom_ids) != set(self.leg_names):
            raise ValueError(f"Could not detect all foot geoms: {self.foot_geom_ids}")
        self.home_foot_pos = self._compute_home_foot_pos()
        self.phase = 0.0
        self._q_prev = Q_HOME.copy()

    def reset(self, rng: np.random.Generator) -> None:
        self.phase = float(rng.uniform(0.0, 2.0 * math.pi))
        self._q_prev = Q_HOME.copy()

    def frequency(self, command: np.ndarray) -> float:
        vx = float(np.asarray(command, dtype=np.float32)[0])
        scale = np.clip(abs(vx) / self.frequency_speed_ref, 0.0, 1.0)
        return float(self.base_freq + (self.max_freq - self.base_freq) * scale)

    def action_label(self, q_teacher: np.ndarray) -> np.ndarray:
        label = (np.asarray(q_teacher, dtype=np.float32) - Q_HOME) / ACTION_SCALE
        return np.clip(label, -1.0, 1.0).astype(np.float32)

    def compute(self, state: dict, command: np.ndarray) -> TeacherOutput:
        command = np.asarray(command, dtype=np.float32)
        vx_cmd = float(command[0])
        yaw_cmd = float(command[2])
        speed_scale = float(np.clip(abs(vx_cmd) / 0.7, 0.0, 1.0))
        yaw_scale = float(np.clip(abs(yaw_cmd) / 0.4, 0.0, 1.0))

        if speed_scale < self.min_gait_speed and yaw_scale < 0.1:
            self._q_prev = (0.4 * Q_HOME + 0.6 * self._q_prev).astype(np.float32)
            return {"q_teacher": self._q_prev.copy(), "phase": self.phase, "extra": {"teacher": "FootspaceCPGIKTeacher", "leg_states": ["stand"] * 4}}

        freq = self.frequency(command)
        self.phase = float((self.phase + 2.0 * math.pi * freq * self.policy_dt) % (2.0 * math.pi))
        step_length = float(self.step_length_min + (self.step_length_max - self.step_length_min) * max(speed_scale, 0.35 * yaw_scale))
        clearance = float(self.clearance_min + (self.clearance_max - self.clearance_min) * max(speed_scale, 0.25 * yaw_scale))

        q_seed = self._q_prev.copy()
        q_target = q_seed.copy()
        leg_states: list[str] = []
        phase_offsets = [0.0, 0.5, 0.5, 0.0]
        yaw_norm = self._yaw_norm(yaw_cmd)
        yaw_stride_scales: dict[str, float] = {}
        leg_step_lengths: dict[str, float] = {}

        for leg_idx, leg in enumerate(self.leg_names):
            phase01 = ((self.phase / (2.0 * math.pi)) + phase_offsets[leg_idx]) % 1.0
            yaw_side = self._yaw_side_sign(leg)
            yaw_scale_factor = self._yaw_stride_scale(leg, yaw_norm)
            leg_step_length = step_length * yaw_scale_factor
            desired, leg_state = self._desired_foot_pos(
                leg=leg,
                phase01=phase01,
                step_length=leg_step_length,
                clearance=clearance,
                yaw_step_bias=self.yaw_step_bias_gain * yaw_norm * yaw_side,
                yaw_lateral_bias=self.yaw_lateral_bias_gain * yaw_norm * yaw_side,
            )
            leg_states.append(leg_state)
            yaw_stride_scales[leg] = yaw_scale_factor
            leg_step_lengths[leg] = leg_step_length
            q_leg = self._solve_leg_ik(leg_idx, desired, q_target)
            q_target[3 * leg_idx : 3 * leg_idx + 3] = q_leg

        alpha = float(np.clip(self.smoothing_alpha, 0.0, 1.0))
        q_teacher = (alpha * q_target + (1.0 - alpha) * self._q_prev).astype(np.float32)
        self._q_prev = q_teacher
        return {
            "q_teacher": q_teacher,
            "phase": self.phase,
            "extra": {
                "teacher": "FootspaceCPGIKTeacher",
                "freq": freq,
                "step_length": step_length,
                "leg_step_lengths": leg_step_lengths,
                "yaw_norm": yaw_norm,
                "yaw_stride_scales": yaw_stride_scales,
                "clearance": clearance,
                "leg_order": list(self.leg_names),
                "leg_states": leg_states,
            },
        }

    def _desired_foot_pos(
        self,
        leg: str,
        phase01: float,
        step_length: float,
        clearance: float,
        yaw_step_bias: float,
        yaw_lateral_bias: float,
    ) -> tuple[np.ndarray, str]:
        home = self.home_foot_pos[leg].copy()
        duty = float(np.clip(self.swing_duty, 0.25, 0.55))
        if phase01 < duty:
            u = phase01 / duty
            smooth = 0.5 - 0.5 * math.cos(math.pi * u)
            lift = math.sin(math.pi * u)
            x = -0.5 * step_length + step_length * smooth
            z = clearance * lift
            yaw_weight = smooth
            state = "swing"
        else:
            u = (phase01 - duty) / (1.0 - duty)
            smooth = 0.5 - 0.5 * math.cos(math.pi * u)
            x = 0.5 * step_length - step_length * smooth
            z = -self.stance_depth * math.sin(math.pi * u)
            yaw_weight = float(np.clip(self.yaw_stance_bias_fraction, 0.0, 1.0))
            state = "stance"
        home[0] += x + yaw_step_bias * yaw_weight
        home[1] += yaw_lateral_bias * yaw_weight
        home[2] += z
        return home.astype(np.float32), state

    def _yaw_norm(self, yaw_cmd: float) -> float:
        limit = max(float(self.yaw_cmd_limit), 1e-6)
        return float(np.clip(yaw_cmd / limit, -1.0, 1.0))

    def _yaw_side_sign(self, leg: str) -> float:
        # Positive yaw turns left, so right-side legs are the outside legs.
        return 1.0 if leg in {"FR", "RR"} else -1.0

    def _yaw_stride_scale(self, leg: str, yaw_norm: float) -> float:
        raw = 1.0 + self._yaw_side_sign(leg) * float(self.yaw_stride_gain) * float(yaw_norm)
        return float(np.clip(raw, self.yaw_stride_min_scale, self.yaw_stride_max_scale))

    def _solve_leg_ik(self, leg_idx: int, desired_pos: np.ndarray, q_all_seed: np.ndarray) -> np.ndarray:
        q_all = np.asarray(q_all_seed, dtype=np.float64).copy()
        leg_slice = slice(3 * leg_idx, 3 * leg_idx + 3)
        geom_id = self.foot_geom_ids[self.leg_names[leg_idx]]
        for _ in range(self.ik_iters):
            self._set_private_q(q_all)
            pos = self.data.geom_xpos[geom_id].copy()
            err = np.asarray(desired_pos, dtype=np.float64) - pos
            if float(np.linalg.norm(err)) < 1e-4:
                break
            jacp = np.zeros((3, self.model.nv), dtype=np.float64)
            jacr = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacGeom(self.model, self.data, jacp, jacr, geom_id)
            j_leg = jacp[:, self.dofadr[leg_slice]]
            lhs = j_leg @ j_leg.T + self.ik_damping * np.eye(3)
            dq = j_leg.T @ np.linalg.solve(lhs, err)
            dq = np.clip(dq, -0.12, 0.12)
            q_all[leg_slice] += dq
            q_all[leg_slice] = self._clip_leg_q(leg_idx, q_all[leg_slice])
        return q_all[leg_slice].astype(np.float32)

    def _set_private_q(self, q: np.ndarray) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.data.qpos[self.qposadr] = np.asarray(q, dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)

    def _clip_leg_q(self, leg_idx: int, q_leg: np.ndarray) -> np.ndarray:
        out = np.asarray(q_leg, dtype=np.float64).copy()
        for local_idx in range(3):
            actuator_idx = 3 * leg_idx + local_idx
            out[local_idx] = np.clip(out[local_idx], self.model.actuator_ctrlrange[actuator_idx, 0], self.model.actuator_ctrlrange[actuator_idx, 1])
        return out

    def _detect_foot_geom_ids(self) -> dict[str, int]:
        ids: dict[str, int] = {}
        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            for leg in self.leg_names:
                if body_name == f"{leg}_calf" and int(self.model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                    ids[leg] = geom_id
        return ids

    def _compute_home_foot_pos(self) -> dict[str, np.ndarray]:
        self._set_private_q(Q_HOME)
        return {leg: self.data.geom_xpos[geom_id].astype(np.float32).copy() for leg, geom_id in self.foot_geom_ids.items()}
