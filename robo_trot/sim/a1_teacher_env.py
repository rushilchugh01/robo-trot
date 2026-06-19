from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from robo_trot.robot.a1 import GRAVITY_WORLD, KD, KP, OBS_DIM_NO_CONTACTS, OBS_DIM_WITH_CONTACTS, Q_HOME, TAU_LIMIT
from robo_trot.robot.kinematics import roll_pitch_from_quat, rotate_world_to_body
from robo_trot.robot.model_info import actuator_joint_maps


def build_actor_obs(
    projected_gravity_body: np.ndarray,
    base_ang_vel_body: np.ndarray,
    base_lin_vel_body: np.ndarray,
    command: np.ndarray,
    q_minus_home: np.ndarray,
    qdot: np.ndarray,
    previous_action: np.ndarray,
    phase: float,
    previous_reward: float,
    reset_flag: bool,
    foot_contacts: np.ndarray,
    use_contacts: bool = True,
) -> np.ndarray:
    """Build the raw actor observation vector from simulator state features.

    Math: angles are expressed in radians unless the caller documents otherwise.
    Frame conventions and equations are made explicit for quaternion, yaw, or IK paths.
    Outputs preserve the repository joint/contact ordering contract.
    """
    parts = [
        np.asarray(projected_gravity_body, dtype=np.float32).reshape(3),
        np.asarray(base_ang_vel_body, dtype=np.float32).reshape(3),
        np.asarray(base_lin_vel_body, dtype=np.float32).reshape(3),
        np.asarray(command, dtype=np.float32).reshape(3),
        np.asarray(q_minus_home, dtype=np.float32).reshape(12),
        np.asarray(qdot, dtype=np.float32).reshape(12),
        np.asarray(previous_action, dtype=np.float32).reshape(12),
        np.array([math.sin(phase), math.cos(phase)], dtype=np.float32),
        np.array([previous_reward], dtype=np.float32),
        np.array([1.0 if reset_flag else 0.0], dtype=np.float32),
    ]
    if use_contacts:
        parts.append(np.asarray(foot_contacts, dtype=np.float32).reshape(4))
    obs = np.concatenate(parts).astype(np.float32)
    expected = OBS_DIM_WITH_CONTACTS if use_contacts else OBS_DIM_NO_CONTACTS
    if obs.shape != (expected,):
        raise ValueError(f"Expected obs shape {(expected,)}, got {obs.shape}")
    return obs


@dataclass(frozen=True)
class A1EnvConfig:
    """Configuration for the MuJoCo A1 teacher rollout environment.

    Instances expose a documented contract used by rollout, data, or policy code.
    """

    physics_dt: float = 0.002
    policy_dt: float = 0.02
    use_contacts: bool = True
    base_height: float = 0.32
    base_height_min: float = 0.18
    max_episode_seconds: float = 12.0
    kp: np.ndarray = field(default_factory=lambda: KP.copy())
    kd: np.ndarray = field(default_factory=lambda: KD.copy())
    tau_limit: np.ndarray = field(default_factory=lambda: TAU_LIMIT.copy())


class A1TeacherEnv:
    """MuJoCo environment wrapper for A1 teacher rollouts.

    Instances expose a documented contract used by rollout, data, or policy code.
    """

    def __init__(self, xml_path: str | Path, cfg: dict[str, Any] | None = None):
        """Load the MuJoCo model and configure control, timing, and metadata.

        It stores configuration and prepares the instance invariants used later.
        """
        self.xml_path = Path(xml_path)
        cfg = dict(cfg or {})
        self.cfg = A1EnvConfig(
            physics_dt=float(cfg.get("physics_dt", 0.002)),
            policy_dt=float(cfg.get("policy_dt", 0.02)),
            use_contacts=bool(cfg.get("use_contacts", True)),
            base_height=float(cfg.get("base_height", 0.32)),
            base_height_min=float(cfg.get("base_height_min", 0.18)),
            max_episode_seconds=float(cfg.get("episode_seconds", cfg.get("max_episode_seconds", 12.0))),
            kp=np.asarray(cfg.get("kp", KP), dtype=np.float32),
            kd=np.asarray(cfg.get("kd", KD), dtype=np.float32),
            tau_limit=np.asarray(cfg.get("tau_limit", TAU_LIMIT), dtype=np.float32),
        )
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.model.opt.timestep = self.cfg.physics_dt
        self.data = mujoco.MjData(self.model)
        self.physics_dt = self.cfg.physics_dt
        self.policy_dt = self.cfg.policy_dt
        self.decimation = max(1, int(round(self.policy_dt / self.physics_dt)))
        self.actuator_maps = actuator_joint_maps(self.model)
        if len(self.actuator_maps) != 12:
            raise ValueError(f"Expected 12 actuators, got {len(self.actuator_maps)}")
        self.qposadr = np.array([item.qposadr for item in self.actuator_maps], dtype=np.int32)
        self.dofadr = np.array([item.dofadr for item in self.actuator_maps], dtype=np.int32)
        self.joint_names = [item.joint_name for item in self.actuator_maps]
        self.actuator_names = [item.actuator_name for item in self.actuator_maps]
        self.foot_geom_ids = self._detect_foot_geom_ids()
        self._renderer: mujoco.Renderer | None = None
        self._last_torque = np.zeros(12, dtype=np.float32)
        self._step_count = 0
        self.actuator_mode = self._detect_actuator_mode()

    def _detect_actuator_mode(self) -> str:
        """Detect whether the model uses position actuators or manual torque control.

        This documents the callable contract used by the surrounding pipeline.
        """
        if self.model.nu != 12:
            return "qfrc_applied"
        # Menagerie A1 defines position actuators. For those, ctrl is the joint
        # target in radians; manual PD is only the fallback path for torque models.
        return "position"

    def reset(self, seed: int | None = None) -> dict:
        """Reset simulator state to the A1 standing home pose.

        It prepares per-episode state before rollout or simulation resumes.
        """
        del seed
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = np.array([0.0, 0.0, self.cfg.base_height], dtype=np.float64)
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.data.qpos[self.qposadr] = Q_HOME.astype(np.float64)
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        if self.actuator_mode == "position":
            self.data.ctrl[:] = Q_HOME.astype(np.float64)
        self._last_torque[:] = 0.0
        self._step_count = 0
        mujoco.mj_forward(self.model, self.data)
        return self.get_state()

    def _detect_foot_geom_ids(self) -> dict[str, int]:
        """Detect spherical foot geoms for contact extraction.

        This documents the callable contract used by the surrounding pipeline.
        """
        ids: dict[str, int] = {}
        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            for leg in ("FR", "FL", "RR", "RL"):
                if body_name == f"{leg}_calf" and int(self.model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                    ids[leg] = geom_id
        return ids

    def get_q_qdot(self) -> tuple[np.ndarray, np.ndarray]:
        """Return actuated joint positions and velocities in actuator order.

        Callers rely on the returned value shape and semantics described here.
        """
        q = self.data.qpos[self.qposadr].astype(np.float32).copy()
        qdot = self.data.qvel[self.dofadr].astype(np.float32).copy()
        return q, qdot

    def _base_quat(self) -> np.ndarray:
        """Return the floating-base quaternion in MuJoCo wxyz order.

        Math: angles are expressed in radians unless the caller documents otherwise.
        Frame conventions and equations are made explicit for quaternion, yaw, or IK paths.
        Outputs preserve the repository joint/contact ordering contract.
        """
        return self.data.qpos[3:7].astype(np.float32).copy()

    def _base_lin_vel_body(self) -> np.ndarray:
        """Return base linear velocity expressed in the body frame.

        Callers rely on the returned value shape and semantics described here.
        """
        return rotate_world_to_body(self._base_quat(), self.data.qvel[0:3].astype(np.float32))

    def _base_ang_vel_body(self) -> np.ndarray:
        """Return base angular velocity expressed in the body frame.

        Callers rely on the returned value shape and semantics described here.
        """
        return rotate_world_to_body(self._base_quat(), self.data.qvel[3:6].astype(np.float32))

    def _foot_contacts(self) -> np.ndarray:
        """Return binary foot-ground contact flags in FR, FL, RR, RL order.

        Callers rely on the returned value shape and semantics described here.
        """
        contacts = np.zeros(4, dtype=np.float32)
        foot_geom_to_idx = {geom_id: idx for idx, geom_id in enumerate(self.foot_geom_ids.get(leg) for leg in ("FR", "FL", "RR", "RL")) if geom_id is not None}
        floor_names = {"floor", "ground"}
        for contact_idx in range(self.data.ncon):
            contact = self.data.contact[contact_idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            name1 = (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1) or "").lower()
            name2 = (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2) or "").lower()
            if name1 not in floor_names and name2 not in floor_names:
                continue
            if geom1 in foot_geom_to_idx:
                contacts[foot_geom_to_idx[geom1]] = 1.0
            if geom2 in foot_geom_to_idx:
                contacts[foot_geom_to_idx[geom2]] = 1.0
        return contacts

    def _foot_pos(self) -> np.ndarray:
        """Return foot geom positions in world coordinates.

        Callers rely on the returned value shape and semantics described here.
        """
        positions = np.zeros((4, 3), dtype=np.float32)
        for idx, leg in enumerate(("FR", "FL", "RR", "RL")):
            geom_id = self.foot_geom_ids.get(leg)
            if geom_id is not None:
                positions[idx] = self.data.geom_xpos[geom_id].astype(np.float32)
        return positions

    def get_state(self) -> dict:
        """Return the raw simulator state fields used by teacher rollouts.

        Callers rely on the returned value shape and semantics described here.
        """
        quat = self._base_quat()
        roll, pitch = roll_pitch_from_quat(quat)
        projected_gravity = rotate_world_to_body(quat, GRAVITY_WORLD)
        return {
            "time": float(self.data.time),
            "base_pos": self.data.qpos[0:3].astype(np.float32).copy(),
            "base_quat": quat,
            "base_lin_vel_body": self._base_lin_vel_body(),
            "base_ang_vel_body": self._base_ang_vel_body(),
            "projected_gravity": projected_gravity,
            "foot_contacts": self._foot_contacts(),
            "foot_pos": self._foot_pos(),
            "roll": roll,
            "pitch": pitch,
            "torque": self._last_torque.copy(),
            "joint_names": list(self.joint_names),
            "actuator_names": list(self.actuator_names),
        }

    def make_obs(
        self,
        command: np.ndarray,
        prev_action: np.ndarray,
        prev_reward: float,
        reset_flag: bool,
        phase: float,
    ) -> np.ndarray:
        """Build the actor observation from current state and previous-step values.

        This documents the callable contract used by the surrounding pipeline.
        """
        q, qdot = self.get_q_qdot()
        state = self.get_state()
        return build_actor_obs(
            projected_gravity_body=state["projected_gravity"],
            base_ang_vel_body=state["base_ang_vel_body"],
            base_lin_vel_body=state["base_lin_vel_body"],
            command=command,
            q_minus_home=q - Q_HOME,
            qdot=qdot,
            previous_action=prev_action,
            phase=phase,
            previous_reward=prev_reward,
            reset_flag=reset_flag,
            foot_contacts=state["foot_contacts"],
            use_contacts=self.cfg.use_contacts,
        )

    def step_q_des(self, q_des: np.ndarray) -> tuple[float, bool, dict]:
        """Apply desired joint targets for one policy step and return rollout feedback.

        Callers rely on the returned value shape and semantics described here.
        """
        q_des = np.asarray(q_des, dtype=np.float32).reshape(12)
        for _ in range(self.decimation):
            q, qdot = self.get_q_qdot()
            if self.actuator_mode == "position":
                ctrl = np.clip(q_des, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
                self.data.ctrl[:] = ctrl
            else:
                tau = self.cfg.kp * (q_des - q) - self.cfg.kd * qdot
                tau = np.clip(tau, -self.cfg.tau_limit, self.cfg.tau_limit).astype(np.float32)
                self.data.qfrc_applied[self.dofadr] = tau
            mujoco.mj_step(self.model, self.data)
            if self.actuator_mode == "position":
                tau = self.data.actuator_force[:12].astype(np.float32).copy()
            self._last_torque = tau
        self._step_count += 1
        state = self.get_state()
        reward = self._reward(state, q_des)
        done, reason = self._done(state)
        info = {
            **state,
            "done_reason": reason,
            "actuator_mode": self.actuator_mode,
            "step_count": self._step_count,
        }
        return reward, done, info

    def _reward(self, state: dict, q_des: np.ndarray) -> float:
        """Compute a lightweight debug reward for logging and filtering.

        This documents the callable contract used by the surrounding pipeline.
        """
        del q_des
        vel = float(state["base_lin_vel_body"][0])
        upright = float(np.clip(-state["projected_gravity"][2], 0.0, 1.0))
        torque_penalty = 1e-4 * float(np.mean(np.square(self._last_torque)))
        return float(0.5 * vel + 0.5 * upright + 0.2 - torque_penalty)

    def _done(self, state: dict) -> tuple[bool, str]:
        """Evaluate terminal conditions and return a reason string.

        Callers rely on the returned value shape and semantics described here.
        """
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
            return True, "nan"
        if float(state["base_pos"][2]) < self.cfg.base_height_min:
            return True, "base_height"
        if abs(float(state["roll"])) > 0.9:
            return True, "roll"
        if abs(float(state["pitch"])) > 0.9:
            return True, "pitch"
        if float(self.data.time) >= self.cfg.max_episode_seconds:
            return True, "timeout"
        return False, ""

    def render_frame(self, width: int = 640, height: int = 360, camera: str | int | None = None) -> np.ndarray:
        """Render one RGB frame from a named, indexed, or default camera.

        This documents the callable contract used by the surrounding pipeline.
        """
        if self._renderer is None or self._renderer.width != width or self._renderer.height != height:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        cam = camera
        if cam is None:
            cam = self._default_camera()
        if cam is None:
            free_cam = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(self.model, free_cam)
            free_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            free_cam.lookat[:] = self.data.qpos[0:3]
            free_cam.lookat[2] += 0.05
            free_cam.distance = 0.85
            free_cam.azimuth = 120.0
            free_cam.elevation = -14.0
            self._renderer.update_scene(self.data, camera=free_cam)
        else:
            self._renderer.update_scene(self.data, camera=cam)
        return self._renderer.render().astype(np.uint8)

    def _default_camera(self) -> str | int | None:
        """Return the preferred model camera name when one exists.

        Callers rely on the returned value shape and semantics described here.
        """
        for name in ("track", "tracking", "side", "fixed"):
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if cam_id >= 0:
                return name
        return None
