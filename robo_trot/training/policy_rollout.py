from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robo_trot.policies.action_adapter import action_label_to_q_des
from robo_trot.policies.base import Policy
from robo_trot.robot.a1 import ACTION_SCALE, OBS_DIM_NO_CONTACTS, OBS_DIM_WITH_CONTACTS, Q_HOME


@dataclass(frozen=True)
class DatasetContract:
    """Dataset/environment contract required before policy rollout or training.

    Instances expose a documented contract used by rollout, data, or policy code.
    """

    joint_names: list[str]
    actuator_names: list[str]
    q_home: np.ndarray
    action_scale: np.ndarray
    obs_dim: int
    action_dim: int


@dataclass(frozen=True)
class PolicyRolloutSummary:
    """Summary statistics from a policy rollout harness run.

    Instances expose a documented contract used by rollout, data, or policy code.
    """

    steps: int
    survived: bool
    done_reason: str
    total_reward: float
    obs_dim: int
    action_dim: int
    max_abs_action: float
    max_joint_delta: float
    had_nan: bool


def load_dataset_contract(metadata_path: str | Path) -> DatasetContract:
    """Load the policy contract fields from dataset metadata JSON.

    This documents the callable contract used by the surrounding pipeline.
    """
    metadata = json.loads(Path(metadata_path).read_text())
    return DatasetContract(
        joint_names=[str(value) for value in metadata["joint_names"]],
        actuator_names=[str(value) for value in metadata["actuator_names"]],
        q_home=np.asarray(metadata["q_home"], dtype=np.float32),
        action_scale=np.asarray(metadata["action_scale"], dtype=np.float32),
        obs_dim=int(metadata["obs_dim"]),
        action_dim=int(metadata["action_dim"]),
    )


def validate_env_contract(env: Any, contract: DatasetContract) -> None:
    """Raise if the current environment does not match dataset ordering metadata.

    Validation failures are surfaced as explicit errors for callers and tests.
    """
    if list(env.joint_names) != contract.joint_names:
        raise ValueError(f"joint_names mismatch: env={list(env.joint_names)} dataset={contract.joint_names}")
    if list(env.actuator_names) != contract.actuator_names:
        raise ValueError(f"actuator_names mismatch: env={list(env.actuator_names)} dataset={contract.actuator_names}")
    if contract.q_home.shape != Q_HOME.shape or not np.allclose(contract.q_home, Q_HOME, rtol=1e-6, atol=1e-6):
        raise ValueError("q_home mismatch between dataset and policy constants")
    if contract.action_scale.shape != ACTION_SCALE.shape or not np.allclose(contract.action_scale, ACTION_SCALE, rtol=1e-6, atol=1e-6):
        raise ValueError("action_scale mismatch between dataset and policy constants")
    expected_obs_dim = OBS_DIM_WITH_CONTACTS if bool(env.cfg.use_contacts) else OBS_DIM_NO_CONTACTS
    if int(contract.obs_dim) != int(expected_obs_dim):
        raise ValueError(f"obs_dim mismatch: env={expected_obs_dim} dataset={contract.obs_dim}")
    if int(contract.action_dim) != 12:
        raise ValueError(f"action_dim mismatch: env=12 dataset={contract.action_dim}")


class PolicyRolloutHarness:
    """Run a live policy-environment loop using normalized policy actions.

    Instances expose a documented contract used by rollout, data, or policy code.
    """

    def __init__(
        self,
        env: Any,
        policy: Policy,
        command: np.ndarray | None = None,
        dataset_contract: DatasetContract | None = None,
    ) -> None:
        """Create a rollout harness and optionally validate dataset ordering.

        It stores configuration and prepares the instance invariants used later.
        """
        self.env = env
        self.policy = policy
        self.command = np.asarray(command if command is not None else np.zeros(3, dtype=np.float32), dtype=np.float32).reshape(3)
        if dataset_contract is not None:
            validate_env_contract(env, dataset_contract)
        self.dataset_contract = dataset_contract

    def run(self, seconds: float, seed: int = 0, render_callback: Any | None = None, print_every: int = 0) -> PolicyRolloutSummary:
        """Run the policy loop for the requested simulated duration.

        The routine owns the command or process lifecycle described by its arguments.
        """
        rng = np.random.default_rng(seed)
        self.env.reset(seed=seed)
        self.policy.reset(rng)
        steps_target = max(1, int(round(float(seconds) / float(self.env.policy_dt))))
        prev_action = np.zeros(12, dtype=np.float32)
        prev_reward = 0.0
        reset_flag = True
        phase = 0.0
        total_reward = 0.0
        max_abs_action = 0.0
        max_joint_delta = 0.0
        had_nan = False
        done_reason = ""
        steps = 0
        q_initial, _ = self.env.get_q_qdot()

        for step in range(steps_target):
            obs = self.env.make_obs(self.command, prev_action, prev_reward, reset_flag, phase)
            action = np.asarray(self.policy.act(obs), dtype=np.float32)
            q_des = self.action_to_q_des(action)
            reward, done, info = self.env.step_q_des(q_des)
            q, _ = self.env.get_q_qdot()
            total_reward += float(reward)
            max_abs_action = max(max_abs_action, float(np.max(np.abs(action))) if action.size else 0.0)
            max_joint_delta = max(max_joint_delta, float(np.max(np.abs(q - q_initial))) if q.size else 0.0)
            had_nan = had_nan or not np.all(np.isfinite(obs)) or not np.all(np.isfinite(action)) or not np.all(np.isfinite(q_des))
            steps = step + 1
            if render_callback is not None:
                render_callback(step, obs, action, q_des, reward, done, info)
            if print_every and step % max(1, int(print_every)) == 0:
                print(
                    f"step={step} reward={float(reward):.4f} "
                    f"action_abs_max={float(np.max(np.abs(action))):.4f} "
                    f"q_des_min={float(np.min(q_des)):.4f} q_des_max={float(np.max(q_des)):.4f}"
                )
            prev_action = action
            prev_reward = float(reward)
            reset_flag = False
            if done:
                done_reason = str(info.get("done_reason", "done"))
                break

        expected_obs_dim = OBS_DIM_WITH_CONTACTS if bool(self.env.cfg.use_contacts) else OBS_DIM_NO_CONTACTS
        return PolicyRolloutSummary(
            steps=steps,
            survived=done_reason == "",
            done_reason=done_reason,
            total_reward=total_reward,
            obs_dim=expected_obs_dim,
            action_dim=12,
            max_abs_action=max_abs_action,
            max_joint_delta=max_joint_delta,
            had_nan=had_nan,
        )

    def action_to_q_des(self, action_label: np.ndarray) -> np.ndarray:
        """Convert one normalized policy action into environment joint targets.

        This documents the callable contract used by the surrounding pipeline.
        """
        return action_label_to_q_des(action_label)
