from pathlib import Path
import subprocess
import sys

import numpy as np

from robo_trot.robot.a1 import OBS_DIM_WITH_CONTACTS, Q_HOME


class RecordingPolicy:
    def __init__(self):
        self.observations = []

    def reset(self, rng):
        self.rng = rng

    def act(self, obs):
        self.observations.append(obs.copy())
        return np.full(12, 0.1, dtype=np.float32)


class FakeEnv:
    policy_dt = 0.02
    joint_names = [f"j{i}" for i in range(12)]
    actuator_names = [f"a{i}" for i in range(12)]

    def __init__(self):
        self.q = Q_HOME.copy()
        self.qdot = np.zeros(12, dtype=np.float32)
        self.q_des_history = []
        self.cfg = type("Cfg", (), {"use_contacts": True})()

    def reset(self, seed=None):
        self.q = Q_HOME.copy()
        return self.get_state()

    def get_state(self):
        return {
            "base_pos": np.array([0.0, 0.0, 0.32], dtype=np.float32),
            "base_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "base_lin_vel_body": np.zeros(3, dtype=np.float32),
            "base_ang_vel_body": np.zeros(3, dtype=np.float32),
            "projected_gravity": np.array([0.0, 0.0, -1.0], dtype=np.float32),
            "foot_contacts": np.zeros(4, dtype=np.float32),
            "torque": np.zeros(12, dtype=np.float32),
        }

    def make_obs(self, command, prev_action, prev_reward, reset_flag, phase):
        obs = np.zeros(OBS_DIM_WITH_CONTACTS, dtype=np.float32)
        obs[9:12] = command
        obs[36:48] = prev_action
        obs[50] = prev_reward
        obs[51] = 1.0 if reset_flag else 0.0
        return obs

    def get_q_qdot(self):
        return self.q.copy(), self.qdot.copy()

    def step_q_des(self, q_des):
        self.q_des_history.append(q_des.copy())
        self.q = np.asarray(q_des, dtype=np.float32).copy()
        return 0.5, False, {"done_reason": "", **self.get_state()}


def test_policy_rollout_harness_calls_policy_and_steps_env():
    from robo_trot.training.policy_rollout import PolicyRolloutHarness

    env = FakeEnv()
    policy = RecordingPolicy()
    harness = PolicyRolloutHarness(env=env, policy=policy, command=np.array([0.2, 0.0, 0.1], dtype=np.float32))

    summary = harness.run(seconds=0.06, seed=7)

    assert summary.steps == 3
    assert len(policy.observations) == 3
    assert len(env.q_des_history) == 3
    assert policy.observations[0].shape == (OBS_DIM_WITH_CONTACTS,)
    assert float(np.max(np.abs(env.q_des_history[-1] - Q_HOME))) > 0.0


def test_short_mujoco_random_policy_rollout_moves_joint_when_assets_exist():
    from robo_trot.policies.random_policy import RandomPolicy
    from robo_trot.sim.a1_teacher_env import A1TeacherEnv
    from robo_trot.training.policy_rollout import PolicyRolloutHarness

    xml_path = Path("assets/mujoco_menagerie/unitree_a1/scene.xml")
    if not xml_path.exists():
        return
    env = A1TeacherEnv(xml_path, {"use_contacts": True, "episode_seconds": 1.0})
    policy = RandomPolicy(action_dim=12, action_limit=0.1)
    harness = PolicyRolloutHarness(env=env, policy=policy)

    summary = harness.run(seconds=0.2, seed=0)

    assert summary.steps > 0
    assert summary.max_joint_delta > 0.0
    assert not summary.had_nan


def test_random_policy_scripts_expose_help():
    for script in ("scripts/play_random_policy.py", "scripts/sanity_check_random_policy.py"):
        result = subprocess.run([sys.executable, script, "--help"], check=False, capture_output=True, text=True)

        assert result.returncode == 0
        assert "--action_limit" in result.stdout
