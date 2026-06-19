import json

import numpy as np
import pytest

from robo_trot.robot.a1 import ACTION_SCALE, OBS_DIM_WITH_CONTACTS, Q_HOME


def test_random_policy_returns_bounded_float32_action():
    from robo_trot.policies.random_policy import RandomPolicy

    rng = np.random.default_rng(123)
    policy = RandomPolicy(action_dim=12, action_limit=0.25)
    policy.reset(rng)

    action = policy.act(np.zeros(OBS_DIM_WITH_CONTACTS, dtype=np.float32))

    assert action.dtype == np.float32
    assert action.shape == (12,)
    assert float(np.max(np.abs(action))) <= 0.25


def test_random_policy_rejects_invalid_action_limit():
    from robo_trot.policies.random_policy import RandomPolicy

    with pytest.raises(ValueError, match="action_limit"):
        RandomPolicy(action_dim=12, action_limit=1.5)


def test_action_label_to_q_des_uses_dataset_convention():
    from robo_trot.policies.action_adapter import action_label_to_q_des

    action = np.array([0.5, -0.25, 1.0] * 4, dtype=np.float32)
    q_des = action_label_to_q_des(action)

    assert q_des.dtype == np.float32
    np.testing.assert_allclose(q_des, Q_HOME + ACTION_SCALE * action)


def test_action_label_to_q_des_rejects_bad_shape_and_out_of_range():
    from robo_trot.policies.action_adapter import action_label_to_q_des

    with pytest.raises(ValueError, match="shape"):
        action_label_to_q_des(np.zeros(11, dtype=np.float32))
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        action_label_to_q_des(np.array([1.1] + [0.0] * 11, dtype=np.float32))


def test_dataset_contract_passes_for_matching_metadata(tmp_path):
    from robo_trot.training.policy_rollout import load_dataset_contract, validate_env_contract

    metadata = {
        "joint_names": [f"j{i}" for i in range(12)],
        "actuator_names": [f"a{i}" for i in range(12)],
        "q_home": Q_HOME.astype(float).tolist(),
        "action_scale": ACTION_SCALE.astype(float).tolist(),
        "obs_dim": 56,
        "action_dim": 12,
    }
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata))

    contract = load_dataset_contract(metadata_path)
    env = type(
            "FakeEnv",
            (),
            {
                "joint_names": [f"j{i}" for i in range(12)],
                "actuator_names": [f"a{i}" for i in range(12)],
                "cfg": type("Cfg", (), {"use_contacts": True})(),
            },
    )()

    validate_env_contract(env, contract)

    assert contract.obs_dim == 56
    assert contract.action_dim == 12


def test_dataset_contract_rejects_reordered_joints(tmp_path):
    from robo_trot.training.policy_rollout import load_dataset_contract, validate_env_contract

    metadata = {
        "joint_names": ["j0", "j1"],
        "actuator_names": ["a0", "a1"],
        "q_home": Q_HOME.astype(float).tolist(),
        "action_scale": ACTION_SCALE.astype(float).tolist(),
        "obs_dim": 56,
        "action_dim": 12,
    }
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata))
    env = type(
        "FakeEnv",
        (),
        {
            "joint_names": ["j1", "j0"],
            "actuator_names": ["a0", "a1"],
            "cfg": type("Cfg", (), {"use_contacts": True})(),
        },
    )()

    with pytest.raises(ValueError, match="joint_names"):
        validate_env_contract(env, load_dataset_contract(metadata_path))
