import numpy as np

from robo_trot.robot.a1 import ACTION_SCALE, Q_HOME


class FakeAuditData:
    """Minimal MuJoCo-like data container for action mapping tests."""

    def __init__(self):
        """Create a fake control vector."""
        self.ctrl = Q_HOME.copy()


class FakeAuditEnv:
    """Environment stub that applies q_des directly to q in actuator order."""

    policy_dt = 0.02
    joint_names = [f"j{i}" for i in range(12)]
    actuator_names = [f"a{i}" for i in range(12)]
    actuator_mode = "position"

    def __init__(self):
        """Initialize the fake environment at home pose."""
        self.data = FakeAuditData()
        self.q = Q_HOME.copy()
        self.qdot = np.zeros(12, dtype=np.float32)
        self.cfg = type("Cfg", (), {"use_contacts": True})()

    def reset(self, seed=None):
        """Reset q and ctrl to home pose."""
        del seed
        self.q = Q_HOME.copy()
        self.data.ctrl = Q_HOME.copy()
        return {}

    def get_q_qdot(self):
        """Return fake joint state in actuator order."""
        return self.q.copy(), self.qdot.copy()

    def step_q_des(self, q_des):
        """Apply q_des directly to q and ctrl."""
        self.data.ctrl = np.asarray(q_des, dtype=np.float32).copy()
        self.q = np.asarray(q_des, dtype=np.float32).copy()
        return 0.0, False, {"done_reason": ""}


def test_audit_action_mapping_reports_expected_index_and_joint_names():
    from robo_trot.training.action_mapping_audit import audit_action_mapping

    env = FakeAuditEnv()

    results = audit_action_mapping(env, action_value=0.5, settle_steps=1, min_observed_delta=1e-5)

    assert len(results) == 12
    for idx, result in enumerate(results):
        assert result.passed
        assert result.index == idx
        assert result.joint_name == f"j{idx}"
        assert result.actuator_name == f"a{idx}"
        assert result.dominant_joint_index == idx
        assert result.dominant_joint_name == f"j{idx}"
        assert np.isclose(result.expected_q_delta, ACTION_SCALE[idx] * 0.5)
        assert np.isclose(result.ctrl_delta, ACTION_SCALE[idx] * 0.5)
        assert np.isclose(result.observed_q_delta, ACTION_SCALE[idx] * 0.5)


def test_audit_action_mapping_fails_when_wrong_joint_moves():
    from robo_trot.training.action_mapping_audit import audit_action_mapping

    class SwappedEnv(FakeAuditEnv):
        """Fake env that incorrectly applies action index 0 to joint index 1."""

        def step_q_des(self, q_des):
            """Apply the first commanded q_des value to the wrong joint."""
            q_des = np.asarray(q_des, dtype=np.float32)
            self.data.ctrl = q_des.copy()
            self.q = Q_HOME.copy()
            self.q[1] = q_des[0]
            return 0.0, False, {"done_reason": ""}

    results = audit_action_mapping(SwappedEnv(), action_value=0.5, settle_steps=1, min_observed_delta=1e-5)

    assert not results[0].passed
    assert results[0].dominant_joint_index == 1
