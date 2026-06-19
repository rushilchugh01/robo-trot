import numpy as np
from pathlib import Path

from robo_trot.robot.a1 import ACTION_SCALE, Q_HOME
from robo_trot.teachers.footspace_cpg_ik import FootspaceCPGIKTeacher


XML_PATH = Path("assets/mujoco_menagerie/unitree_a1/scene.xml")


def test_a1_action_constants_have_expected_shapes_and_values():
    assert Q_HOME.shape == (12,)
    assert ACTION_SCALE.shape == (12,)
    np.testing.assert_allclose(
        Q_HOME,
        np.array(
            [
                0.0, 0.9, -1.8,
                0.0, 0.9, -1.8,
                0.0, 0.9, -1.8,
                0.0, 0.9, -1.8,
            ],
            dtype=np.float32,
        ),
    )


def test_footspace_teacher_stands_for_zero_command_when_assets_exist():
    if not XML_PATH.exists():
        return
    teacher = FootspaceCPGIKTeacher(XML_PATH, policy_dt=0.02)
    teacher.reset(np.random.default_rng(0))
    out = teacher.compute({}, np.zeros(3, dtype=np.float32))
    assert out["q_teacher"].shape == (12,)
    np.testing.assert_allclose(out["q_teacher"], Q_HOME, atol=0.25)
    assert out["extra"]["leg_states"] == ["stand"] * 4


def test_footspace_action_label_clips_to_unit_range_when_assets_exist():
    if not XML_PATH.exists():
        return
    teacher = FootspaceCPGIKTeacher(XML_PATH, policy_dt=0.02)
    q_teacher = Q_HOME + ACTION_SCALE * 2.0
    label = teacher.action_label(q_teacher)
    assert label.shape == (12,)
    assert np.all(label <= 1.0)
    assert np.all(label >= -1.0)


def test_footspace_teacher_defaults_use_strict_slip_tuned_speed_envelope_when_assets_exist():
    if not XML_PATH.exists():
        return
    teacher = FootspaceCPGIKTeacher(XML_PATH, policy_dt=0.02)

    assert teacher.step_length_max == 0.18
    assert teacher.max_freq == 2.8


def test_footspace_teacher_reports_diagonal_swing_pairs_when_assets_exist():
    if not XML_PATH.exists():
        return
    teacher = FootspaceCPGIKTeacher(XML_PATH, policy_dt=0.0, smoothing_alpha=1.0)
    teacher.phase = 0.1 * 2.0 * np.pi
    teacher._q_prev = Q_HOME.copy()
    out = teacher.compute({}, np.array([0.4, 0.0, 0.0], dtype=np.float32))
    states = out["extra"]["leg_states"]
    assert states[0] == states[3]
    assert states[1] == states[2]
    assert states[0] != states[1]
    assert out["q_teacher"].shape == (12,)
    np.testing.assert_allclose(
        ACTION_SCALE,
        np.array(
            [
                0.25, 0.60, 0.60,
                0.25, 0.60, 0.60,
                0.25, 0.60, 0.60,
                0.25, 0.60, 0.60,
            ],
            dtype=np.float32,
        ),
    )


def test_footspace_teacher_yaw_command_scales_left_and_right_strides_when_assets_exist():
    if not XML_PATH.exists():
        return
    teacher = FootspaceCPGIKTeacher(
        XML_PATH,
        policy_dt=0.0,
        smoothing_alpha=1.0,
        yaw_stride_gain=0.6,
        yaw_cmd_limit=0.8,
    )
    teacher.phase = 0.1 * 2.0 * np.pi
    teacher._q_prev = Q_HOME.copy()

    out = teacher.compute({}, np.array([0.5, 0.0, -0.8], dtype=np.float32))

    lengths = out["extra"]["leg_step_lengths"]
    scales = out["extra"]["yaw_stride_scales"]
    assert lengths["FL"] > lengths["FR"]
    assert lengths["RL"] > lengths["RR"]
    assert scales["FL"] > 1.0
    assert scales["RL"] > 1.0
    assert scales["FR"] < 1.0
    assert scales["RR"] < 1.0


def test_footspace_teacher_zero_yaw_keeps_left_and_right_stride_lengths_symmetric_when_assets_exist():
    if not XML_PATH.exists():
        return
    teacher = FootspaceCPGIKTeacher(
        XML_PATH,
        policy_dt=0.0,
        smoothing_alpha=1.0,
        yaw_stride_gain=0.6,
    )
    teacher.phase = 0.1 * 2.0 * np.pi
    teacher._q_prev = Q_HOME.copy()

    out = teacher.compute({}, np.array([0.5, 0.0, 0.0], dtype=np.float32))

    lengths = out["extra"]["leg_step_lengths"]
    np.testing.assert_allclose(lengths["FR"], lengths["FL"], atol=1e-6)
    np.testing.assert_allclose(lengths["RR"], lengths["RL"], atol=1e-6)
