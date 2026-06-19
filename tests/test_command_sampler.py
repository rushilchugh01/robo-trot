import numpy as np
from pathlib import Path

from robo_trot.demos.record_teacher_demos import (
    CATEGORY_COMMAND_RANGES,
    contact_slip_metrics,
    make_teacher,
    parse_fixed_command,
    sample_category_command,
    sample_command,
    should_accept,
)


XML_PATH = Path("assets/mujoco_menagerie/unitree_a1/scene.xml")


def _quat_from_yaw(yaw: np.ndarray) -> np.ndarray:
    yaw = np.asarray(yaw, dtype=np.float32)
    quat = np.zeros((yaw.shape[0], 4), dtype=np.float32)
    quat[:, 0] = np.cos(yaw / 2.0)
    quat[:, 3] = np.sin(yaw / 2.0)
    return quat


def _healthy_episode(command: np.ndarray, yaw_delta: float = 0.0, steps: int = 320) -> dict:
    episode = {
        "reward": np.ones((steps,), dtype=np.float32),
        "base_pos": np.zeros((steps, 3), dtype=np.float32),
        "base_quat": _quat_from_yaw(np.linspace(0.0, yaw_delta, steps, dtype=np.float32)),
        "command": np.tile(command.astype(np.float32), (steps, 1)),
        "action_label": np.zeros((steps, 12), dtype=np.float32),
        "foot_contacts": np.zeros((steps, 4), dtype=np.float32),
        "foot_pos": np.zeros((steps, 4, 3), dtype=np.float32),
    }
    episode["base_pos"][:, 0] = np.linspace(0.0, 2.0, steps, dtype=np.float32)
    episode["base_pos"][:, 2] = 0.25
    episode["foot_pos"][:, :, 2] = 0.02
    episode["foot_pos"][::20, :, 2] = 0.12
    return episode


def test_sample_command_returns_supported_ranges():
    rng = np.random.default_rng(123)
    for _ in range(500):
        command, kind = sample_command(rng)
        assert command.shape == (3,)
        assert command.dtype == np.float32
        assert kind in {"forward", "slow", "turning"}
        assert command[1] == 0.0
        if kind == "forward":
            assert 0.2 <= command[0] <= 0.7
            assert -0.1 <= command[2] <= 0.1
        elif kind == "slow":
            assert 0.0 <= command[0] <= 0.2
            assert -0.2 <= command[2] <= 0.2
        else:
            assert 0.1 <= command[0] <= 0.5
            assert -0.4 <= command[2] <= 0.4


def test_fast_probe_command_profile_samples_higher_forward_speeds():
    rng = np.random.default_rng(123)
    seen_high = False
    for _ in range(500):
        command, kind = sample_command(rng, profile="fast_probe")
        assert command.shape == (3,)
        assert command[1] == 0.0
        if kind == "forward":
            assert 0.5 <= command[0] <= 0.9
            seen_high = seen_high or bool(command[0] > 0.75)
    assert seen_high


def test_sample_category_command_uses_exact_5m_category_ranges():
    rng = np.random.default_rng(123)
    for category, spec in CATEGORY_COMMAND_RANGES.items():
        for _ in range(100):
            command, kind = sample_category_command(rng, category)
            assert kind == category
            assert command.shape == (3,)
            assert command.dtype == np.float32
            assert spec["vx"][0] <= command[0] <= spec["vx"][1]
            assert command[1] == 0.0
            if "yaw_abs" in spec:
                assert spec["yaw_abs"][0] <= abs(float(command[2])) <= spec["yaw_abs"][1]
            else:
                assert spec["yaw"][0] <= command[2] <= spec["yaw"][1]


def test_sample_category_command_rejects_unknown_category():
    rng = np.random.default_rng(123)

    try:
        sample_category_command(rng, "bad_category")
    except ValueError as exc:
        assert "Unknown command category" in str(exc)
    else:
        raise AssertionError("sample_category_command should reject unknown categories")


def test_make_teacher_applies_footspace_speed_profile():
    if not XML_PATH.exists():
        return
    teacher = make_teacher("footspace", str(XML_PATH), 0.02, profile="cruise_walk")

    assert teacher.step_length_max == 0.20
    assert teacher.max_freq == 2.8


def test_parse_fixed_command_returns_float32_vector():
    command = parse_fixed_command([0.8, 0.0, -0.2])
    assert command is not None
    assert command.dtype == np.float32
    np.testing.assert_allclose(command, np.array([0.8, 0.0, -0.2], dtype=np.float32))


def test_should_accept_rejects_forward_sliding_without_foot_clearance():
    steps = 320
    episode = {
        "reward": np.ones((steps,), dtype=np.float32),
        "base_pos": np.zeros((steps, 3), dtype=np.float32),
        "command": np.tile(np.array([0.4, 0.0, 0.0], dtype=np.float32), (steps, 1)),
        "action_label": np.zeros((steps, 12), dtype=np.float32),
        "foot_pos": np.zeros((steps, 4, 3), dtype=np.float32),
    }
    episode["base_pos"][:, 0] = np.linspace(0.0, 2.0, steps, dtype=np.float32)
    episode["base_pos"][:, 2] = 0.25
    episode["foot_pos"][:, :, 2] = 0.02

    accepted, reason = should_accept(episode, done_reason="")

    assert not accepted
    assert reason == "low_foot_clearance"


def test_should_accept_rejects_forward_contact_foot_sliding():
    steps = 320
    episode = {
        "reward": np.ones((steps,), dtype=np.float32),
        "base_pos": np.zeros((steps, 3), dtype=np.float32),
        "command": np.tile(np.array([0.4, 0.0, 0.0], dtype=np.float32), (steps, 1)),
        "action_label": np.zeros((steps, 12), dtype=np.float32),
        "foot_contacts": np.ones((steps, 4), dtype=np.float32),
        "foot_pos": np.zeros((steps, 4, 3), dtype=np.float32),
    }
    episode["base_pos"][:, 0] = np.linspace(0.0, 2.0, steps, dtype=np.float32)
    episode["base_pos"][:, 2] = 0.25
    episode["foot_pos"][:, :, 0] = np.linspace(0.0, 4.0, steps, dtype=np.float32)[:, None]
    episode["foot_pos"][:, :, 2] = 0.02
    episode["foot_pos"][::20, :, 2] = 0.12

    accepted, reason = should_accept(episode, done_reason="")

    assert not accepted
    assert reason == "foot_sliding"


def test_should_accept_allows_more_slip_when_thresholds_are_configured():
    steps = 320
    episode = {
        "reward": np.ones((steps,), dtype=np.float32),
        "base_pos": np.zeros((steps, 3), dtype=np.float32),
        "command": np.tile(np.array([0.4, 0.0, 0.0], dtype=np.float32), (steps, 1)),
        "action_label": np.zeros((steps, 12), dtype=np.float32),
        "foot_contacts": np.ones((steps, 4), dtype=np.float32),
        "foot_pos": np.zeros((steps, 4, 3), dtype=np.float32),
    }
    episode["base_pos"][:, 0] = np.linspace(0.0, 2.0, steps, dtype=np.float32)
    episode["base_pos"][:, 2] = 0.25
    episode["foot_pos"][:, :, 0] = np.linspace(0.0, 4.0, steps, dtype=np.float32)[:, None]
    episode["foot_pos"][:, :, 2] = 0.02
    episode["foot_pos"][::20, :, 2] = 0.12

    accepted, reason = should_accept(
        episode,
        done_reason="",
        max_contact_slip_mean=0.8,
        max_contact_slip_p95=3.0,
    )

    assert accepted
    assert reason == ""


def test_contact_slip_metrics_reports_contact_foot_speed():
    foot_pos = np.zeros((3, 4, 3), dtype=np.float32)
    foot_contacts = np.ones((3, 4), dtype=np.float32)
    foot_pos[:, :, 0] = np.array([0.0, 0.02, 0.04], dtype=np.float32)[:, None]

    metrics = contact_slip_metrics(foot_pos, foot_contacts, policy_dt=0.02)

    assert metrics["contact_samples"] == 8
    assert metrics["mean"] == np.float32(1.0)


def test_should_accept_rejects_turning_episode_with_low_yaw_response_when_required():
    episode = _healthy_episode(np.array([0.4, 0.0, -0.6], dtype=np.float32), yaw_delta=-0.05)

    accepted, reason = should_accept(
        episode,
        done_reason="",
        require_yaw_response=True,
        yaw_cmd_threshold=0.2,
        min_yaw_delta=0.25,
    )

    assert not accepted
    assert reason == "low_yaw_response"


def test_should_accept_rejects_turning_episode_with_wrong_yaw_direction_when_required():
    episode = _healthy_episode(np.array([0.4, 0.0, -0.6], dtype=np.float32), yaw_delta=0.4)

    accepted, reason = should_accept(
        episode,
        done_reason="",
        require_yaw_response=True,
        yaw_cmd_threshold=0.2,
        min_yaw_delta=0.25,
    )

    assert not accepted
    assert reason == "wrong_yaw_direction"


def test_should_accept_does_not_apply_yaw_gate_to_straight_walk_commands():
    episode = _healthy_episode(np.array([0.4, 0.0, 0.0], dtype=np.float32), yaw_delta=0.0)

    accepted, reason = should_accept(
        episode,
        done_reason="",
        require_yaw_response=True,
        yaw_cmd_threshold=0.2,
        min_yaw_delta=0.25,
    )

    assert accepted
    assert reason == ""
