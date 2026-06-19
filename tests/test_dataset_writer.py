import json

import pytest
import numpy as np

from robo_trot.demos.dataset_writer import DatasetWriter
from robo_trot.demos.record_teacher_demos import (
    expand_frames_for_playback,
    initial_recording_counters,
    render_q_teacher_episode,
    rollout_episode,
    write_debug_exports,
)
from robo_trot.robot.a1 import Q_HOME
from scripts.inspect_dataset import inspect_dataset


def make_episode(length: int = 3) -> dict:
    obs = np.zeros((length, 56), dtype=np.float32)
    obs[:, 49] = 1.0
    obs[0, 51] = 1.0
    q_teacher = np.tile(Q_HOME, (length, 1)).astype(np.float32)
    return {
        "obs": obs,
        "action_label": np.zeros((length, 12), dtype=np.float32),
        "q_teacher": q_teacher,
        "q": np.tile(Q_HOME, (length, 1)).astype(np.float32),
        "qdot": np.zeros((length, 12), dtype=np.float32),
        "command": np.zeros((length, 3), dtype=np.float32),
        "reward": np.zeros((length,), dtype=np.float32),
        "done": np.zeros((length,), dtype=bool),
        "reset_flag": np.array([idx == 0 for idx in range(length)], dtype=bool),
        "phase": np.zeros((length,), dtype=np.float32),
        "base_pos": np.zeros((length, 3), dtype=np.float32),
        "base_quat": np.zeros((length, 4), dtype=np.float32),
        "base_lin_vel_body": np.zeros((length, 3), dtype=np.float32),
        "base_ang_vel_body": np.zeros((length, 3), dtype=np.float32),
        "projected_gravity": np.zeros((length, 3), dtype=np.float32),
        "foot_contacts": np.zeros((length, 4), dtype=np.float32),
        "torque": np.zeros((length, 12), dtype=np.float32),
    }


def test_dataset_writer_preserves_episode_boundaries(tmp_path):
    writer = DatasetWriter(tmp_path, metadata={"obs_dim": 56, "action_dim": 12})
    ep_path = writer.write_episode(0, make_episode(4), accepted=True, stats={"fell": False})
    assert ep_path.name == "ep_000000.npz"
    assert ep_path.exists()
    loaded = np.load(ep_path)
    assert loaded["obs"].shape == (4, 56)
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["obs_dim"] == 56


def test_dataset_writer_accepts_no_contact_observation_episode(tmp_path):
    episode = make_episode(4)
    episode["obs"] = episode["obs"][:, :52]
    writer = DatasetWriter(tmp_path, metadata={"obs_dim": 52, "action_dim": 12})

    ep_path = writer.write_episode(0, episode, accepted=True, stats={"fell": False})

    loaded = np.load(ep_path)
    assert loaded["obs"].shape == (4, 52)


def test_dataset_writer_rejects_wrong_episode_shapes(tmp_path):
    writer = DatasetWriter(tmp_path, metadata={"obs_dim": 56, "action_dim": 12})
    episode = make_episode(4)
    episode["action_label"] = np.zeros((4, 11), dtype=np.float32)

    try:
        writer.write_episode(0, episode, accepted=True, stats={"fell": False})
    except ValueError as exc:
        assert "action_label" in str(exc)
    else:
        raise AssertionError("Expected writer to reject wrong action_label shape")


def test_dataset_writer_rejects_wrong_optional_debug_array_shapes(tmp_path):
    writer = DatasetWriter(tmp_path, metadata={"obs_dim": 56, "action_dim": 12})
    episode = make_episode(4)
    episode["foot_pos"] = np.zeros((4, 3, 3), dtype=np.float32)

    try:
        writer.write_episode(0, episode, accepted=True, stats={"fell": False})
    except ValueError as exc:
        assert "foot_pos" in str(exc)
    else:
        raise AssertionError("Expected writer to reject wrong foot_pos shape")


def test_dataset_writer_rejects_episode_without_initial_reset_flag(tmp_path):
    writer = DatasetWriter(tmp_path, metadata={"obs_dim": 56, "action_dim": 12})
    episode = make_episode(4)
    episode["reset_flag"] = np.zeros((4,), dtype=bool)

    try:
        writer.write_episode(0, episode, accepted=True, stats={"fell": False})
    except ValueError as exc:
        assert "reset_flag[0]" in str(exc)
    else:
        raise AssertionError("Expected writer to reject missing initial reset flag")


def test_dataset_writer_rejects_mid_episode_reset_flags(tmp_path):
    writer = DatasetWriter(tmp_path, metadata={"obs_dim": 56, "action_dim": 12})
    episode = make_episode(4)
    episode["reset_flag"] = np.array([True, False, True, False], dtype=bool)

    try:
        writer.write_episode(0, episode, accepted=True, stats={"fell": False})
    except ValueError as exc:
        assert "mid-episode reset_flag" in str(exc)
    else:
        raise AssertionError("Expected writer to reject mid-episode reset flag")


def test_dataset_writer_rejects_mid_episode_done_flags(tmp_path):
    writer = DatasetWriter(tmp_path, metadata={"obs_dim": 56, "action_dim": 12})
    episode = make_episode(4)
    episode["done"] = np.array([False, True, False, False], dtype=bool)

    try:
        writer.write_episode(0, episode, accepted=True, stats={"fell": False})
    except ValueError as exc:
        assert "mid-episode done" in str(exc)
    else:
        raise AssertionError("Expected writer to reject mid-episode done flag")


def test_dataset_writer_rejects_observation_phase_mismatch(tmp_path):
    writer = DatasetWriter(tmp_path, metadata={"obs_dim": 56, "action_dim": 12})
    episode = make_episode(4)
    episode["phase"] = np.linspace(0.0, 0.6, 4, dtype=np.float32)

    try:
        writer.write_episode(0, episode, accepted=True, stats={"fell": False})
    except ValueError as exc:
        assert "obs phase mismatch" in str(exc)
    else:
        raise AssertionError("Expected writer to reject mismatched observation phase")


def test_dataset_writer_rejects_mismatched_action_label_contract(tmp_path):
    writer = DatasetWriter(tmp_path, metadata={"obs_dim": 56, "action_dim": 12})
    episode = make_episode(4)
    episode["action_label"][:, 1] = 0.5

    try:
        writer.write_episode(0, episode, accepted=True, stats={"fell": False})
    except ValueError as exc:
        assert "action_label mismatch" in str(exc)
    else:
        raise AssertionError("Expected writer to reject mismatched action_label")


def test_dataset_writer_rejects_observation_state_mismatch(tmp_path):
    writer = DatasetWriter(tmp_path, metadata={"obs_dim": 56, "action_dim": 12})
    episode = make_episode(4)
    episode["obs"][:, 12] = 0.25

    try:
        writer.write_episode(0, episode, accepted=True, stats={"fell": False})
    except ValueError as exc:
        assert "obs state mismatch" in str(exc)
    else:
        raise AssertionError("Expected writer to reject mismatched observation state")


def test_dataset_writer_resume_uses_next_episode_id_and_existing_steps(tmp_path):
    writer = DatasetWriter(tmp_path, metadata={"obs_dim": 56, "action_dim": 12})
    writer.write_episode(0, make_episode(4), accepted=True, stats={"fell": False, "survival_steps": 4})

    resumed = DatasetWriter(tmp_path, metadata={"obs_dim": 56, "action_dim": 12}, resume=True)

    assert resumed.next_episode_id == 1
    assert resumed.accepted_steps == 4


def test_dataset_writer_resume_preserves_existing_metadata_config(tmp_path):
    writer = DatasetWriter(
        tmp_path,
        metadata={"obs_dim": 56, "action_dim": 12, "teacher_profile": "strict_walk"},
    )
    writer.write_episode(0, make_episode(4), accepted=True, stats={"fell": False, "survival_steps": 4})

    resumed = DatasetWriter(
        tmp_path,
        metadata={"obs_dim": 52, "action_dim": 12, "teacher_profile": "turn_walk", "new_key": "added"},
        resume=True,
    )

    assert resumed.metadata["obs_dim"] == 56
    assert resumed.metadata["teacher_profile"] == "strict_walk"
    assert resumed.metadata["new_key"] == "added"


def test_dataset_writer_records_rejection_counts_and_recent_summaries(tmp_path):
    writer = DatasetWriter(tmp_path, metadata={"obs_dim": 56, "action_dim": 12})

    writer.record_rejection(
        3,
        {
            "reject_reason": "foot_sliding",
            "survival_steps": 320,
            "forward_progress": 2.0,
            "contact_slip": {"mean": 0.4, "p95": 1.3},
        },
    )
    writer.record_rejection(4, {"reject_reason": "foot_sliding", "survival_steps": 280})
    writer.record_rejection(5, {"reject_reason": "base_height", "survival_steps": 120})

    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["rejection_counts"] == {"base_height": 1, "foot_sliding": 2}
    assert metadata["recent_rejections"][0]["attempt_id"] == 3
    assert metadata["recent_rejections"][0]["reject_reason"] == "foot_sliding"
    assert metadata["recent_rejections"][0]["contact_slip_mean"] == 0.4


def test_initial_recording_counters_preserve_attempts_on_resume(tmp_path):
    writer = DatasetWriter(tmp_path, metadata={"obs_dim": 56, "action_dim": 12})
    writer.write_episode(0, make_episode(4), accepted=True, stats={"fell": False, "survival_steps": 4})
    writer.record_rejection(1, {"reject_reason": "foot_sliding", "survival_steps": 320})
    writer.update_metadata(
        accepted_episodes=1,
        attempted_episodes=2,
        accepted_steps=4,
        saved_review_gifs=1,
    )
    resumed = DatasetWriter(tmp_path, metadata={"obs_dim": 56, "action_dim": 12}, resume=True)

    counters = initial_recording_counters(resumed, tmp_path, resume=True)

    assert counters["accepted_steps"] == 4
    assert counters["accepted_eps"] == 1
    assert counters["attempted_eps"] == 2
    assert counters["saved_review_gifs"] == 1


def test_expand_frames_for_playback_duplicates_sparse_frames_for_realtime_gif():
    frames = [
        np.full((2, 2, 3), 1, dtype=np.uint8),
        np.full((2, 2, 3), 2, dtype=np.uint8),
    ]

    expanded = expand_frames_for_playback(frames, policy_dt=0.02, render_every=5, gif_fps=60)

    assert len(expanded) == 12
    assert [int(frame[0, 0, 0]) for frame in expanded[:6]] == [1] * 6
    assert [int(frame[0, 0, 0]) for frame in expanded[6:]] == [2] * 6


def test_dataset_writer_writes_video_when_backend_available(tmp_path):
    writer = DatasetWriter(tmp_path, metadata={"obs_dim": 56, "action_dim": 12})
    frames = [
        np.zeros((16, 16, 3), dtype=np.uint8),
        np.full((16, 16, 3), 255, dtype=np.uint8),
    ]

    path = writer.write_video(0, frames, accepted=True, stats={"survival_steps": 2}, fps=10)
    if path is None:
        pytest.skip("MP4 backend not available")

    assert path.exists()
    assert path.suffix == ".mp4"
    assert path.with_suffix(".json").exists()


def test_render_q_teacher_episode_replays_targets_before_rendering():
    class FakeEnv:
        def __init__(self):
            self.targets = []

        def reset(self, seed=None):
            self.targets.clear()

        def step_q_des(self, q_des):
            self.targets.append(np.asarray(q_des, dtype=np.float32).copy())
            return 0.0, False, {}

        def render_frame(self, width, height):
            return np.full((height, width, 3), len(self.targets), dtype=np.uint8)

    env = FakeEnv()
    q_teacher = np.zeros((5, 12), dtype=np.float32)

    frames = render_q_teacher_episode(env, q_teacher, render_every=2, gif_width=3, gif_height=2)

    assert len(frames) == 3
    assert [int(frame[0, 0, 0]) for frame in frames] == [1, 3, 5]
    assert len(env.targets) == 5


def test_rollout_episode_observation_phase_matches_recorded_phase():
    class FakeEnv:
        def __init__(self):
            self.policy_dt = 0.02
            self.step = 0

        def reset(self, seed=None):
            self.step = 0

        def get_state(self):
            return {
                "base_pos": np.array([0.01 * self.step, 0.0, 0.3], dtype=np.float32),
                "base_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                "base_lin_vel_body": np.zeros(3, dtype=np.float32),
                "base_ang_vel_body": np.zeros(3, dtype=np.float32),
                "projected_gravity": np.array([0.0, 0.0, -1.0], dtype=np.float32),
                "foot_contacts": np.zeros(4, dtype=np.float32),
                "foot_pos": np.zeros((4, 3), dtype=np.float32),
                "torque": np.zeros(12, dtype=np.float32),
            }

        def make_obs(self, command, prev_action, prev_reward, reset_flag, phase):
            obs = np.zeros(56, dtype=np.float32)
            obs[48] = np.sin(phase)
            obs[49] = np.cos(phase)
            return obs

        def get_q_qdot(self):
            return np.zeros(12, dtype=np.float32), np.zeros(12, dtype=np.float32)

        def step_q_des(self, q_des):
            self.step += 1
            return 1.0, False, {"done_reason": ""}

    class PhaseTeacher:
        def __init__(self):
            self.phase = 0.0

        def reset(self, rng):
            self.phase = 0.0

        def compute(self, state, command):
            self.phase += 0.4
            return {
                "q_teacher": np.zeros(12, dtype=np.float32),
                "phase": self.phase,
                "extra": {},
            }

        def action_label(self, q_teacher):
            return np.zeros(12, dtype=np.float32)

    episode, _, _ = rollout_episode(
        FakeEnv(),
        PhaseTeacher(),
        np.random.default_rng(0),
        episode_steps=3,
        render=False,
        debug_failed_gifs=False,
        fixed_command=np.array([0.3, 0.0, 0.0], dtype=np.float32),
    )

    np.testing.assert_allclose(episode["obs"][:, 48], np.sin(episode["phase"]), atol=1e-6)
    np.testing.assert_allclose(episode["obs"][:, 49], np.cos(episode["phase"]), atol=1e-6)


def test_rollout_episode_records_debug_state_from_observation_timestep():
    class FakeEnv:
        def __init__(self):
            self.policy_dt = 0.02
            self.step = 0

        def reset(self, seed=None):
            self.step = 0

        def get_state(self):
            value = float(self.step)
            return {
                "base_pos": np.array([value, 0.0, 0.3], dtype=np.float32),
                "base_quat": np.array([1.0, 0.0, 0.0, value], dtype=np.float32),
                "base_lin_vel_body": np.array([value, 0.0, 0.0], dtype=np.float32),
                "base_ang_vel_body": np.array([0.0, value, 0.0], dtype=np.float32),
                "projected_gravity": np.array([0.0, 0.0, -1.0 + value], dtype=np.float32),
                "foot_contacts": np.full(4, value, dtype=np.float32),
                "foot_pos": np.full((4, 3), value, dtype=np.float32),
                "torque": np.full(12, 10.0 + value, dtype=np.float32),
            }

        def make_obs(self, command, prev_action, prev_reward, reset_flag, phase):
            return np.zeros(56, dtype=np.float32)

        def get_q_qdot(self):
            return np.zeros(12, dtype=np.float32), np.zeros(12, dtype=np.float32)

        def step_q_des(self, q_des):
            self.step += 1
            return 1.0, False, {"done_reason": ""}

    class FakeTeacher:
        def reset(self, rng):
            pass

        def compute(self, state, command):
            return {"q_teacher": Q_HOME.copy(), "phase": 0.0, "extra": {}}

        def action_label(self, q_teacher):
            return np.zeros(12, dtype=np.float32)

    episode, _, _ = rollout_episode(
        FakeEnv(),
        FakeTeacher(),
        np.random.default_rng(0),
        episode_steps=3,
        render=False,
        debug_failed_gifs=False,
        fixed_command=np.array([0.3, 0.0, 0.0], dtype=np.float32),
    )

    np.testing.assert_allclose(episode["base_pos"][:, 0], np.array([0.0, 1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(episode["foot_contacts"][:, 0], np.array([0.0, 1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(episode["torque"][:, 0], np.array([11.0, 12.0, 13.0], dtype=np.float32))


def test_write_debug_exports_can_write_gif_and_video_from_one_replay():
    class FakeEnv:
        policy_dt = 0.02

        def __init__(self):
            self.render_calls = 0

        def reset(self, seed=None):
            pass

        def step_q_des(self, q_des):
            return 0.0, False, {}

        def render_frame(self, width, height):
            self.render_calls += 1
            return np.full((height, width, 3), self.render_calls, dtype=np.uint8)

    class FakeWriter:
        def __init__(self):
            self.gif_calls = []
            self.video_calls = []

        def write_gif(self, ep_id, frames, accepted, stats, fps):
            self.gif_calls.append((ep_id, len(frames), accepted, stats, fps))
            return "gif"

        def write_video(self, ep_id, frames, accepted, stats, fps):
            self.video_calls.append((ep_id, len(frames), accepted, stats, fps))
            return "video"

    env = FakeEnv()
    writer = FakeWriter()
    episode = {"q_teacher": np.zeros((5, 12), dtype=np.float32)}
    stats = {"survival_steps": 5}

    wrote = write_debug_exports(
        writer=writer,
        env=env,
        ep_id=7,
        episode=episode,
        stats=stats,
        accepted=True,
        render_every=2,
        gif_width=3,
        gif_height=2,
        gif_fps=60,
        save_gif=True,
        save_video=True,
        video_fps=30,
    )

    assert wrote == {"gif": True, "video": True}
    assert env.render_calls == 3
    assert writer.gif_calls == [(7, 6, True, stats, 60)]
    assert writer.video_calls == [(7, 3, True, stats, 30)]


def test_rollout_episode_switches_commands_relative_to_previous_switch(monkeypatch):
    class FakeRng:
        def __init__(self):
            self.values = iter([123, 100, 150, 150])

        def integers(self, low, high=None):
            return next(self.values)

    class FakeTeacher:
        def reset(self, rng):
            pass

        def compute(self, state, command):
            return {"q_teacher": np.zeros(12, dtype=np.float32), "phase": 0.0, "extra": {}}

        def action_label(self, q_teacher):
            return np.zeros(12, dtype=np.float32)

    class FakeEnv:
        def reset(self, seed=None):
            pass

        def get_state(self):
            return {
                "base_pos": np.array([0.0, 0.0, 0.25], dtype=np.float32),
                "base_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                "base_lin_vel_body": np.zeros(3, dtype=np.float32),
                "base_ang_vel_body": np.zeros(3, dtype=np.float32),
                "projected_gravity": np.array([0.0, 0.0, -1.0], dtype=np.float32),
                "foot_contacts": np.zeros(4, dtype=np.float32),
                "foot_pos": np.zeros((4, 3), dtype=np.float32),
                "torque": np.zeros(12, dtype=np.float32),
            }

        def make_obs(self, command, prev_action, prev_reward, reset_flag, phase):
            return np.zeros(56, dtype=np.float32)

        def get_q_qdot(self):
            return np.zeros(12, dtype=np.float32), np.zeros(12, dtype=np.float32)

        def step_q_des(self, q_des):
            return 0.0, False, {}

    commands = [
        np.array([0.1, 0.0, 0.0], dtype=np.float32),
        np.array([0.2, 0.0, 0.0], dtype=np.float32),
        np.array([0.3, 0.0, 0.0], dtype=np.float32),
    ]
    command_iter = iter(commands)

    def fake_sample_command(rng, profile="default"):
        return next(command_iter).copy(), "sampled"

    monkeypatch.setattr("data.record_teacher_demos.sample_command", fake_sample_command)

    episode, _frames, _meta = rollout_episode(
        env=FakeEnv(),
        teacher=FakeTeacher(),
        rng=FakeRng(),
        episode_steps=260,
        render=False,
        debug_failed_gifs=False,
    )

    np.testing.assert_allclose(episode["command"][0:100], np.tile(commands[0], (100, 1)))
    np.testing.assert_allclose(episode["command"][100:250], np.tile(commands[1], (150, 1)))
    np.testing.assert_allclose(episode["command"][250:260], np.tile(commands[2], (10, 1)))


def test_inspect_dataset_reports_fall_reject_rate(tmp_path):
    writer = DatasetWriter(tmp_path, metadata={"obs_dim": 56, "action_dim": 12})
    writer.write_episode(0, make_episode(4), accepted=True, stats={"fell": False, "survival_steps": 4})
    writer.metadata["episodes"].append(
        {
            "episode_id": 1,
            "accepted": False,
            "path": "episodes/ep_000001.npz",
            "stats": {"fell": True, "survival_steps": 2},
        }
    )
    writer.update_metadata()

    report = inspect_dataset(tmp_path)

    assert "fall/reject count: 1" in report
    assert "fall/reject rate: 50.000%" in report


def test_inspect_dataset_uses_attempted_episode_counters_for_rejects(tmp_path):
    writer = DatasetWriter(tmp_path, metadata={"obs_dim": 56, "action_dim": 12})
    writer.write_episode(0, make_episode(4), accepted=True, stats={"fell": False, "survival_steps": 4})
    writer.write_episode(1, make_episode(4), accepted=True, stats={"fell": False, "survival_steps": 4})
    writer.update_metadata(attempted_episodes=4, accepted_episodes=2)

    report = inspect_dataset(tmp_path)

    assert "fall/reject count: 2" in report
    assert "fall/reject rate: 50.000%" in report


def test_inspect_dataset_reports_contact_slip_stats_when_available(tmp_path):
    writer = DatasetWriter(tmp_path, metadata={"obs_dim": 56, "action_dim": 12})
    writer.write_episode(
        0,
        make_episode(4),
        accepted=True,
        stats={"fell": False, "survival_steps": 4, "contact_slip": {"mean": 0.12, "p95": 0.34}},
    )

    report = inspect_dataset(tmp_path)

    assert "contact slip mean: 0.1200" in report
    assert "contact slip p95 mean: 0.3400" in report


def test_inspect_dataset_reports_yaw_stats_when_available(tmp_path):
    writer = DatasetWriter(tmp_path, metadata={"obs_dim": 56, "action_dim": 12})
    writer.write_episode(
        0,
        make_episode(4),
        accepted=True,
        stats={"fell": False, "survival_steps": 4, "yaw_delta": -0.8, "mean_yaw_rate": -0.1},
    )
    writer.write_episode(
        1,
        make_episode(4),
        accepted=True,
        stats={"fell": False, "survival_steps": 4, "yaw_delta": 1.2, "mean_yaw_rate": 0.15},
    )

    report = inspect_dataset(tmp_path)

    assert "yaw delta mean/abs_mean/max_abs: 0.2000/1.0000/1.2000" in report
    assert "mean yaw rate mean/abs_mean: 0.0250/0.1250" in report


def test_inspect_dataset_reports_observation_phase_error(tmp_path):
    writer = DatasetWriter(tmp_path, metadata={"obs_dim": 56, "action_dim": 12})
    writer.write_episode(0, make_episode(4), accepted=True, stats={"fell": False, "survival_steps": 4})

    report = inspect_dataset(tmp_path)

    assert "obs phase max error: 0.000000" in report


def test_inspect_dataset_reports_torque_abs_mean(tmp_path):
    episode = make_episode(4)
    episode["torque"] = np.full((4, 12), 2.5, dtype=np.float32)
    writer = DatasetWriter(tmp_path, metadata={"obs_dim": 56, "action_dim": 12})
    writer.write_episode(0, episode, accepted=True, stats={"fell": False, "survival_steps": 4})

    report = inspect_dataset(tmp_path)

    assert "torque abs mean: 2.5000" in report


def test_inspect_dataset_reports_rejection_reason_counts(tmp_path):
    writer = DatasetWriter(tmp_path, metadata={"obs_dim": 56, "action_dim": 12})
    writer.write_episode(0, make_episode(4), accepted=True, stats={"fell": False, "survival_steps": 4})
    writer.record_rejection(1, {"reject_reason": "foot_sliding", "survival_steps": 320})
    writer.record_rejection(2, {"reject_reason": "base_height", "survival_steps": 80})
    writer.update_metadata(attempted_episodes=3, accepted_episodes=1)

    report = inspect_dataset(tmp_path)

    assert "reject reasons: base_height=1, foot_sliding=1" in report


def test_inspect_dataset_reports_rejections_when_no_episodes_were_accepted(tmp_path):
    writer = DatasetWriter(tmp_path, metadata={"obs_dim": 56, "action_dim": 12})
    writer.record_rejection(1, {"reject_reason": "foot_sliding", "survival_steps": 320})
    writer.record_rejection(2, {"reject_reason": "foot_sliding", "survival_steps": 280})
    writer.update_metadata(attempted_episodes=2, accepted_episodes=0)

    report = inspect_dataset(tmp_path)

    assert "number of episodes: 0" in report
    assert "fall/reject count: 2" in report
    assert "fall/reject rate: 100.000%" in report
    assert "reject reasons: foot_sliding=2" in report
