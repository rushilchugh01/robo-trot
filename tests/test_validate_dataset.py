import numpy as np

from data.dataset_writer import DatasetWriter
from robo_trot.a1_constants import ACTION_SCALE, Q_HOME
from scripts.validate_dataset import validate_dataset
from scripts.inspect_dataset import inspect_dataset
from tests.test_dataset_writer import make_episode


def test_validate_dataset_accepts_matching_teacher_action_labels(tmp_path):
    episode = make_episode(4)
    q_teacher = np.tile(Q_HOME + 0.25 * ACTION_SCALE, (4, 1)).astype(np.float32)
    episode["q_teacher"] = q_teacher
    episode["action_label"] = np.clip((q_teacher - Q_HOME) / ACTION_SCALE, -1.0, 1.0).astype(np.float32)
    episode["obs"][1:, 36:48] = episode["action_label"][:-1]
    writer = DatasetWriter(tmp_path, metadata={"obs_dim": 56, "action_dim": 12})
    writer.write_episode(0, episode, accepted=True, stats={"survival_steps": 4})

    result = validate_dataset(tmp_path)

    assert result.ok
    assert result.episodes == 1
    assert result.transitions == 4


def test_validate_dataset_accepts_no_contact_observation_episode(tmp_path):
    episode = make_episode(4)
    episode["obs"] = episode["obs"][:, :52]
    writer = DatasetWriter(tmp_path, metadata={"obs_dim": 52, "action_dim": 12})
    writer.write_episode(0, episode, accepted=True, stats={"survival_steps": 4})

    result = validate_dataset(tmp_path)

    assert result.ok
    assert result.transitions == 4


def test_validate_dataset_rejects_mismatched_teacher_action_labels(tmp_path):
    episode = make_episode(4)
    episode["q_teacher"] = np.tile(Q_HOME + 0.25 * ACTION_SCALE, (4, 1)).astype(np.float32)
    episode["action_label"] = np.zeros((4, 12), dtype=np.float32)
    writer = DatasetWriter(tmp_path, metadata={"obs_dim": 56, "action_dim": 12})
    path = writer.write_episode(0, make_episode(4), accepted=True, stats={"survival_steps": 4})
    np.savez_compressed(path, **episode)

    result = validate_dataset(tmp_path)

    assert not result.ok
    assert any("action_label mismatch" in message for message in result.errors)


def test_validate_dataset_rejects_metadata_dimension_mismatch(tmp_path):
    episode = make_episode(4)
    episode["q_teacher"] = np.tile(Q_HOME, (4, 1)).astype(np.float32)
    episode["action_label"] = np.zeros((4, 12), dtype=np.float32)
    episodes_dir = tmp_path / "episodes"
    episodes_dir.mkdir()
    (tmp_path / "metadata.json").write_text('{"obs_dim": 52, "action_dim": 11}')
    np.savez_compressed(episodes_dir / "ep_000000.npz", **episode)

    result = validate_dataset(tmp_path)

    assert not result.ok
    assert any("metadata obs_dim" in message for message in result.errors)
    assert any("metadata action_dim" in message for message in result.errors)


def test_validate_dataset_rejects_later_episode_obs_dim_mismatch(tmp_path):
    writer = DatasetWriter(tmp_path, metadata={"obs_dim": 56, "action_dim": 12})
    writer.write_episode(0, make_episode(4), accepted=True, stats={"survival_steps": 4})
    bad_episode = make_episode(4)
    bad_episode["obs"] = bad_episode["obs"][:, :52]
    np.savez_compressed(tmp_path / "episodes" / "ep_000001.npz", **bad_episode)

    result = validate_dataset(tmp_path)

    assert not result.ok
    assert any("obs shape" in message for message in result.errors)


def test_validate_dataset_rejects_optional_array_length_mismatch(tmp_path):
    episode = make_episode(4)
    episode["q_teacher"] = np.tile(Q_HOME, (4, 1)).astype(np.float32)
    episode["action_label"] = np.zeros((4, 12), dtype=np.float32)
    episode["foot_pos"] = np.zeros((3, 4, 3), dtype=np.float32)
    writer = DatasetWriter(tmp_path, metadata={"obs_dim": 56, "action_dim": 12})
    path = writer.write_episode(0, make_episode(4), accepted=True, stats={"survival_steps": 4})
    np.savez_compressed(path, **episode)

    result = validate_dataset(tmp_path)

    assert not result.ok
    assert any("foot_pos length" in message for message in result.errors)


def test_validate_dataset_rejects_optional_array_shape_mismatch(tmp_path):
    episode = make_episode(4)
    episode["q_teacher"] = np.tile(Q_HOME, (4, 1)).astype(np.float32)
    episode["action_label"] = np.zeros((4, 12), dtype=np.float32)
    episode["foot_pos"] = np.zeros((4, 3, 3), dtype=np.float32)
    writer = DatasetWriter(tmp_path, metadata={"obs_dim": 56, "action_dim": 12})
    path = writer.write_episode(0, make_episode(4), accepted=True, stats={"survival_steps": 4})
    np.savez_compressed(path, **episode)

    result = validate_dataset(tmp_path)

    assert not result.ok
    assert any("foot_pos shape" in message for message in result.errors)


def test_validate_dataset_rejects_missing_accepted_episode_file_referenced_by_metadata(tmp_path):
    episode = make_episode(4)
    episode["q_teacher"] = np.tile(Q_HOME, (4, 1)).astype(np.float32)
    episode["action_label"] = np.zeros((4, 12), dtype=np.float32)
    writer = DatasetWriter(tmp_path, metadata={"obs_dim": 56, "action_dim": 12})
    writer.write_episode(0, episode, accepted=True, stats={"survival_steps": 4})
    metadata = writer.metadata
    metadata["episodes"].append(
        {
            "episode_id": 1,
            "accepted": True,
            "path": "episodes/ep_000001.npz",
            "stats": {"survival_steps": 4},
        }
    )
    writer.update_metadata(episodes=metadata["episodes"])

    result = validate_dataset(tmp_path)

    assert not result.ok
    assert any("metadata episode path missing" in message for message in result.errors)


def test_validate_dataset_rejects_episode_file_missing_from_metadata(tmp_path):
    writer = DatasetWriter(tmp_path, metadata={"obs_dim": 56, "action_dim": 12})
    writer.write_episode(0, make_episode(4), accepted=True, stats={"survival_steps": 4})
    np.savez_compressed(tmp_path / "episodes" / "ep_000001.npz", **make_episode(4))

    result = validate_dataset(tmp_path)

    assert not result.ok
    assert any("episode file missing from metadata" in message for message in result.errors)


def test_validate_dataset_rejects_accepted_episode_counter_mismatch(tmp_path):
    episode = make_episode(4)
    episode["q_teacher"] = np.tile(Q_HOME, (4, 1)).astype(np.float32)
    episode["action_label"] = np.zeros((4, 12), dtype=np.float32)
    writer = DatasetWriter(tmp_path, metadata={"obs_dim": 56, "action_dim": 12})
    writer.write_episode(0, episode, accepted=True, stats={"survival_steps": 4})
    writer.update_metadata(accepted_episodes=2)

    result = validate_dataset(tmp_path)

    assert not result.ok
    assert any("metadata accepted_episodes" in message for message in result.errors)


def test_validate_dataset_rejects_accepted_steps_counter_mismatch(tmp_path):
    episode = make_episode(4)
    episode["q_teacher"] = np.tile(Q_HOME, (4, 1)).astype(np.float32)
    episode["action_label"] = np.zeros((4, 12), dtype=np.float32)
    writer = DatasetWriter(tmp_path, metadata={"obs_dim": 56, "action_dim": 12})
    writer.write_episode(0, episode, accepted=True, stats={"survival_steps": 4})
    writer.update_metadata(accepted_steps=8)

    result = validate_dataset(tmp_path)

    assert not result.ok
    assert any("metadata accepted_steps" in message for message in result.errors)


def test_validate_dataset_rejects_episode_without_initial_reset_flag(tmp_path):
    episode = make_episode(4)
    episode["q_teacher"] = np.tile(Q_HOME, (4, 1)).astype(np.float32)
    episode["action_label"] = np.zeros((4, 12), dtype=np.float32)
    episode["reset_flag"] = np.zeros((4,), dtype=bool)
    writer = DatasetWriter(tmp_path, metadata={"obs_dim": 56, "action_dim": 12})
    path = writer.write_episode(0, make_episode(4), accepted=True, stats={"survival_steps": 4})
    np.savez_compressed(path, **episode)

    result = validate_dataset(tmp_path)

    assert not result.ok
    assert any("reset_flag[0]" in message for message in result.errors)


def test_inspect_dataset_summarizes_sharded_manifest(tmp_path):
    (tmp_path / "dataset_manifest.json").write_text(
        """
        {
          "accepted_steps": 19,
          "accepted_episodes": 3,
          "target_steps": 18,
          "category_steps": {"forward": 11, "turn": 8},
          "shard_count": 2,
          "rejection_counts": {"foot_sliding": 1},
          "shards": [
            {
              "name": "shard_00_forward",
              "category": "forward",
              "target_steps": 10,
              "accepted_steps": 11,
              "accepted_episodes": 2,
              "attempted_episodes": 3,
              "rejection_counts": {"foot_sliding": 1}
            },
            {
              "name": "shard_04_turn",
              "category": "turn",
              "target_steps": 8,
              "accepted_steps": 8,
              "accepted_episodes": 1,
              "attempted_episodes": 1,
              "rejection_counts": {}
            }
          ]
        }
        """
    )

    report = inspect_dataset(tmp_path)

    assert "sharded dataset: True" in report
    assert "total transitions: 19" in report
    assert "target transitions: 18" in report
    assert "category transitions: forward=11, turn=8" in report
    assert "fall/reject count: 1" in report


def test_validate_dataset_rejects_mid_episode_reset_flag(tmp_path):
    episode = make_episode(4)
    episode["q_teacher"] = np.tile(Q_HOME, (4, 1)).astype(np.float32)
    episode["action_label"] = np.zeros((4, 12), dtype=np.float32)
    episode["reset_flag"] = np.array([True, False, True, False], dtype=bool)
    writer = DatasetWriter(tmp_path, metadata={"obs_dim": 56, "action_dim": 12})
    path = writer.write_episode(0, make_episode(4), accepted=True, stats={"survival_steps": 4})
    np.savez_compressed(path, **episode)

    result = validate_dataset(tmp_path)

    assert not result.ok
    assert any("mid-episode reset_flag" in message for message in result.errors)


def test_validate_dataset_rejects_mid_episode_done_flag(tmp_path):
    episode = make_episode(4)
    episode["q_teacher"] = np.tile(Q_HOME, (4, 1)).astype(np.float32)
    episode["action_label"] = np.zeros((4, 12), dtype=np.float32)
    episode["done"] = np.array([False, True, False, False], dtype=bool)
    writer = DatasetWriter(tmp_path, metadata={"obs_dim": 56, "action_dim": 12})
    path = writer.write_episode(0, make_episode(4), accepted=True, stats={"survival_steps": 4})
    np.savez_compressed(path, **episode)

    result = validate_dataset(tmp_path)

    assert not result.ok
    assert any("mid-episode done" in message for message in result.errors)


def test_validate_dataset_rejects_observation_phase_mismatch(tmp_path):
    episode = make_episode(4)
    episode["q_teacher"] = np.tile(Q_HOME, (4, 1)).astype(np.float32)
    episode["action_label"] = np.zeros((4, 12), dtype=np.float32)
    episode["phase"] = np.linspace(0.0, 0.6, 4, dtype=np.float32)
    writer = DatasetWriter(tmp_path, metadata={"obs_dim": 56, "action_dim": 12})
    path = writer.write_episode(0, make_episode(4), accepted=True, stats={"survival_steps": 4})
    np.savez_compressed(path, **episode)

    result = validate_dataset(tmp_path)

    assert not result.ok
    assert any("obs phase mismatch" in message for message in result.errors)


def test_validate_dataset_rejects_observation_state_mismatch(tmp_path):
    episode = make_episode(4)
    episode["obs"][:, 12] = 0.25
    writer = DatasetWriter(tmp_path, metadata={"obs_dim": 56, "action_dim": 12})
    path = writer.write_episode(0, make_episode(4), accepted=True, stats={"survival_steps": 4})
    np.savez_compressed(path, **episode)

    result = validate_dataset(tmp_path)

    assert not result.ok
    assert any("obs state mismatch" in message for message in result.errors)
