import json

from data.build_5m_manifest import build_manifest


def _write_fake_shard(root, name, category, target_steps, teacher_profile, episode_steps):
    shard_dir = root / "shards" / name
    episodes_dir = shard_dir / "episodes"
    episodes_dir.mkdir(parents=True)
    episodes = []
    for idx, steps in enumerate(episode_steps):
        path = episodes_dir / f"ep_{idx:06d}.npz"
        path.write_bytes(b"fake")
        episodes.append(
            {
                "episode_id": idx,
                "accepted": True,
                "path": f"episodes/ep_{idx:06d}.npz",
                "stats": {
                    "survival_steps": steps,
                    "clip_fraction": 0.0,
                    "contact_slip": {"mean": 0.12, "p95": 0.5},
                    "yaw_delta": 0.4 if category == "turn" else 0.0,
                    "mean_yaw_rate": 0.05 if category == "turn" else 0.0,
                },
            }
        )
    metadata = {
        "obs_dim": 56,
        "action_dim": 12,
        "teacher": "footspace",
        "teacher_profile": teacher_profile,
        "command_category": category,
        "accepted_steps": sum(episode_steps),
        "accepted_episodes": len(episode_steps),
        "attempted_episodes": len(episode_steps),
        "episodes": episodes,
    }
    (shard_dir / "metadata.json").write_text(json.dumps(metadata))
    return {
        "name": name,
        "category": category,
        "target_steps": target_steps,
        "seed": 1,
    }


def test_build_manifest_reports_shards_categories_and_splits(tmp_path):
    shards = [
        _write_fake_shard(tmp_path, "shard_00_forward", "forward", 10, "strict_walk", [6, 5]),
        _write_fake_shard(tmp_path, "shard_04_turn", "turn", 8, "turn_walk", [8]),
    ]
    (tmp_path / "launcher_config.json").write_text(
        json.dumps({"target_steps": 18, "category_steps": {"forward": 10, "turn": 8}, "shards": shards})
    )

    manifest = build_manifest(tmp_path, create_links=True)

    assert manifest["accepted_steps"] == 19
    assert manifest["target_steps"] == 18
    assert manifest["category_steps"] == {"forward": 11, "turn": 8}
    assert manifest["shard_count"] == 2
    splits = json.loads((tmp_path / "splits.json").read_text())
    assert len(splits["forward"]) == 2
    assert len(splits["turn"]) == 1
    assert (tmp_path / "dataset_manifest.json").exists()
    assert (tmp_path / "merged" / "metadata.json").exists()
    assert (tmp_path / "merged" / "episodes" / "forward" / "shard_00_forward_ep_000000.npz").exists()


def test_build_manifest_rejects_missing_episode_file(tmp_path):
    shard = _write_fake_shard(tmp_path, "shard_00_forward", "forward", 10, "strict_walk", [10])
    (tmp_path / "launcher_config.json").write_text(json.dumps({"target_steps": 10, "shards": [shard]}))
    (tmp_path / "shards" / "shard_00_forward" / "episodes" / "ep_000000.npz").unlink()

    try:
        build_manifest(tmp_path)
    except FileNotFoundError as exc:
        assert "missing episode file" in str(exc)
    else:
        raise AssertionError("build_manifest should reject missing episode files")
