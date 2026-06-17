from pathlib import Path

from data.launch_5m_shards import (
    SHARDS,
    build_shard_command,
    category_step_totals,
    scaled_shards,
    total_target_steps,
)


def test_5m_shard_config_sums_to_requested_category_totals():
    assert total_target_steps(SHARDS) == 5_000_000
    assert category_step_totals(SHARDS) == {
        "forward": 3_000_000,
        "turn": 1_000_000,
        "slow": 750_000,
        "fast_probe": 250_000,
    }


def test_scaled_shards_keep_every_shard_nonzero_for_dry_runs():
    shards = scaled_shards(SHARDS, scale=0.001)

    assert total_target_steps(shards) == 5_000
    assert min(int(shard["target_steps"]) for shard in shards) > 0


def test_build_shard_command_uses_category_profile_and_disables_media():
    shard = {
        "name": "shard_04_turn",
        "category": "turn",
        "target_steps": 500_000,
        "seed": 5200,
    }

    command = build_shard_command(
        shard,
        out_dir=Path("datasets/a1_teacher_flat_5m_v001"),
        xml_path="assets/mujoco_menagerie/unitree_a1/scene.xml",
        resume=True,
    )

    assert command[:2] == ["python", "data/record_teacher_demos.py"]
    assert "--out_dir" in command
    assert "datasets/a1_teacher_flat_5m_v001/shards/shard_04_turn" in command
    assert command[command.index("--target_steps") + 1] == "500000"
    assert command[command.index("--teacher_profile") + 1] == "turn_walk"
    assert command[command.index("--command_category") + 1] == "turn"
    assert command[command.index("--gif_every") + 1] == "0"
    assert command[command.index("--review_gifs") + 1] == "0"
    assert "--resume" in command
    assert "--debug_failed_gifs" not in command
    assert "--save_videos" not in command
