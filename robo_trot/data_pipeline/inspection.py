from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from robo_trot.data_pipeline.dataset_writer import obs_phase_max_error


def load_episode_paths(dataset_dir: Path) -> list[Path]:
    """Return sorted episode NPZ paths from an unsharded dataset directory.

    Callers rely on the returned value shape and semantics described here.
    """
    return sorted((dataset_dir / "episodes").glob("ep_*.npz"))


def inspect_sharded_dataset(dataset_dir: Path) -> str:
    """Format a summary for a sharded dataset manifest.

    This documents the callable contract used by the surrounding pipeline.
    """
    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text())
    category_steps = manifest.get("category_steps", {})
    category_text = ", ".join(f"{category}={int(steps)}" for category, steps in sorted(category_steps.items()))
    shards = manifest.get("shards", [])
    attempted = sum(int(shard.get("attempted_episodes", 0)) for shard in shards)
    accepted = sum(int(shard.get("accepted_episodes", 0)) for shard in shards)
    rejection_counts = manifest.get("rejection_counts", {})
    reject_count = sum(int(count) for count in rejection_counts.values())
    if attempted:
        reject_count = max(reject_count, attempted - accepted)
    reject_rate = 100.0 * float(reject_count / attempted) if attempted > 0 else 0.0
    quality = manifest.get("quality", {})
    lines = [
        f"dataset: {dataset_dir}",
        "sharded dataset: True",
        f"shard count: {int(manifest.get('shard_count', len(shards)))}",
        f"total transitions: {int(manifest.get('accepted_steps', 0))}",
        f"target transitions: {int(manifest.get('target_steps', 0))}",
        f"number of episodes: {int(manifest.get('accepted_episodes', accepted))}",
        f"category transitions: {category_text}",
        f"fall/reject count: {reject_count}",
        f"fall/reject rate: {reject_rate:.3f}%",
    ]
    if quality:
        lines.extend(
            [
                f"clip fraction mean: {float(quality.get('clip_fraction_mean', 0.0)):.6f}",
                f"contact slip mean: {float(quality.get('contact_slip_mean', 0.0)):.4f}",
                f"contact slip p95 mean: {float(quality.get('contact_slip_p95_mean', 0.0)):.4f}",
                f"yaw delta abs mean: {float(quality.get('yaw_delta_abs_mean', 0.0)):.4f}",
                f"yaw rate abs mean: {float(quality.get('yaw_rate_abs_mean', 0.0)):.4f}",
            ]
        )
    if rejection_counts:
        reason_text = ", ".join(f"{reason}={int(count)}" for reason, count in sorted(rejection_counts.items()))
        lines.append(f"reject reasons: {reason_text}")
    return "\n".join(lines)


def inspect_dataset(dataset_dir: Path) -> str:
    """Format dataset statistics for either sharded or single-directory datasets.

    This documents the callable contract used by the surrounding pipeline.
    """
    if (dataset_dir / "dataset_manifest.json").exists():
        return inspect_sharded_dataset(dataset_dir)
    paths = load_episode_paths(dataset_dir)
    metadata_path = dataset_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if not paths:
        attempted_count = int(metadata.get("attempted_episodes", 0))
        accepted_count = int(metadata.get("accepted_episodes", 0))
        rejection_counts = metadata.get("rejection_counts", {})
        reject_count = sum(int(count) for count in rejection_counts.values())
        if attempted_count:
            reject_count = max(reject_count, attempted_count - accepted_count)
        denominator = attempted_count or reject_count
        reject_rate = 100.0 * float(reject_count / denominator) if denominator > 0 else 0.0
        lines = [
            f"dataset: {dataset_dir}",
            "number of episodes: 0",
            "total transitions: 0",
            f"fall/reject count: {reject_count}",
            f"fall/reject rate: {reject_rate:.3f}%",
        ]
        if rejection_counts:
            reason_text = ", ".join(f"{reason}={int(count)}" for reason, count in sorted(rejection_counts.items()))
            lines.append(f"reject reasons: {reason_text}")
        return "\n".join(lines)
    lengths = []
    rewards = []
    actions = []
    commands = []
    torques = []
    phase_errors = []
    reject_count = 0
    for path in paths:
        data = np.load(path)
        lengths.append(int(data["obs"].shape[0]))
        rewards.append(data["reward"])
        actions.append(data["action_label"])
        commands.append(data["command"])
        torques.append(data["torque"])
        phase_errors.append(obs_phase_max_error(data["obs"], data["phase"]))
    action_all = np.concatenate(actions, axis=0)
    command_all = np.concatenate(commands, axis=0)
    torque_all = np.concatenate(torques, axis=0)
    reward_all = np.concatenate(rewards, axis=0)
    for entry in metadata.get("episodes", []):
        if not entry.get("accepted", True):
            reject_count += 1
    attempted_count = int(metadata.get("attempted_episodes", 0))
    accepted_count = int(metadata.get("accepted_episodes", 0))
    if attempted_count and accepted_count:
        reject_count = max(reject_count, attempted_count - accepted_count)
    denominator = attempted_count or len(metadata.get("episodes", [])) or len(paths)
    reject_rate = 100.0 * float(reject_count / denominator) if denominator > 0 else 0.0
    lengths_arr = np.asarray(lengths, dtype=np.int32)
    lines = [
        f"dataset: {dataset_dir}",
        f"number of episodes: {len(paths)}",
        f"total transitions: {int(lengths_arr.sum())}",
        f"obs_dim: {int(np.load(paths[0])['obs'].shape[1])}",
        f"action_dim: {int(action_all.shape[1])}",
        f"action mean: {np.round(action_all.mean(axis=0), 4).tolist()}",
        f"action std: {np.round(action_all.std(axis=0), 4).tolist()}",
        f"clipped action labels: {100.0 * float(np.mean(np.abs(action_all) >= 0.999)):.3f}%",
        f"episode length min/mean/max: {int(lengths_arr.min())}/{float(lengths_arr.mean()):.1f}/{int(lengths_arr.max())}",
        f"mean reward: {float(reward_all.mean()):.4f}",
        f"fall/reject count: {reject_count}",
        f"fall/reject rate: {reject_rate:.3f}%",
        f"command mean: {np.round(command_all.mean(axis=0), 4).tolist()}",
        f"command min: {np.round(command_all.min(axis=0), 4).tolist()}",
        f"command max: {np.round(command_all.max(axis=0), 4).tolist()}",
        f"obs phase max error: {float(np.max(phase_errors)):.6f}",
        f"torque abs mean: {float(np.mean(np.abs(torque_all))):.4f}",
    ]
    contact_slip = [
        entry.get("stats", {}).get("contact_slip", {})
        for entry in metadata.get("episodes", [])
        if entry.get("stats", {}).get("contact_slip")
    ]
    if contact_slip:
        slip_mean = np.asarray([item.get("mean", 0.0) for item in contact_slip], dtype=np.float32)
        slip_p95 = np.asarray([item.get("p95", 0.0) for item in contact_slip], dtype=np.float32)
        lines.extend(
            [
                f"contact slip mean: {float(slip_mean.mean()):.4f}",
                f"contact slip p95 mean: {float(slip_p95.mean()):.4f}",
            ]
        )
    yaw_stats = [
        entry.get("stats", {})
        for entry in metadata.get("episodes", [])
        if "yaw_delta" in entry.get("stats", {}) and "mean_yaw_rate" in entry.get("stats", {})
    ]
    if yaw_stats:
        yaw_delta = np.asarray([item.get("yaw_delta", 0.0) for item in yaw_stats], dtype=np.float32)
        mean_yaw_rate = np.asarray([item.get("mean_yaw_rate", 0.0) for item in yaw_stats], dtype=np.float32)
        lines.extend(
            [
                "yaw delta mean/abs_mean/max_abs: "
                f"{float(yaw_delta.mean()):.4f}/{float(np.abs(yaw_delta).mean()):.4f}/{float(np.abs(yaw_delta).max()):.4f}",
                f"mean yaw rate mean/abs_mean: {float(mean_yaw_rate.mean()):.4f}/{float(np.abs(mean_yaw_rate).mean()):.4f}",
            ]
        )
    rejection_counts = metadata.get("rejection_counts", {})
    if rejection_counts:
        reason_text = ", ".join(f"{reason}={int(count)}" for reason, count in sorted(rejection_counts.items()))
        lines.append(f"reject reasons: {reason_text}")
    return "\n".join(lines)


def main() -> None:
    """Run the dataset inspection command-line entry point.

    This is the direct execution entry point for the module.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir")
    args = parser.parse_args()
    print(inspect_dataset(Path(args.dataset_dir)))


if __name__ == "__main__":
    main()
