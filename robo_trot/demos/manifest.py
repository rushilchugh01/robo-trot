from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from robo_trot.demos.sharded_generation import SHARDS, category_step_totals, total_target_steps
from robo_trot.demos.record_teacher_demos import CATEGORY_COMMAND_RANGES


def load_shard_config(dataset_dir: Path) -> dict[str, Any]:
    """Load launcher shard configuration or synthesize the default 5M layout."""
    config_path = dataset_dir / "launcher_config.json"
    if config_path.exists():
        return json.loads(config_path.read_text())
    return {
        "target_steps": total_target_steps(SHARDS),
        "category_steps": category_step_totals(SHARDS),
        "shards": SHARDS,
    }


def _require(condition: bool, message: str) -> None:
    """Raise a manifest validation error when a required condition is false."""
    if not condition:
        raise ValueError(message)


def _read_metadata(path: Path) -> dict[str, Any]:
    """Read a shard metadata JSON file, requiring that it exists."""
    if not path.exists():
        raise FileNotFoundError(f"missing shard metadata: {path}")
    return json.loads(path.read_text())


def _episode_steps(entry: dict[str, Any]) -> int:
    """Extract accepted transition count from an episode metadata entry."""
    return int(entry.get("stats", {}).get("survival_steps", 0))


def _validate_shard(shard: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Validate one shard's metadata against the launcher configuration."""
    category = str(shard["category"])
    expected_profile = str(CATEGORY_COMMAND_RANGES[category]["teacher_profile"])
    _require(int(metadata.get("obs_dim", 0)) == 56, f"{shard['name']} obs_dim must be 56")
    _require(metadata.get("teacher") == "footspace", f"{shard['name']} teacher must be footspace")
    _require(
        metadata.get("teacher_profile") == expected_profile,
        f"{shard['name']} teacher_profile must be {expected_profile}",
    )
    _require(
        metadata.get("command_category") == category,
        f"{shard['name']} command_category must be {category}",
    )
    _require(
        int(metadata.get("accepted_steps", 0)) >= int(shard["target_steps"]),
        f"{shard['name']} accepted_steps below target",
    )
    for entry in metadata.get("episodes", []):
        if not entry.get("accepted", False):
            continue
        stats = entry.get("stats", {})
        _require("clip_fraction" in stats, f"{shard['name']} episode missing clip_fraction")
        _require("contact_slip" in stats, f"{shard['name']} episode missing contact_slip")


def _create_episode_link(source: Path, target: Path) -> None:
    """Create or replace a merged-view symlink for one episode file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    os.symlink(source.resolve(), target)


def build_manifest(dataset_dir: str | Path, create_links: bool = False) -> dict[str, Any]:
    """Build sharded dataset manifest, splits, and optional merged episode links."""
    dataset_dir = Path(dataset_dir)
    config = load_shard_config(dataset_dir)
    shards = list(config.get("shards", []))
    splits: dict[str, list[dict[str, Any]]] = {}
    shard_summaries: list[dict[str, Any]] = []
    category_steps: dict[str, int] = {}
    rejection_counts: dict[str, int] = {}
    clip_fractions: list[float] = []
    contact_slip_means: list[float] = []
    contact_slip_p95s: list[float] = []
    yaw_deltas: list[float] = []
    yaw_rates: list[float] = []
    accepted_steps = 0
    accepted_episodes = 0

    for shard in shards:
        shard_name = str(shard["name"])
        category = str(shard["category"])
        shard_dir = dataset_dir / "shards" / shard_name
        metadata = _read_metadata(shard_dir / "metadata.json")
        _validate_shard(shard, metadata)

        shard_steps = int(metadata.get("accepted_steps", 0))
        shard_episodes = 0
        for reason, count in metadata.get("rejection_counts", {}).items():
            rejection_counts[str(reason)] = rejection_counts.get(str(reason), 0) + int(count)

        for entry in metadata.get("episodes", []):
            if not entry.get("accepted", False):
                continue
            relative_episode_path = str(entry["path"])
            episode_path = shard_dir / relative_episode_path
            if not episode_path.exists():
                raise FileNotFoundError(f"missing episode file: {episode_path}")
            steps = _episode_steps(entry)
            item = {
                "category": category,
                "shard": shard_name,
                "episode_id": int(entry.get("episode_id", shard_episodes)),
                "steps": steps,
                "path": str(episode_path.relative_to(dataset_dir)),
            }
            if create_links:
                link_path = (
                    dataset_dir
                    / "merged"
                    / "episodes"
                    / category
                    / f"{shard_name}_ep_{int(entry.get('episode_id', shard_episodes)):06d}.npz"
                )
                _create_episode_link(episode_path, link_path)
                item["merged_path"] = str(link_path.relative_to(dataset_dir))
            splits.setdefault(category, []).append(item)
            stats = entry.get("stats", {})
            clip_fractions.append(float(stats.get("clip_fraction", 0.0)))
            contact_slip = stats.get("contact_slip", {})
            if isinstance(contact_slip, dict):
                contact_slip_means.append(float(contact_slip.get("mean", 0.0)))
                contact_slip_p95s.append(float(contact_slip.get("p95", 0.0)))
            if "yaw_delta" in stats:
                yaw_deltas.append(float(stats.get("yaw_delta", 0.0)))
            if "mean_yaw_rate" in stats:
                yaw_rates.append(float(stats.get("mean_yaw_rate", 0.0)))
            shard_episodes += 1

        accepted_steps += shard_steps
        accepted_episodes += shard_episodes
        category_steps[category] = category_steps.get(category, 0) + sum(item["steps"] for item in splits.get(category, []) if item["shard"] == shard_name)
        shard_summaries.append(
            {
                "name": shard_name,
                "category": category,
                "target_steps": int(shard["target_steps"]),
                "accepted_steps": shard_steps,
                "accepted_episodes": shard_episodes,
                "attempted_episodes": int(metadata.get("attempted_episodes", shard_episodes)),
                "rejection_counts": metadata.get("rejection_counts", {}),
            }
        )

    manifest = {
        "dataset_dir": str(dataset_dir),
        "target_steps": int(config.get("target_steps", total_target_steps(shards))),
        "target_category_steps": config.get("category_steps", category_step_totals(shards)),
        "accepted_steps": accepted_steps,
        "accepted_episodes": accepted_episodes,
        "category_steps": category_steps,
        "shard_count": len(shards),
        "shards": shard_summaries,
        "rejection_counts": rejection_counts,
        "quality": {
            "clip_fraction_mean": sum(clip_fractions) / len(clip_fractions) if clip_fractions else 0.0,
            "contact_slip_mean": sum(contact_slip_means) / len(contact_slip_means) if contact_slip_means else 0.0,
            "contact_slip_p95_mean": sum(contact_slip_p95s) / len(contact_slip_p95s) if contact_slip_p95s else 0.0,
            "yaw_delta_abs_mean": sum(abs(value) for value in yaw_deltas) / len(yaw_deltas) if yaw_deltas else 0.0,
            "yaw_rate_abs_mean": sum(abs(value) for value in yaw_rates) / len(yaw_rates) if yaw_rates else 0.0,
        },
    }
    (dataset_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (dataset_dir / "splits.json").write_text(json.dumps(splits, indent=2, sort_keys=True))
    merged_dir = dataset_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    (merged_dir / "metadata.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for manifest generation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir")
    parser.add_argument("--no_links", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the manifest generation command-line entry point."""
    args = parse_args()
    manifest = build_manifest(args.dataset_dir, create_links=not args.no_links)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
