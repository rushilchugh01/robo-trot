from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from robo_trot.data_pipeline.dataset_writer import OPTIONAL_ARRAY_TAIL_SHAPES, REQUIRED_ARRAYS, obs_phase_max_error, obs_state_max_error
from robo_trot.robot.a1 import ACTION_SCALE, Q_HOME

@dataclass
class ValidationResult:
    """Summary of structural dataset validation."""

    ok: bool
    episodes: int
    transitions: int
    errors: list[str] = field(default_factory=list)

    def report(self) -> str:
        """Format validation results for command-line output."""
        lines = [
            f"valid: {self.ok}",
            f"episodes: {self.episodes}",
            f"transitions: {self.transitions}",
            f"errors: {len(self.errors)}",
        ]
        lines.extend(f"- {error}" for error in self.errors)
        return "\n".join(lines)


def _episode_paths(dataset_dir: Path) -> list[Path]:
    """Return sorted episode NPZ paths from a single shard or dataset."""
    return sorted((dataset_dir / "episodes").glob("ep_*.npz"))


def _validate_episode(path: Path, expected_obs_dim: int | None = None, expected_action_dim: int | None = None) -> tuple[int, list[str]]:
    """Validate one episode file and return its length plus errors."""
    errors: list[str] = []
    with np.load(path) as data:
        keys = set(data.files)
        missing = [key for key in REQUIRED_ARRAYS if key not in keys]
        if missing:
            return 0, [f"{path.name}: missing arrays {missing}"]
        obs = data["obs"]
        if obs.ndim != 2:
            return 0, [f"{path.name}: obs must be rank 2, got shape {obs.shape}"]
        length = int(obs.shape[0])
        if expected_obs_dim is not None and int(obs.shape[1]) != int(expected_obs_dim):
            errors.append(f"{path.name}: obs shape {obs.shape} != ({length}, {int(expected_obs_dim)})")
        for key in data.files:
            array = data[key]
            if array.ndim > 0 and array.shape[0] != length:
                errors.append(f"{path.name}: {key} length {array.shape[0]} != obs length {length}")
        for key, tail_shape in OPTIONAL_ARRAY_TAIL_SHAPES.items():
            if key in data:
                expected = (length, *tail_shape)
                if data[key].shape != expected:
                    errors.append(f"{path.name}: {key} shape {data[key].shape} != {expected}")
        expected_shapes = {
            "action_label": (length, 12),
            "q_teacher": (length, 12),
            "q": (length, 12),
            "qdot": (length, 12),
            "command": (length, 3),
            "reward": (length,),
            "done": (length,),
            "reset_flag": (length,),
            "phase": (length,),
            "base_pos": (length, 3),
            "base_quat": (length, 4),
            "base_lin_vel_body": (length, 3),
            "base_ang_vel_body": (length, 3),
            "projected_gravity": (length, 3),
            "foot_contacts": (length, 4),
            "torque": (length, 12),
        }
        for key, expected in expected_shapes.items():
            if data[key].shape != expected:
                errors.append(f"{path.name}: {key} shape {data[key].shape} != {expected}")
        if expected_action_dim is not None and int(data["action_label"].shape[1]) != int(expected_action_dim):
            errors.append(
                f"{path.name}: action_label shape {data['action_label'].shape} != ({length}, {int(expected_action_dim)})"
            )
        expected_label = np.clip((data["q_teacher"].astype(np.float32) - Q_HOME) / ACTION_SCALE, -1.0, 1.0)
        max_error = float(np.max(np.abs(data["action_label"].astype(np.float32) - expected_label))) if length else 0.0
        if max_error > 1e-5:
            errors.append(f"{path.name}: action_label mismatch max_error={max_error:.6g}")
        phase_error = obs_phase_max_error(data["obs"], data["phase"])
        if phase_error > 1e-5:
            errors.append(f"{path.name}: obs phase mismatch max_error={phase_error:.6g}")
        episode_for_obs = {key: data[key] for key in data.files}
        obs_error = obs_state_max_error(episode_for_obs)
        if obs_error > 1e-5:
            errors.append(f"{path.name}: obs state mismatch max_error={obs_error:.6g}")
        if length > 0:
            reset_flag = data["reset_flag"].astype(bool)
            if not bool(reset_flag[0]):
                errors.append(f"{path.name}: reset_flag[0] must be True")
            if np.any(reset_flag[1:]):
                errors.append(f"{path.name}: mid-episode reset_flag values must be False")
            done = data["done"].astype(bool)
            if np.any(done[:-1]):
                errors.append(f"{path.name}: mid-episode done values must be False")
        for key in REQUIRED_ARRAYS:
            if key in data and not np.all(np.isfinite(data[key])):
                errors.append(f"{path.name}: {key} contains non-finite values")
    return length, errors


def _validate_metadata(dataset_dir: Path, episode_paths: list[Path]) -> list[str]:
    """Validate metadata consistency against episode files on disk."""
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        return []
    errors: list[str] = []
    metadata = json.loads(metadata_path.read_text())
    if episode_paths:
        with np.load(episode_paths[0]) as data:
            if "obs_dim" in metadata and int(metadata["obs_dim"]) != int(data["obs"].shape[1]):
                errors.append(f"metadata obs_dim {metadata['obs_dim']} != episode obs_dim {int(data['obs'].shape[1])}")
            if "action_dim" in metadata and int(metadata["action_dim"]) != int(data["action_label"].shape[1]):
                errors.append(f"metadata action_dim {metadata['action_dim']} != episode action_dim {int(data['action_label'].shape[1])}")
    metadata_paths = {str(entry.get("path", "")) for entry in metadata.get("episodes", []) if entry.get("path")}
    actual_paths = {str(path.relative_to(dataset_dir)) for path in episode_paths}
    for path_value in sorted(actual_paths - metadata_paths):
        errors.append(f"episode file missing from metadata: {path_value}")
    for entry in metadata.get("episodes", []):
        if not entry.get("accepted", False):
            continue
        path_value = entry.get("path", "")
        if not path_value:
            errors.append(f"metadata accepted episode {entry.get('episode_id', '<unknown>')} missing path")
            continue
        episode_path = dataset_dir / str(path_value)
        if not episode_path.exists():
            errors.append(f"metadata episode path missing: {path_value}")
    accepted_entries = [entry for entry in metadata.get("episodes", []) if entry.get("accepted", False)]
    if "accepted_episodes" in metadata and int(metadata["accepted_episodes"]) != len(accepted_entries):
        errors.append(f"metadata accepted_episodes {metadata['accepted_episodes']} != accepted metadata entries {len(accepted_entries)}")
    if "accepted_steps" in metadata:
        metadata_steps = int(metadata["accepted_steps"])
        accepted_steps = sum(int(entry.get("stats", {}).get("survival_steps", 0)) for entry in accepted_entries)
        if metadata_steps != accepted_steps:
            errors.append(f"metadata accepted_steps {metadata_steps} != accepted metadata survival_steps {accepted_steps}")
    return errors


def validate_dataset(dataset_dir: str | Path) -> ValidationResult:
    """Validate episode files and metadata for one dataset directory."""
    dataset_dir = Path(dataset_dir)
    paths = _episode_paths(dataset_dir)
    errors: list[str] = []
    transitions = 0
    if not paths:
        errors.append(f"no episode files found in {dataset_dir / 'episodes'}")
    errors.extend(_validate_metadata(dataset_dir, paths))
    metadata_path = dataset_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    expected_obs_dim = int(metadata["obs_dim"]) if "obs_dim" in metadata else None
    expected_action_dim = int(metadata["action_dim"]) if "action_dim" in metadata else None
    for path in paths:
        length, episode_errors = _validate_episode(path, expected_obs_dim, expected_action_dim)
        transitions += length
        errors.extend(episode_errors)
    return ValidationResult(ok=not errors, episodes=len(paths), transitions=transitions, errors=errors)


def main() -> None:
    """Run the dataset validation command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir")
    args = parser.parse_args()
    result = validate_dataset(args.dataset_dir)
    print(result.report())
    raise SystemExit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
