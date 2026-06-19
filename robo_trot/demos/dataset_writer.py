from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from robo_trot.robot.a1 import ACTION_SCALE, Q_HOME

REQUIRED_ARRAYS = (
    "obs",
    "action_label",
    "q_teacher",
    "q",
    "qdot",
    "command",
    "reward",
    "done",
    "reset_flag",
    "phase",
    "base_pos",
    "base_quat",
    "base_lin_vel_body",
    "base_ang_vel_body",
    "projected_gravity",
    "foot_contacts",
    "torque",
)

ARRAY_TAIL_SHAPES = {
    "obs": None,
    "action_label": (12,),
    "q_teacher": (12,),
    "q": (12,),
    "qdot": (12,),
    "command": (3,),
    "reward": (),
    "done": (),
    "reset_flag": (),
    "phase": (),
    "base_pos": (3,),
    "base_quat": (4,),
    "base_lin_vel_body": (3,),
    "base_ang_vel_body": (3,),
    "projected_gravity": (3,),
    "foot_contacts": (4,),
    "torque": (12,),
}

OPTIONAL_ARRAY_TAIL_SHAPES = {
    "foot_pos": (4, 3),
}

OBS_PHASE_SIN_INDEX = 48
OBS_PHASE_COS_INDEX = 49


def obs_phase_max_error(obs: np.ndarray, phase: np.ndarray) -> float:
    """Return the maximum mismatch between observation phase channels and saved phase."""
    obs = np.asarray(obs, dtype=np.float32)
    phase = np.asarray(phase, dtype=np.float32)
    if obs.ndim != 2 or obs.shape[1] <= OBS_PHASE_COS_INDEX or phase.ndim != 1 or obs.shape[0] != phase.shape[0]:
        return float("inf")
    expected = np.stack((np.sin(phase), np.cos(phase)), axis=1).astype(np.float32)
    actual = obs[:, [OBS_PHASE_SIN_INDEX, OBS_PHASE_COS_INDEX]]
    return float(np.max(np.abs(actual - expected))) if phase.size else 0.0


def action_label_max_error(q_teacher: np.ndarray, action_label: np.ndarray) -> float:
    """Return the maximum mismatch between labels and normalized teacher targets."""
    q_teacher = np.asarray(q_teacher, dtype=np.float32)
    action_label = np.asarray(action_label, dtype=np.float32)
    if q_teacher.shape != action_label.shape or q_teacher.ndim != 2 or q_teacher.shape[1] != 12:
        return float("inf")
    expected = np.clip((q_teacher - Q_HOME) / ACTION_SCALE, -1.0, 1.0).astype(np.float32)
    return float(np.max(np.abs(action_label - expected))) if q_teacher.size else 0.0


def obs_state_max_error(episode: dict[str, np.ndarray]) -> float:
    """Return the maximum mismatch between observation slices and saved state arrays."""
    obs = np.asarray(episode["obs"], dtype=np.float32)
    length = int(obs.shape[0])
    if obs.ndim != 2 or obs.shape[1] not in {52, 56}:
        return float("inf")
    checks = [
        (obs[:, 0:3], np.asarray(episode["projected_gravity"], dtype=np.float32)),
        (obs[:, 3:6], np.asarray(episode["base_ang_vel_body"], dtype=np.float32)),
        (obs[:, 6:9], np.asarray(episode["base_lin_vel_body"], dtype=np.float32)),
        (obs[:, 9:12], np.asarray(episode["command"], dtype=np.float32)),
        (obs[:, 12:24], np.asarray(episode["q"], dtype=np.float32) - Q_HOME),
        (obs[:, 24:36], np.asarray(episode["qdot"], dtype=np.float32)),
    ]
    previous_action = np.zeros((length, 12), dtype=np.float32)
    previous_reward = np.zeros((length,), dtype=np.float32)
    if length > 1:
        previous_action[1:] = np.asarray(episode["action_label"], dtype=np.float32)[:-1]
        previous_reward[1:] = np.asarray(episode["reward"], dtype=np.float32)[:-1]
    checks.extend(
        [
            (obs[:, 36:48], previous_action),
            (obs[:, 50], previous_reward),
            (obs[:, 51], np.asarray(episode["reset_flag"], dtype=np.float32)),
        ]
    )
    if obs.shape[1] == 56:
        checks.append((obs[:, 52:56], np.asarray(episode["foot_contacts"], dtype=np.float32)))
    max_error = 0.0
    for actual, expected in checks:
        if actual.shape != expected.shape:
            return float("inf")
        if actual.size:
            max_error = max(max_error, float(np.max(np.abs(actual - expected))))
    return max_error


class DatasetWriter:
    """Write accepted teacher episodes and media artifacts with metadata."""

    def __init__(self, out_dir: str | Path, metadata: dict[str, Any], resume: bool = False):
        """Initialize output directories and load or create dataset metadata."""
        self.out_dir = Path(out_dir)
        self.episodes_dir = self.out_dir / "episodes"
        self.gifs_dir = self.out_dir / "gifs"
        self.videos_dir = self.out_dir / "videos"
        self.failed_gifs_dir = self.out_dir / "failed_gifs"
        for path in (self.episodes_dir, self.gifs_dir, self.videos_dir, self.failed_gifs_dir):
            path.mkdir(parents=True, exist_ok=True)
        metadata_path = self.out_dir / "metadata.json"
        if resume and metadata_path.exists():
            self.metadata = json.loads(metadata_path.read_text())
            for key, value in metadata.items():
                if key != "episodes":
                    self.metadata.setdefault(key, value)
        else:
            self.metadata = dict(metadata)
            self.metadata.setdefault("episodes", [])
        self._write_metadata()

    @property
    def next_episode_id(self) -> int:
        """Return the next episode identifier after existing metadata entries."""
        episodes = self.metadata.get("episodes", [])
        if not episodes:
            return 0
        return max(int(entry.get("episode_id", -1)) for entry in episodes) + 1

    @property
    def accepted_steps(self) -> int:
        """Return the number of accepted transitions recorded in metadata."""
        total = 0
        for entry in self.metadata.get("episodes", []):
            if entry.get("accepted", False):
                total += int(entry.get("stats", {}).get("survival_steps", 0))
        return total

    def _write_metadata(self) -> None:
        """Persist current metadata to the dataset root."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "metadata.json").write_text(json.dumps(self.metadata, indent=2, sort_keys=True))

    def update_metadata(self, **updates: Any) -> None:
        """Merge metadata fields and persist the updated metadata file."""
        self.metadata.update(updates)
        self._write_metadata()

    def record_rejection(self, attempt_id: int, stats: dict[str, Any], recent_limit: int = 100) -> None:
        """Record a rejected episode summary without writing episode arrays."""
        reason = str(stats.get("reject_reason") or stats.get("done_reason") or "unknown")
        counts = dict(self.metadata.get("rejection_counts", {}))
        counts[reason] = int(counts.get(reason, 0)) + 1
        self.metadata["rejection_counts"] = counts

        summary = {
            "attempt_id": int(attempt_id),
            "reject_reason": reason,
            "survival_steps": int(stats.get("survival_steps", 0)),
            "forward_progress": float(stats.get("forward_progress", 0.0)),
            "min_base_height": float(stats.get("min_base_height", 0.0)),
            "clip_fraction": float(stats.get("clip_fraction", 0.0)),
        }
        contact_slip = stats.get("contact_slip")
        if isinstance(contact_slip, dict):
            summary["contact_slip_mean"] = float(contact_slip.get("mean", 0.0))
            summary["contact_slip_p95"] = float(contact_slip.get("p95", 0.0))

        recent = list(self.metadata.get("recent_rejections", []))
        recent.append(summary)
        self.metadata["recent_rejections"] = recent[-max(1, int(recent_limit)) :]
        self._write_metadata()

    def write_episode(self, ep_id: int, episode: dict[str, np.ndarray], accepted: bool, stats: dict[str, Any]) -> Path:
        """Validate and write one episode NPZ plus its metadata entry."""
        missing = [key for key in REQUIRED_ARRAYS if key not in episode]
        if missing:
            raise KeyError(f"Episode missing arrays: {missing}")
        self._validate_episode_shapes(episode)
        path = self.episodes_dir / f"ep_{ep_id:06d}.npz"
        np.savez_compressed(path, **episode)
        entry = {
            "episode_id": ep_id,
            "accepted": bool(accepted),
            "path": str(path.relative_to(self.out_dir)),
            "stats": stats,
        }
        self.metadata.setdefault("episodes", []).append(entry)
        self._write_metadata()
        return path

    def _validate_episode_shapes(self, episode: dict[str, np.ndarray]) -> None:
        """Validate episode array shapes and observation/label consistency."""
        length = int(np.asarray(episode["obs"]).shape[0])
        obs_dim = int(self.metadata.get("obs_dim", np.asarray(episode["obs"]).shape[1]))
        for key in REQUIRED_ARRAYS:
            array = np.asarray(episode[key])
            if array.shape[0] != length:
                raise ValueError(f"{key} length {array.shape[0]} does not match obs length {length}")
            expected_tail = (obs_dim,) if key == "obs" else ARRAY_TAIL_SHAPES[key]
            expected_shape = (length, *expected_tail)
            if array.shape != expected_shape:
                raise ValueError(f"{key} shape {array.shape} does not match expected {expected_shape}")
        if length > 0:
            reset_flag = np.asarray(episode["reset_flag"], dtype=bool)
            if not bool(reset_flag[0]):
                raise ValueError("reset_flag[0] must be True")
            if np.any(reset_flag[1:]):
                raise ValueError("mid-episode reset_flag values must be False")
            done = np.asarray(episode["done"], dtype=bool)
            if np.any(done[:-1]):
                raise ValueError("mid-episode done values must be False")
            phase_error = obs_phase_max_error(episode["obs"], episode["phase"])
            if phase_error > 1e-5:
                raise ValueError(f"obs phase mismatch max_error={phase_error:.6g}")
            label_error = action_label_max_error(episode["q_teacher"], episode["action_label"])
            if label_error > 1e-5:
                raise ValueError(f"action_label mismatch max_error={label_error:.6g}")
            obs_error = obs_state_max_error(episode)
            if obs_error > 1e-5:
                raise ValueError(f"obs state mismatch max_error={obs_error:.6g}")
        for key, expected_tail in OPTIONAL_ARRAY_TAIL_SHAPES.items():
            if key not in episode:
                continue
            array = np.asarray(episode[key])
            expected_shape = (length, *expected_tail)
            if array.shape != expected_shape:
                raise ValueError(f"{key} shape {array.shape} does not match expected {expected_shape}")

    def write_gif(
        self,
        ep_id: int,
        frames: list[np.ndarray],
        accepted: bool,
        stats: dict[str, Any],
        fps: int = 20,
    ) -> Path | None:
        """Write a GIF preview and matching JSON stats sidecar when frames exist."""
        if not frames:
            return None
        directory = self.gifs_dir if accepted else self.failed_gifs_dir
        path = directory / f"ep_{ep_id:06d}.gif"
        pil_frames = [Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB") for frame in frames]
        duration_ms = int(round(1000.0 / max(1, fps)))
        pil_frames[0].save(
            path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
            disposal=2,
            optimize=False,
        )
        stats_path = path.with_suffix(".json")
        stats_payload = {"episode_id": ep_id, "accepted": bool(accepted), **stats}
        stats_path.write_text(json.dumps(stats_payload, indent=2, sort_keys=True))
        return path

    def write_video(
        self,
        ep_id: int,
        frames: list[np.ndarray],
        accepted: bool,
        stats: dict[str, Any],
        fps: int = 30,
    ) -> Path | None:
        """Write an MP4 preview and matching JSON stats sidecar when supported."""
        if not frames:
            return None
        directory = self.videos_dir if accepted else (self.out_dir / "failed_videos")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"ep_{ep_id:06d}.mp4"
        try:
            with imageio.get_writer(path, fps=max(1, int(fps)), codec="libx264", macro_block_size=1) as writer:
                for frame in frames:
                    writer.append_data(np.asarray(frame, dtype=np.uint8))
        except Exception:
            if path.exists():
                path.unlink()
            return None
        stats_payload = {"episode_id": ep_id, "accepted": bool(accepted), **stats}
        path.with_suffix(".json").write_text(json.dumps(stats_payload, indent=2, sort_keys=True))
        return path
