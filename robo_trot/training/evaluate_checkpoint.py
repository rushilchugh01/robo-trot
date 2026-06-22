from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from robo_trot.policies.action_adapter import action_label_to_q_des
from robo_trot.training.action_mapping_audit import audit_action_mapping
from robo_trot.training.checkpointing import CheckpointRecord, find_complete_checkpoints, load_checkpoint
from robo_trot.training.dataset import BehaviorCloningDataset
from robo_trot.training.eval_reward import EvalReward, compute_eval_reward, summarize_reward_terms
from robo_trot.training.policy_rollout import load_dataset_contract, validate_env_contract
from robo_trot.training.torch_utils import import_torch


@dataclass(frozen=True)
class EvalCommand:
    """One fixed command used for checkpoint evaluation.

    Labels are used for metric rows and rollout media file names.
    """

    label: str
    command: np.ndarray


@dataclass(frozen=True)
class RolloutEvalResult:
    """Metrics and media path for one command rollout.

    The evaluator aggregates these across commands for checkpoint comparison.
    """

    label: str
    reward_mean: float
    survival_seconds: float
    fell: bool
    forward_velocity_mean: float
    yaw_rate_mean: float
    roll_pitch_max: float
    foot_slip_mean: float
    reward_terms: dict[str, float]
    media_path: str | None


FIXED_EVAL_COMMANDS = (
    EvalCommand("vx00", np.array([0.0, 0.0, 0.0], dtype=np.float32)),
    EvalCommand("vx03", np.array([0.3, 0.0, 0.0], dtype=np.float32)),
    EvalCommand("vx06", np.array([0.6, 0.0, 0.0], dtype=np.float32)),
    EvalCommand("yaw_left", np.array([0.4, 0.0, 0.5], dtype=np.float32)),
    EvalCommand("yaw_right", np.array([0.4, 0.0, -0.5], dtype=np.float32)),
)


def select_eval_commands(command_labels: str | list[str] | tuple[str, ...] | None = None) -> tuple[EvalCommand, ...]:
    """Return fixed eval commands selected by label.

    `None` and `all` preserve the full evaluator command suite.
    """
    if command_labels is None:
        return FIXED_EVAL_COMMANDS
    if isinstance(command_labels, str):
        labels = tuple(part.strip() for part in command_labels.split(",") if part.strip())
    else:
        labels = tuple(str(part).strip() for part in command_labels if str(part).strip())
    if not labels or labels == ("all",):
        return FIXED_EVAL_COMMANDS
    by_label = {command.label: command for command in FIXED_EVAL_COMMANDS}
    invalid = [label for label in labels if label not in by_label]
    if invalid:
        raise ValueError(f"unknown eval command label(s): {invalid}")
    return tuple(by_label[label] for label in dict.fromkeys(labels))


def collect_policy_checkpoints(run_dir: str | Path, model: str) -> list[CheckpointRecord]:
    """Return complete checkpoints for one model under a comparison run.

    Incomplete checkpoint directories are omitted by the checkpointing utility.
    """
    return find_complete_checkpoints(Path(run_dir) / str(model) / "checkpoints")


def evaluate_checkpoint(
    checkpoint: str | Path,
    model_type: str,
    xml_path: str | Path,
    dataset_metadata: str | Path | None = None,
    seconds: float = 20.0,
    command: np.ndarray | None = None,
    save_media: str | Path | None = None,
    save_gif: str | Path | None = None,
    viewer: bool = False,
    seed: int = 0,
    gif_fps: int = 60,
    gif_seconds: float | None = None,
    gif_width: int = 480,
    gif_height: int = 270,
    dataset_dir: str | Path | None = None,
    dataset_eval_split: str = "test",
    dataset_eval_batch_size: int = 4096,
    dataset_eval_max_batches: int = 16,
    sequence_length: int = 64,
) -> dict[str, Any]:
    """Evaluate one checkpoint in the MuJoCo A1 environment.

    Reward metrics are for checkpoint comparison only and are not training losses.
    """
    from robo_trot.sim.a1_teacher_env import A1TeacherEnv

    policy = load_policy_from_checkpoint(checkpoint, model_type)
    env = A1TeacherEnv(xml_path, {"episode_seconds": float(seconds), "use_contacts": True})
    if dataset_metadata is not None:
        contract = load_dataset_contract(dataset_metadata)
        validate_env_contract(env, contract)
    audit_results = audit_action_mapping(env, action_value=0.5, settle_steps=3, min_observed_delta=1e-5)
    if not all(result.passed for result in audit_results):
        failed = [result.reason for result in audit_results if not result.passed]
        raise ValueError(f"action mapping audit failed before eval: {failed}")
    eval_command = EvalCommand("manual", np.asarray(command if command is not None else np.zeros(3), dtype=np.float32))
    result = run_policy_eval_episode(
        env=env,
        policy=policy,
        eval_command=eval_command,
        seconds=float(seconds),
        seed=int(seed),
        save_media=_media_output_path(save_media if save_media is not None else save_gif),
        gif_fps=int(gif_fps),
        gif_seconds=gif_seconds,
        gif_width=int(gif_width),
        gif_height=int(gif_height),
        viewer=bool(viewer),
    )
    metrics = rollout_result_to_metrics(
        result,
        model_type=model_type,
        checkpoint_path=Path(checkpoint),
        checkpoint_update=_checkpoint_update(checkpoint),
    )
    if dataset_dir is not None:
        metrics.update(
            evaluate_dataset_action_loss(
                policy=policy,
                model_type=model_type,
                dataset_dir=dataset_dir,
                split=str(dataset_eval_split),
                batch_size=int(dataset_eval_batch_size),
                sequence_length=int(sequence_length),
                max_batches=int(dataset_eval_max_batches),
                seed=int(seed),
            )
        )
    return metrics


def load_policy_from_checkpoint(checkpoint: str | Path, model_type: str) -> Any:
    """Load an MLP or TXL policy from a complete checkpoint directory.

    The function imports torch lazily so dashboard-only usage stays lightweight.
    """
    from robo_trot.training.torch_utils import import_torch

    torch = import_torch()
    loaded = load_checkpoint(checkpoint)
    config = dict(loaded.metadata.get("model_config", {}))
    if str(model_type) == "mlp":
        from robo_trot.policies.mlp_policy import MLPPolicy

        policy = MLPPolicy(**config)
    elif str(model_type) == "txl":
        from robo_trot.policies.txl_policy import TXLPolicy

        policy = TXLPolicy(**config)
    else:
        raise ValueError(f"unsupported model type: {model_type}")
    model_path = Path(checkpoint) / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"checkpoint missing model.pt: {model_path}")
    state_dict = torch.load(model_path, map_location="cpu")
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def run_policy_eval_episode(
    env: Any,
    policy: Any,
    eval_command: EvalCommand,
    seconds: float,
    seed: int,
    save_media: Path | None = None,
    save_gif: Path | None = None,
    gif_fps: int = 60,
    gif_seconds: float | None = None,
    gif_width: int = 480,
    gif_height: int = 270,
    viewer: bool = False,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> RolloutEvalResult:
    """Run one command rollout and return reward, survival, and media metrics.

    The loop uses the same normalized action adapter as the policy rollout harness.
    """
    media_path = _media_output_path(save_media if save_media is not None else save_gif)
    command = np.asarray(eval_command.command, dtype=np.float32).reshape(3)
    env.reset(seed=seed)
    policy.reset(np.random.default_rng(seed))
    steps_target = max(1, int(round(float(seconds) / float(env.policy_dt))))
    prev_action = np.zeros(12, dtype=np.float32)
    prev_reward = 0.0
    prev_foot_pos = None
    phase = 0.0
    reset_flag = True
    reward_rows: list[EvalReward] = []
    frames: list[np.ndarray] = []
    forward_velocities: list[float] = []
    yaw_rates: list[float] = []
    roll_pitch: list[float] = []
    slips: list[float] = []
    done_reason = ""
    render_every = max(1, int(round(1.0 / (float(env.policy_dt) * max(1, int(gif_fps))))))
    if gif_seconds is None:
        render_steps_target = steps_target
    else:
        render_steps_target = max(0, min(steps_target, int(round(float(gif_seconds) / float(env.policy_dt)))))

    viewer_context = _viewer_context(env, enabled=viewer)
    with viewer_context as passive_viewer:
        for step in range(steps_target):
            obs = env.make_obs(command, prev_action, prev_reward, reset_flag, phase)
            action = np.asarray(policy.act(obs), dtype=np.float32).reshape(12)
            q_des = action_label_to_q_des(action)
            _, done, info = env.step_q_des(q_des)
            reward = compute_eval_reward(info, command, action, prev_action, prev_foot_pos=prev_foot_pos, done=done)
            reward_rows.append(reward)
            forward_velocities.append(float(np.asarray(info.get("base_lin_vel_body", np.zeros(3)))[0]))
            yaw_rates.append(float(np.asarray(info.get("base_ang_vel_body", np.zeros(3)))[2]))
            roll_pitch.append(max(abs(float(info.get("roll", 0.0))), abs(float(info.get("pitch", 0.0)))))
            slips.append(abs(float(reward.terms.get("foot_slip_penalty", 0.0))))
            if media_path is not None and step < render_steps_target and step % render_every == 0:
                frames.append(env.render_frame(width=int(gif_width), height=int(gif_height)))
            if progress_callback is not None:
                progress_callback(step + 1, steps_target, len(frames))
            if passive_viewer is not None:
                passive_viewer.sync()
            prev_action = action
            prev_reward = float(reward.total)
            prev_foot_pos = np.asarray(info.get("foot_pos"), dtype=np.float32).copy() if "foot_pos" in info else None
            reset_flag = False
            phase = advance_command_phase(phase, command, env.policy_dt)
            if done:
                done_reason = str(info.get("done_reason", "done"))
                break

    media_path_value: str | None = None
    if media_path is not None and frames:
        target_frame_count = (render_steps_target + render_every - 1) // render_every if render_steps_target > 0 else 0
        _write_mp4(media_path, frames, fps=max(1, int(gif_fps)), min_frame_count=target_frame_count)
        media_path_value = media_path.as_posix()
    survived_seconds = len(reward_rows) * float(env.policy_dt)
    term_summary = summarize_reward_terms(reward_rows)
    return RolloutEvalResult(
        label=eval_command.label,
        reward_mean=float(term_summary.get("reward_mean", 0.0)),
        survival_seconds=float(survived_seconds),
        fell=bool(done_reason and done_reason != "timeout"),
        forward_velocity_mean=float(np.mean(forward_velocities)) if forward_velocities else 0.0,
        yaw_rate_mean=float(np.mean(yaw_rates)) if yaw_rates else 0.0,
        roll_pitch_max=float(np.max(roll_pitch)) if roll_pitch else 0.0,
        foot_slip_mean=float(np.mean(slips)) if slips else 0.0,
        reward_terms=term_summary,
        media_path=media_path_value,
    )


def advance_command_phase(phase: float, command: np.ndarray, policy_dt: float) -> float:
    """Advance the gait phase with the teacher CPG frequency law.

    The dataset records phase from `FootspaceCPGIKTeacher.compute`, whose
    frequency is speed-dependent rather than the fixed 1 Hz clock used before.
    """
    frequency = command_phase_frequency(command)
    return float((float(phase) + 2.0 * np.pi * frequency * float(policy_dt)) % (2.0 * np.pi))


def command_phase_frequency(command: np.ndarray) -> float:
    """Return the teacher CPG phase frequency for a command.

    This mirrors `FootspaceCPGIKTeacher.frequency` without constructing a teacher.
    """
    values = np.asarray(command, dtype=np.float32).reshape(3)
    vx = float(values[0])
    yaw = float(values[2])
    if abs(vx) / 0.7 < 0.08 and abs(yaw) / 0.4 < 0.1:
        return 0.0
    base_freq = 1.6
    max_freq = 2.8
    speed_ref = 1.1
    scale = float(np.clip(abs(vx) / speed_ref, 0.0, 1.0))
    return float(base_freq + (max_freq - base_freq) * scale)


def evaluate_checkpoint_set(
    checkpoint: str | Path,
    model_type: str,
    xml_path: str | Path,
    dataset_metadata: str | Path | None,
    out_dir: str | Path,
    seconds: float = 20.0,
    checkpoint_update: int | None = None,
    gif_every_eval: int = 1,
    gif_fps: int = 5,
    gif_seconds: float | None = 10.0,
    gif_width: int = 320,
    gif_height: int = 180,
    eval_index: int = 0,
    seed: int = 0,
    command_labels: str | list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate a checkpoint on selected fixed commands and persist MP4s.

    The returned metrics can be appended directly to `eval/metrics.jsonl`.
    """
    from robo_trot.sim.a1_teacher_env import A1TeacherEnv

    update = int(checkpoint_update if checkpoint_update is not None else _checkpoint_update(checkpoint))
    policy = load_policy_from_checkpoint(checkpoint, model_type)
    env = A1TeacherEnv(xml_path, {"episode_seconds": float(seconds), "use_contacts": True})
    if dataset_metadata is not None:
        contract = load_dataset_contract(dataset_metadata)
        validate_env_contract(env, contract)
    audit_results = audit_action_mapping(env, action_value=0.5, settle_steps=3, min_observed_delta=1e-5)
    if not all(result.passed for result in audit_results):
        raise ValueError("action mapping audit failed before eval")
    step_dir = Path(out_dir) / "media" / f"step_{update:09d}"
    rows: list[dict[str, Any]] = []
    for command_index, eval_command in enumerate(select_eval_commands(command_labels)):
        save_media = None
        if int(gif_every_eval) > 0 and int(eval_index) % int(gif_every_eval) == 0:
            save_media = step_dir / f"{model_type}_{eval_command.label}.mp4"
        result = run_policy_eval_episode(
            env=env,
            policy=policy,
            eval_command=eval_command,
            seconds=float(seconds),
            seed=int(seed) + command_index,
            save_media=save_media,
            gif_fps=int(gif_fps),
            gif_seconds=gif_seconds,
            gif_width=int(gif_width),
            gif_height=int(gif_height),
        )
        rows.append(rollout_result_to_metrics(result, model_type, Path(checkpoint), update, root_dir=Path(out_dir).parent))
    return rows


def evaluate_dataset_action_loss(
    policy: Any,
    model_type: str,
    dataset_dir: str | Path,
    split: str = "test",
    batch_size: int = 4096,
    sequence_length: int = 64,
    max_batches: int = 16,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute supervised action-label diagnostics for one policy checkpoint.

    These metrics compare policy outputs to dataset `action_label` values; they are not rewards.
    """
    torch = import_torch()
    dataset = BehaviorCloningDataset(dataset_dir, split=str(split), seed=int(seed))
    if str(model_type) == "mlp":
        return _evaluate_mlp_dataset_loss(torch, policy, dataset, str(split), int(batch_size), int(max_batches))
    if str(model_type) == "txl":
        return _evaluate_txl_dataset_loss(
            torch,
            policy,
            dataset,
            str(split),
            int(batch_size),
            int(sequence_length),
            int(max_batches),
        )
    raise ValueError(f"unsupported model type for dataset evaluation: {model_type}")


def _evaluate_mlp_dataset_loss(
    torch: Any,
    policy: Any,
    dataset: BehaviorCloningDataset,
    split: str,
    batch_size: int,
    max_batches: int,
) -> dict[str, Any]:
    """Evaluate feed-forward action prediction loss on deterministic batches.

    The pass is bounded by `max_batches` so checkpoint evaluation remains predictable.
    """
    policy.eval()
    total_squared = 0.0
    total_absolute = 0.0
    total_clipped = 0.0
    total_values = 0
    batches = 0
    with torch.no_grad():
        for batch in dataset.iter_transition_batches(max(1, int(batch_size))):
            obs = torch.as_tensor(batch.obs, dtype=torch.float32)
            target = torch.as_tensor(batch.action_label, dtype=torch.float32)
            pred = policy(obs)
            total_squared += float(torch.sum((pred - target) ** 2))
            total_absolute += float(torch.sum(torch.abs(pred - target)))
            total_clipped += float(torch.sum((torch.abs(pred) > 0.995).float()))
            total_values += int(np.prod(target.shape))
            batches += 1
            if 0 < int(max_batches) <= batches:
                break
    return _dataset_loss_metrics(split, total_squared, total_absolute, total_clipped, total_values, batches)


def _evaluate_txl_dataset_loss(
    torch: Any,
    policy: Any,
    dataset: BehaviorCloningDataset,
    split: str,
    batch_size: int,
    sequence_length: int,
    max_batches: int,
) -> dict[str, Any]:
    """Evaluate sequence action prediction loss on deterministic windows.

    Memory is carried across contiguous chunks within each episode.
    """
    policy.eval()
    total_squared = 0.0
    total_absolute = 0.0
    total_clipped = 0.0
    total_values = 0
    batches = 0
    with torch.no_grad():
        for episode_index, episode in enumerate(dataset.episodes):
            memory = None
            for start in range(0, episode.length, int(sequence_length)):
                batch = dataset.make_stream_chunk_batch([(episode_index, start)], int(sequence_length))
                obs = torch.as_tensor(batch.obs, dtype=torch.float32)
                target = torch.as_tensor(batch.action_label, dtype=torch.float32)
                valid = torch.as_tensor(batch.valid_mask, dtype=torch.float32).unsqueeze(-1)
                episode_reset = torch.as_tensor(batch.episode_reset_mask, dtype=torch.bool)
                token_valid = torch.as_tensor(batch.valid_mask, dtype=torch.bool)
                pred, memory = policy(
                    obs,
                    reset_mask=episode_reset,
                    memory=memory,
                    valid_mask=token_valid,
                    return_memory=True,
                )
                total_squared += float(torch.sum(((pred - target) ** 2) * valid))
                total_absolute += float(torch.sum(torch.abs(pred - target) * valid))
                total_clipped += float(torch.sum((torch.abs(pred) > 0.995).float() * valid))
                total_values += int(torch.sum(valid).item()) * int(target.shape[-1])
                batches += 1
                if 0 < int(max_batches) <= batches:
                    return _dataset_loss_metrics(split, total_squared, total_absolute, total_clipped, total_values, batches)
    return _dataset_loss_metrics(split, total_squared, total_absolute, total_clipped, total_values, batches)


def _dataset_loss_metrics(
    split: str,
    total_squared: float,
    total_absolute: float,
    total_clipped: float,
    total_values: int,
    batches: int,
) -> dict[str, Any]:
    """Format dataset action-label diagnostics for eval metrics JSONL.

    Zero-token splits produce infinite losses so dashboard consumers can flag them.
    """
    if int(total_values) <= 0:
        mse = float("inf")
        l1 = float("inf")
        clip_fraction = 0.0
    else:
        mse = float(total_squared) / float(total_values)
        l1 = float(total_absolute) / float(total_values)
        clip_fraction = float(total_clipped) / float(total_values)
    return {
        "dataset_eval_split": str(split),
        "dataset_eval_action_mse": mse,
        "dataset_eval_action_l1": l1,
        "dataset_eval_action_clip_fraction": clip_fraction,
        "dataset_eval_values": int(total_values),
        "dataset_eval_batches": int(batches),
    }


def aggregate_eval_rows(rows: list[dict[str, Any]], model_type: str, checkpoint_update: int) -> dict[str, Any]:
    """Aggregate per-command eval rows into one dashboard-friendly row.

    Media paths are preserved by command label for the latest checkpoint panel.
    """
    if not rows:
        return {"model_type": model_type, "checkpoint_update": int(checkpoint_update)}
    reward_terms: dict[str, float] = {}
    term_keys = sorted({key for row in rows for key in row.get("reward_terms", {})})
    for key in term_keys:
        reward_terms[key] = float(np.mean([row.get("reward_terms", {}).get(key, 0.0) for row in rows]))
    media_paths = {str(row["command_label"]): row["media_path"] for row in rows if row.get("media_path")}
    gif_paths = {label: path for label, path in media_paths.items() if str(path).lower().endswith(".gif")}
    output = {
        "model_type": model_type,
        "checkpoint_update": int(checkpoint_update),
        "eval_reward_mean": float(np.mean([row.get("eval_reward_mean", 0.0) for row in rows])),
        "eval_survival_seconds_mean": float(np.mean([row.get("eval_survival_seconds_mean", 0.0) for row in rows])),
        "eval_fall_rate": float(np.mean([1.0 if row.get("fell") else 0.0 for row in rows])),
        "eval_forward_velocity_mean": float(np.mean([row.get("eval_forward_velocity_mean", 0.0) for row in rows])),
        "eval_yaw_rate_mean": float(np.mean([row.get("eval_yaw_rate_mean", 0.0) for row in rows])),
        "eval_roll_pitch_max": float(np.max([row.get("eval_roll_pitch_max", 0.0) for row in rows])),
        "eval_foot_slip_mean": float(np.mean([row.get("eval_foot_slip_mean", 0.0) for row in rows])),
        "reward_terms": reward_terms,
        "media_paths": media_paths,
        "wall_time": time.time(),
    }
    if gif_paths:
        output["gif_paths"] = gif_paths
    return output


def append_eval_metrics(path: str | Path, row: dict[str, Any]) -> None:
    """Append one evaluation metrics row as JSONL.

    Parent directories are created automatically for evaluator workers.
    """
    metrics_path = Path(path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def rollout_result_to_metrics(
    result: RolloutEvalResult,
    model_type: str,
    checkpoint_path: Path,
    checkpoint_update: int,
    root_dir: Path | None = None,
) -> dict[str, Any]:
    """Convert one command rollout result into a JSON metrics row.

    Media paths are made relative to the run directory when that root is known.
    """
    media_path = result.media_path
    if media_path is not None and root_dir is not None:
        try:
            media_path = Path(media_path).relative_to(root_dir).as_posix()
        except ValueError:
            media_path = Path(media_path).as_posix()
    return {
        "model_type": str(model_type),
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_update": int(checkpoint_update),
        "command_label": result.label,
        "eval_reward_mean": result.reward_mean,
        "eval_survival_seconds_mean": result.survival_seconds,
        "eval_fall_rate": 1.0 if result.fell else 0.0,
        "fell": result.fell,
        "eval_forward_velocity_mean": result.forward_velocity_mean,
        "eval_yaw_rate_mean": result.yaw_rate_mean,
        "eval_roll_pitch_max": result.roll_pitch_max,
        "eval_foot_slip_mean": result.foot_slip_mean,
        "reward_terms": result.reward_terms,
        "media_path": media_path,
        "wall_time": time.time(),
    }


def _media_output_path(path: str | Path | None) -> Path | None:
    """Return a normalized MP4 output path for eval media.

    Legacy `.gif` names and extensionless paths are redirected to `.mp4`.
    """
    if path is None:
        return None
    output = Path(path)
    return output if output.suffix.lower() == ".mp4" else output.with_suffix(".mp4")


def _gif_output_path(path: str | Path | None) -> Path | None:
    """Return the legacy normalized eval media path.

    This compatibility alias now redirects GIF-era callers to MP4 output.
    """
    return _media_output_path(path)


def _write_mp4(path: Path, frames: list[np.ndarray], fps: int, min_frame_count: int = 0) -> None:
    """Encode RGB frames as browser-compatible H.264 MP4 media.

    Short fall rollouts are padded with the final frame before encoding.
    """
    prepared = [_as_rgb_frame(frame) for frame in frames]
    if not prepared:
        return
    prepared = _pad_frames_to_count(prepared, int(min_frame_count))
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode checkpoint eval MP4 media")
    height, width = prepared[0].shape[:2]
    if any(frame.shape[:2] != (height, width) for frame in prepared):
        raise ValueError("all MP4 frames must have matching dimensions")
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(max(1, int(fps))),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        path.as_posix(),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert process.stdin is not None
        for frame in prepared:
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr is not None else ""
        returncode = process.wait()
    except Exception:
        process.kill()
        process.wait()
        if path.exists():
            path.unlink()
        raise
    if returncode != 0:
        if path.exists():
            path.unlink()
        raise RuntimeError(f"ffmpeg failed to encode MP4 media: {stderr.strip()}")


def _pad_frames_to_count(frames: list[np.ndarray], min_frame_count: int) -> list[np.ndarray]:
    """Return frames padded by repeating the final image when needed.

    This keeps failed policy videos dashboard-playable for the requested duration.
    """
    target = max(0, int(min_frame_count))
    if not frames or len(frames) >= target:
        return frames
    padded = list(frames)
    padded.extend([frames[-1]] * (target - len(frames)))
    return padded


def _write_gif(path: Path, frames: list[np.ndarray], fps: int) -> None:
    """Encode RGB frames as an animated GIF preview.

    GIF output matches the training dashboard and checkpoint-eval requirements.
    """
    import imageio.v2 as imageio

    prepared = [_as_rgb_frame(frame) for frame in frames]
    if not prepared:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        imageio.mimsave(path, prepared, duration=1.0 / max(1, int(fps)))
    except Exception:
        if path.exists():
            path.unlink()
        raise


def _as_rgb_frame(frame: np.ndarray) -> np.ndarray:
    """Return one contiguous uint8 RGB frame for GIF encoding.

    Eval renderers are expected to provide HxWx3 arrays; alpha channels are dropped.
    """
    array = np.asarray(frame, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError(f"expected RGB/RGBA frame, got shape {array.shape}")
    if array.shape[2] == 4:
        array = array[:, :, :3]
    return np.ascontiguousarray(array)


def _checkpoint_update(checkpoint: str | Path) -> int:
    """Infer a checkpoint update from metadata or directory naming.

    The directory convention is `step_000001000`.
    """
    try:
        metadata = load_checkpoint(checkpoint).metadata
        if "update" in metadata:
            return int(metadata["update"])
    except FileNotFoundError:
        pass
    name = Path(checkpoint).name
    if name.startswith("step_"):
        return int(name.replace("step_", ""))
    return -1


class _NullViewer:
    """No-op context manager used when live viewing is disabled.

    It keeps the rollout loop independent from optional MuJoCo viewer imports.
    """

    def __enter__(self) -> None:
        """Enter a disabled viewer context.

        The returned value is `None` so callers can branch cheaply.
        """
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Exit a disabled viewer context.

        Exceptions are not suppressed.
        """
        del exc_type, exc, tb
        return None


def _viewer_context(env: Any, enabled: bool) -> Any:
    """Return a MuJoCo passive viewer context when requested.

    Viewer imports stay lazy to keep headless checkpoint evaluation lightweight.
    """
    if not enabled:
        return _NullViewer()
    import mujoco.viewer

    return mujoco.viewer.launch_passive(env.model, env.data)
