from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from robo_trot.data_pipeline.dataset_writer import DatasetWriter
from robo_trot.sim.a1_teacher_env import A1TeacherEnv
from robo_trot.robot.a1 import ACTION_SCALE, OBS_DIM_NO_CONTACTS, OBS_DIM_WITH_CONTACTS, Q_HOME
from robo_trot.teachers.footspace_cpg_ik import FootspaceCPGIKTeacher

DEFAULT_MIN_FOOT_CLEARANCE = 0.025
DEFAULT_MAX_CONTACT_SLIP_MEAN = 0.25
DEFAULT_MAX_CONTACT_SLIP_P95 = 1.0

COMMAND_PROFILES = {
    "default": {
        "forward": ((0.2, 0.7), (-0.1, 0.1)),
        "slow": ((0.0, 0.2), (-0.2, 0.2)),
        "turning": ((0.1, 0.5), (-0.4, 0.4)),
    },
    "fast_probe": {
        "forward": ((0.5, 0.9), (-0.08, 0.08)),
        "slow": ((0.2, 0.5), (-0.15, 0.15)),
        "turning": ((0.3, 0.7), (-0.3, 0.3)),
    },
    "turn_probe": {
        "forward": ((0.2, 0.6), (-0.15, 0.15)),
        "slow": ((0.0, 0.25), (-0.5, 0.5)),
        "turning": ((0.1, 0.55), (-0.8, 0.8)),
    },
}

CATEGORY_COMMAND_RANGES = {
    "forward": {
        "teacher_profile": "strict_walk",
        "vx": (0.2, 0.7),
        "yaw": (-0.1, 0.1),
        "require_yaw_response": False,
    },
    "turn": {
        "teacher_profile": "turn_walk",
        "vx": (0.1, 0.55),
        "yaw_abs": (0.35, 0.8),
        "require_yaw_response": True,
    },
    "slow": {
        "teacher_profile": "strict_walk",
        "vx": (0.0, 0.2),
        "yaw": (-0.2, 0.2),
        "require_yaw_response": False,
    },
    "fast_probe": {
        "teacher_profile": "strict_walk",
        "vx": (0.7, 0.9),
        "yaw": (-0.08, 0.08),
        "require_yaw_response": False,
    },
}

FOOTSPACE_TEACHER_PROFILES = {
    "strict_walk": {
        "max_freq": 2.8,
        "step_length_max": 0.18,
    },
    "cruise_walk": {
        "max_freq": 2.8,
        "step_length_max": 0.20,
    },
    "legacy_fast": {
        "max_freq": 3.2,
        "step_length_max": 0.24,
    },
    "turn_walk": {
        "max_freq": 2.8,
        "step_length_max": 0.18,
        "yaw_cmd_limit": 0.8,
        "yaw_stride_gain": 0.10,
        "yaw_step_bias_gain": 0.035,
        "yaw_lateral_bias_gain": 0.0,
        "yaw_stance_bias_fraction": 0.12,
    },
}


def sample_command(rng: np.random.Generator, profile: str = "default") -> tuple[np.ndarray, str]:
    """Sample a command from a mixed forward, slow, and turning profile.

    This documents the callable contract used by the surrounding pipeline.
    """
    if profile not in COMMAND_PROFILES:
        raise ValueError(f"Unknown command profile: {profile}")
    ranges = COMMAND_PROFILES[profile]
    p = float(rng.random())
    if p < 0.60:
        vx_range, yaw_range = ranges["forward"]
        return np.array([rng.uniform(*vx_range), 0.0, rng.uniform(*yaw_range)], dtype=np.float32), "forward"
    if p < 0.80:
        vx_range, yaw_range = ranges["slow"]
        return np.array([rng.uniform(*vx_range), 0.0, rng.uniform(*yaw_range)], dtype=np.float32), "slow"
    vx_range, yaw_range = ranges["turning"]
    return np.array([rng.uniform(*vx_range), 0.0, rng.uniform(*yaw_range)], dtype=np.float32), "turning"


def sample_category_command(rng: np.random.Generator, category: str) -> tuple[np.ndarray, str]:
    """Sample a command from one fixed dataset category.

    This documents the callable contract used by the surrounding pipeline.
    """
    if category not in CATEGORY_COMMAND_RANGES:
        raise ValueError(f"Unknown command category: {category}")
    spec = CATEGORY_COMMAND_RANGES[category]
    vx = float(rng.uniform(*spec["vx"]))
    if "yaw_abs" in spec:
        yaw = float(rng.uniform(*spec["yaw_abs"]))
        yaw *= -1.0 if float(rng.random()) < 0.5 else 1.0
    else:
        yaw = float(rng.uniform(*spec["yaw"]))
    return np.array([vx, 0.0, yaw], dtype=np.float32), category


def parse_fixed_command(values: list[float] | None) -> np.ndarray | None:
    """Convert optional CLI fixed-command values into a command vector.

    This documents the callable contract used by the surrounding pipeline.
    """
    if values is None:
        return None
    if len(values) != 3:
        raise ValueError("--fixed_command requires exactly three values: vx vy yaw_rate")
    return np.asarray(values, dtype=np.float32)


def make_teacher(name: str, xml_path: str, policy_dt: float, profile: str = "strict_walk"):
    """Instantiate a teacher controller by name and profile.

    This documents the callable contract used by the surrounding pipeline.
    """
    if name == "footspace":
        if profile not in FOOTSPACE_TEACHER_PROFILES:
            raise ValueError(f"Unknown footspace teacher profile: {profile}")
        return FootspaceCPGIKTeacher(xml_path=xml_path, policy_dt=policy_dt, **FOOTSPACE_TEACHER_PROFILES[profile])
    raise ValueError(f"Unknown teacher: {name}")


def _append_step(buffers: dict[str, list], values: dict[str, Any]) -> None:
    """Append one timestep's arrays into rollout buffers.

    This documents the callable contract used by the surrounding pipeline.
    """
    for key, value in values.items():
        buffers[key].append(value)


def _stack_episode(buffers: dict[str, list]) -> dict[str, np.ndarray]:
    """Stack rollout buffers into typed per-episode arrays.

    This documents the callable contract used by the surrounding pipeline.
    """
    arrays: dict[str, np.ndarray] = {}
    for key, values in buffers.items():
        if key in {"done", "reset_flag"}:
            arrays[key] = np.asarray(values, dtype=bool)
        else:
            arrays[key] = np.asarray(values, dtype=np.float32)
    return arrays


def _clip_fraction(action_labels: np.ndarray) -> float:
    """Return the fraction of normalized labels at the clipping boundary.

    Callers rely on the returned value shape and semantics described here.
    """
    if action_labels.size == 0:
        return 0.0
    return float(np.mean(np.abs(action_labels) >= 0.999))


def quat_wxyz_to_yaw(quat: np.ndarray) -> float:
    """Convert a MuJoCo wxyz quaternion into yaw angle in radians.

    Math: angles are expressed in radians unless the caller documents otherwise.
    Frame conventions and equations are made explicit for quaternion, yaw, or IK paths.
    Outputs preserve the repository joint/contact ordering contract.
    """
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def episode_yaw_delta(episode: dict[str, np.ndarray]) -> float:
    """Return unwrapped yaw change across an episode.

    Math: angles are expressed in radians unless the caller documents otherwise.
    Frame conventions and equations are made explicit for quaternion, yaw, or IK paths.
    Outputs preserve the repository joint/contact ordering contract.
    """
    quats = np.asarray(episode.get("base_quat", np.zeros((0, 4), dtype=np.float32)), dtype=np.float32)
    if quats.ndim != 2 or quats.shape[0] < 2 or quats.shape[1] != 4:
        return 0.0
    yaws = np.unwrap([quat_wxyz_to_yaw(quat) for quat in quats])
    return float(yaws[-1] - yaws[0])


def contact_slip_metrics(
    foot_pos: np.ndarray,
    foot_contacts: np.ndarray,
    policy_dt: float = 0.02,
) -> dict[str, float | int]:
    """Measure contacted-foot horizontal slip speeds over an episode.

    Math: angles are expressed in radians unless the caller documents otherwise.
    Frame conventions and equations are made explicit for quaternion, yaw, or IK paths.
    Outputs preserve the repository joint/contact ordering contract.
    """
    foot_pos = np.asarray(foot_pos, dtype=np.float32)
    foot_contacts = np.asarray(foot_contacts, dtype=np.float32) > 0.5
    if foot_pos.ndim != 3 or foot_pos.shape[0] < 2 or foot_pos.shape[1:] != (4, 3):
        return {"contact_samples": 0, "mean": 0.0, "p95": 0.0, "max": 0.0}
    speed = np.linalg.norm(np.diff(foot_pos[:, :, :2], axis=0), axis=2) / max(float(policy_dt), 1e-6)
    contact_mask = foot_contacts[:-1]
    values = speed[contact_mask]
    if values.size == 0:
        return {"contact_samples": 0, "mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "contact_samples": int(values.size),
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def expand_frames_for_playback(
    frames: list[np.ndarray],
    policy_dt: float,
    render_every: int,
    gif_fps: int,
) -> list[np.ndarray]:
    """Repeat rendered frames so GIF playback approximates simulation time.

    This documents the callable contract used by the surrounding pipeline.
    """
    if not frames:
        return frames
    frame_step_seconds = float(policy_dt) * max(1, int(render_every))
    repeats = max(1, int(round(frame_step_seconds * max(1, int(gif_fps)))))
    if repeats == 1:
        return frames
    expanded: list[np.ndarray] = []
    for frame in frames:
        expanded.extend([frame] * repeats)
    return expanded


def render_q_teacher_episode(
    env,
    q_teacher: np.ndarray,
    render_every: int,
    gif_width: int,
    gif_height: int,
) -> list[np.ndarray]:
    """Render an episode by replaying saved teacher joint targets.

    This documents the callable contract used by the surrounding pipeline.
    """
    env.reset(seed=0)
    frames: list[np.ndarray] = []
    for step, q_des in enumerate(np.asarray(q_teacher, dtype=np.float32)):
        env.step_q_des(q_des)
        if step % max(1, int(render_every)) == 0:
            frames.append(env.render_frame(width=gif_width, height=gif_height))
    return frames


def write_debug_exports(
    writer: DatasetWriter,
    env,
    ep_id: int,
    episode: dict[str, np.ndarray],
    stats: dict[str, Any],
    accepted: bool,
    render_every: int,
    gif_width: int,
    gif_height: int,
    gif_fps: int,
    save_gif: bool,
    save_video: bool,
    video_fps: int,
) -> dict[str, bool]:
    """Render and write requested GIF or video debug exports.

    The side effect is part of the dataset or debug artifact contract.
    """
    if not save_gif and not save_video:
        return {"gif": False, "video": False}

    frames = render_q_teacher_episode(env, episode["q_teacher"], render_every, gif_width, gif_height)
    wrote_gif = False
    wrote_video = False
    if save_gif:
        expanded = expand_frames_for_playback(frames, env.policy_dt, render_every, gif_fps)
        wrote_gif = writer.write_gif(ep_id, expanded, accepted=accepted, stats=stats, fps=gif_fps) is not None
    if save_video:
        wrote_video = writer.write_video(ep_id, frames, accepted=accepted, stats=stats, fps=video_fps) is not None
    return {"gif": wrote_gif, "video": wrote_video}


def initial_recording_counters(writer: DatasetWriter, out_dir: str | Path, resume: bool) -> dict[str, int]:
    """Return recorder counters for a fresh or resumed dataset run.

    Callers rely on the returned value shape and semantics described here.
    """
    if not resume:
        return {
            "accepted_steps": 0,
            "accepted_eps": 0,
            "attempted_eps": 0,
            "saved_review_gifs": 0,
        }
    metadata = writer.metadata
    accepted_eps = writer.next_episode_id
    attempted_eps = int(metadata.get("attempted_episodes", accepted_eps))
    saved_review_gifs = int(metadata.get("saved_review_gifs", len(list((Path(out_dir) / "gifs").glob("ep_*.gif")))))
    return {
        "accepted_steps": writer.accepted_steps,
        "accepted_eps": accepted_eps,
        "attempted_eps": attempted_eps,
        "saved_review_gifs": saved_review_gifs,
    }


def episode_stats(episode: dict[str, np.ndarray], done_reason: str, accepted: bool, reject_reason: str) -> dict[str, Any]:
    """Compute per-episode quality and acceptance summary statistics.

    This documents the callable contract used by the surrounding pipeline.
    """
    base_pos = episode["base_pos"]
    commands = episode["command"]
    rewards = episode["reward"]
    steps = int(len(rewards))
    progress = float(base_pos[-1, 0] - base_pos[0, 0]) if steps else 0.0
    survival_seconds = 0.02 * steps
    stats = {
        "accepted": bool(accepted),
        "survival_steps": steps,
        "survival_seconds": survival_seconds,
        "mean_command": commands.mean(axis=0).astype(float).tolist() if steps else [0.0, 0.0, 0.0],
        "mean_forward_velocity": float(progress / survival_seconds) if survival_seconds > 0 else 0.0,
        "forward_progress": progress,
        "fell": bool(done_reason not in {"", "timeout"}),
        "done_reason": done_reason,
        "reject_reason": reject_reason,
        "clip_fraction": _clip_fraction(episode["action_label"]),
        "mean_reward": float(rewards.mean()) if steps else 0.0,
        "min_base_height": float(base_pos[:, 2].min()) if steps else 0.0,
        "yaw_delta": episode_yaw_delta(episode),
    }
    stats["mean_yaw_rate"] = float(stats["yaw_delta"] / survival_seconds) if survival_seconds > 0 else 0.0
    if "foot_pos" in episode and len(episode["foot_pos"]):
        foot_z = episode["foot_pos"][:, :, 2]
        stats["foot_clearance_ptp"] = np.ptp(foot_z, axis=0).astype(float).tolist()
        stats["foot_contact_fraction"] = episode["foot_contacts"].mean(axis=0).astype(float).tolist()
        stats["contact_slip"] = contact_slip_metrics(episode["foot_pos"], episode["foot_contacts"])
    return stats


def should_accept(
    episode: dict[str, np.ndarray],
    done_reason: str,
    clip_limit: float = 0.25,
    min_foot_clearance: float = DEFAULT_MIN_FOOT_CLEARANCE,
    max_contact_slip_mean: float = DEFAULT_MAX_CONTACT_SLIP_MEAN,
    max_contact_slip_p95: float = DEFAULT_MAX_CONTACT_SLIP_P95,
    require_yaw_response: bool = False,
    yaw_cmd_threshold: float = 0.2,
    min_yaw_delta: float = 0.25,
) -> tuple[bool, str]:
    """Apply quality gates to decide whether an episode enters the dataset.

    This documents the callable contract used by the surrounding pipeline.
    """
    steps = len(episode["reward"])
    if steps < 100:
        return False, "too_short"
    if done_reason not in {"", "timeout"}:
        return False, done_reason
    if not all(np.all(np.isfinite(value)) for value in episode.values()):
        return False, "nan"
    if steps < 300:
        return False, "short_episode"
    if float(episode["base_pos"][:, 2].min()) < 0.18:
        return False, "base_height"
    if _clip_fraction(episode["action_label"]) > clip_limit:
        return False, "action_clip"
    mean_vx_cmd = float(np.mean(episode["command"][:, 0]))
    progress = float(episode["base_pos"][-1, 0] - episode["base_pos"][0, 0])
    if mean_vx_cmd > 0.15 and progress < max(0.05, 0.15 * mean_vx_cmd * steps * 0.02):
        return False, "low_forward_progress"
    if mean_vx_cmd > 0.15 and "foot_pos" in episode:
        foot_clearance = np.ptp(episode["foot_pos"][:, :, 2], axis=0)
        if int(np.count_nonzero(foot_clearance > float(min_foot_clearance))) < 2:
            return False, "low_foot_clearance"
    if mean_vx_cmd > 0.15 and "foot_pos" in episode and "foot_contacts" in episode:
        slip = contact_slip_metrics(episode["foot_pos"], episode["foot_contacts"])
        if int(slip["contact_samples"]) > 20 and (
            float(slip["mean"]) > float(max_contact_slip_mean) or float(slip["p95"]) > float(max_contact_slip_p95)
        ):
            return False, "foot_sliding"
    if require_yaw_response:
        mean_yaw_cmd = float(np.mean(episode["command"][:, 2]))
        if abs(mean_yaw_cmd) >= float(yaw_cmd_threshold):
            yaw_delta = episode_yaw_delta(episode)
            if yaw_delta * mean_yaw_cmd < 0.0:
                return False, "wrong_yaw_direction"
            if abs(yaw_delta) < float(min_yaw_delta):
                return False, "low_yaw_response"
    return True, ""


def rollout_episode(
    env: A1TeacherEnv,
    teacher,
    rng: np.random.Generator,
    episode_steps: int,
    render: bool,
    debug_failed_gifs: bool,
    fixed_command: np.ndarray | None = None,
    render_every: int = 10,
    gif_width: int = 320,
    gif_height: int = 180,
    command_profile: str = "default",
    command_category: str | None = None,
) -> tuple[dict[str, np.ndarray], list[np.ndarray], dict[str, Any]]:
    """Roll out one teacher-controlled episode and return arrays, frames, and metadata.

    Callers rely on the returned value shape and semantics described here.
    """
    del debug_failed_gifs
    env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    teacher.reset(rng)
    if fixed_command is None and command_category is not None:
        command, command_kind = sample_category_command(rng, command_category)
    elif fixed_command is None:
        command, command_kind = sample_command(rng, profile=command_profile)
    else:
        command = fixed_command.astype(np.float32).copy()
        command_kind = "fixed"
    next_switch_step = int(rng.integers(100, 251))
    prev_action = np.zeros(12, dtype=np.float32)
    prev_reward = 0.0
    reset_flag = True
    phase = 0.0
    done_reason = ""
    frames: list[np.ndarray] = []
    buffers: dict[str, list] = defaultdict(list)

    for step in range(episode_steps):
        if fixed_command is None and step > 0 and step >= next_switch_step:
            if command_category is not None:
                command, command_kind = sample_category_command(rng, command_category)
            else:
                command, command_kind = sample_command(rng, profile=command_profile)
            next_switch_step = step + int(rng.integers(100, 251))
        state = env.get_state()
        output = teacher.compute(state, command)
        q_teacher = output["q_teacher"]
        phase = float(output["phase"])
        action_label = teacher.action_label(q_teacher)
        obs = env.make_obs(command, prev_action, prev_reward, reset_flag, phase)
        q, qdot = env.get_q_qdot()
        reward, done, info = env.step_q_des(q_teacher)
        next_state = env.get_state()
        _append_step(
            buffers,
            {
                "obs": obs,
                "action_label": action_label,
                "q_teacher": q_teacher,
                "q": q,
                "qdot": qdot,
                "command": command.copy(),
                "reward": np.float32(reward),
                "done": bool(done),
                "reset_flag": bool(reset_flag),
                "phase": np.float32(phase),
                "base_pos": state["base_pos"],
                "base_quat": state["base_quat"],
                "base_lin_vel_body": state["base_lin_vel_body"],
                "base_ang_vel_body": state["base_ang_vel_body"],
                "projected_gravity": state["projected_gravity"],
                "foot_contacts": state["foot_contacts"],
                "foot_pos": state["foot_pos"],
                "torque": next_state["torque"],
            },
        )
        if render and step % render_every == 0:
            frames.append(env.render_frame(width=gif_width, height=gif_height))
        prev_action = action_label
        prev_reward = float(reward)
        reset_flag = False
        if done:
            done_reason = str(info.get("done_reason", "done"))
            break

    episode = _stack_episode(buffers)
    meta = {"command_kind": command_kind, "done_reason": done_reason}
    return episode, frames, meta


def run_recording(args: argparse.Namespace) -> None:
    """Run the teacher demonstration recording loop from parsed CLI arguments.

    The routine owns the command or process lifecycle described by its arguments.
    """
    rng = np.random.default_rng(args.seed)
    env = A1TeacherEnv(args.xml_path, {"use_contacts": args.use_contacts})
    teacher = make_teacher(args.teacher, args.xml_path, env.policy_dt, profile=args.teacher_profile)
    fixed_command = parse_fixed_command(args.fixed_command)
    category_config = CATEGORY_COMMAND_RANGES.get(args.command_category, {}) if args.command_category else {}
    require_yaw_response = bool(
        args.require_yaw_response
        or args.command_profile == "turn_probe"
        or bool(category_config.get("require_yaw_response", False))
    )
    obs_dim = OBS_DIM_WITH_CONTACTS if args.use_contacts else OBS_DIM_NO_CONTACTS
    writer = DatasetWriter(
        args.out_dir,
        {
            "obs_dim": obs_dim,
            "action_dim": 12,
            "q_home": Q_HOME.astype(float).tolist(),
            "action_scale": ACTION_SCALE.astype(float).tolist(),
            "xml_path": str(args.xml_path),
            "joint_names": env.joint_names,
            "actuator_names": env.actuator_names,
            "teacher": args.teacher,
            "teacher_profile": args.teacher_profile,
            "teacher_config": FOOTSPACE_TEACHER_PROFILES.get(args.teacher_profile, {}) if args.teacher == "footspace" else {},
            "command_profile": args.command_profile,
            "command_profile_config": COMMAND_PROFILES[args.command_profile],
            "command_category": args.command_category,
            "command_category_config": category_config,
            "acceptance": {
                "clip_limit": float(args.clip_limit),
                "min_foot_clearance": float(args.min_foot_clearance),
                "max_contact_slip_mean": float(args.max_contact_slip_mean),
                "max_contact_slip_p95": float(args.max_contact_slip_p95),
                "require_yaw_response": require_yaw_response,
                "yaw_cmd_threshold": float(args.yaw_cmd_threshold),
                "min_yaw_delta": float(args.min_yaw_delta),
            },
        },
        resume=args.resume,
    )
    counters = initial_recording_counters(writer, args.out_dir, args.resume)
    accepted_steps = counters["accepted_steps"]
    accepted_eps = counters["accepted_eps"]
    attempted_eps = counters["attempted_eps"]
    saved_review_gifs = counters["saved_review_gifs"]
    pbar = tqdm(total=args.target_steps, initial=min(accepted_steps, args.target_steps), desc="accepted transitions")
    while accepted_steps < args.target_steps:
        if args.max_episodes and attempted_eps >= args.max_episodes:
            break
        if args.review_gifs and saved_review_gifs >= args.review_gifs:
            break
        episode_steps = int(rng.integers(300, 601))
        wants_gifs = args.gif_every > 0 or args.review_gifs > 0 or args.debug_failed_gifs
        wants_videos = args.save_videos or args.debug_failed_videos
        episode, frames, meta = rollout_episode(
            env,
            teacher,
            rng,
            episode_steps,
            render=False,
            debug_failed_gifs=args.debug_failed_gifs,
            fixed_command=fixed_command,
            render_every=args.render_every,
            gif_width=args.gif_width,
            gif_height=args.gif_height,
            command_profile=args.command_profile,
            command_category=args.command_category,
        )
        accepted, reject_reason = should_accept(
            episode,
            meta["done_reason"],
            clip_limit=args.clip_limit,
            min_foot_clearance=args.min_foot_clearance,
            max_contact_slip_mean=args.max_contact_slip_mean,
            max_contact_slip_p95=args.max_contact_slip_p95,
            require_yaw_response=require_yaw_response,
            yaw_cmd_threshold=args.yaw_cmd_threshold,
            min_yaw_delta=args.min_yaw_delta,
        )
        stats = episode_stats(episode, meta["done_reason"], accepted, reject_reason)
        stats["command_kind"] = meta["command_kind"]
        attempted_eps += 1
        if accepted:
            writer.write_episode(accepted_eps, episode, accepted=True, stats=stats)
            accepted_steps += len(episode["reward"])
            pbar.update(len(episode["reward"]))
            should_gif = wants_gifs and (
                accepted_eps < 5 or (args.gif_every > 0 and accepted_eps % args.gif_every == 0) or bool(args.review_gifs)
            )
            should_video = wants_videos and should_gif
            if should_gif or should_video:
                gif_fps = args.gif_fps or max(1, int(round(1.0 / (env.policy_dt * args.render_every))))
                video_fps = args.video_fps or max(1, int(round(1.0 / (env.policy_dt * args.render_every))))
                wrote = write_debug_exports(
                    writer=writer,
                    env=env,
                    ep_id=accepted_eps,
                    episode=episode,
                    stats=stats,
                    accepted=True,
                    render_every=args.render_every,
                    gif_width=args.gif_width,
                    gif_height=args.gif_height,
                    gif_fps=gif_fps,
                    save_gif=should_gif,
                    save_video=should_video,
                    video_fps=video_fps,
                )
                saved_review_gifs += 1
                if args.verbose and should_video and not wrote["video"]:
                    tqdm.write("video export skipped: MP4 backend unavailable")
            accepted_eps += 1
        else:
            writer.record_rejection(attempted_eps, stats)
            if args.debug_failed_gifs or args.debug_failed_videos:
                gif_fps = args.gif_fps or max(1, int(round(1.0 / (env.policy_dt * args.render_every))))
                video_fps = args.video_fps or max(1, int(round(1.0 / (env.policy_dt * args.render_every))))
                wrote = write_debug_exports(
                    writer=writer,
                    env=env,
                    ep_id=attempted_eps,
                    episode=episode,
                    stats=stats,
                    accepted=False,
                    render_every=args.render_every,
                    gif_width=args.gif_width,
                    gif_height=args.gif_height,
                    gif_fps=gif_fps,
                    save_gif=args.debug_failed_gifs,
                    save_video=args.debug_failed_videos,
                    video_fps=video_fps,
                )
                if args.verbose and args.debug_failed_videos and not wrote["video"]:
                    tqdm.write("failed video export skipped: MP4 backend unavailable")
        if args.verbose:
            tqdm.write(
                f"episode attempt={attempted_eps} accepted={accepted} "
                f"steps={len(episode['reward'])} reason={reject_reason or 'accepted'} "
                f"progress={stats['forward_progress']:.3f} clip={stats['clip_fraction']:.3f}"
            )
        writer.update_metadata(
            accepted_episodes=accepted_eps,
            attempted_episodes=attempted_eps,
            accepted_steps=accepted_steps,
            saved_review_gifs=saved_review_gifs,
        )
    pbar.close()
    print(json.dumps(writer.metadata, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for teacher demonstration recording.

    The returned namespace is consumed by the corresponding command-line entry point.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_path", default="assets/mujoco_menagerie/unitree_a1/scene.xml")
    parser.add_argument("--out_dir", default="datasets/a1_teacher_flat_v001")
    parser.add_argument("--target_steps", type=int, default=200000)
    parser.add_argument("--gif_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_contacts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug_failed_gifs", action="store_true")
    parser.add_argument("--review_gifs", type=int, default=0)
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--render_every", type=int, default=25)
    parser.add_argument("--gif_width", type=int, default=240)
    parser.add_argument("--gif_height", type=int, default=135)
    parser.add_argument("--gif_fps", type=int, default=0)
    parser.add_argument("--save_videos", action="store_true")
    parser.add_argument("--debug_failed_videos", action="store_true")
    parser.add_argument("--video_fps", type=int, default=0)
    parser.add_argument("--teacher", choices=("footspace",), default="footspace")
    parser.add_argument("--teacher_profile", choices=tuple(FOOTSPACE_TEACHER_PROFILES), default="strict_walk")
    parser.add_argument("--command_profile", choices=tuple(COMMAND_PROFILES), default="default")
    parser.add_argument("--command_category", choices=tuple(CATEGORY_COMMAND_RANGES), default=None)
    parser.add_argument("--fixed_command", nargs=3, type=float, default=None, metavar=("VX", "VY", "YAW"))
    parser.add_argument("--clip_limit", type=float, default=0.25)
    parser.add_argument("--min_foot_clearance", type=float, default=DEFAULT_MIN_FOOT_CLEARANCE)
    parser.add_argument("--max_contact_slip_mean", type=float, default=DEFAULT_MAX_CONTACT_SLIP_MEAN)
    parser.add_argument("--max_contact_slip_p95", type=float, default=DEFAULT_MAX_CONTACT_SLIP_P95)
    parser.add_argument("--require_yaw_response", action="store_true")
    parser.add_argument("--yaw_cmd_threshold", type=float, default=0.2)
    parser.add_argument("--min_yaw_delta", type=float, default=0.25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the teacher demonstration recorder command-line entry point.

    This is the direct execution entry point for the module.
    """
    run_recording(parse_args())


if __name__ == "__main__":
    main()
