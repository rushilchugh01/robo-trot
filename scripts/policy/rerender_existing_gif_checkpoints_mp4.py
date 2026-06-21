#!/usr/bin/env python3
"""Rerender MP4 rollouts for checkpoint/command pairs that already have GIFs.

The script uses MuJoCo checkpoints as the source of truth; it does not transcode
existing GIFs.  It appends dashboard-visible eval rows that point at MP4 files.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def _early_refuse_unsafe_cli(argv: list[str]) -> None:
    """Refuse broad forced rerenders before importing MuJoCo or torch.

    This protects live training from stale background workers that repeatedly
    launch all-model, forced media rerenders.
    """
    if os.environ.get("ROBO_TROT_ALLOW_BROAD_MP4_RERENDER") == "I_UNDERSTAND_THIS_STOPS_TRAINING":
        return
    models = "mlp,txl"
    shard_count = 1
    force = "--force" in argv
    for index, item in enumerate(argv):
        if item == "--models" and index + 1 < len(argv):
            models = argv[index + 1]
        elif item.startswith("--models="):
            models = item.split("=", 1)[1]
        elif item == "--shard_count" and index + 1 < len(argv):
            try:
                shard_count = int(argv[index + 1])
            except ValueError:
                shard_count = 1
        elif item.startswith("--shard_count="):
            try:
                shard_count = int(item.split("=", 1)[1])
            except ValueError:
                shard_count = 1
    model_set = {model.strip() for model in models.split(",") if model.strip()}
    if model_set - {"txl"} and (force or shard_count > 1):
        raise SystemExit(
            "Refusing forced or sharded non-TXL MP4 rerender while training may be active. "
            "Set ROBO_TROT_ALLOW_BROAD_MP4_RERENDER=1 to override intentionally."
        )


_early_refuse_unsafe_cli(sys.argv[1:])

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robo_trot.sim.a1_teacher_env import A1TeacherEnv
from robo_trot.training.action_mapping_audit import audit_action_mapping
from robo_trot.training.evaluate_checkpoint import (
    FIXED_EVAL_COMMANDS,
    aggregate_eval_rows,
    append_eval_metrics,
    load_policy_from_checkpoint,
    rollout_result_to_metrics,
    run_policy_eval_episode,
)
from robo_trot.training.policy_rollout import load_dataset_contract, validate_env_contract


COMMANDS = {command.label: command for command in FIXED_EVAL_COMMANDS}
COMMAND_ORDER = {label: index for index, label in enumerate(COMMANDS)}


class ProgressPrinter:
    """Emit compact progress for one MuJoCo MP4 rerender.

    The runner uses this to keep attached shells and log tails visibly active.
    """

    def __init__(self, model: str, update: int, label: str, every_steps: int) -> None:
        """Store rollout identity and throttling state.

        Progress is emitted by step interval or when wall time has gone quiet.
        """
        self.model = model
        self.update = int(update)
        self.label = label
        self.every_steps = max(1, int(every_steps))
        self.last_step = 0
        self.last_wall = time.monotonic()

    def __call__(self, step: int, target_steps: int, frame_count: int) -> None:
        """Print a one-line progress heartbeat when enough work has advanced.

        The callback receives current rollout step, target steps, and frames held.
        """
        now = time.monotonic()
        should_print = (
            step == 1
            or step >= target_steps
            or step - self.last_step >= self.every_steps
            or now - self.last_wall >= 15.0
        )
        if not should_print:
            return
        self.last_step = int(step)
        self.last_wall = now
        print(
            f"[rerender] progress {self.model} step_{self.update:09d} {self.label} "
            f"rollout_step={step}/{target_steps} frames={frame_count}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    """Parse rerender command options.

    Defaults target the local `bc_compare_v001` run and produce 10 second MP4s.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default="runs/bc_compare_v001")
    parser.add_argument("--models", default="mlp,txl")
    parser.add_argument("--xml_path", default="assets/mujoco_menagerie/unitree_a1/scene.xml")
    parser.add_argument(
        "--dataset_metadata",
        default="datasets/a1_teacher_flat_7m_v001_main/shards/shard_00_forward/metadata.json",
    )
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max_groups", type=int, default=0)
    parser.add_argument("--max_commands", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--order", choices=("oldest", "newest"), default="oldest")
    return parser.parse_args()


def main() -> None:
    """Run the exact GIF-derived checkpoint tasks.

    One dashboard row is appended after each command so interrupted runs still
    expose completed MP4s in the dashboard.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass
    args = parse_args()
    run_dir = Path(args.run_dir)
    models = {item.strip() for item in str(args.models).split(",") if item.strip()}
    enforce_operational_safety(args, models)
    groups = discover_groups(run_dir, models)
    groups = sorted(groups, key=lambda group: (group["model"], group["update"]))
    if args.order == "newest":
        groups = list(reversed(groups))
    shard_count = max(1, int(args.shard_count))
    shard_index = int(args.shard_index)
    if shard_index < 0 or shard_index >= shard_count:
        raise SystemExit(f"shard_index must be in [0, {shard_count - 1}]")
    groups = [group for index, group in enumerate(groups) if index % shard_count == shard_index]
    if int(args.max_groups) > 0:
        groups = groups[: int(args.max_groups)]
    print(
        f"[rerender] groups={len(groups)} commands={sum(len(group['labels']) for group in groups)} "
        f"seconds={args.seconds} fps={args.fps} force={args.force} "
        f"shard={shard_index}/{shard_count}",
        flush=True,
    )
    command_budget = int(args.max_commands)
    command_count = 0
    for group in groups:
        remaining = 0 if command_budget <= 0 else command_budget - command_count
        if command_budget > 0 and remaining <= 0:
            break
        command_count += rerender_group(args, run_dir, group, remaining)
    print(f"[rerender] finished rendered_or_verified_commands={command_count}", flush=True)


def enforce_operational_safety(args: argparse.Namespace, models: set[str]) -> None:
    """Refuse broad background rerenders unless explicitly enabled.

    TXL-only MP4 backfills are allowed; forced MLP/all-model rerenders can starve
    live training and require an environment opt-in.
    """
    allow_broad = os.environ.get("ROBO_TROT_ALLOW_BROAD_MP4_RERENDER") == "I_UNDERSTAND_THIS_STOPS_TRAINING"
    touches_non_txl = bool(models - {"txl"})
    uses_parallel_shards = int(args.shard_count) > 1
    if allow_broad or not touches_non_txl:
        return
    if bool(args.force) or uses_parallel_shards:
        raise SystemExit(
            "Refusing forced or sharded non-TXL MP4 rerender while training may be active. "
            "Set ROBO_TROT_ALLOW_BROAD_MP4_RERENDER=1 to override intentionally."
        )


def discover_groups(run_dir: Path, models: set[str]) -> list[dict[str, Any]]:
    """Return unique model/update/command groups represented by existing GIFs.

    GIFs under archives and active eval output are both considered.
    """
    grouped: dict[tuple[str, int], set[str]] = defaultdict(set)
    sources: dict[tuple[str, int], list[str]] = defaultdict(list)
    for gif in sorted(run_dir.glob("**/*.gif")):
        parsed = parse_gif_path(gif)
        if parsed is None:
            continue
        model, update, label = parsed
        if model not in models or label not in COMMANDS:
            continue
        grouped[(model, update)].add(label)
        sources[(model, update)].append(gif.as_posix())
    output: list[dict[str, Any]] = []
    for (model, update), labels in grouped.items():
        checkpoint = resolve_checkpoint(run_dir, model, update)
        if checkpoint is None:
            print(f"[rerender] missing checkpoint for {model} step_{update:09d}; skipping", flush=True)
            continue
        output.append(
            {
                "model": model,
                "update": update,
                "labels": sorted(labels, key=lambda label: COMMAND_ORDER[label]),
                "checkpoint": checkpoint,
                "sources": sorted(sources[(model, update)]),
            }
        )
    return output


def parse_gif_path(path: Path) -> tuple[str, int, str] | None:
    """Extract model, update, and command label from one GIF path.

    Expected filenames are `mlp_vx03.gif` or `txl_yaw_left.gif` below a
    `step_000001000` directory.
    """
    stem = path.stem
    if "_" not in stem:
        return None
    model, label = stem.split("_", 1)
    if model not in {"mlp", "txl"}:
        return None
    update = None
    for part in path.parts:
        if part.startswith("step_"):
            try:
                update = int(part.removeprefix("step_"))
            except ValueError:
                return None
            break
    if update is None:
        return None
    return model, update, label


def resolve_checkpoint(run_dir: Path, model: str, update: int) -> Path | None:
    """Find the checkpoint directory for one GIF-derived model/update pair.

    Current run checkpoints are preferred; archive checkpoints are a fallback.
    """
    step_name = f"step_{update:09d}"
    candidates = [
        run_dir / model / "checkpoints" / step_name,
        run_dir / "archive" / f"{model}_checkpoints_backup_20260621T110251Z" / "checkpoints" / step_name,
    ]
    candidates.extend(sorted((run_dir / "archive").glob(f"**/{model}/checkpoints/{step_name}")))
    candidates.extend(sorted((run_dir / "archive").glob(f"**/checkpoints/{step_name}")))
    for candidate in candidates:
        if (candidate / "_SUCCESS").exists() and (candidate / "model.pt").exists():
            return candidate
    return None


def rerender_group(args: argparse.Namespace, run_dir: Path, group: dict[str, Any], remaining: int) -> int:
    """Rerender one checkpoint group and append dashboard rows.

    Returns the number of command labels rendered or verified for this group.
    """
    model = str(group["model"])
    update = int(group["update"])
    labels = list(group["labels"])
    if remaining > 0:
        labels = labels[:remaining]
    if not bool(args.force) and group_has_all_valid_mp4s(run_dir, model, update, labels, float(args.seconds)):
        print(f"[rerender] skip complete {model} step_{update:09d} labels={labels}", flush=True)
        return len(labels)
    checkpoint = Path(group["checkpoint"])
    print(f"[rerender] group {model} step_{update:09d} labels={labels}", flush=True)
    policy = load_policy_from_checkpoint(checkpoint, model)
    env = A1TeacherEnv(args.xml_path, {"episode_seconds": float(args.seconds), "use_contacts": True})
    if args.dataset_metadata:
        validate_env_contract(env, load_dataset_contract(args.dataset_metadata))
    audit = audit_action_mapping(env, action_value=0.5, settle_steps=3, min_observed_delta=1e-5)
    if not all(result.passed for result in audit):
        failed = [result.reason for result in audit if not result.passed]
        raise RuntimeError(f"action mapping audit failed for {model} step_{update:09d}: {failed}")
    rows: list[dict[str, Any]] = []
    rendered = 0
    for label in labels:
        existing_path = first_valid_mp4_path(run_dir, model, update, label, float(args.seconds))
        if existing_path is not None and not bool(args.force):
            print(f"[rerender] verified existing {existing_path}", flush=True)
            rows.append(media_only_row(run_dir, checkpoint, model, update, label, existing_path))
            append_partial_row(run_dir, rows, model, update)
            rendered += 1
            continue
        media_path = canonical_mp4_path(run_dir, model, update, label)
        media_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[rerender] render {model} step_{update:09d} {label} -> {media_path}", flush=True)
        progress_every = max(1, int(round(1.0 / float(env.policy_dt))))
        result = run_policy_eval_episode(
            env=env,
            policy=policy,
            eval_command=COMMANDS[label],
            seconds=float(args.seconds),
            seed=int(args.seed) + update + COMMAND_ORDER[label],
            save_media=media_path,
            gif_fps=int(args.fps),
            gif_seconds=float(args.seconds),
            gif_width=int(args.width),
            gif_height=int(args.height),
            progress_callback=ProgressPrinter(model, update, label, progress_every),
        )
        print(f"[rerender] encoded {media_path}", flush=True)
        row = rollout_result_to_metrics(result, model, checkpoint, update, root_dir=run_dir)
        row["rerendered_from_existing_gif"] = True
        rows.append(row)
        append_partial_row(run_dir, rows, model, update)
        rendered += 1
    return rendered


def group_has_all_valid_mp4s(run_dir: Path, model: str, update: int, labels: list[str], seconds: float) -> bool:
    """Return whether all labels already have valid rerendered MP4 files.

    This avoids reloading policies and MuJoCo environments for complete groups.
    """
    if not labels:
        return True
    for label in labels:
        if first_valid_mp4_path(run_dir, model, update, label, seconds) is None:
            return False
    return True


def canonical_mp4_path(run_dir: Path, model: str, update: int, label: str) -> Path:
    """Return the dashboard-primary MP4 path for one checkpoint command.

    New broad rerenders write beside evaluator media under `eval/media`.
    """
    return run_dir / "eval" / "media" / f"step_{update:09d}" / f"{model}_{label}.mp4"


def mp4_path_candidates(run_dir: Path, model: str, update: int, label: str) -> list[Path]:
    """Return known MP4 locations for old and current rerender helpers.

    The canonical evaluator media path is preferred for dashboard continuity.
    """
    return [
        canonical_mp4_path(run_dir, model, update, label),
        run_dir / "eval" / "rerendered_media" / f"step_{update:09d}" / f"{model}_{label}.mp4",
        run_dir / "eval" / "gifs" / f"step_{update:09d}" / f"{model}_{label}.mp4",
    ]


def first_valid_mp4_path(run_dir: Path, model: str, update: int, label: str, seconds: float) -> Path | None:
    """Return the first valid MP4 path for one checkpoint command.

    Missing or invalid files are ignored so callers can decide whether to render.
    """
    for path in mp4_path_candidates(run_dir, model, update, label):
        if valid_mp4(path, seconds):
            return path
    return None


def media_only_row(run_dir: Path, checkpoint: Path, model: str, update: int, label: str, media_path: Path) -> dict[str, Any]:
    """Return a dashboard row for an already rendered command MP4.

    This keeps resumed runs dashboard-visible even when reward terms are not
    recomputed for skipped clips.
    """
    return {
        "model_type": model,
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_update": int(update),
        "command_label": label,
        "media_path": media_path.relative_to(run_dir).as_posix(),
        "reward_terms": {},
        "rerendered_from_existing_gif": True,
        "wall_time": time.time(),
    }


def append_partial_row(run_dir: Path, rows: list[dict[str, Any]], model: str, update: int) -> None:
    """Append a dashboard row with all media rendered so far for a checkpoint.

    The dashboard dedupes by model/checkpoint and later rows win.
    """
    aggregate = aggregate_eval_rows(rows, model, update)
    aggregate["rerendered_from_existing_gif"] = True
    aggregate["rerendered_media_kind"] = "checkpoint_mujoco_mp4"
    append_eval_metrics(run_dir / "eval" / "metrics.jsonl", aggregate)


def valid_mp4(path: Path, seconds: float) -> bool:
    """Return whether a file is a browser-compatible 8-10 second MP4.

    The duration tolerance is intentionally broad enough for muxing metadata.
    """
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        output = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,pix_fmt,duration,avg_frame_rate,width,height",
                "-of",
                "json",
                path.as_posix(),
            ],
            text=True,
        )
        stream = json.loads(output)["streams"][0]
    except Exception:
        return False
    duration = float(stream.get("duration") or 0.0)
    return (
        stream.get("codec_name") == "h264"
        and stream.get("pix_fmt") == "yuv420p"
        and 7.5 <= duration <= max(10.5, float(seconds) + 0.5)
    )


if __name__ == "__main__":
    main()
