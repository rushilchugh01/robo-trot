from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from robo_trot.training.checkpointing import update_checkpoint_alias
from robo_trot.training.evaluate_checkpoint import (
    aggregate_eval_rows,
    append_eval_metrics,
    collect_policy_checkpoints,
    evaluate_checkpoint_set,
    evaluate_dataset_action_loss,
    load_policy_from_checkpoint,
    rollout_result_to_metrics,
    run_policy_eval_episode,
    select_eval_commands,
)
from robo_trot.training.torch_utils import configure_single_thread_torch


@dataclass(frozen=True)
class BackfillTask:
    """Serializable checkpoint evaluation task.

    Ray and local process pools pass this small payload to worker functions.
    """

    model_type: str
    checkpoint: str
    checkpoint_update: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line flags for checkpoint eval backfill.

    Defaults match the scratch BC run media cadence: MP4s every 100 updates.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--models", default="mlp,txl")
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--min_update", type=int, default=0)
    parser.add_argument("--max_update", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpus_per_task", type=float, default=1.0)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--xml_path", default="assets/mujoco_menagerie/unitree_a1/scene.xml")
    parser.add_argument("--dataset_dir", default="datasets/a1_teacher_flat_7m_v001_main")
    parser.add_argument(
        "--dataset_metadata",
        default="datasets/a1_teacher_flat_7m_v001_main/shards/shard_00_forward/metadata.json",
    )
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--command_labels", default="vx03")
    parser.add_argument("--media_seconds", type=float, default=10.0)
    parser.add_argument("--media_fps", type=int, default=30)
    parser.add_argument("--media_width", type=int, default=320)
    parser.add_argument("--media_height", type=int, default=180)
    parser.add_argument("--dataset_eval_split", default="test")
    parser.add_argument("--dataset_eval_batch_size", type=int, default=4096)
    parser.add_argument("--dataset_eval_max_batches", type=int, default=16)
    parser.add_argument("--sequence_length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ray", action="store_true")
    parser.add_argument("--ray_address", default="auto")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run checkpoint eval backfill from the command line.

    This entry point is used by the script wrapper in `scripts/policy`.
    """
    args = parse_args(argv)
    run_backfill(args)


def run_backfill(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Evaluate missing checkpoint media and metrics for a run directory.

    Tasks are skipped when a complete metrics row and all command MP4s exist.
    """
    configure_single_thread_torch()
    run_dir = Path(args.run_dir)
    models = parse_model_selector(args.models)
    tasks = discover_backfill_tasks(
        run_dir=run_dir,
        models=models,
        eval_every=int(args.eval_every),
        min_update=int(args.min_update),
        max_update=int(args.max_update),
        force=bool(args.force),
        command_labels=args.command_labels,
    )
    print(f"[backfill] tasks={len(tasks)} models={','.join(models)} run_dir={run_dir}", flush=True)
    if not tasks:
        return []
    config = vars(args).copy()
    if bool(args.ray):
        return _run_backfill_with_ray(config, tasks)
    return _run_backfill_locally(config, tasks)


def parse_model_selector(raw_value: str) -> tuple[str, ...]:
    """Return normalized model names from a comma-separated selector.

    The selector accepts `mlp`, `txl`, or both without duplicates.
    """
    models = tuple(part.strip().lower() for part in str(raw_value).split(",") if part.strip())
    if not models:
        raise ValueError("--models must select at least one model")
    invalid = [model for model in models if model not in {"mlp", "txl"}]
    if invalid:
        raise ValueError(f"unsupported model selector(s): {invalid}")
    return tuple(dict.fromkeys(models))


def discover_backfill_tasks(
    run_dir: str | Path,
    models: tuple[str, ...],
    eval_every: int,
    min_update: int = 0,
    max_update: int = 0,
    force: bool = False,
    command_labels: str | list[str] | tuple[str, ...] | None = None,
) -> list[BackfillTask]:
    """Return checkpoint eval tasks that still need dashboard-ready outputs.

    Complete checkpoint directories are required, and incomplete eval rows are rerun.
    """
    root = Path(run_dir)
    tasks: list[BackfillTask] = []
    for model_type in models:
        for record in collect_policy_checkpoints(root, model_type):
            update = int(record.update)
            if update <= 0 or update % max(1, int(eval_every)) != 0:
                continue
            if int(min_update) > 0 and update < int(min_update):
                continue
            if int(max_update) > 0 and update > int(max_update):
                continue
            if not bool(force) and checkpoint_eval_complete(root, model_type, update, command_labels=command_labels):
                continue
            tasks.append(BackfillTask(model_type=model_type, checkpoint=record.path.as_posix(), checkpoint_update=update))
    model_order = {"mlp": 0, "txl": 1}
    return sorted(tasks, key=lambda task: (int(task.checkpoint_update), model_order[task.model_type]))


def checkpoint_eval_complete(
    run_dir: str | Path,
    model_type: str,
    checkpoint_update: int,
    command_labels: str | list[str] | tuple[str, ...] | None = None,
) -> bool:
    """Return whether dashboard eval output is complete for one checkpoint.

    A complete row must include reward, dataset loss, and selected command MP4s.
    """
    row = latest_eval_row(Path(run_dir) / "eval" / "metrics.jsonl", model_type, int(checkpoint_update))
    if row is None or "eval_reward_mean" not in row or row.get("eval_error"):
        return False
    if "dataset_eval_action_mse" not in row and "dataset_eval_error" not in row:
        return False
    media_paths = row.get("media_paths")
    if not isinstance(media_paths, dict):
        return False
    labels = {command.label for command in select_eval_commands(command_labels)}
    if labels - set(str(label) for label in media_paths):
        return False
    root = Path(run_dir)
    return all(_media_file_exists(root, media_paths[label]) for label in labels)


def latest_eval_row(metrics_path: str | Path, model_type: str, checkpoint_update: int) -> dict[str, Any] | None:
    """Return the latest JSONL row for a model/update pair.

    Invalid or partially written JSON lines are ignored during recovery.
    """
    path = Path(metrics_path)
    if not path.exists():
        return None
    selected: dict[str, Any] | None = None
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("model_type") == model_type and int(row.get("checkpoint_update", -1)) == int(checkpoint_update):
            selected = row
    return selected


def evaluate_backfill_task(config: dict[str, Any], task_dict: dict[str, Any]) -> dict[str, Any]:
    """Run one checkpoint evaluation task and append a locked metrics row.

    Workers use the same MuJoCo evaluator as the live checkpoint watcher.
    """
    configure_single_thread_torch()
    task = BackfillTask(**task_dict)
    run_dir = Path(config["run_dir"])
    eval_dir = run_dir / "eval"
    try:
        rows = evaluate_checkpoint_set(
            checkpoint=task.checkpoint,
            model_type=task.model_type,
            xml_path=config["xml_path"],
            dataset_metadata=_optional_path(config.get("dataset_metadata")),
            out_dir=eval_dir,
            seconds=float(config["seconds"]),
            checkpoint_update=int(task.checkpoint_update),
            gif_every_eval=1,
            gif_fps=int(config["media_fps"]),
            gif_seconds=float(config["media_seconds"]),
            gif_width=int(config["media_width"]),
            gif_height=int(config["media_height"]),
            eval_index=0,
            seed=int(config["seed"]) + int(task.checkpoint_update),
            command_labels=config.get("command_labels", "vx03"),
        )
        row = aggregate_eval_rows(rows, task.model_type, int(task.checkpoint_update))
        row.update(_dataset_action_metrics(config, task))
        append_eval_metrics_locked(run_dir, row)
        return {"status": "ok", "model_type": task.model_type, "checkpoint_update": int(task.checkpoint_update)}
    except Exception as exc:
        row = {
            "model_type": task.model_type,
            "checkpoint_update": int(task.checkpoint_update),
            "eval_error": f"{type(exc).__name__}: {exc}",
            "wall_time": time.time(),
        }
        append_eval_metrics_locked(run_dir, row)
        return {
            "status": "error",
            "model_type": task.model_type,
            "checkpoint_update": int(task.checkpoint_update),
            "error": row["eval_error"],
        }


def evaluate_backfill_chunk(config: dict[str, Any], task_dicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Evaluate a sequence of checkpoints with one MuJoCo env setup.

    The action-mapping audit and dataset contract validation run once per worker chunk.
    """
    configure_single_thread_torch()
    tasks = [BackfillTask(**task_dict) for task_dict in task_dicts]
    try:
        env = _build_checked_eval_env(config)
    except Exception as exc:
        return [_append_task_error(config, task, exc) for task in tasks]
    results: list[dict[str, Any]] = []
    for task in tasks:
        result = _evaluate_backfill_task_with_env(config, task, env)
        results.append(result)
        print(
            f"[backfill-worker] {task.model_type} step_{int(task.checkpoint_update):09d} {result['status']}",
            flush=True,
        )
    return results


def chunk_backfill_tasks(tasks: list[BackfillTask], chunk_size: int) -> list[list[BackfillTask]]:
    """Split tasks into fixed-size chunks for Ray worker reuse.

    Small chunks keep load balancing responsive while amortizing MuJoCo setup.
    """
    size = max(1, int(chunk_size))
    return [tasks[index : index + size] for index in range(0, len(tasks), size)]


def append_eval_metrics_locked(run_dir: str | Path, row: dict[str, Any]) -> None:
    """Append one eval row while holding a run-local file lock.

    The lock also protects best-eval alias updates from concurrent backfill workers.
    """
    root = Path(run_dir)
    metrics_path = root / "eval" / "metrics.jsonl"
    lock_path = root / "eval" / ".metrics.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            append_eval_metrics(metrics_path, row)
            _update_best_eval_alias(root, str(row.get("model_type", "")))
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _run_backfill_with_ray(config: dict[str, Any], tasks: list[BackfillTask]) -> list[dict[str, Any]]:
    """Dispatch checkpoint eval tasks to an existing Ray cluster.

    Ray resource reservations throttle concurrency to the cluster CPU budget.
    """
    try:
        import ray
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("ray is required when --ray is passed") from exc
    ray.init(address=config.get("ray_address", "auto"))
    chunks = chunk_backfill_tasks(tasks, int(config.get("chunk_size", 1)))
    remote_eval = ray.remote(num_cpus=float(config.get("cpus_per_task", 1.0)))(evaluate_backfill_chunk)
    pending = [remote_eval.remote(config, [asdict(task) for task in chunk]) for chunk in chunks]
    results: list[dict[str, Any]] = []
    while pending:
        done, pending = ray.wait(pending, num_returns=1)
        chunk_results = ray.get(done[0])
        results.extend(chunk_results)
        for result in chunk_results:
            print(
                f"[backfill] {len(results)}/{len(tasks)} {result['model_type']} "
                f"step_{int(result['checkpoint_update']):09d} {result['status']}",
                flush=True,
            )
    return results


def _run_backfill_locally(config: dict[str, Any], tasks: list[BackfillTask]) -> list[dict[str, Any]]:
    """Run checkpoint eval tasks in local processes.

    This fallback keeps the backfill usable without a Ray runtime.
    """
    workers = max(1, int(config.get("workers", 1)))
    if workers == 1:
        results: list[dict[str, Any]] = []
        for chunk in chunk_backfill_tasks(tasks, int(config.get("chunk_size", 1))):
            results.extend(evaluate_backfill_chunk(config, [asdict(task) for task in chunk]))
        return results
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        chunks = chunk_backfill_tasks(tasks, int(config.get("chunk_size", 1)))
        futures = [pool.submit(evaluate_backfill_chunk, config, [asdict(task) for task in chunk]) for chunk in chunks]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            chunk_results = future.result()
            results.extend(chunk_results)
            print(f"[backfill] chunk {index}/{len(chunks)} completed total_results={len(results)}", flush=True)
    return results


def _evaluate_backfill_task_with_env(config: dict[str, Any], task: BackfillTask, env: Any) -> dict[str, Any]:
    """Evaluate one checkpoint using an already validated MuJoCo env.

    Rows are appended immediately so interrupted chunk workers keep progress.
    """
    run_dir = Path(config["run_dir"])
    try:
        policy = load_policy_from_checkpoint(task.checkpoint, task.model_type)
        rows = []
        eval_dir = run_dir / "eval"
        step_dir = eval_dir / "media" / f"step_{int(task.checkpoint_update):09d}"
        for command_index, eval_command in enumerate(select_eval_commands(config.get("command_labels", "vx03"))):
            media_path = step_dir / f"{task.model_type}_{eval_command.label}.mp4"
            result = run_policy_eval_episode(
                env=env,
                policy=policy,
                eval_command=eval_command,
                seconds=float(config["seconds"]),
                seed=int(config["seed"]) + int(task.checkpoint_update) + int(command_index),
                save_media=media_path,
                gif_fps=int(config["media_fps"]),
                gif_seconds=float(config["media_seconds"]),
                gif_width=int(config["media_width"]),
                gif_height=int(config["media_height"]),
            )
            rows.append(
                rollout_result_to_metrics(
                    result,
                    task.model_type,
                    Path(task.checkpoint),
                    int(task.checkpoint_update),
                    root_dir=run_dir,
                )
            )
        row = aggregate_eval_rows(rows, task.model_type, int(task.checkpoint_update))
        row.update(_dataset_action_metrics_for_policy(config, task, policy))
        append_eval_metrics_locked(run_dir, row)
        return {"status": "ok", "model_type": task.model_type, "checkpoint_update": int(task.checkpoint_update)}
    except Exception as exc:
        return _append_task_error(config, task, exc)


def _append_task_error(config: dict[str, Any], task: BackfillTask, exc: Exception) -> dict[str, Any]:
    """Append one checkpoint eval error row and return worker status.

    Error rows prevent silent loss when a checkpoint cannot be evaluated.
    """
    row = {
        "model_type": task.model_type,
        "checkpoint_update": int(task.checkpoint_update),
        "eval_error": f"{type(exc).__name__}: {exc}",
        "wall_time": time.time(),
    }
    append_eval_metrics_locked(Path(config["run_dir"]), row)
    return {
        "status": "error",
        "model_type": task.model_type,
        "checkpoint_update": int(task.checkpoint_update),
        "error": row["eval_error"],
    }


def _build_checked_eval_env(config: dict[str, Any]) -> Any:
    """Create one MuJoCo eval environment and validate fixed contracts.

    The action mapping audit is intentionally once per worker chunk.
    """
    from robo_trot.sim.a1_teacher_env import A1TeacherEnv
    from robo_trot.training.action_mapping_audit import audit_action_mapping
    from robo_trot.training.policy_rollout import load_dataset_contract, validate_env_contract

    env = A1TeacherEnv(config["xml_path"], {"episode_seconds": float(config["seconds"]), "use_contacts": True})
    dataset_metadata = _optional_path(config.get("dataset_metadata"))
    if dataset_metadata is not None:
        validate_env_contract(env, load_dataset_contract(dataset_metadata))
    audit_results = audit_action_mapping(env, action_value=0.5, settle_steps=3, min_observed_delta=1e-5)
    if not all(result.passed for result in audit_results):
        failed = [result.reason for result in audit_results if not result.passed]
        raise ValueError(f"action mapping audit failed before eval: {failed}")
    return env


def _dataset_action_metrics(config: dict[str, Any], task: BackfillTask) -> dict[str, Any]:
    """Compute held-out action-label metrics for one checkpoint.

    Errors are recorded in the eval row so MuJoCo media remains visible.
    """
    try:
        policy = load_policy_from_checkpoint(task.checkpoint, task.model_type)
        return _dataset_action_metrics_for_policy(config, task, policy)
    except Exception as exc:
        return {
            "dataset_eval_split": str(config.get("dataset_eval_split", "test")),
            "dataset_eval_error": f"{type(exc).__name__}: {exc}",
        }


def _dataset_action_metrics_for_policy(config: dict[str, Any], task: BackfillTask, policy: Any) -> dict[str, Any]:
    """Compute held-out action-label metrics for an already loaded policy.

    Reusing the rollout policy avoids loading checkpoint weights twice.
    """
    try:
        return evaluate_dataset_action_loss(
            policy=policy,
            model_type=task.model_type,
            dataset_dir=config["dataset_dir"],
            split=str(config.get("dataset_eval_split", "test")),
            batch_size=int(config.get("dataset_eval_batch_size", 4096)),
            sequence_length=int(config.get("sequence_length", 64)),
            max_batches=int(config.get("dataset_eval_max_batches", 16)),
            seed=int(config.get("seed", 0)) + int(task.checkpoint_update),
        )
    except Exception as exc:
        return {
            "dataset_eval_split": str(config.get("dataset_eval_split", "test")),
            "dataset_eval_error": f"{type(exc).__name__}: {exc}",
        }


def _update_best_eval_alias(run_dir: Path, model_type: str) -> None:
    """Point `best_eval_reward` at the highest-reward evaluated checkpoint.

    Rows with errors or missing checkpoint directories are ignored.
    """
    if model_type not in {"mlp", "txl"}:
        return
    metrics_path = run_dir / "eval" / "metrics.jsonl"
    if not metrics_path.exists():
        return
    best_row: dict[str, Any] | None = None
    for line in metrics_path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("model_type") != model_type or "eval_reward_mean" not in row:
            continue
        if best_row is None or float(row["eval_reward_mean"]) > float(best_row["eval_reward_mean"]):
            best_row = row
    if best_row is None:
        return
    checkpoint_dir = run_dir / model_type / "checkpoints" / f"step_{int(best_row['checkpoint_update']):09d}"
    if checkpoint_dir.exists():
        update_checkpoint_alias(run_dir / model_type / "checkpoints" / "best_eval_reward", checkpoint_dir)


def _media_file_exists(run_dir: Path, media_path: Any) -> bool:
    """Return whether a dashboard media reference points to a non-empty file.

    Relative media paths are resolved from the run directory.
    """
    path = Path(str(media_path))
    if not path.is_absolute():
        path = run_dir / path
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _optional_path(value: Any) -> str | None:
    """Normalize optional path-like configuration values.

    Empty strings are treated as unset for evaluator compatibility.
    """
    if value is None or str(value).strip() == "":
        return None
    return str(value)
