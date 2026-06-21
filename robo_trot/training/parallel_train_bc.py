from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robo_trot.training.checkpointing import find_complete_checkpoints, save_torch_checkpoint_atomic, update_checkpoint_alias
from robo_trot.training.dataset import BehaviorCloningDataset
from robo_trot.training.evaluate_checkpoint import (
    aggregate_eval_rows,
    append_eval_metrics,
    collect_policy_checkpoints,
    evaluate_checkpoint_set,
)
from robo_trot.training.torch_utils import configure_single_thread_torch, import_torch


@dataclass(frozen=True)
class TrainGroupConfig:
    """Runtime configuration for one model training group.

    The orchestrator creates separate configs for MLP and TXL processes.
    """

    model_type: str
    dataset_dir: str
    out_dir: str
    workers: int
    cores: tuple[int, ...]
    batch_size: int
    sequence_length: int
    lr: float
    max_updates: int
    metrics_every: int
    checkpoint_every: int
    seed: int
    txl_memory_length: int
    resume: bool = False
    obs_dim: int = 56
    action_dim: int = 12


@dataclass
class TXLStreamState:
    """Mutable per-worker state for contiguous TXL episode streams.

    Memory is owned by the worker process and detached between chunks.
    """

    episode_indices: np.ndarray
    offsets: np.ndarray
    memory: Any | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for parallel BC training.

    Defaults match the requested local MLP/TXL comparison process layout.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--models", default="mlp,txl")
    parser.add_argument("--mlp_workers", type=int, default=4)
    parser.add_argument("--txl_workers", type=int, default=4)
    parser.add_argument("--eval_workers", type=int, default=1)
    parser.add_argument("--mlp_cores", default="0,1")
    parser.add_argument("--txl_cores", default="2,3")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--sequence_length", type=int, default=64)
    parser.add_argument("--txl_memory_seconds", type=float, default=20.0)
    parser.add_argument("--txl_memory_length", type=int, default=None)
    parser.add_argument("--policy_dt", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_updates", type=int, default=200000)
    parser.add_argument("--metrics_every", type=int, default=100)
    parser.add_argument("--checkpoint_every", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--gif_every_eval", type=int, default=1)
    parser.add_argument("--eval_gif_fps", type=int, default=30)
    parser.add_argument("--eval_gif_seconds", type=float, default=10.0)
    parser.add_argument("--eval_gif_width", type=int, default=320)
    parser.add_argument("--eval_gif_height", type=int, default=180)
    parser.add_argument("--dataset_eval_split", default="test")
    parser.add_argument("--dataset_eval_batch_size", type=int, default=4096)
    parser.add_argument("--dataset_eval_max_batches", type=int, default=16)
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--dashboard_host", default="0.0.0.0")
    parser.add_argument("--dashboard_port", type=int, default=8002)
    parser.add_argument("--xml_path", default="assets/mujoco_menagerie/unitree_a1/scene.xml")
    parser.add_argument("--dataset_metadata", default=None)
    parser.add_argument("--eval_seconds", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ray", action="store_true")
    parser.add_argument("--ray_address", default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run parallel MLP/TXL behavior-cloning training.

    This is the module entry point used by the script wrapper.
    """
    args = parse_args(argv)
    run_parallel_training(args)


def run_parallel_training(args: argparse.Namespace) -> None:
    """Create the run directory and launch all requested processes.

    Local multiprocessing is the default; Ray/JOVA mode delegates group runners to Ray.
    """
    configure_single_thread_torch()
    out_dir = Path(args.out_dir)
    _prepare_run_dir(out_dir, args)
    if bool(args.ray):
        _run_with_ray(args)
        return
    _run_with_local_multiprocessing(args)


def _run_with_local_multiprocessing(args: argparse.Namespace) -> None:
    """Launch local train, eval, and dashboard processes.

    Training groups run concurrently and each group uses its configured workers.
    """
    ctx = mp.get_context("spawn")
    processes: list[mp.Process] = []
    configs = _selected_group_configs(args)
    for config in configs:
        process = ctx.Process(target=train_group_main, args=(asdict(config),), name=f"{config.model_type}-group")
        process.start()
        processes.append(process)
    if int(args.eval_workers) > 0:
        evaluator = ctx.Process(target=evaluator_loop_main, args=(vars(args),), name="bc-evaluator")
        evaluator.start()
        processes.append(evaluator)
    if bool(args.dashboard):
        dashboard = ctx.Process(target=dashboard_main, args=(vars(args),), name="bc-dashboard")
        dashboard.start()
        processes.append(dashboard)
    _wait_for_training_processes(processes[: len(configs)])
    for process in processes[len(configs) :]:
        process.terminate()
    for process in processes:
        process.join(timeout=10)


def _run_with_ray(args: argparse.Namespace) -> None:
    """Run group and evaluator entry points through an existing Ray cluster.

    The same worker code is used so local and JOVA/Ray behavior stays aligned.
    """
    try:
        import ray
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("ray is required when --ray is passed") from exc
    ray.init(address=args.ray_address)

    @ray.remote
    def run_train_group_remote(config: dict[str, Any]) -> str:
        """Ray wrapper for a local multiprocessing training group.

        The return value is a small status string for orchestration.
        """
        train_group_main(config)
        return f"{config['model_type']} done"

    @ray.remote
    def run_eval_remote(config: dict[str, Any]) -> str:
        """Ray wrapper for the evaluator loop.

        The loop exits once training reaches `max_updates` for both models.
        """
        evaluator_loop_main(config)
        return "eval done"

    configs = _selected_group_configs(args)
    refs = [
        run_train_group_remote.options(num_cpus=_ray_group_cpus(config)).remote(asdict(config))
        for config in configs
    ]
    if int(args.eval_workers) > 0:
        refs.append(run_eval_remote.options(num_cpus=0).remote(vars(args)))
    dashboard_process: mp.Process | None = None
    if bool(args.dashboard):
        ctx = mp.get_context("spawn")
        dashboard_process = ctx.Process(target=dashboard_main, args=(vars(args),), name="bc-dashboard")
        dashboard_process.start()
    try:
        ray.get(refs)
    finally:
        if dashboard_process is not None:
            dashboard_process.terminate()
            dashboard_process.join(timeout=10)


def train_group_main(config_dict: dict[str, Any]) -> None:
    """Launch the worker processes for one policy family.

    Each worker samples batches and updates a shared model with one torch thread.
    """
    configure_single_thread_torch()
    config = TrainGroupConfig(**config_dict)
    _set_cpu_affinity(config.cores)
    torch = import_torch()
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context("spawn")
    train_data = BehaviorCloningDataset(config.dataset_dir, split="train", seed=config.seed)
    val_data = BehaviorCloningDataset(config.dataset_dir, split="val", seed=config.seed + 1000)
    model = _build_model(config)
    resume_update = _load_resume_checkpoint(torch, config, model) if bool(config.resume) else 0
    model.share_memory()
    update_counter = ctx.Value("i", int(resume_update))
    samples_counter = ctx.Value("q", _latest_samples_seen(Path(config.out_dir) / config.model_type / "metrics.jsonl"))
    best_val_loss = ctx.Value("d", _best_val_loss(Path(config.out_dir) / config.model_type / "metrics.jsonl"))
    update_lock = ctx.Lock()
    metric_lock = ctx.Lock()
    workers: list[mp.Process] = []
    for rank in range(int(config.workers)):
        process = ctx.Process(
            target=_train_worker_main,
            args=(
                asdict(config),
                rank,
                model,
                train_data,
                val_data,
                update_counter,
                samples_counter,
                best_val_loss,
                update_lock,
                metric_lock,
            ),
            name=f"{config.model_type}-worker-{rank}",
        )
        process.start()
        workers.append(process)
    for process in workers:
        process.join()
        if process.exitcode not in (0, None):
            raise RuntimeError(f"{process.name} exited with {process.exitcode}")
    del torch


def _train_worker_main(
    config_dict: dict[str, Any],
    rank: int,
    model: Any,
    train_data: BehaviorCloningDataset,
    val_data: BehaviorCloningDataset,
    update_counter: Any,
    samples_counter: Any,
    best_val_loss: Any,
    update_lock: Any,
    metric_lock: Any,
) -> None:
    """Run the per-process training loop for one worker.

    Shared counters coordinate checkpoint cadence across all workers in the group.
    """
    configure_single_thread_torch()
    config = TrainGroupConfig(**config_dict)
    _set_cpu_affinity(config.cores)
    torch = import_torch()
    rng = np.random.default_rng(config.seed + int(rank) * 9973)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr))
    start_time = time.time()
    worker_batch_size = _per_worker_batch_size(config)
    stream_state = _init_txl_stream_state(train_data, config, worker_batch_size, rng) if config.model_type == "txl" else None

    while True:
        with update_lock:
            if int(update_counter.value) >= int(config.max_updates):
                break
            update_counter.value += 1
            update = int(update_counter.value)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if config.model_type == "txl":
            loss, diagnostics = _txl_streaming_training_loss(
                torch,
                model,
                config,
                train_data,
                stream_state,
                rng,
            )
        else:
            loss, diagnostics = _training_loss(torch, model, config, train_data, worker_batch_size, rng)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with samples_counter.get_lock():
            samples_counter.value += int(worker_batch_size)

        if update % int(config.metrics_every) == 0:
            with metric_lock:
                val_loss, val_diag = _validation_loss(torch, model, config, val_data)
                row = {
                    "model_type": config.model_type,
                    "update": update,
                    "train_loss": float(diagnostics["loss"]),
                    "val_loss": float(val_loss),
                    "action_mse": float(diagnostics["mse"]),
                    "action_l1": float(diagnostics["l1"]),
                    "action_clip_fraction": float(diagnostics["clip_fraction"]),
                    "wall_time": float(time.time() - start_time),
                    "samples_seen": int(samples_counter.value),
                    **{f"val_{key}": float(value) for key, value in val_diag.items()},
                }
                _append_jsonl(Path(config.out_dir) / config.model_type / "metrics.jsonl", row)
        if update % int(config.checkpoint_every) == 0:
            with metric_lock:
                val_loss, _ = _validation_loss(torch, model, config, val_data)
                checkpoint_path = _save_model_checkpoint(torch, config, model, optimizer, update, float(val_loss), tag=f"step_{update:09d}")
                update_checkpoint_alias(Path(config.out_dir) / config.model_type / "checkpoints" / "latest", checkpoint_path)
                if val_loss < float(best_val_loss.value):
                    best_val_loss.value = float(val_loss)
                    update_checkpoint_alias(Path(config.out_dir) / config.model_type / "checkpoints" / "best_val_loss", checkpoint_path)


def _training_loss(
    torch: Any,
    model: Any,
    config: TrainGroupConfig,
    dataset: BehaviorCloningDataset,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[Any, dict[str, float]]:
    """Compute one supervised action loss for a sampled batch.

    MLP uses transition batches; TXL uses sequence batches with valid masks.
    """
    if config.model_type == "mlp":
        batch = dataset.sample_transition_batch(batch_size, rng=rng)
        obs = torch.as_tensor(batch.obs, dtype=torch.float32)
        target = torch.as_tensor(batch.action_label, dtype=torch.float32)
        pred = model(obs)
        loss = torch.mean((pred - target) ** 2)
    else:
        sequence_rows = _txl_sequence_rows(batch_size, config.sequence_length)
        batch = dataset.sample_sequence_batch(sequence_rows, config.sequence_length, rng=rng)
        obs = torch.as_tensor(batch.obs, dtype=torch.float32)
        target = torch.as_tensor(batch.action_label, dtype=torch.float32)
        valid = torch.as_tensor(batch.valid_mask, dtype=torch.float32).unsqueeze(-1)
        reset = torch.as_tensor(batch.reset_mask, dtype=torch.bool)
        pred = model(obs, reset_mask=reset, memory=None, valid_mask=torch.as_tensor(batch.valid_mask, dtype=torch.bool))
        loss = torch.sum(((pred - target) ** 2) * valid) / torch.clamp(torch.sum(valid) * target.shape[-1], min=1.0)
    with torch.no_grad():
        diff = pred - target
        if config.model_type == "txl":
            mask = valid
            mse = torch.sum((diff ** 2) * mask) / torch.clamp(torch.sum(mask) * target.shape[-1], min=1.0)
            l1 = torch.sum(torch.abs(diff) * mask) / torch.clamp(torch.sum(mask) * target.shape[-1], min=1.0)
            clip_fraction = torch.sum((torch.abs(pred) > 0.995).float() * mask) / torch.clamp(torch.sum(mask) * target.shape[-1], min=1.0)
        else:
            mse = torch.mean(diff ** 2)
            l1 = torch.mean(torch.abs(diff))
            clip_fraction = torch.mean((torch.abs(pred) > 0.995).float())
    return loss, {"loss": float(loss.detach()), "mse": float(mse), "l1": float(l1), "clip_fraction": float(clip_fraction)}


def _init_txl_stream_state(
    dataset: BehaviorCloningDataset,
    config: TrainGroupConfig,
    batch_size: int,
    rng: np.random.Generator,
) -> TXLStreamState:
    """Create active episode streams for one TXL worker.

    Each stream starts at a true episode boundary so memory begins clean.
    """
    stream_count = _txl_sequence_rows(batch_size, config.sequence_length)
    episode_indices = rng.choice(
        len(dataset.episodes),
        size=int(stream_count),
        replace=True,
        p=dataset._episode_sampling_prob,
    ).astype(np.int64)
    offsets = np.zeros((int(stream_count),), dtype=np.int64)
    return TXLStreamState(episode_indices=episode_indices, offsets=offsets, memory=None)


def _txl_streaming_training_loss(
    torch: Any,
    model: Any,
    config: TrainGroupConfig,
    dataset: BehaviorCloningDataset,
    state: TXLStreamState,
    rng: np.random.Generator,
) -> tuple[Any, dict[str, float]]:
    """Compute one streamed TXL BC loss with carried detached memory.

    Chunks are contiguous within each active episode stream and never cross episodes.
    """
    rows = [(int(episode_index), int(offset)) for episode_index, offset in zip(state.episode_indices, state.offsets, strict=True)]
    batch = dataset.make_stream_chunk_batch(rows, config.sequence_length)
    obs = torch.as_tensor(batch.obs, dtype=torch.float32)
    target = torch.as_tensor(batch.action_label, dtype=torch.float32)
    valid = torch.as_tensor(batch.valid_mask, dtype=torch.float32).unsqueeze(-1)
    episode_reset = torch.as_tensor(batch.episode_reset_mask, dtype=torch.bool)
    token_valid = torch.as_tensor(batch.valid_mask, dtype=torch.bool)
    pred, state.memory = model(
        obs,
        reset_mask=episode_reset,
        memory=state.memory,
        valid_mask=token_valid,
        return_memory=True,
    )
    denom = torch.clamp(torch.sum(valid) * target.shape[-1], min=1.0)
    loss = torch.sum(((pred - target) ** 2) * valid) / denom
    with torch.no_grad():
        diff = pred - target
        mse = torch.sum((diff ** 2) * valid) / denom
        l1 = torch.sum(torch.abs(diff) * valid) / denom
        clip_fraction = torch.sum((torch.abs(pred) > 0.995).float() * valid) / denom
    _advance_txl_streams(dataset, state, batch.valid_mask, rng)
    return loss, {"loss": float(loss.detach()), "mse": float(mse), "l1": float(l1), "clip_fraction": float(clip_fraction)}


def _advance_txl_streams(
    dataset: BehaviorCloningDataset,
    state: TXLStreamState,
    valid_mask: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Advance each stream and replace finished episodes.

    Replacement happens after the current chunk, so next chunk receives an episode reset.
    """
    valid_lengths = np.sum(valid_mask, axis=1).astype(np.int64)
    for row, valid_len in enumerate(valid_lengths):
        episode_index = int(state.episode_indices[row])
        next_offset = int(state.offsets[row]) + int(valid_len)
        if next_offset >= dataset.episodes[episode_index].length:
            state.episode_indices[row] = int(
                rng.choice(len(dataset.episodes), replace=True, p=dataset._episode_sampling_prob)
            )
            state.offsets[row] = 0
        else:
            state.offsets[row] = next_offset


def _validation_loss(torch: Any, model: Any, config: TrainGroupConfig, dataset: BehaviorCloningDataset) -> tuple[float, dict[str, float]]:
    """Compute a bounded validation estimate for metrics and best checkpoints.

    The validation pass caps work so frequent metrics remain lightweight.
    """
    if config.model_type == "txl":
        return _txl_streaming_validation_loss(torch, model, config, dataset)
    model.eval()
    losses: list[float] = []
    l1s: list[float] = []
    clips: list[float] = []
    with torch.no_grad():
        iterator = (
            dataset.iter_transition_batches(config.batch_size)
            if config.model_type == "mlp"
            else dataset.iter_sequence_batches(_txl_sequence_rows(config.batch_size, config.sequence_length), config.sequence_length)
        )
        for index, batch in enumerate(iterator):
            if index >= 8:
                break
            if config.model_type == "mlp":
                obs = torch.as_tensor(batch.obs, dtype=torch.float32)
                target = torch.as_tensor(batch.action_label, dtype=torch.float32)
                pred = model(obs)
                loss = torch.mean((pred - target) ** 2)
                l1 = torch.mean(torch.abs(pred - target))
                clip = torch.mean((torch.abs(pred) > 0.995).float())
            else:
                obs = torch.as_tensor(batch.obs, dtype=torch.float32)
                target = torch.as_tensor(batch.action_label, dtype=torch.float32)
                valid = torch.as_tensor(batch.valid_mask, dtype=torch.float32).unsqueeze(-1)
                reset = torch.as_tensor(batch.reset_mask, dtype=torch.bool)
                pred = model(obs, reset_mask=reset, memory=None, valid_mask=torch.as_tensor(batch.valid_mask, dtype=torch.bool))
                denom = torch.clamp(torch.sum(valid) * target.shape[-1], min=1.0)
                loss = torch.sum(((pred - target) ** 2) * valid) / denom
                l1 = torch.sum(torch.abs(pred - target) * valid) / denom
                clip = torch.sum((torch.abs(pred) > 0.995).float() * valid) / denom
            losses.append(float(loss))
            l1s.append(float(l1))
            clips.append(float(clip))
    if not losses:
        return float("inf"), {"action_l1": float("inf"), "action_clip_fraction": 0.0}
    return float(np.mean(losses)), {"action_l1": float(np.mean(l1s)), "action_clip_fraction": float(np.mean(clips))}


def _txl_streaming_validation_loss(
    torch: Any,
    model: Any,
    config: TrainGroupConfig,
    dataset: BehaviorCloningDataset,
) -> tuple[float, dict[str, float]]:
    """Compute bounded streamed validation loss for TXL.

    Memory is carried across contiguous chunks within each validation episode.
    """
    model.eval()
    losses: list[float] = []
    l1s: list[float] = []
    clips: list[float] = []
    with torch.no_grad():
        for episode_index, episode in enumerate(dataset.episodes):
            memory = None
            for start in range(0, episode.length, int(config.sequence_length)):
                batch = dataset.make_stream_chunk_batch([(episode_index, start)], config.sequence_length)
                obs = torch.as_tensor(batch.obs, dtype=torch.float32)
                target = torch.as_tensor(batch.action_label, dtype=torch.float32)
                valid = torch.as_tensor(batch.valid_mask, dtype=torch.float32).unsqueeze(-1)
                episode_reset = torch.as_tensor(batch.episode_reset_mask, dtype=torch.bool)
                token_valid = torch.as_tensor(batch.valid_mask, dtype=torch.bool)
                pred, memory = model(
                    obs,
                    reset_mask=episode_reset,
                    memory=memory,
                    valid_mask=token_valid,
                    return_memory=True,
                )
                denom = torch.clamp(torch.sum(valid) * target.shape[-1], min=1.0)
                losses.append(float(torch.sum(((pred - target) ** 2) * valid) / denom))
                l1s.append(float(torch.sum(torch.abs(pred - target) * valid) / denom))
                clips.append(float(torch.sum((torch.abs(pred) > 0.995).float() * valid) / denom))
                if len(losses) >= 8:
                    return float(np.mean(losses)), {
                        "action_l1": float(np.mean(l1s)),
                        "action_clip_fraction": float(np.mean(clips)),
                    }
    if not losses:
        return float("inf"), {"action_l1": float("inf"), "action_clip_fraction": 0.0}
    return float(np.mean(losses)), {"action_l1": float(np.mean(l1s)), "action_clip_fraction": float(np.mean(clips))}


def _build_model(config: TrainGroupConfig) -> Any:
    """Construct the requested policy model.

    Architecture defaults are conservative for CPU behavior cloning.
    """
    if config.model_type == "mlp":
        from robo_trot.policies.mlp_policy import MLPPolicy

        return MLPPolicy(obs_dim=config.obs_dim, action_dim=config.action_dim)
    if config.model_type == "txl":
        from robo_trot.policies.txl_policy import TXLPolicy

        return TXLPolicy(obs_dim=config.obs_dim, action_dim=config.action_dim, memory_length=config.txl_memory_length)
    raise ValueError(f"unsupported model type: {config.model_type}")


def _txl_sequence_rows(batch_size: int, sequence_length: int) -> int:
    """Return the number of TXL sequences for a token budget.

    The public `--batch_size` flag is interpreted as supervised tokens for TXL.
    """
    return max(1, int(batch_size) // max(1, int(sequence_length)))


def _per_worker_batch_size(config: TrainGroupConfig) -> int:
    """Return the mini-batch budget assigned to one worker process.

    The CLI batch size is a group budget, sharded across worker processes.
    """
    workers = max(1, int(config.workers))
    batch_size = max(1, int(config.batch_size))
    worker_batch = max(1, int(np.ceil(batch_size / workers)))
    if config.model_type == "txl":
        sequence_length = max(1, int(config.sequence_length))
        sequence_rows = max(1, int(np.ceil(worker_batch / sequence_length)))
        return sequence_rows * sequence_length
    return worker_batch


def _load_resume_checkpoint(torch: Any, config: TrainGroupConfig, model: Any) -> int:
    """Load model weights from the latest complete checkpoint.

    Resume restores the model and update counter; per-worker optimizers restart.
    """
    checkpoint_root = Path(config.out_dir) / config.model_type / "checkpoints"
    records = find_complete_checkpoints(checkpoint_root)
    if not records:
        return 0
    latest = records[-1]
    model_path = latest.path / "model.pt"
    if not model_path.exists():
        return 0
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return int(latest.update)


def _latest_samples_seen(metrics_path: Path) -> int:
    """Return the latest sample counter from a metrics JSONL file.

    Missing or empty metrics files resume sample accounting at zero.
    """
    row = _latest_metric(metrics_path)
    if row is None:
        return 0
    return int(row.get("samples_seen", 0))


def _best_val_loss(metrics_path: Path) -> float:
    """Return the best validation loss already recorded for a model.

    This keeps best-val aliases monotonic when a run is relaunched.
    """
    if not metrics_path.exists():
        return float("inf")
    values: list[float] = []
    for line in metrics_path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "val_loss" in row:
            values.append(float(row["val_loss"]))
    return min(values) if values else float("inf")


def _save_model_checkpoint(
    torch: Any,
    config: TrainGroupConfig,
    model: Any,
    optimizer: Any,
    update: int,
    val_loss: float,
    tag: str,
) -> Path:
    """Persist a model checkpoint and return its directory path.

    Checkpoint directories are written atomically and contain metadata plus torch state.
    """
    checkpoint_dir = Path(config.out_dir) / config.model_type / "checkpoints" / tag
    metadata = {
        "model": config.model_type,
        "model_type": config.model_type,
        "update": int(update),
        "val_loss": float(val_loss),
        "model_config": model.config_dict(),
        "saved_at": time.time(),
    }
    del torch
    return save_torch_checkpoint_atomic(checkpoint_dir, metadata=metadata, state_dict=model.state_dict(), optimizer_state=optimizer.state_dict())


def evaluator_loop_main(config: dict[str, Any]) -> None:
    """Watch MLP and TXL checkpoints and append evaluation metrics.

    Available latest checkpoints are evaluated even when the other model lags.
    """
    args = argparse.Namespace(**config)
    run_dir = Path(args.out_dir)
    eval_dir = run_dir / "eval"
    evaluated = _load_evaluated_checkpoints(eval_dir / "metrics.jsonl")
    eval_index = 0
    models = _enabled_models(args)
    while True:
        candidates = _interleaved_eval_candidates(run_dir, int(args.eval_every), evaluated, models=models)
        if not candidates:
            if _evaluator_ready_to_exit(run_dir, int(args.max_updates), int(args.eval_every), evaluated, models=models):
                break
            time.sleep(2.0)
            continue
        model_type, record = candidates[0]
        key = (model_type, int(record.update))
        try:
            rows = evaluate_checkpoint_set(
                checkpoint=record.path,
                model_type=model_type,
                xml_path=args.xml_path,
                dataset_metadata=_resolve_dataset_metadata(args),
                out_dir=eval_dir,
                seconds=float(args.eval_seconds),
                checkpoint_update=int(record.update),
                gif_every_eval=int(args.gif_every_eval),
                gif_fps=int(args.eval_gif_fps),
                gif_seconds=float(args.eval_gif_seconds),
                gif_width=int(args.eval_gif_width),
                gif_height=int(args.eval_gif_height),
                eval_index=eval_index,
                seed=int(args.seed) + int(record.update),
            )
            eval_row = aggregate_eval_rows(rows, model_type, int(record.update))
            eval_row.update(
                _checkpoint_dataset_action_metrics(
                    args=args,
                    checkpoint=record.path,
                    model_type=model_type,
                    seed=int(args.seed) + int(record.update),
                )
            )
            append_eval_metrics(eval_dir / "metrics.jsonl", eval_row)
            _update_best_eval_alias(run_dir, model_type)
        except Exception as exc:
            append_eval_metrics(
                eval_dir / "metrics.jsonl",
                {
                    "model_type": model_type,
                    "checkpoint_update": int(record.update),
                    "eval_error": f"{type(exc).__name__}: {exc}",
                    "wall_time": time.time(),
                },
            )
        evaluated.add(key)
        eval_index += 1


def _evaluator_ready_to_exit(
    run_dir: Path,
    max_updates: int,
    eval_every: int,
    evaluated: set[tuple[str, int]],
    models: tuple[str, ...] = ("mlp", "txl"),
) -> bool:
    """Return whether evaluator work is complete for selected models.

    Training completion alone is not enough because MuJoCo eval can lag behind.
    """
    if not _training_complete(run_dir, int(max_updates), models=models):
        return False
    return not _interleaved_eval_candidates(run_dir, int(eval_every), evaluated, models=models)


def _interleaved_eval_candidates(
    run_dir: Path,
    eval_every: int,
    evaluated: set[tuple[str, int]],
    models: tuple[str, ...] = ("mlp", "txl"),
) -> list[tuple[str, Any]]:
    """Return complete checkpoint candidates sorted by update across models.

    Interleaving prevents one model's backlog from starving the other model's evals.
    """
    model_order = {"mlp": 0, "txl": 1}
    candidates: list[tuple[str, Any]] = []
    for model_type in models:
        for record in collect_policy_checkpoints(run_dir, model_type):
            key = (model_type, int(record.update))
            if key in evaluated or int(record.update) % int(eval_every) != 0:
                continue
            candidates.append((model_type, record))
    return sorted(candidates, key=lambda item: (int(item[1].update), model_order[item[0]]))


def _checkpoint_dataset_action_metrics(
    args: argparse.Namespace,
    checkpoint: Path,
    model_type: str,
    seed: int,
) -> dict[str, Any]:
    """Return held-out dataset action-label metrics for a checkpoint.

    Errors are captured as metrics so MuJoCo reward logging can still continue.
    """
    try:
        from robo_trot.training.evaluate_checkpoint import evaluate_dataset_action_loss, load_policy_from_checkpoint

        policy = load_policy_from_checkpoint(checkpoint, model_type)
        return evaluate_dataset_action_loss(
            policy=policy,
            model_type=model_type,
            dataset_dir=args.dataset_dir,
            split=str(args.dataset_eval_split),
            batch_size=int(args.dataset_eval_batch_size),
            sequence_length=int(args.sequence_length),
            max_batches=int(args.dataset_eval_max_batches),
            seed=int(seed),
        )
    except Exception as exc:
        return {
            "dataset_eval_split": str(getattr(args, "dataset_eval_split", "test")),
            "dataset_eval_error": f"{type(exc).__name__}: {exc}",
        }


def _load_evaluated_checkpoints(metrics_path: Path) -> set[tuple[str, int]]:
    """Load completed or errored eval checkpoint keys from JSONL metrics.

    This lets a restarted evaluator continue from persisted progress.
    """
    evaluated: set[tuple[str, int]] = set()
    if not metrics_path.exists():
        return evaluated
    for line in metrics_path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        model_type = row.get("model_type")
        update = row.get("checkpoint_update")
        if model_type is not None and update is not None:
            evaluated.add((str(model_type), int(update)))
    return evaluated


def dashboard_main(config: dict[str, Any]) -> None:
    """Run the lightweight dashboard server process.

    The server reads metrics and rollout media directly from the run directory.
    """
    from robo_trot.training.dashboard import serve_dashboard

    args = argparse.Namespace(**config)
    serve_dashboard(args.out_dir, host=args.dashboard_host, port=args.dashboard_port)


def _update_best_eval_alias(run_dir: Path, model_type: str) -> None:
    """Update the `best_eval_reward` alias from eval metric history.

    Errors and incomplete checkpoints are ignored.
    """
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


def _training_complete(run_dir: Path, max_updates: int, models: tuple[str, ...] = ("mlp", "txl")) -> bool:
    """Return whether selected model groups have reached `max_updates`.

    The evaluator uses this to exit after final checkpoint scans.
    """
    for model_type in models:
        row = _latest_metric(run_dir / model_type / "metrics.jsonl")
        if row is None or int(row.get("update", 0)) < int(max_updates):
            return False
    return True


def _latest_metric(path: Path) -> dict[str, Any] | None:
    """Return the last parseable JSONL row from a metrics file.

    Missing or partially written files are treated as no data yet.
    """
    if not path.exists():
        return None
    for line in reversed(path.read_text().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def _make_group_configs(args: argparse.Namespace) -> tuple[TrainGroupConfig, TrainGroupConfig]:
    """Build MLP and TXL training group configs from parsed args.

    CPU affinity strings are parsed into tuples of integer core IDs.
    """
    common = {
        "dataset_dir": args.dataset_dir,
        "out_dir": args.out_dir,
        "batch_size": int(args.batch_size),
        "sequence_length": int(args.sequence_length),
        "lr": float(args.lr),
        "max_updates": int(args.max_updates),
        "metrics_every": int(args.metrics_every),
        "checkpoint_every": int(args.checkpoint_every),
        "seed": int(args.seed),
        "txl_memory_length": _txl_memory_length(args),
        "resume": bool(args.resume),
    }
    return (
        TrainGroupConfig(model_type="mlp", workers=int(args.mlp_workers), cores=_parse_cores(args.mlp_cores), **common),
        TrainGroupConfig(model_type="txl", workers=int(args.txl_workers), cores=_parse_cores(args.txl_cores), **common),
    )


def _selected_group_configs(args: argparse.Namespace) -> tuple[TrainGroupConfig, ...]:
    """Return training group configs requested by `--models`.

    This allows a run to resume only TXL or only MLP without changing checkpoint layout.
    """
    enabled = set(_enabled_models(args))
    return tuple(config for config in _make_group_configs(args) if config.model_type in enabled)


def _enabled_models(args: argparse.Namespace) -> tuple[str, ...]:
    """Parse the comma-separated model selector from CLI args.

    Valid values are `mlp`, `txl`, or both in comma-separated form.
    """
    raw_value = str(getattr(args, "models", "mlp,txl"))
    models = tuple(part.strip().lower() for part in raw_value.split(",") if part.strip())
    if not models:
        raise ValueError("--models must select at least one of: mlp, txl")
    invalid = [model for model in models if model not in {"mlp", "txl"}]
    if invalid:
        raise ValueError(f"unsupported model selector(s): {invalid}")
    return tuple(dict.fromkeys(models))


def _parse_cores(value: str | None) -> tuple[int, ...]:
    """Parse a comma-separated CPU core list.

    Empty values disable affinity pinning for that process.
    """
    if value is None or str(value).strip() == "":
        return ()
    return tuple(int(part) for part in str(value).split(",") if part.strip() != "")


def _ray_group_cpus(config: TrainGroupConfig) -> float:
    """Return Ray CPU reservations for one training group.

    Core-pin lists reserve their full width so MLP and TXL occupy four CPUs total.
    """
    if config.cores:
        return float(len(config.cores))
    return float(max(1, min(int(config.workers), 2)))


def _txl_memory_length(args: argparse.Namespace) -> int:
    """Return TXL memory tokens from explicit tokens or seconds.

    At the default policy rate of 0.02s, 20 seconds becomes 1000 cached tokens.
    """
    explicit = getattr(args, "txl_memory_length", None)
    if explicit is not None:
        return max(0, int(explicit))
    policy_dt = max(float(getattr(args, "policy_dt", 0.02)), 1e-9)
    return max(1, int(round(float(args.txl_memory_seconds) / policy_dt)))


def _set_cpu_affinity(cores: tuple[int, ...]) -> None:
    """Pin the current process to CPU cores when the OS supports it.

    Unsupported platforms silently continue unpinned.
    """
    if not cores or not hasattr(os, "sched_setaffinity"):
        return
    try:
        os.sched_setaffinity(0, set(int(core) for core in cores))
    except OSError:
        return


def _prepare_run_dir(out_dir: Path, args: argparse.Namespace) -> None:
    """Create run directories and write the immutable config snapshot.

    The layout matches the MLP/TXL/eval/dashboard directory contract.
    """
    for path in (
        out_dir / "mlp" / "checkpoints",
        out_dir / "txl" / "checkpoints",
        out_dir / "eval" / "media",
        out_dir / "eval" / "gifs",
        out_dir / "eval" / "dashboard",
    ):
        path.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))


def _resolve_dataset_metadata(args: argparse.Namespace) -> str | None:
    """Return an explicit or conventional dataset metadata path.

    Missing metadata returns `None` so evaluation can still run with XML only.
    """
    if args.dataset_metadata:
        return str(args.dataset_metadata)
    conventional = Path(args.dataset_dir) / "shards" / "shard_00_forward" / "metadata.json"
    if conventional.exists():
        return conventional.as_posix()
    root_metadata = Path(args.dataset_dir) / "metadata.json"
    if root_metadata.exists():
        return root_metadata.as_posix()
    return None


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one metrics row to a JSONL file.

    Parent directories are created before writing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _wait_for_training_processes(processes: list[mp.Process]) -> None:
    """Wait for training group processes and raise on failures.

    Evaluator and dashboard processes are terminated by the caller afterwards.
    """
    for process in processes:
        process.join()
    failures = [process for process in processes if process.exitcode not in (0, None)]
    if failures:
        names = ", ".join(f"{process.name}:{process.exitcode}" for process in failures)
        raise RuntimeError(f"training processes failed: {names}")


if __name__ == "__main__":
    main()
