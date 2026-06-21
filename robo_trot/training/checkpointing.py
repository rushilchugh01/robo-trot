from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

COMPLETE_MARKER = "_SUCCESS"


@dataclass(frozen=True)
class CheckpointRecord:
    """Metadata for one complete checkpoint directory.

    Evaluators use these records to avoid loading partial checkpoint writes.
    """

    path: Path
    update: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Loaded metadata and NumPy arrays from a checkpoint directory.

    Torch state files are loaded separately by model-specific callers.
    """

    path: Path
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]


def save_checkpoint_atomic(
    target_dir: str | Path,
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray] | None = None,
) -> Path:
    """Write a checkpoint directory through a temporary sibling directory.

    The completion marker is written last so readers can ignore partial output.
    """
    target = Path(target_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.parent / f".{target.name}.tmp-{os.getpid()}-{time.time_ns()}"
    temp.mkdir(parents=True)
    try:
        (temp / "metadata.json").write_text(json.dumps(dict(metadata), indent=2, sort_keys=True))
        if arrays:
            normalized = {key: np.asarray(value) for key, value in arrays.items()}
            np.savez_compressed(temp / "arrays.npz", **normalized)
        (temp / COMPLETE_MARKER).write_text("ok\n")
        if target.exists() or target.is_symlink():
            _remove_existing_checkpoint_path(target)
        os.replace(temp, target)
    except Exception:
        if temp.exists():
            shutil.rmtree(temp, ignore_errors=True)
        raise
    return target


def save_torch_checkpoint_atomic(
    target_dir: str | Path,
    metadata: dict[str, Any],
    state_dict: Any,
    optimizer_state: Any | None = None,
) -> Path:
    """Write a torch checkpoint directory atomically when torch is installed.

    The function imports torch lazily so non-training tests do not require it.
    """
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("torch is required to save model checkpoints") from exc
    target = Path(target_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.parent / f".{target.name}.tmp-{os.getpid()}-{time.time_ns()}"
    temp.mkdir(parents=True)
    try:
        (temp / "metadata.json").write_text(json.dumps(dict(metadata), indent=2, sort_keys=True))
        torch.save(state_dict, temp / "model.pt")
        if optimizer_state is not None:
            torch.save(optimizer_state, temp / "optimizer.pt")
        (temp / COMPLETE_MARKER).write_text("ok\n")
        if target.exists() or target.is_symlink():
            _remove_existing_checkpoint_path(target)
        os.replace(temp, target)
    except Exception:
        if temp.exists():
            shutil.rmtree(temp, ignore_errors=True)
        raise
    return target


def checkpoint_is_complete(path: str | Path) -> bool:
    """Return whether a checkpoint directory has its completion marker.

    Symlink aliases are accepted when their resolved target is complete.
    """
    checkpoint_path = Path(path)
    return checkpoint_path.is_dir() and (checkpoint_path / COMPLETE_MARKER).exists() and (checkpoint_path / "metadata.json").exists()


def load_checkpoint(path: str | Path) -> LoadedCheckpoint:
    """Load metadata and optional NumPy arrays from one complete checkpoint.

    Partial checkpoint directories raise `FileNotFoundError`.
    """
    checkpoint_path = Path(path)
    if not checkpoint_is_complete(checkpoint_path):
        raise FileNotFoundError(f"incomplete checkpoint: {checkpoint_path}")
    metadata = json.loads((checkpoint_path / "metadata.json").read_text())
    arrays: dict[str, np.ndarray] = {}
    arrays_path = checkpoint_path / "arrays.npz"
    if arrays_path.exists():
        with np.load(arrays_path) as data:
            arrays = {key: np.asarray(data[key]) for key in data.files}
    return LoadedCheckpoint(path=checkpoint_path, metadata=metadata, arrays=arrays)


def find_complete_checkpoints(root_dir: str | Path) -> list[CheckpointRecord]:
    """Return complete `step_*` checkpoint directories sorted by update.

    Directories without `_SUCCESS` are intentionally omitted.
    """
    root = Path(root_dir)
    if not root.exists():
        return []
    records: list[CheckpointRecord] = []
    for path in sorted(root.iterdir()):
        if not path.name.startswith("step_"):
            continue
        if not checkpoint_is_complete(path):
            continue
        metadata = json.loads((path / "metadata.json").read_text())
        records.append(CheckpointRecord(path=path, update=_checkpoint_update(path, metadata), metadata=metadata))
    return sorted(records, key=lambda record: record.update)


def update_checkpoint_alias(alias_dir: str | Path, checkpoint_dir: str | Path) -> Path:
    """Atomically point an alias such as `latest` at a complete checkpoint.

    A symlink is used when supported so alias updates are cheap and atomic.
    """
    alias = Path(alias_dir)
    checkpoint = Path(checkpoint_dir)
    if not checkpoint_is_complete(checkpoint):
        raise FileNotFoundError(f"cannot alias incomplete checkpoint: {checkpoint}")
    alias.parent.mkdir(parents=True, exist_ok=True)
    temp = alias.parent / f".{alias.name}.link-{os.getpid()}-{time.time_ns()}"
    try:
        rel_target = os.path.relpath(checkpoint, alias.parent)
        temp.symlink_to(rel_target, target_is_directory=True)
        os.replace(temp, alias)
    except OSError:
        if temp.exists() or temp.is_symlink():
            temp.unlink()
        copy_temp = alias.parent / f".{alias.name}.copy-{os.getpid()}-{time.time_ns()}"
        shutil.copytree(checkpoint, copy_temp, symlinks=True)
        if alias.exists() or alias.is_symlink():
            _remove_existing_checkpoint_path(alias)
        os.replace(copy_temp, alias)
    return alias


def _checkpoint_update(path: Path, metadata: dict[str, Any]) -> int:
    """Infer the update number from metadata or the directory name.

    Directory names use the `step_000001000` convention.
    """
    if "update" in metadata:
        return int(metadata["update"])
    if "step" in metadata:
        return int(metadata["step"])
    try:
        return int(path.name.replace("step_", ""))
    except ValueError:
        return -1


def _remove_existing_checkpoint_path(path: Path) -> None:
    """Remove an existing checkpoint path before replacing it.

    This helper handles both symlink aliases and real directories.
    """
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)
