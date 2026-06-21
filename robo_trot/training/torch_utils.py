from __future__ import annotations

import os
from typing import Any


def import_torch() -> Any:
    """Import torch lazily and return the module.

    A clear dependency message is raised when training is requested without torch.
    """
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torch is required for BC training and policy checkpoint evaluation; "
            "install the training extra or a Python-compatible torch build."
        ) from exc
    return torch


def configure_single_thread_torch() -> None:
    """Constrain BLAS and torch threading for multiprocessing trainers.

    This keeps each worker conservative when multiple processes share CPU cores.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        torch = import_torch()
    except ModuleNotFoundError:
        return
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
