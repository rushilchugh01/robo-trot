from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class EpisodeReference:
    """Reference to one accepted episode file inside a BC dataset.

    The relative path is normalized against the requested dataset root.
    """

    path: Path
    relative_path: str
    episode_id: int


@dataclass(frozen=True)
class TransitionBatch:
    """Transition mini-batch for feed-forward behavior cloning.

    Each row maps `obs_t` to the normalized `action_label_t`.
    """

    obs: np.ndarray
    action_label: np.ndarray
    reset_mask: np.ndarray
    episode_id: np.ndarray
    timestep: np.ndarray


@dataclass(frozen=True)
class SequenceBatch:
    """Sequence mini-batch for recurrent or TXL behavior cloning.

    Valid and reset masks are aligned to every `[batch, time]` token.
    """

    obs: np.ndarray
    action_label: np.ndarray
    valid_mask: np.ndarray
    reset_mask: np.ndarray
    episode_id: np.ndarray
    timestep: np.ndarray


@dataclass(frozen=True)
class StreamChunkBatch:
    """Contiguous episode chunks for Transformer-XL truncated BPTT.

    Episode reset masks clear memory; trial reset masks preserve dataset `reset_flag`.
    """

    obs: np.ndarray
    action_label: np.ndarray
    valid_mask: np.ndarray
    episode_reset_mask: np.ndarray
    trial_reset_mask: np.ndarray
    episode_id: np.ndarray
    timestep: np.ndarray


@dataclass(frozen=True)
class EpisodeArrays:
    """In-memory arrays needed for supervised policy training.

    The loader keeps only observations, action labels, and boundary masks.
    """

    reference: EpisodeReference
    obs: np.ndarray
    action_label: np.ndarray
    reset_flag: np.ndarray
    done: np.ndarray

    @property
    def length(self) -> int:
        """Return the number of transitions in this episode.

        Callers use this value for weighted sampling and sequence windows.
        """
        return int(self.obs.shape[0])


def discover_episode_paths(
    dataset_dir: str | Path,
    split: str = "train",
    seed: int = 0,
    val_fraction: float = 0.1,
) -> list[EpisodeReference]:
    """Return accepted episode paths for a named split.

    If `splits.json` is absent, a seeded deterministic train/val/test split is used.
    """
    dataset_root = Path(dataset_dir)
    references = _all_episode_references(dataset_root)
    if not references:
        raise FileNotFoundError(f"no episode npz files found under {dataset_root}")
    split_path = dataset_root / "splits.json"
    if split_path.exists():
        split_data = json.loads(split_path.read_text())
        if not _split_data_has_named_split(split_data, split):
            return _fallback_split(references, split=split, seed=seed, val_fraction=val_fraction)
        return _references_from_splits(references, split_data, split)
    return _fallback_split(references, split=split, seed=seed, val_fraction=val_fraction)


def load_episode(reference: EpisodeReference) -> EpisodeArrays:
    """Load one episode NPZ into training arrays.

    The returned arrays are copied to float32/bool types for stable batching.
    """
    with np.load(reference.path) as data:
        obs = np.asarray(data["obs"], dtype=np.float32)
        action_label = np.asarray(data["action_label"], dtype=np.float32)
        reset_flag = np.asarray(data["reset_flag"], dtype=bool) if "reset_flag" in data else _default_reset(obs.shape[0])
        done = np.asarray(data["done"], dtype=bool) if "done" in data else _default_done(obs.shape[0])
    _validate_episode_arrays(reference.path, obs, action_label, reset_flag, done)
    return EpisodeArrays(
        reference=reference,
        obs=obs,
        action_label=action_label,
        reset_flag=reset_flag,
        done=done,
    )


class BehaviorCloningDataset:
    """Random-access BC sampler for transition and sequence batches.

    The dataset respects split files, episode boundaries, and reset masks.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str = "train",
        seed: int = 0,
        val_fraction: float = 0.1,
    ) -> None:
        """Load the selected split into memory for repeated sampling.

        The public sampler methods return NumPy arrays ready for torch conversion.
        """
        self.dataset_dir = Path(dataset_dir)
        self.split = str(split)
        self.seed = int(seed)
        self.references = discover_episode_paths(
            self.dataset_dir,
            split=self.split,
            seed=self.seed,
            val_fraction=float(val_fraction),
        )
        self.episodes = [load_episode(reference) for reference in self.references]
        self.episode_lengths = np.asarray([episode.length for episode in self.episodes], dtype=np.int64)
        self.cumulative_lengths = np.cumsum(self.episode_lengths, dtype=np.int64)
        self.total_transitions = int(self.cumulative_lengths[-1]) if self.cumulative_lengths.size else 0
        if self.total_transitions <= 0:
            raise ValueError(f"split {self.split!r} has no transitions")
        self.obs_dim = int(self.episodes[0].obs.shape[1])
        self.action_dim = int(self.episodes[0].action_label.shape[1])
        self._episode_sampling_prob = self.episode_lengths.astype(np.float64) / float(self.total_transitions)

    def sample_transition_batch(self, batch_size: int, rng: np.random.Generator | None = None) -> TransitionBatch:
        """Sample `obs_t -> action_label_t` transitions with replacement.

        Reset masks are copied from the source episode's `reset_flag` array.
        """
        rng = rng or np.random.default_rng(self.seed)
        size = int(batch_size)
        if size <= 0:
            raise ValueError("batch_size must be positive")
        flat_indices = rng.integers(0, self.total_transitions, size=size, endpoint=False)
        episode_indices, timesteps = self._locate_flat_indices(flat_indices)
        obs = np.empty((size, self.obs_dim), dtype=np.float32)
        action = np.empty((size, self.action_dim), dtype=np.float32)
        reset = np.empty((size,), dtype=bool)
        episode_ids = np.empty((size,), dtype=np.int64)
        for row, (episode_index, timestep) in enumerate(zip(episode_indices, timesteps, strict=True)):
            episode = self.episodes[int(episode_index)]
            obs[row] = episode.obs[int(timestep)]
            action[row] = episode.action_label[int(timestep)]
            reset[row] = bool(episode.reset_flag[int(timestep)])
            episode_ids[row] = int(episode.reference.episode_id)
        return TransitionBatch(obs=obs, action_label=action, reset_mask=reset, episode_id=episode_ids, timestep=timesteps)

    def sample_sequence_batch(
        self,
        batch_size: int,
        sequence_length: int,
        rng: np.random.Generator | None = None,
    ) -> SequenceBatch:
        """Sample fixed-length causal windows without crossing episodes.

        Short tail windows are padded and marked false in `valid_mask`.
        """
        rng = rng or np.random.default_rng(self.seed)
        rows = int(batch_size)
        seq_len = int(sequence_length)
        if rows <= 0:
            raise ValueError("batch_size must be positive")
        if seq_len <= 0:
            raise ValueError("sequence_length must be positive")
        obs = np.zeros((rows, seq_len, self.obs_dim), dtype=np.float32)
        action = np.zeros((rows, seq_len, self.action_dim), dtype=np.float32)
        valid = np.zeros((rows, seq_len), dtype=bool)
        reset = np.zeros((rows, seq_len), dtype=bool)
        episode_ids = np.full((rows, seq_len), -1, dtype=np.int64)
        timesteps = np.full((rows, seq_len), -1, dtype=np.int64)
        selected = rng.choice(len(self.episodes), size=rows, replace=True, p=self._episode_sampling_prob)
        for row, episode_index in enumerate(selected):
            episode = self.episodes[int(episode_index)]
            start = _sample_sequence_start(rng, episode.length, seq_len)
            valid_len = min(seq_len, episode.length - start)
            end = start + valid_len
            obs[row, :valid_len] = episode.obs[start:end]
            action[row, :valid_len] = episode.action_label[start:end]
            valid[row, :valid_len] = True
            reset[row, :valid_len] = episode.reset_flag[start:end]
            episode_ids[row, :valid_len] = int(episode.reference.episode_id)
            timesteps[row, :valid_len] = np.arange(start, end, dtype=np.int64)
        return SequenceBatch(
            obs=obs,
            action_label=action,
            valid_mask=valid,
            reset_mask=reset,
            episode_id=episode_ids,
            timestep=timesteps,
        )

    def iter_transition_batches(self, batch_size: int) -> Iterable[TransitionBatch]:
        """Yield deterministic transition batches over the loaded split.

        The final batch may be smaller than `batch_size`.
        """
        size = int(batch_size)
        if size <= 0:
            raise ValueError("batch_size must be positive")
        for start in range(0, self.total_transitions, size):
            flat = np.arange(start, min(start + size, self.total_transitions), dtype=np.int64)
            episode_indices, timesteps = self._locate_flat_indices(flat)
            obs = np.empty((flat.shape[0], self.obs_dim), dtype=np.float32)
            action = np.empty((flat.shape[0], self.action_dim), dtype=np.float32)
            reset = np.empty((flat.shape[0],), dtype=bool)
            episode_ids = np.empty((flat.shape[0],), dtype=np.int64)
            for row, (episode_index, timestep) in enumerate(zip(episode_indices, timesteps, strict=True)):
                episode = self.episodes[int(episode_index)]
                obs[row] = episode.obs[int(timestep)]
                action[row] = episode.action_label[int(timestep)]
                reset[row] = bool(episode.reset_flag[int(timestep)])
                episode_ids[row] = int(episode.reference.episode_id)
            yield TransitionBatch(obs=obs, action_label=action, reset_mask=reset, episode_id=episode_ids, timestep=timesteps)

    def iter_sequence_batches(self, batch_size: int, sequence_length: int) -> Iterable[SequenceBatch]:
        """Yield deterministic sequence windows within each episode.

        This iterator is intended for validation loss and small audits.
        """
        rows: list[tuple[EpisodeArrays, int]] = []
        for episode in self.episodes:
            for start in range(0, episode.length, int(sequence_length)):
                rows.append((episode, start))
        for offset in range(0, len(rows), int(batch_size)):
            yield self._make_sequence_batch_from_rows(rows[offset : offset + int(batch_size)], int(sequence_length))

    def make_stream_chunk_batch(self, rows: list[tuple[int, int]], sequence_length: int) -> StreamChunkBatch:
        """Build contiguous stream chunks from explicit episode/start rows.

        The returned episode reset mask is true only at true episode starts.
        """
        seq_len = int(sequence_length)
        if seq_len <= 0:
            raise ValueError("sequence_length must be positive")
        obs = np.zeros((len(rows), seq_len, self.obs_dim), dtype=np.float32)
        action = np.zeros((len(rows), seq_len, self.action_dim), dtype=np.float32)
        valid = np.zeros((len(rows), seq_len), dtype=bool)
        episode_reset = np.zeros((len(rows), seq_len), dtype=bool)
        trial_reset = np.zeros((len(rows), seq_len), dtype=bool)
        episode_ids = np.full((len(rows), seq_len), -1, dtype=np.int64)
        timesteps = np.full((len(rows), seq_len), -1, dtype=np.int64)
        for row, (episode_index, start) in enumerate(rows):
            episode = self.episodes[int(episode_index)]
            start_index = int(start)
            valid_len = min(seq_len, episode.length - start_index)
            if start_index < 0 or valid_len <= 0:
                raise ValueError(f"invalid stream row episode={episode_index} start={start}")
            end = start_index + valid_len
            obs[row, :valid_len] = episode.obs[start_index:end]
            action[row, :valid_len] = episode.action_label[start_index:end]
            valid[row, :valid_len] = True
            episode_reset[row, 0] = start_index == 0
            trial_reset[row, :valid_len] = episode.reset_flag[start_index:end]
            episode_ids[row, :valid_len] = int(episode.reference.episode_id)
            timesteps[row, :valid_len] = np.arange(start_index, end, dtype=np.int64)
        return StreamChunkBatch(
            obs=obs,
            action_label=action,
            valid_mask=valid,
            episode_reset_mask=episode_reset,
            trial_reset_mask=trial_reset,
            episode_id=episode_ids,
            timestep=timesteps,
        )

    def _locate_flat_indices(self, flat_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map flattened transition indices to episode/timestep pairs.

        The returned vectors are aligned with the input index order.
        """
        episode_indices = np.searchsorted(self.cumulative_lengths, flat_indices, side="right").astype(np.int64)
        previous = np.zeros_like(episode_indices)
        nonzero = episode_indices > 0
        previous[nonzero] = self.cumulative_lengths[episode_indices[nonzero] - 1]
        timesteps = (flat_indices - previous).astype(np.int64)
        return episode_indices, timesteps

    def _make_sequence_batch_from_rows(self, rows: list[tuple[EpisodeArrays, int]], sequence_length: int) -> SequenceBatch:
        """Build a padded sequence batch from explicit episode/start rows.

        This helper preserves episode boundaries for deterministic validation.
        """
        obs = np.zeros((len(rows), sequence_length, self.obs_dim), dtype=np.float32)
        action = np.zeros((len(rows), sequence_length, self.action_dim), dtype=np.float32)
        valid = np.zeros((len(rows), sequence_length), dtype=bool)
        reset = np.zeros((len(rows), sequence_length), dtype=bool)
        episode_ids = np.full((len(rows), sequence_length), -1, dtype=np.int64)
        timesteps = np.full((len(rows), sequence_length), -1, dtype=np.int64)
        for row, (episode, start) in enumerate(rows):
            valid_len = min(sequence_length, episode.length - start)
            end = start + valid_len
            obs[row, :valid_len] = episode.obs[start:end]
            action[row, :valid_len] = episode.action_label[start:end]
            valid[row, :valid_len] = True
            reset[row, :valid_len] = episode.reset_flag[start:end]
            episode_ids[row, :valid_len] = int(episode.reference.episode_id)
            timesteps[row, :valid_len] = np.arange(start, end, dtype=np.int64)
        return SequenceBatch(obs=obs, action_label=action, valid_mask=valid, reset_mask=reset, episode_id=episode_ids, timestep=timesteps)


def _all_episode_references(dataset_root: Path) -> list[EpisodeReference]:
    """Collect accepted episode references from root and shard metadata.

    Metadata entries are preferred; globbing is the compatibility fallback.
    """
    references: list[EpisodeReference] = []
    references.extend(_metadata_references(dataset_root, dataset_root))
    shards_dir = dataset_root / "shards"
    if shards_dir.exists():
        for shard_dir in sorted(path for path in shards_dir.iterdir() if path.is_dir()):
            references.extend(_metadata_references(shard_dir, dataset_root))
    if references:
        return sorted(references, key=lambda item: item.relative_path)
    paths = sorted(dataset_root.glob("episodes/ep_*.npz"))
    paths.extend(sorted(dataset_root.glob("shards/*/episodes/ep_*.npz")))
    return [
        EpisodeReference(path=path, relative_path=path.relative_to(dataset_root).as_posix(), episode_id=idx)
        for idx, path in enumerate(paths)
    ]


def _metadata_references(metadata_dir: Path, dataset_root: Path) -> list[EpisodeReference]:
    """Read accepted episode references from one `metadata.json` file.

    Missing metadata returns an empty list so shard discovery can continue.
    """
    metadata_path = metadata_dir / "metadata.json"
    if not metadata_path.exists():
        return []
    metadata = json.loads(metadata_path.read_text())
    references: list[EpisodeReference] = []
    for fallback_id, entry in enumerate(metadata.get("episodes", [])):
        if not bool(entry.get("accepted", True)):
            continue
        path_value = str(entry.get("path", ""))
        if not path_value:
            continue
        relative_to_metadata = Path(path_value)
        path = metadata_dir / relative_to_metadata
        episode_id = int(entry.get("episode_id", fallback_id))
        references.append(
            EpisodeReference(
                path=path,
                relative_path=path.relative_to(dataset_root).as_posix(),
                episode_id=episode_id,
            )
        )
    return references


def _references_from_splits(
    references: list[EpisodeReference],
    split_data: dict[str, Any],
    split: str,
) -> list[EpisodeReference]:
    """Filter episode references using flexible `splits.json` values.

    Split values may be paths, basenames, dictionaries, or episode IDs.
    """
    values = None
    for key in _split_lookup_keys(split):
        if key in split_data:
            values = split_data[key]
            break
    if values is None:
        raise KeyError(f"splits.json does not contain split {split!r}")
    wanted = {_normalize_split_value(value) for value in values}
    selected = [reference for reference in references if _reference_matches_split(reference, wanted)]
    if not selected:
        raise ValueError(f"split {split!r} matched no episode files")
    return selected


def _split_data_has_named_split(split_data: dict[str, Any], split: str) -> bool:
    """Return whether `splits.json` contains an explicit train/val split.

    Category-indexed split files fall back to deterministic train/val partitioning.
    """
    return any(key in split_data for key in _split_lookup_keys(split))


def _fallback_split(
    references: list[EpisodeReference],
    split: str,
    seed: int,
    val_fraction: float,
) -> list[EpisodeReference]:
    """Create a deterministic split when no `splits.json` exists.

    The validation/test split is non-empty for datasets with more than one episode.
    """
    if split not in {"train", "val", "validation", "test", "all"}:
        raise ValueError(f"unsupported split {split!r}")
    if split == "all":
        return references
    if len(references) == 1:
        return references if split == "train" else []
    rng = np.random.default_rng(int(seed))
    shuffled = np.arange(len(references), dtype=np.int64)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(references) * float(val_fraction))))
    val_count = min(val_count, len(references) - 1)
    val_indices = set(int(index) for index in shuffled[:val_count])
    if split in {"val", "validation", "test"}:
        return [reference for index, reference in enumerate(references) if index in val_indices]
    return [reference for index, reference in enumerate(references) if index not in val_indices]


def _split_lookup_keys(split: str) -> tuple[str, ...]:
    """Return explicit split keys to try for a requested split name.

    A missing `test` split intentionally falls back to validation-style held-out data.
    """
    if split == "validation":
        return ("val", "validation")
    if split == "test":
        return ("test", "val", "validation")
    return (split,)


def _normalize_split_value(value: Any) -> str:
    """Normalize one split entry into a comparable string token.

    Dictionaries may contain either `path` or `episode_id` fields.
    """
    if isinstance(value, dict):
        if "path" in value:
            return str(value["path"])
        if "episode_id" in value:
            return str(int(value["episode_id"]))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _reference_matches_split(reference: EpisodeReference, wanted: set[str]) -> bool:
    """Return whether an episode reference is selected by split tokens.

    Matching accepts exact relative paths, path suffixes, basenames, and IDs.
    """
    candidates = {
        reference.relative_path,
        Path(reference.relative_path).name,
        str(reference.episode_id),
    }
    return any(candidate in wanted or any(candidate.endswith(f"/{token}") for token in wanted) for candidate in candidates)


def _validate_episode_arrays(
    path: Path,
    obs: np.ndarray,
    action_label: np.ndarray,
    reset_flag: np.ndarray,
    done: np.ndarray,
) -> None:
    """Validate the minimal arrays required for BC training.

    Shape and finiteness errors are raised before the trainer sees data.
    """
    if obs.ndim != 2:
        raise ValueError(f"{path}: obs must be rank 2, got {obs.shape}")
    if action_label.ndim != 2 or action_label.shape[1] != 12:
        raise ValueError(f"{path}: action_label must have shape (T, 12), got {action_label.shape}")
    if action_label.shape[0] != obs.shape[0]:
        raise ValueError(f"{path}: action_label length {action_label.shape[0]} != obs length {obs.shape[0]}")
    if reset_flag.shape != (obs.shape[0],):
        raise ValueError(f"{path}: reset_flag shape {reset_flag.shape} != {(obs.shape[0],)}")
    if done.shape != (obs.shape[0],):
        raise ValueError(f"{path}: done shape {done.shape} != {(obs.shape[0],)}")
    if obs.shape[0] and not bool(reset_flag[0]):
        raise ValueError(f"{path}: reset_flag[0] must be true")
    if not np.all(np.isfinite(obs)) or not np.all(np.isfinite(action_label)):
        raise ValueError(f"{path}: obs/action_label contain non-finite values")


def _default_reset(length: int) -> np.ndarray:
    """Return a reset mask for datasets missing `reset_flag`.

    Only the first timestep is marked as a reset boundary.
    """
    reset = np.zeros((int(length),), dtype=bool)
    if int(length) > 0:
        reset[0] = True
    return reset


def _default_done(length: int) -> np.ndarray:
    """Return a done mask for datasets missing `done`.

    Only the final timestep is marked as done when the episode is non-empty.
    """
    done = np.zeros((int(length),), dtype=bool)
    if int(length) > 0:
        done[-1] = True
    return done


def _sample_sequence_start(rng: np.random.Generator, episode_length: int, sequence_length: int) -> int:
    """Sample a start index for one fixed-length episode-local window.

    Starts are never chosen outside the source episode.
    """
    length = int(episode_length)
    seq_len = int(sequence_length)
    if length <= seq_len:
        return 0
    return int(rng.integers(0, length, endpoint=False))
