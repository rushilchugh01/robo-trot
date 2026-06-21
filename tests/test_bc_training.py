import json
from pathlib import Path

import numpy as np
import pytest


def make_bc_episode(length: int, offset: float) -> dict[str, np.ndarray]:
    """Return a compact behavior-cloning episode fixture.

    The observation offset makes episode-boundary assertions unambiguous.
    """
    obs = np.zeros((length, 56), dtype=np.float32)
    obs[:, 0] = np.arange(length, dtype=np.float32) + offset
    obs[:, 49] = 1.0
    obs[0, 51] = 1.0
    action = np.zeros((length, 12), dtype=np.float32)
    action[:, 0] = np.linspace(-0.5, 0.5, length, dtype=np.float32)
    return {
        "obs": obs,
        "action_label": action,
        "reset_flag": np.array([idx == 0 for idx in range(length)], dtype=bool),
        "done": np.array([idx == length - 1 for idx in range(length)], dtype=bool),
    }


def write_bc_dataset(root: Path, with_splits: bool = True) -> Path:
    """Write a small on-disk dataset matching the teacher NPZ layout.

    The fixture includes train and validation episodes for split handling.
    """
    episodes_dir = root / "episodes"
    episodes_dir.mkdir(parents=True)
    entries = []
    for episode_id, (length, offset) in enumerate(((5, 0.0), (4, 100.0), (3, 200.0))):
        path = episodes_dir / f"ep_{episode_id:06d}.npz"
        np.savez_compressed(path, **make_bc_episode(length, offset))
        entries.append({"episode_id": episode_id, "accepted": True, "path": f"episodes/{path.name}"})
    metadata = {
        "obs_dim": 56,
        "action_dim": 12,
        "joint_names": [f"j{idx}" for idx in range(12)],
        "actuator_names": [f"a{idx}" for idx in range(12)],
        "q_home": [0.0] * 12,
        "action_scale": [1.0] * 12,
        "episodes": entries,
    }
    (root / "metadata.json").write_text(json.dumps(metadata))
    if with_splits:
        (root / "splits.json").write_text(
            json.dumps(
                {
                    "train": ["episodes/ep_000000.npz", "episodes/ep_000001.npz"],
                    "val": ["episodes/ep_000002.npz"],
                }
            )
        )
    return root


def test_mlp_transition_loader_shapes_from_splits(tmp_path):
    """MLP batches expose observation/action labels from the requested split."""
    from robo_trot.training.dataset import BehaviorCloningDataset

    dataset = BehaviorCloningDataset(write_bc_dataset(tmp_path), split="val", seed=7)

    batch = dataset.sample_transition_batch(batch_size=4, rng=np.random.default_rng(0))

    assert batch.obs.shape == (4, 56)
    assert batch.action_label.shape == (4, 12)
    assert batch.reset_mask.shape == (4,)
    assert set(batch.episode_id.tolist()) == {2}
    assert np.all(batch.obs[:, 0] >= 200.0)


def test_deterministic_fallback_split_is_stable(tmp_path):
    """Fallback train/val splits are deterministic when splits.json is absent."""
    from robo_trot.training.dataset import discover_episode_paths

    root = write_bc_dataset(tmp_path, with_splits=False)

    first = discover_episode_paths(root, split="train", seed=123, val_fraction=0.34)
    second = discover_episode_paths(root, split="train", seed=123, val_fraction=0.34)
    val = discover_episode_paths(root, split="val", seed=123, val_fraction=0.34)

    assert [item.relative_path for item in first] == [item.relative_path for item in second]
    assert first
    assert val
    assert set(item.relative_path for item in first).isdisjoint(item.relative_path for item in val)


def test_fallback_test_split_matches_held_out_validation_split(tmp_path):
    """The test split uses deterministic held-out data without explicit split metadata."""
    from robo_trot.training.dataset import discover_episode_paths

    root = write_bc_dataset(tmp_path, with_splits=False)

    validation = discover_episode_paths(root, split="val", seed=99, val_fraction=0.34)
    test = discover_episode_paths(root, split="test", seed=99, val_fraction=0.34)

    assert [item.relative_path for item in test] == [item.relative_path for item in validation]


def test_txl_sequence_loader_shapes_and_masks(tmp_path):
    """TXL batches expose fixed-length sequences with valid and reset masks."""
    from robo_trot.training.dataset import BehaviorCloningDataset

    dataset = BehaviorCloningDataset(write_bc_dataset(tmp_path), split="train", seed=3)

    batch = dataset.sample_sequence_batch(batch_size=3, sequence_length=4, rng=np.random.default_rng(4))

    assert batch.obs.shape == (3, 4, 56)
    assert batch.action_label.shape == (3, 4, 12)
    assert batch.valid_mask.shape == (3, 4)
    assert batch.reset_mask.shape == (3, 4)
    assert batch.valid_mask.dtype == np.bool_
    assert batch.reset_mask.dtype == np.bool_
    assert np.all(batch.valid_mask.any(axis=1))


def test_txl_sequence_batches_do_not_cross_episode_boundaries(tmp_path):
    """Sequence windows never mix timesteps from different source episodes."""
    from robo_trot.training.dataset import BehaviorCloningDataset

    dataset = BehaviorCloningDataset(write_bc_dataset(tmp_path), split="train", seed=5)

    batch = dataset.sample_sequence_batch(batch_size=20, sequence_length=4, rng=np.random.default_rng(5))

    for row in range(batch.episode_id.shape[0]):
        valid_episode_ids = batch.episode_id[row][batch.valid_mask[row]]
        assert len(set(valid_episode_ids.tolist())) == 1


def test_txl_stream_chunks_separate_episode_and_trial_resets(tmp_path):
    """Stream chunks clear memory only at true episode starts."""
    from robo_trot.training.dataset import BehaviorCloningDataset

    dataset_dir = write_bc_dataset(tmp_path, with_splits=False)
    episode_path = dataset_dir / "episodes" / "ep_000000.npz"
    with np.load(episode_path) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["reset_flag"] = arrays["reset_flag"].copy()
    arrays["reset_flag"][2] = True
    np.savez_compressed(episode_path, **arrays)
    dataset = BehaviorCloningDataset(dataset_dir, split="all", seed=5)

    first_chunk = dataset.make_stream_chunk_batch([(0, 0)], sequence_length=4)
    later_chunk = dataset.make_stream_chunk_batch([(0, 2)], sequence_length=3)

    assert first_chunk.episode_reset_mask.tolist() == [[True, False, False, False]]
    assert first_chunk.trial_reset_mask[0, 2]
    assert later_chunk.episode_reset_mask.tolist() == [[False, False, False]]
    assert later_chunk.trial_reset_mask[0, 0]


def test_checkpoint_atomic_save_load_and_incomplete_dirs_are_ignored(tmp_path):
    """Checkpoint helpers write complete directories and ignore partial output."""
    from robo_trot.training.checkpointing import find_complete_checkpoints, load_checkpoint, save_checkpoint_atomic

    complete = tmp_path / "step_000001000"
    save_checkpoint_atomic(
        complete,
        metadata={"model": "mlp", "update": 1000},
        arrays={"weights": np.ones((2, 3), dtype=np.float32)},
    )
    incomplete = tmp_path / "step_000002000"
    incomplete.mkdir()
    (incomplete / "metadata.json").write_text("{}")

    loaded = load_checkpoint(complete)
    found = find_complete_checkpoints(tmp_path)

    assert loaded.metadata["update"] == 1000
    assert np.array_equal(loaded.arrays["weights"], np.ones((2, 3), dtype=np.float32))
    assert [item.path.name for item in found] == ["step_000001000"]


def test_evaluator_ignores_incomplete_checkpoint_dirs(tmp_path):
    """Evaluator checkpoint discovery only returns directories with completion markers."""
    from robo_trot.training.checkpointing import save_checkpoint_atomic
    from robo_trot.training.evaluate_checkpoint import collect_policy_checkpoints

    checkpoint_root = tmp_path / "mlp" / "checkpoints"
    save_checkpoint_atomic(checkpoint_root / "step_000001000", metadata={"model": "mlp", "update": 1000})
    incomplete = checkpoint_root / "step_000002000"
    incomplete.mkdir(parents=True)
    (incomplete / "metadata.json").write_text(json.dumps({"model": "mlp", "update": 2000}))

    found = collect_policy_checkpoints(tmp_path, "mlp")

    assert [item.update for item in found] == [1000]


def test_evaluator_candidates_respect_model_filter(tmp_path):
    """Checkpoint watcher can restrict eval work to TXL-only runs."""
    from robo_trot.training.checkpointing import save_checkpoint_atomic
    from robo_trot.training.parallel_train_bc import _interleaved_eval_candidates

    save_checkpoint_atomic(tmp_path / "mlp" / "checkpoints" / "step_000001000", metadata={"model": "mlp", "update": 1000})
    save_checkpoint_atomic(tmp_path / "txl" / "checkpoints" / "step_000001000", metadata={"model": "txl", "update": 1000})

    candidates = _interleaved_eval_candidates(tmp_path, eval_every=1000, evaluated=set(), models=("txl",))

    assert [(model, record.update) for model, record in candidates] == [("txl", 1000)]


def test_evaluator_waits_for_backlog_after_training_completion(tmp_path):
    """Evaluator exit requires both final training metrics and drained evals."""
    from robo_trot.training.checkpointing import save_checkpoint_atomic
    from robo_trot.training.parallel_train_bc import _evaluator_ready_to_exit

    save_checkpoint_atomic(tmp_path / "mlp" / "checkpoints" / "step_000000100", metadata={"model": "mlp", "update": 100})
    metrics_path = tmp_path / "mlp" / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({"model_type": "mlp", "update": 100}) + "\n")

    assert not _evaluator_ready_to_exit(tmp_path, max_updates=100, eval_every=100, evaluated=set(), models=("mlp",))
    assert _evaluator_ready_to_exit(
        tmp_path,
        max_updates=100,
        eval_every=100,
        evaluated={("mlp", 100)},
        models=("mlp",),
    )


def test_eval_reward_terms_are_logged_separately():
    """Reward computation returns a scalar total plus named diagnostic terms."""
    from robo_trot.training.eval_reward import compute_eval_reward

    state = {
        "base_lin_vel_body": np.array([0.25, 0.0, 0.0], dtype=np.float32),
        "base_ang_vel_body": np.array([0.0, 0.0, 0.2], dtype=np.float32),
        "projected_gravity": np.array([0.0, 0.0, -1.0], dtype=np.float32),
        "base_pos": np.array([0.0, 0.0, 0.31], dtype=np.float32),
        "roll": 0.05,
        "pitch": -0.04,
        "torque": np.ones(12, dtype=np.float32),
        "foot_contacts": np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
        "foot_pos": np.zeros((4, 3), dtype=np.float32),
    }

    result = compute_eval_reward(
        state,
        command=np.array([0.3, 0.0, 0.2], dtype=np.float32),
        action=np.full(12, 0.1, dtype=np.float32),
        prev_action=np.zeros(12, dtype=np.float32),
        prev_foot_pos=np.zeros((4, 3), dtype=np.float32),
        done=False,
    )

    assert result.total > 0.0
    assert "forward_velocity" in result.terms
    assert "yaw_tracking" in result.terms
    assert "torque_penalty" in result.terms
    assert "foot_slip_penalty" in result.terms


def test_dataset_action_loss_evaluates_held_out_action_labels(tmp_path):
    """Checkpoint diagnostics compare policy outputs against held-out action labels."""
    pytest.importorskip("torch")
    from robo_trot.policies.mlp_policy import MLPPolicy
    from robo_trot.training.evaluate_checkpoint import evaluate_dataset_action_loss

    dataset_dir = write_bc_dataset(tmp_path, with_splits=False)
    policy = MLPPolicy(obs_dim=56, action_dim=12, hidden_sizes=(16,))

    metrics = evaluate_dataset_action_loss(
        policy=policy,
        model_type="mlp",
        dataset_dir=dataset_dir,
        split="test",
        batch_size=2,
        max_batches=2,
        seed=11,
    )

    assert metrics["dataset_eval_split"] == "test"
    assert metrics["dataset_eval_values"] > 0
    assert metrics["dataset_eval_batches"] > 0
    assert metrics["dataset_eval_action_mse"] >= 0.0
    assert metrics["dataset_eval_action_l1"] >= 0.0


def test_parallel_train_script_exposes_parallel_and_ray_flags():
    """The training CLI exposes local process, CPU affinity, dashboard, and Ray flags."""
    from scripts.policy.parallel_train_bc import parse_args

    args = parse_args(
        [
            "--dataset_dir",
            "datasets/a1_teacher_flat_7m_v001_main",
            "--out_dir",
            "runs/bc_compare_v001",
            "--ray",
        ]
    )

    assert args.mlp_workers == 4
    assert args.txl_workers == 4
    assert args.eval_workers == 1
    assert args.models == "mlp,txl"
    assert args.mlp_cores == "0,1"
    assert args.txl_cores == "2,3"
    assert args.dashboard_host == "0.0.0.0"
    assert args.dashboard_port == 8002
    assert args.txl_memory_seconds == 20.0
    assert args.policy_dt == 0.02
    assert args.eval_gif_fps == 30
    assert args.eval_gif_seconds == 10.0
    assert args.dataset_eval_split == "test"
    assert args.dataset_eval_batch_size == 4096
    assert args.dataset_eval_max_batches == 16
    assert args.ray
    assert args.ray_address == "auto"


def test_parallel_train_can_select_txl_only():
    """The training orchestrator can run TXL without launching MLP workers."""
    from robo_trot.training.parallel_train_bc import _enabled_models, _selected_group_configs, parse_args

    args = parse_args(
        [
            "--dataset_dir",
            "datasets/a1_teacher_flat_7m_v001_main",
            "--out_dir",
            "runs/bc_compare_v001",
            "--models",
            "txl",
            "--txl_workers",
            "8",
            "--txl_cores",
            "0,1,2,3",
        ]
    )

    configs = _selected_group_configs(args)

    assert _enabled_models(args) == ("txl",)
    assert [config.model_type for config in configs] == ["txl"]
    assert configs[0].workers == 8
    assert configs[0].cores == (0, 1, 2, 3)


def test_train_batch_size_is_sharded_across_workers():
    """The public batch-size flag is a group budget, not per-worker memory."""
    from robo_trot.training.parallel_train_bc import _make_group_configs, _per_worker_batch_size, parse_args

    args = parse_args(
        [
            "--dataset_dir",
            "datasets/a1_teacher_flat_7m_v001_main",
            "--out_dir",
            "runs/bc_compare_v001",
            "--batch_size",
            "4096",
            "--sequence_length",
            "64",
            "--mlp_workers",
            "4",
            "--txl_workers",
            "8",
        ]
    )
    mlp_config, txl_config = _make_group_configs(args)

    assert _per_worker_batch_size(mlp_config) == 1024
    assert _per_worker_batch_size(txl_config) == 512


def test_dashboard_script_exposes_host_and_port():
    """The standalone dashboard CLI accepts a run directory, host, and port."""
    from scripts.policy.serve_training_dashboard import parse_args

    args = parse_args(["--run_dir", "runs/bc_compare_v001", "--host", "0.0.0.0", "--port", "8002"])

    assert args.run_dir == "runs/bc_compare_v001"
    assert args.host == "0.0.0.0"
    assert args.port == 8002


def test_checkpoint_eval_script_exposes_save_media_alias():
    """The standalone eval CLI accepts MP4 media output paths."""
    from scripts.policy.evaluate_checkpoint import parse_args

    args = parse_args(["--checkpoint", "ckpt", "--model", "mlp", "--save_media", "rollout.mp4"])

    assert args.save_media == "rollout.mp4"
    assert args.save_gif is None


def test_dashboard_html_is_client_app_shell(tmp_path):
    """Dashboard HTML is a static client application shell."""
    from robo_trot.training.dashboard import render_dashboard_html

    html = render_dashboard_html(tmp_path)

    assert "https://cdn.plot.ly/plotly-2.35.2.min.js" in html
    assert 'id="dashboard-root"' in html
    assert 'id="refresh-button"' in html
    assert "/api/summary" in html
    assert "/api/train-metrics" in html
    assert "/api/eval-metrics" in html
    assert "/api/gifs" in html


def test_dashboard_api_payload_handles_missing_metrics(tmp_path):
    """Dashboard APIs return empty structures before the first metrics arrive."""
    from robo_trot.training.dashboard import build_dashboard_payload

    payload = build_dashboard_payload(tmp_path)

    assert payload["summary"]["models"]["mlp"]["train"] is None
    assert payload["summary"]["models"]["mlp"]["eval"] is None
    assert payload["summary"]["models"]["txl"]["train"] is None
    assert payload["summary"]["models"]["txl"]["eval"] is None
    assert payload["train_metrics"] == {"mlp": [], "txl": []}
    assert payload["eval_metrics"] == []
    assert payload["gifs"] == {"mlp": [], "txl": []}


def test_dashboard_does_not_auto_refresh(tmp_path):
    """Dashboard HTML avoids browser refresh loops during long training review."""
    from robo_trot.training.dashboard import render_dashboard_html

    html = render_dashboard_html(tmp_path)

    assert "http-equiv=\"refresh\"" not in html.lower()
    assert "setinterval" not in html.lower()
    assert "refreshes every" not in html.lower()
    assert "addEventListener(\"click\", render)" in html


def test_dashboard_gif_index_separates_models(tmp_path):
    """Media API entries keep MLP and TXL rollout evidence separate."""
    from robo_trot.training.dashboard import build_gif_index

    media_dir = tmp_path / "eval" / "media" / "step_000001000"
    media_dir.mkdir(parents=True)
    (media_dir / "mlp_vx03.mp4").write_bytes(b"mp4")
    (media_dir / "txl_vx03.mp4").write_bytes(b"mp4")
    rows = [
        {
            "model_type": "mlp",
            "checkpoint_update": 1000,
            "media_paths": {"vx03": "eval/media/step_000001000/mlp_vx03.mp4"},
        },
        {
            "model_type": "txl",
            "checkpoint_update": 1000,
            "media_paths": {"vx03": "eval/media/step_000001000/txl_vx03.mp4"},
        },
    ]

    gifs = build_gif_index(tmp_path, rows)

    assert gifs["mlp"][0]["gifs"]["vx03"]["path"] == "eval/media/step_000001000/mlp_vx03.mp4"
    assert gifs["mlp"][0]["gifs"]["vx03"]["kind"] == "video"
    assert gifs["txl"][0]["gifs"]["vx03"]["path"] == "eval/media/step_000001000/txl_vx03.mp4"


def test_dashboard_gif_index_does_not_mix_checkpoint_steps(tmp_path):
    """Latest eval media API rows stay within their checkpoint directory."""
    from robo_trot.training.dashboard import build_gif_index

    old_dir = tmp_path / "eval" / "gifs" / "step_000001000"
    new_dir = tmp_path / "eval" / "gifs" / "step_000002000"
    old_dir.mkdir(parents=True)
    new_dir.mkdir(parents=True)
    (old_dir / "mlp_vx06.gif").write_bytes(b"GIF89a")
    (new_dir / "mlp_vx03.gif").write_bytes(b"GIF89a")
    rows = [
        {
            "model_type": "mlp",
            "checkpoint_update": 2000,
            "gif_paths": {"vx03": "eval/gifs/step_000002000/mlp_vx03.gif"},
        }
    ]

    gifs = build_gif_index(tmp_path, rows)

    assert gifs["mlp"][0]["checkpoint_update"] == 2000
    assert set(gifs["mlp"][0]["gifs"]) == {"vx03"}
    assert gifs["mlp"][0]["gifs"]["vx03"]["path"] == "eval/gifs/step_000002000/mlp_vx03.gif"


def test_dashboard_summary_uses_highest_update_after_checkpoint_resume(tmp_path):
    """Dashboard training summary stays monotonic after checkpoint resumes."""
    from robo_trot.training.dashboard import build_summary

    metrics = tmp_path / "txl" / "metrics.jsonl"
    metrics.parent.mkdir(parents=True)
    metrics.write_text(
        "\n".join(
            [
                json.dumps({"model_type": "txl", "update": 1400, "train_loss": 0.14}),
                json.dumps({"model_type": "txl", "update": 1100, "train_loss": 0.11}),
            ]
        )
    )

    summary = build_summary(tmp_path)

    assert summary["models"]["txl"]["train"]["update"] == 1400
    assert summary["models"]["txl"]["train"]["train_loss"] == 0.14


def test_dashboard_payload_exposes_held_out_action_label_metrics(tmp_path):
    """Dashboard APIs expose checkpoint dataset action-label diagnostics."""
    from robo_trot.training.dashboard import build_dashboard_payload

    metrics = tmp_path / "eval" / "metrics.jsonl"
    metrics.parent.mkdir(parents=True)
    metrics.write_text(
        json.dumps(
            {
                "model_type": "mlp",
                "checkpoint_update": 1000,
                "dataset_eval_split": "test",
                "dataset_eval_action_mse": 0.0123,
                "dataset_eval_action_l1": 0.0456,
                "dataset_eval_action_clip_fraction": 0.0789,
                "dataset_eval_values": 120,
            }
        )
    )

    payload = build_dashboard_payload(tmp_path)
    eval_row = payload["summary"]["models"]["mlp"]["eval"]

    assert eval_row["dataset_eval_split"] == "test"
    assert eval_row["dataset_eval_action_mse"] == 0.0123
    assert eval_row["dataset_eval_action_l1"] == 0.0456
    assert eval_row["dataset_eval_action_clip_fraction"] == 0.0789
    assert eval_row["dataset_eval_values"] == 120


def test_dashboard_shell_uses_comparable_notation_and_timestamps(tmp_path):
    """Client rendering uses shared scientific notation and timestamped step badges."""
    from robo_trot.training.dashboard import render_dashboard_html

    html = render_dashboard_html(tmp_path)

    assert "return number.toExponential(4).replace" in html
    assert 'tickformat: ".2e"' in html
    assert "function stepBadge(evalRow, trainRow)" in html
    assert "timeLabel(evalRow?.wall_time)" in html
    assert "step ${stepLabel(update)}" in html
    assert "test action MSE" in html


def test_dashboard_shell_pads_scientific_exponents(tmp_path):
    """Client scientific notation uses padded exponents for comparable values."""
    from robo_trot.training.dashboard import render_dashboard_html

    html = render_dashboard_html(tmp_path)

    assert "padStart(2, \"0\")" in html
    assert ".replace(/e([+-])(\\d)$/" in html


def test_dashboard_shell_surfaces_policy_metrics_before_curves(tmp_path):
    """Detailed MLP/TXL policy metric panels render before large graph widgets."""
    from robo_trot.training.dashboard import render_dashboard_html

    html = render_dashboard_html(tmp_path)

    assert "Policy Metrics" in html
    assert "renderStatus(data.summary.models) + renderModels" in html
    assert "renderModels(data.summary.models, data.gifs) + renderCharts" in html


def test_dashboard_shell_has_zoomable_plotly_widgets_and_split_loss_curves(tmp_path):
    """Dashboard client defines Plotly widgets for separate train and val curves."""
    from robo_trot.training.dashboard import render_dashboard_html

    html = render_dashboard_html(tmp_path)

    assert "Plotly.newPlot" in html
    assert "scrollZoom: true" in html
    assert "Training action MSE" in html
    assert "Validation action MSE" in html
    assert "MuJoCo eval reward" in html
    assert "Held-out dataset action MSE" in html
    assert 'plotLoss("plot-train-loss", trainMetrics, "train_loss"' in html
    assert 'plotLoss("plot-val-loss", trainMetrics, "val_loss"' in html


def test_dashboard_shell_renders_looping_muted_video_media(tmp_path):
    """Dashboard client renders MP4 clips as autoplaying muted loop videos."""
    from robo_trot.training.dashboard import render_dashboard_html

    html = render_dashboard_html(tmp_path)

    assert "<video autoplay muted loop playsinline preload=\"auto\"" in html
    assert "function mediaElement(media, label)" in html
    assert "media.kind === \"video\"" in html


def test_dashboard_gif_index_renders_all_eval_gifs_and_dynamic_commands(tmp_path):
    """Media API history includes every evaluated checkpoint command."""
    from robo_trot.training.dashboard import build_gif_index

    gif_dir_1000 = tmp_path / "eval" / "gifs" / "step_000001000"
    gif_dir_2000 = tmp_path / "eval" / "gifs" / "step_000002000"
    gif_dir_1000.mkdir(parents=True)
    gif_dir_2000.mkdir(parents=True)
    for path in (
        gif_dir_1000 / "mlp_vx03.gif",
        gif_dir_1000 / "mlp_side_step.gif",
        gif_dir_2000 / "txl_vx06.gif",
    ):
        path.write_bytes(b"GIF89a")
    rows = [
        {
            "model_type": "mlp",
            "checkpoint_update": 1000,
            "eval_reward_mean": 1.2,
            "gif_paths": {
                "vx03": "eval/gifs/step_000001000/mlp_vx03.gif",
                "side_step": "eval/gifs/step_000001000/mlp_side_step.gif",
            },
        },
        {
            "model_type": "txl",
            "checkpoint_update": 2000,
            "eval_reward_mean": 1.4,
            "gif_paths": {"vx06": "eval/gifs/step_000002000/txl_vx06.gif"},
        },
    ]

    gifs = build_gif_index(tmp_path, rows)

    assert set(gifs["mlp"][0]["gifs"]) == {"vx03", "side_step"}
    assert gifs["mlp"][0]["gifs"]["side_step"]["path"] == "eval/gifs/step_000001000/mlp_side_step.gif"
    assert set(gifs["txl"][0]["gifs"]) == {"vx06"}
    assert gifs["txl"][0]["gifs"]["vx06"]["path"] == "eval/gifs/step_000002000/txl_vx06.gif"


def test_dashboard_media_index_prefers_new_mp4_paths_over_legacy_gifs(tmp_path):
    """Dashboard indexes new MP4 media rows while preserving legacy GIF support."""
    from robo_trot.training.dashboard import build_gif_index

    media_dir = tmp_path / "eval" / "media" / "step_000003000"
    gif_dir = tmp_path / "eval" / "gifs" / "step_000002000"
    media_dir.mkdir(parents=True)
    gif_dir.mkdir(parents=True)
    (media_dir / "txl_vx03.mp4").write_bytes(b"mp4")
    (gif_dir / "mlp_vx03.gif").write_bytes(b"GIF89a")
    rows = [
        {
            "model_type": "txl",
            "checkpoint_update": 3000,
            "media_paths": {"vx03": "eval/media/step_000003000/txl_vx03.mp4"},
        },
        {
            "model_type": "mlp",
            "checkpoint_update": 2000,
            "gif_paths": {"vx03": "eval/gifs/step_000002000/mlp_vx03.gif"},
        },
    ]

    gifs = build_gif_index(tmp_path, rows)

    assert gifs["txl"][0]["gifs"]["vx03"]["kind"] == "video"
    assert gifs["txl"][0]["gifs"]["vx03"]["mime_type"] == "video/mp4"
    assert gifs["mlp"][0]["gifs"]["vx03"]["kind"] == "image"


def test_dashboard_media_index_uses_legacy_gif_when_mp4_path_is_missing(tmp_path):
    """Dashboard falls back to legacy GIF media when a new media path is stale."""
    from robo_trot.training.dashboard import build_gif_index

    gif_dir = tmp_path / "eval" / "gifs" / "step_000002000"
    gif_dir.mkdir(parents=True)
    (gif_dir / "mlp_vx03.gif").write_bytes(b"GIF89a")
    rows = [
        {
            "model_type": "mlp",
            "checkpoint_update": 2000,
            "media_paths": {"vx03": "eval/media/step_000002000/mlp_vx03.mp4"},
            "gif_paths": {"vx03": "eval/gifs/step_000002000/mlp_vx03.gif"},
        }
    ]

    gifs = build_gif_index(tmp_path, rows)

    assert gifs["mlp"][0]["gifs"]["vx03"]["path"] == "eval/gifs/step_000002000/mlp_vx03.gif"
    assert gifs["mlp"][0]["gifs"]["vx03"]["kind"] == "image"


def test_dashboard_gif_index_groups_history_by_model(tmp_path):
    """Historical eval GIFs are grouped under the owning model key."""
    from robo_trot.training.dashboard import build_gif_index

    gif_dir_1000 = tmp_path / "eval" / "gifs" / "step_000001000"
    gif_dir_2000 = tmp_path / "eval" / "gifs" / "step_000002000"
    gif_dir_1000.mkdir(parents=True)
    gif_dir_2000.mkdir(parents=True)
    for path in (
        gif_dir_1000 / "mlp_vx03.gif",
        gif_dir_1000 / "mlp_side_step.gif",
        gif_dir_2000 / "txl_vx06.gif",
    ):
        path.write_bytes(b"GIF89a")
    rows = [
        {
            "model_type": "mlp",
            "checkpoint_update": 1000,
            "gif_paths": {
                "vx03": "eval/gifs/step_000001000/mlp_vx03.gif",
                "side_step": "eval/gifs/step_000001000/mlp_side_step.gif",
            },
        },
        {
            "model_type": "txl",
            "checkpoint_update": 2000,
            "gif_paths": {"vx06": "eval/gifs/step_000002000/txl_vx06.gif"},
        },
    ]

    gifs = build_gif_index(tmp_path, rows)

    assert [row["checkpoint_update"] for row in gifs["mlp"]] == [1000]
    assert [row["checkpoint_update"] for row in gifs["txl"]] == [2000]
    assert "txl_vx06.gif" not in json.dumps(gifs["mlp"])
    assert "mlp_vx03.gif" not in json.dumps(gifs["txl"])


def test_eval_aggregation_publishes_mp4_media_paths_without_gif_aliases():
    """Eval aggregation publishes MP4 media paths without legacy GIF aliases."""
    from robo_trot.training.evaluate_checkpoint import aggregate_eval_rows

    row = aggregate_eval_rows(
        [
            {
                "command_label": "vx03",
                "media_path": "eval/media/step_000001000/txl_vx03.mp4",
                "eval_reward_mean": 1.0,
                "eval_survival_seconds_mean": 2.0,
                "fell": False,
            }
        ],
        model_type="txl",
        checkpoint_update=1000,
    )

    assert row["media_paths"] == {"vx03": "eval/media/step_000001000/txl_vx03.mp4"}
    assert "gif_paths" not in row


def test_eval_aggregation_keeps_gif_paths_for_legacy_rows():
    """Eval aggregation keeps GIF aliases only for actual legacy GIF media."""
    from robo_trot.training.evaluate_checkpoint import aggregate_eval_rows

    row = aggregate_eval_rows(
        [
            {
                "command_label": "vx03",
                "media_path": "eval/gifs/step_000001000/txl_vx03.gif",
                "eval_reward_mean": 1.0,
                "eval_survival_seconds_mean": 2.0,
                "fell": False,
            }
        ],
        model_type="txl",
        checkpoint_update=1000,
    )

    assert row["media_paths"] == {"vx03": "eval/gifs/step_000001000/txl_vx03.gif"}
    assert row["gif_paths"] == {"vx03": "eval/gifs/step_000001000/txl_vx03.gif"}


def test_eval_gif_writer_normalizes_legacy_gif_paths_to_mp4(tmp_path):
    """Eval media writer redirects legacy GIF filenames to MP4 output."""
    import subprocess

    from robo_trot.training.evaluate_checkpoint import _media_output_path, _write_mp4

    output = _media_output_path(tmp_path / "manual.gif")
    frames = [
        np.zeros((5, 7, 3), dtype=np.uint8),
        np.full((5, 7, 3), 255, dtype=np.uint8),
    ]

    _write_mp4(output, frames, fps=2)

    assert output == tmp_path / "manual.mp4"
    assert output.exists()
    assert output.read_bytes()[4:8] == b"ftyp"
    assert output.stat().st_size > 0
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,pix_fmt,width,height",
                "-of",
                "json",
                output.as_posix(),
            ],
            text=True,
        )
    )
    stream = probe["streams"][0]
    assert stream["codec_name"] == "h264"
    assert stream["pix_fmt"] == "yuv420p"
    assert stream["width"] % 2 == 0
    assert stream["height"] % 2 == 0


def test_eval_mp4_writer_pads_short_rollout_to_requested_duration(tmp_path):
    """Eval MP4 writer repeats the final frame for early-fall rollouts."""
    import subprocess

    from robo_trot.training.evaluate_checkpoint import _write_mp4

    output = tmp_path / "fallen_policy.mp4"
    frames = [
        np.zeros((6, 8, 3), dtype=np.uint8),
        np.full((6, 8, 3), 255, dtype=np.uint8),
    ]

    _write_mp4(output, frames, fps=10, min_frame_count=80)

    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,pix_fmt,duration",
                "-of",
                "json",
                output.as_posix(),
            ],
            text=True,
        )
    )
    stream = probe["streams"][0]
    assert stream["codec_name"] == "h264"
    assert stream["pix_fmt"] == "yuv420p"
    assert float(stream["duration"]) >= 7.5


def test_checkpoint_eval_set_uses_mp4_media_directory(monkeypatch, tmp_path):
    """Checkpoint-set eval writes command media under eval/media as MP4 files."""
    import robo_trot.training.evaluate_checkpoint as module

    class FakeEnv:
        """Minimal environment fixture for checkpoint-set media path assertions.

        The evaluator function is patched, so only construction is exercised.
        """

        def __init__(self, xml_path, config):
            """Record constructor inputs for the patched eval flow.

            The real MuJoCo environment is intentionally not launched here.
            """
            self.xml_path = xml_path
            self.config = config

    media_paths: list[Path | None] = []
    clip_seconds: list[float | None] = []

    def fake_run_policy_eval_episode(**kwargs):
        """Capture save_media paths and return deterministic rollout metrics.

        This avoids MuJoCo rollout work while validating evaluator path contracts.
        """
        media_paths.append(kwargs["save_media"])
        clip_seconds.append(kwargs["gif_seconds"])
        return module.RolloutEvalResult(
            label=kwargs["eval_command"].label,
            reward_mean=1.0,
            survival_seconds=2.0,
            fell=False,
            forward_velocity_mean=0.0,
            yaw_rate_mean=0.0,
            roll_pitch_max=0.0,
            foot_slip_mean=0.0,
            reward_terms={},
            media_path=kwargs["save_media"].as_posix(),
        )

    monkeypatch.setattr(module, "load_policy_from_checkpoint", lambda checkpoint, model_type: object())
    monkeypatch.setattr(module, "audit_action_mapping", lambda *args, **kwargs: [])
    monkeypatch.setitem(__import__("sys").modules, "robo_trot.sim.a1_teacher_env", type("M", (), {"A1TeacherEnv": FakeEnv}))
    monkeypatch.setattr(module, "run_policy_eval_episode", fake_run_policy_eval_episode)

    rows = module.evaluate_checkpoint_set(
        checkpoint=tmp_path / "step_000001000",
        model_type="txl",
        xml_path="scene.xml",
        dataset_metadata=None,
        out_dir=tmp_path / "eval",
        checkpoint_update=1000,
    )

    assert media_paths[0] == tmp_path / "eval" / "media" / "step_000001000" / "txl_vx00.mp4"
    assert clip_seconds[0] == 10.0
    assert rows[0]["media_path"] == "eval/media/step_000001000/txl_vx00.mp4"


def test_mlp_policy_output_shape_and_range_when_torch_available():
    """MLP policy emits bounded normalized action labels."""
    torch = pytest.importorskip("torch")
    from robo_trot.policies.mlp_policy import MLPPolicy

    model = MLPPolicy(obs_dim=56, action_dim=12, hidden_sizes=(32, 32))
    output = model(torch.zeros((2, 56), dtype=torch.float32))

    assert tuple(output.shape) == (2, 12)
    assert torch.all(output <= 1.0)
    assert torch.all(output >= -1.0)


def test_txl_policy_output_shape_and_range_when_torch_available():
    """TXL policy emits bounded per-token normalized action labels."""
    torch = pytest.importorskip("torch")
    from robo_trot.policies.txl_policy import TXLPolicy

    model = TXLPolicy(obs_dim=56, action_dim=12, d_model=32, n_head=4, num_layers=2, memory_length=4)
    output, memory = model(
        torch.zeros((2, 5, 56), dtype=torch.float32),
        reset_mask=torch.zeros((2, 5), dtype=torch.bool),
        memory=None,
        return_memory=True,
    )

    assert tuple(output.shape) == (2, 5, 12)
    assert torch.all(output <= 1.0)
    assert torch.all(output >= -1.0)
    assert memory is not None
    assert tuple(memory["valid_mask"].shape) == (2, 4)


def test_txl_policy_act_does_not_clear_memory_on_observation_reset_flag():
    """TXL rollout memory persists across non-episode reset flags in observations."""
    pytest.importorskip("torch")
    from robo_trot.policies.txl_policy import TXLPolicy

    model = TXLPolicy(obs_dim=56, action_dim=12, d_model=32, n_head=4, num_layers=2, memory_length=8)
    first_obs = np.zeros(56, dtype=np.float32)
    second_obs = np.zeros(56, dtype=np.float32)
    second_obs[51] = 1.0

    model.act(first_obs)
    first_lengths = model._rollout_memory["valid_mask"].sum(dim=1).tolist()
    model.act(second_obs)
    second_lengths = model._rollout_memory["valid_mask"].sum(dim=1).tolist()

    assert first_lengths == [1]
    assert second_lengths == [2]


def test_txl_memory_valid_mask_resets_only_episode_rows():
    """TXL memory masks prior cache rows at true episode boundaries."""
    torch = pytest.importorskip("torch")
    from robo_trot.policies.txl_policy import TXLPolicy

    model = TXLPolicy(obs_dim=56, action_dim=12, d_model=32, n_head=4, num_layers=2, memory_length=8)
    _, memory = model(
        torch.zeros((2, 3, 56), dtype=torch.float32),
        reset_mask=torch.zeros((2, 3), dtype=torch.bool),
        valid_mask=torch.ones((2, 3), dtype=torch.bool),
        return_memory=True,
    )
    reset = torch.tensor([[True, False], [False, False]], dtype=torch.bool)
    _, memory = model(
        torch.zeros((2, 2, 56), dtype=torch.float32),
        reset_mask=reset,
        memory=memory,
        valid_mask=torch.ones((2, 2), dtype=torch.bool),
        return_memory=True,
    )

    assert memory["valid_mask"][0].tolist() == [False, False, False, True, True]
    assert memory["valid_mask"][1].tolist() == [True, True, True, True, True]
