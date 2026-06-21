from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robo_trot.training.evaluate_checkpoint import evaluate_checkpoint


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for standalone checkpoint evaluation.

    The CLI supports MP4 headless eval and live MuJoCo viewing.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", choices=("mlp", "txl"), required=True)
    parser.add_argument("--xml_path", default="assets/mujoco_menagerie/unitree_a1/scene.xml")
    parser.add_argument("--dataset_metadata", default=None)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--command", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument("--save_media", default=None, help="Write rollout media as H.264 MP4; .gif names are redirected.")
    parser.add_argument("--save_gif", default=None, help="Legacy alias for --save_media; output is normalized to .mp4.")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gif_fps", type=int, default=60)
    parser.add_argument("--gif_seconds", type=float, default=None)
    parser.add_argument("--gif_width", type=int, default=480)
    parser.add_argument("--gif_height", type=int, default=270)
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--dataset_eval_split", default="test")
    parser.add_argument("--dataset_eval_batch_size", type=int, default=4096)
    parser.add_argument("--dataset_eval_max_batches", type=int, default=16)
    parser.add_argument("--sequence_length", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run one checkpoint evaluation and print JSON metrics.

    This is the direct execution entry point for the script wrapper.
    """
    args = parse_args(argv)
    metrics = evaluate_checkpoint(
        checkpoint=args.checkpoint,
        model_type=args.model,
        xml_path=args.xml_path,
        dataset_metadata=args.dataset_metadata,
        seconds=args.seconds,
        command=np.asarray(args.command, dtype=np.float32),
        save_media=args.save_media,
        save_gif=args.save_gif,
        viewer=args.viewer,
        seed=args.seed,
        gif_fps=args.gif_fps,
        gif_seconds=args.gif_seconds,
        gif_width=args.gif_width,
        gif_height=args.gif_height,
        dataset_dir=args.dataset_dir,
        dataset_eval_split=args.dataset_eval_split,
        dataset_eval_batch_size=args.dataset_eval_batch_size,
        dataset_eval_max_batches=args.dataset_eval_max_batches,
        sequence_length=args.sequence_length,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
