from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from robo_trot.demos.record_teacher_demos import CATEGORY_COMMAND_RANGES

SHARDS: list[dict[str, Any]] = [
    {"name": "shard_00_forward", "category": "forward", "target_steps": 750_000, "seed": 5100},
    {"name": "shard_01_forward", "category": "forward", "target_steps": 750_000, "seed": 5101},
    {"name": "shard_02_forward", "category": "forward", "target_steps": 750_000, "seed": 5102},
    {"name": "shard_03_forward", "category": "forward", "target_steps": 750_000, "seed": 5103},
    {"name": "shard_04_turn", "category": "turn", "target_steps": 500_000, "seed": 5200},
    {"name": "shard_05_turn", "category": "turn", "target_steps": 500_000, "seed": 5201},
    {"name": "shard_06_slow", "category": "slow", "target_steps": 750_000, "seed": 5300},
    {"name": "shard_07_fast_probe", "category": "fast_probe", "target_steps": 250_000, "seed": 5400},
]


def total_target_steps(shards: list[dict[str, Any]]) -> int:
    return sum(int(shard["target_steps"]) for shard in shards)


def category_step_totals(shards: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for shard in shards:
        category = str(shard["category"])
        totals[category] = totals.get(category, 0) + int(shard["target_steps"])
    return totals


def shards_for_total(total_steps: int, base_shards: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    base = base_shards or SHARDS
    base_total = total_target_steps(base)
    if int(total_steps) <= 0:
        raise ValueError("--total_steps must be positive")
    scaled: list[dict[str, Any]] = []
    assigned = 0
    for index, shard in enumerate(base):
        item = dict(shard)
        if index == len(base) - 1:
            target = int(total_steps) - assigned
        else:
            target = int(round(int(shard["target_steps"]) * int(total_steps) / base_total))
            assigned += target
        item["target_steps"] = max(1, target)
        scaled.append(item)
    return scaled


def scaled_shards(shards: list[dict[str, Any]], scale: float) -> list[dict[str, Any]]:
    if scale <= 0.0:
        raise ValueError("--scale must be positive")
    scaled: list[dict[str, Any]] = []
    for shard in shards:
        item = dict(shard)
        item["target_steps"] = max(1, int(round(int(shard["target_steps"]) * float(scale))))
        scaled.append(item)
    return scaled


def build_shard_command(
    shard: dict[str, Any],
    out_dir: Path,
    xml_path: str,
    resume: bool = False,
    python_executable: str = "python",
) -> list[str]:
    category = str(shard["category"])
    if category not in CATEGORY_COMMAND_RANGES:
        raise ValueError(f"Unknown shard category: {category}")
    teacher_profile = str(CATEGORY_COMMAND_RANGES[category]["teacher_profile"])
    command = [
        python_executable,
        "data/record_teacher_demos.py",
        "--xml_path",
        xml_path,
        "--out_dir",
        str(out_dir / "shards" / str(shard["name"])),
        "--target_steps",
        str(int(shard["target_steps"])),
        "--teacher",
        "footspace",
        "--teacher_profile",
        teacher_profile,
        "--command_category",
        category,
        "--gif_every",
        "0",
        "--review_gifs",
        "0",
        "--seed",
        str(int(shard["seed"])),
    ]
    if resume:
        command.append("--resume")
    return command


def shard_is_complete(out_dir: Path, shard: dict[str, Any]) -> bool:
    metadata_path = out_dir / "shards" / str(shard["name"]) / "metadata.json"
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        return False
    return int(metadata.get("accepted_steps", 0)) >= int(shard["target_steps"])


def launch_shards(
    out_dir: Path,
    xml_path: str,
    workers: int,
    scale: float,
    total_steps: int,
    resume: bool,
    dry_run: bool = False,
) -> int:
    shards = scaled_shards(shards_for_total(total_steps), scale)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    launcher_config = {
        "target_steps": total_target_steps(shards),
        "category_steps": category_step_totals(shards),
        "workers": int(workers),
        "scale": float(scale),
        "shards": shards,
    }
    (out_dir / "launcher_config.json").write_text(json.dumps(launcher_config, indent=2, sort_keys=True))

    pending = [
        shard
        for shard in shards
        if not (resume and shard_is_complete(out_dir, shard))
    ]
    commands = [
        build_shard_command(shard, out_dir=out_dir, xml_path=xml_path, resume=resume)
        for shard in pending
    ]
    if dry_run:
        for command in commands:
            print(" ".join(command))
        return 0

    active: list[tuple[dict[str, Any], subprocess.Popen, Any]] = []
    next_index = 0
    failures: list[tuple[str, int]] = []
    worker_count = max(1, int(workers))
    while next_index < len(pending) or active:
        while next_index < len(pending) and len(active) < worker_count:
            shard = pending[next_index]
            command = commands[next_index]
            log_file = (logs_dir / f"{shard['name']}.log").open("ab")
            log_file.write((" ".join(command) + "\n").encode("utf-8"))
            log_file.flush()
            process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
            active.append((shard, process, log_file))
            next_index += 1

        remaining: list[tuple[dict[str, Any], subprocess.Popen, Any]] = []
        for shard, process, log_file in active:
            code = process.poll()
            if code is None:
                remaining.append((shard, process, log_file))
                continue
            log_file.close()
            if code != 0:
                failures.append((str(shard["name"]), int(code)))
        active = remaining
        if active:
            time.sleep(5.0)

    if failures:
        for shard_name, code in failures:
            print(f"shard failed: {shard_name} exit={code}")
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="datasets/a1_teacher_flat_5m_v001")
    parser.add_argument("--xml_path", default="assets/mujoco_menagerie/unitree_a1/scene.xml")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--total_steps", type=int, default=5_000_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(
        launch_shards(
            out_dir=Path(args.out_dir),
            xml_path=args.xml_path,
            workers=args.workers,
            scale=args.scale,
            total_steps=args.total_steps,
            resume=args.resume,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
