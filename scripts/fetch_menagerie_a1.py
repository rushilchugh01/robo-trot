from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
UNITREE_A1_DIR = "unitree_a1"


def fetch_unitree_a1(out_dir: Path, force: bool = False) -> Path:
    target = out_dir / UNITREE_A1_DIR
    scene = target / "scene.xml"
    if scene.exists() and not force:
        return target
    if target.exists() and force:
        shutil.rmtree(target)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="menagerie_a1_") as tmp:
        repo = Path(tmp) / "mujoco_menagerie"
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", MENAGERIE_URL, str(repo)],
            check=True,
        )
        subprocess.run(["git", "-C", str(repo), "sparse-checkout", "set", UNITREE_A1_DIR], check=True)
        shutil.copytree(repo / UNITREE_A1_DIR, target)
    if not scene.exists():
        raise FileNotFoundError(f"Expected Menagerie scene not found: {scene}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="assets/mujoco_menagerie")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    target = fetch_unitree_a1(Path(args.out_dir), force=args.force)
    print(target)


if __name__ == "__main__":
    main()
