from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mujoco

from robo_trot.robot.model_info import describe_model


def main() -> None:
    """Document the main callable contract.

    This is the direct execution entry point for the module.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_path", default="assets/mujoco_menagerie/unitree_a1/scene.xml")
    args = parser.parse_args()
    xml_path = Path(args.xml_path)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    print(describe_model(model))


if __name__ == "__main__":
    main()
