from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robo_trot.sim.a1_teacher_env import A1TeacherEnv
from robo_trot.training.action_mapping_audit import ActionMappingAuditResult, audit_action_mapping
from robo_trot.training.policy_rollout import load_dataset_contract, validate_env_contract


def result_to_dict(result: ActionMappingAuditResult) -> dict:
    """Convert one audit result to a JSON-serializable dictionary.

    This documents the callable contract used by the surrounding pipeline.
    """
    return {
        "index": result.index,
        "joint_name": result.joint_name,
        "actuator_name": result.actuator_name,
        "expected_q_delta": result.expected_q_delta,
        "ctrl_delta": result.ctrl_delta,
        "observed_q_delta": result.observed_q_delta,
        "dominant_joint_index": result.dominant_joint_index,
        "dominant_joint_name": result.dominant_joint_name,
        "passed": result.passed,
        "reason": result.reason,
    }


def print_table(results: list[ActionMappingAuditResult]) -> None:
    """Print action mapping audit results as a compact table.

    This documents the callable contract used by the surrounding pipeline.
    """
    header = (
        "idx  actuator      joint           expected   ctrl       observed   dominant       status  reason"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"{result.index:>3}  "
            f"{result.actuator_name:<13} "
            f"{result.joint_name:<15} "
            f"{result.expected_q_delta:>8.4f}   "
            f"{result.ctrl_delta:>8.4f}   "
            f"{result.observed_q_delta:>8.4f}   "
            f"{result.dominant_joint_name:<14} "
            f"{status:<6}  "
            f"{result.reason}"
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the action mapping audit.

    The returned namespace is consumed by the corresponding command-line entry point.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_path", default="assets/mujoco_menagerie/unitree_a1/scene.xml")
    parser.add_argument("--dataset_metadata", default="datasets/a1_teacher_flat_7m_v001_main/shards/shard_00_forward/metadata.json")
    parser.add_argument("--action_value", type=float, default=0.5)
    parser.add_argument("--settle_steps", type=int, default=10)
    parser.add_argument("--min_observed_delta", type=float, default=1e-3)
    parser.add_argument("--use_contacts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table.")
    return parser.parse_args()


def main() -> None:
    """Run the action-index to joint-index mapping audit.

    This is the direct execution entry point for the module.
    """
    args = parse_args()
    env = A1TeacherEnv(args.xml_path, {"use_contacts": args.use_contacts, "episode_seconds": 2.0})
    contract = load_dataset_contract(args.dataset_metadata)
    validate_env_contract(env, contract)
    results = audit_action_mapping(
        env,
        action_value=args.action_value,
        settle_steps=args.settle_steps,
        min_observed_delta=args.min_observed_delta,
    )
    if args.json:
        print(json.dumps([result_to_dict(result) for result in results], indent=2, sort_keys=True))
    else:
        print_table(results)
    if not all(result.passed for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
