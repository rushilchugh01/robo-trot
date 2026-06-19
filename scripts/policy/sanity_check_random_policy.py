from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from robo_trot.policies.random_policy import RandomPolicy
from robo_trot.policies.probe_policy import SineFlailPolicy, SineJointProbePolicy, SineJointScanPolicy
from robo_trot.sim.a1_teacher_env import A1TeacherEnv
from robo_trot.training.policy_rollout import PolicyRolloutHarness, load_dataset_contract, validate_env_contract


def run_check(args: argparse.Namespace) -> dict:
    """Run a headless random-policy smoke check and return a report."""
    env = A1TeacherEnv(args.xml_path, {"use_contacts": args.use_contacts, "episode_seconds": args.seconds + 1.0})
    contract = load_dataset_contract(args.dataset_metadata)
    validate_env_contract(env, contract)
    if args.policy_mode == "random":
        policy = RandomPolicy(action_dim=12, action_limit=args.action_limit)
    elif args.policy_mode == "joint_probe":
        policy = SineJointProbePolicy(
            action_dim=12,
            amplitude=args.probe_amplitude,
            frequency_hz=args.probe_frequency,
            policy_dt=env.policy_dt,
            joint_index=args.probe_joint,
        )
    elif args.policy_mode == "flail":
        policy = SineFlailPolicy(
            action_dim=12,
            amplitude=args.flail_amplitude,
            frequency_hz=args.flail_frequency,
            policy_dt=env.policy_dt,
            randomize_phases=not args.no_randomize_flail,
        )
    else:
        policy = SineJointScanPolicy(
            action_dim=12,
            amplitude=args.scan_amplitude,
            frequency_hz=args.scan_frequency,
            policy_dt=env.policy_dt,
            steps_per_joint=args.scan_steps_per_joint,
        )
    harness = PolicyRolloutHarness(env=env, policy=policy, command=np.asarray(args.command, dtype=np.float32), dataset_contract=contract)
    summary = harness.run(seconds=args.seconds, seed=args.seed)
    report = {
        **summary.__dict__,
        "contract_ok": True,
        "joint_names": env.joint_names,
        "actuator_names": env.actuator_names,
    }
    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the random-policy sanity check."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_path", default="assets/mujoco_menagerie/unitree_a1/scene.xml")
    parser.add_argument("--dataset_metadata", default="datasets/a1_teacher_flat_7m_v001_main/shards/shard_00_forward/metadata.json")
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--policy_mode", choices=("random", "joint_probe", "flail", "joint_scan"), default="random")
    parser.add_argument("--action_limit", type=float, default=0.25)
    parser.add_argument("--probe_amplitude", type=float, default=0.35)
    parser.add_argument("--probe_frequency", type=float, default=0.5)
    parser.add_argument("--probe_joint", type=int, default=1)
    parser.add_argument("--flail_amplitude", type=float, default=0.8)
    parser.add_argument("--flail_frequency", type=float, default=0.7)
    parser.add_argument("--no_randomize_flail", action="store_true")
    parser.add_argument("--scan_amplitude", type=float, default=0.6)
    parser.add_argument("--scan_frequency", type=float, default=0.5)
    parser.add_argument("--scan_steps_per_joint", type=int, default=100)
    parser.add_argument("--min_joint_delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--command", nargs=3, type=float, default=[0.0, 0.0, 0.0], metavar=("VX", "VY", "YAW"))
    parser.add_argument("--use_contacts", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    """Run the random-policy sanity-check command-line entry point."""
    args = parse_args()
    report = run_check(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    if (
        not bool(report["contract_ok"])
        or bool(report["had_nan"])
        or int(report["steps"]) <= 0
        or float(report["max_joint_delta"]) < float(args.min_joint_delta)
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
