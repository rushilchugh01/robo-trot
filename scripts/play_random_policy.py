from __future__ import annotations

import argparse
import json
import time

import mujoco.viewer
import numpy as np

from robo_trot.policies.random_policy import RandomPolicy
from robo_trot.sim.a1_teacher_env import A1TeacherEnv
from robo_trot.training.policy_rollout import PolicyRolloutHarness, load_dataset_contract


def parse_command(values: list[float] | None) -> np.ndarray:
    """Parse a three-value command vector from CLI values."""
    if values is None:
        return np.zeros(3, dtype=np.float32)
    if len(values) != 3:
        raise ValueError("--command requires exactly VX VY YAW")
    return np.asarray(values, dtype=np.float32)


def build_harness(args: argparse.Namespace) -> PolicyRolloutHarness:
    """Build the random-policy rollout harness from CLI arguments."""
    env = A1TeacherEnv(args.xml_path, {"use_contacts": args.use_contacts, "episode_seconds": args.seconds + 1.0})
    policy = RandomPolicy(action_dim=12, action_limit=args.action_limit)
    contract = load_dataset_contract(args.dataset_metadata) if args.dataset_metadata else None
    return PolicyRolloutHarness(env=env, policy=policy, command=parse_command(args.command), dataset_contract=contract)


def run_headless(args: argparse.Namespace) -> None:
    """Run the random policy without opening a MuJoCo viewer."""
    harness = build_harness(args)
    summary = harness.run(seconds=args.seconds, seed=args.seed, print_every=args.print_every)
    print(json.dumps(summary.__dict__, indent=2, sort_keys=True))


def run_viewer(args: argparse.Namespace) -> None:
    """Run the random policy loop while syncing the MuJoCo viewer."""
    harness = build_harness(args)
    rng = np.random.default_rng(args.seed)
    env = harness.env
    policy = harness.policy
    env.reset(seed=args.seed)
    policy.reset(rng)
    prev_action = np.zeros(12, dtype=np.float32)
    prev_reward = 0.0
    reset_flag = True
    phase = 0.0
    steps_target = max(1, int(round(float(args.seconds) / float(env.policy_dt))))

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for step in range(steps_target):
            loop_start = time.time()
            obs = env.make_obs(harness.command, prev_action, prev_reward, reset_flag, phase)
            action = policy.act(obs)
            q_des = harness.action_to_q_des(action)
            reward, done, info = env.step_q_des(q_des)
            prev_action = action
            prev_reward = float(reward)
            reset_flag = False
            if args.print_every and step % max(1, int(args.print_every)) == 0:
                print(
                    f"step={step} reward={float(reward):.4f} "
                    f"action_abs_max={float(np.max(np.abs(action))):.4f} "
                    f"q_des_min={float(np.min(q_des)):.4f} q_des_max={float(np.max(q_des)):.4f}"
                )
            viewer.sync()
            if done:
                print(f"done step={step} reason={info.get('done_reason', 'done')}")
                break
            sleep_time = float(env.policy_dt) - (time.time() - loop_start)
            if sleep_time > 0.0:
                time.sleep(sleep_time)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the random-policy viewer."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_path", default="assets/mujoco_menagerie/unitree_a1/scene.xml")
    parser.add_argument("--dataset_metadata", default="datasets/a1_teacher_flat_7m_v001_main/shards/shard_00_forward/metadata.json")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--action_limit", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--command", nargs=3, type=float, default=None, metavar=("VX", "VY", "YAW"))
    parser.add_argument("--print_every", type=int, default=25)
    parser.add_argument("--use_contacts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no_viewer", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the random-policy playback command-line entry point."""
    args = parse_args()
    if args.no_viewer:
        run_headless(args)
    else:
        run_viewer(args)


if __name__ == "__main__":
    main()
