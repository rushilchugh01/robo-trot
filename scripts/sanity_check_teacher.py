from __future__ import annotations

import argparse
import json

import numpy as np

from robo_trot.demos.record_teacher_demos import FOOTSPACE_TEACHER_PROFILES, make_teacher
from robo_trot.sim.a1_teacher_env import A1TeacherEnv
from robo_trot.robot.a1 import Q_HOME
try:
    from scripts.play_teacher import summarize_rollout
except ModuleNotFoundError:
    from play_teacher import summarize_rollout


def evaluate_rollout(
    summary: dict,
    min_seconds: float,
    min_forward_progress: float,
    min_base_height: float,
    max_abs_roll: float,
    max_abs_pitch: float,
) -> dict:
    reasons: list[str] = []
    if not bool(summary.get("survived", False)):
        reasons.append(f"terminated: {summary.get('done_reason') or 'unknown'}")
    if float(summary.get("survival_seconds", 0.0)) < float(min_seconds):
        reasons.append(f"survival_seconds {float(summary.get('survival_seconds', 0.0)):.3f} < {float(min_seconds):.3f}")
    if float(summary.get("forward_progress", 0.0)) < float(min_forward_progress):
        reasons.append(
            f"forward_progress {float(summary.get('forward_progress', 0.0)):.3f} < {float(min_forward_progress):.3f}"
        )
    if float(summary.get("min_base_height", 0.0)) < float(min_base_height):
        reasons.append(f"min_base_height {float(summary.get('min_base_height', 0.0)):.3f} < {float(min_base_height):.3f}")
    if float(summary.get("max_abs_roll", 0.0)) > float(max_abs_roll):
        reasons.append(f"max_abs_roll {float(summary.get('max_abs_roll', 0.0)):.3f} > {float(max_abs_roll):.3f}")
    if float(summary.get("max_abs_pitch", 0.0)) > float(max_abs_pitch):
        reasons.append(f"max_abs_pitch {float(summary.get('max_abs_pitch', 0.0)):.3f} > {float(max_abs_pitch):.3f}")
    return {"ok": not reasons, "reasons": reasons}


def rollout_constant_command(
    xml_path: str,
    teacher_profile: str,
    seconds: float,
    command: np.ndarray,
    seed: int,
    mode: str,
) -> dict:
    env = A1TeacherEnv(xml_path, {"use_contacts": True, "episode_seconds": seconds + 1.0})
    teacher = make_teacher("footspace", xml_path, env.policy_dt, profile=teacher_profile)
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    teacher.reset(rng)
    states: list[dict] = []
    done_reason = ""
    for step in range(int(seconds / env.policy_dt)):
        state = env.get_state()
        if mode == "home":
            q_des = Q_HOME
        else:
            q_des = teacher.compute(state, command)["q_teacher"]
        _reward, done, info = env.step_q_des(q_des)
        states.append(env.get_state())
        if done:
            done_reason = str(info.get("done_reason", "done"))
            return summarize_rollout(states, (step + 1) * env.policy_dt, done_reason)
    return summarize_rollout(states, seconds, done_reason)


def run_checks(args: argparse.Namespace) -> dict:
    stand_summary = rollout_constant_command(
        xml_path=args.xml_path,
        teacher_profile=args.teacher_profile,
        seconds=args.stand_seconds,
        command=np.zeros(3, dtype=np.float32),
        seed=args.seed,
        mode="home",
    )
    walk_summary = rollout_constant_command(
        xml_path=args.xml_path,
        teacher_profile=args.teacher_profile,
        seconds=args.walk_seconds,
        command=np.array([args.walk_vx, 0.0, 0.0], dtype=np.float32),
        seed=args.seed,
        mode="trot",
    )
    stand_gate = evaluate_rollout(
        stand_summary,
        min_seconds=args.stand_seconds,
        min_forward_progress=-float("inf"),
        min_base_height=args.min_base_height,
        max_abs_roll=args.max_abs_roll,
        max_abs_pitch=args.max_abs_pitch,
    )
    walk_gate = evaluate_rollout(
        walk_summary,
        min_seconds=args.walk_seconds,
        min_forward_progress=args.min_walk_progress,
        min_base_height=args.min_base_height,
        max_abs_roll=args.max_abs_roll,
        max_abs_pitch=args.max_abs_pitch,
    )
    return {
        "ok": bool(stand_gate["ok"] and walk_gate["ok"]),
        "stand": {"summary": stand_summary, "gate": stand_gate},
        "walk": {"summary": walk_summary, "gate": walk_gate},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_path", default="assets/mujoco_menagerie/unitree_a1/scene.xml")
    parser.add_argument("--teacher_profile", choices=tuple(FOOTSPACE_TEACHER_PROFILES), default="strict_walk")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stand_seconds", type=float, default=30.0)
    parser.add_argument("--walk_seconds", type=float, default=20.0)
    parser.add_argument("--walk_vx", type=float, default=0.5)
    parser.add_argument("--min_walk_progress", type=float, default=1.0)
    parser.add_argument("--min_base_height", type=float, default=0.18)
    parser.add_argument("--max_abs_roll", type=float, default=0.9)
    parser.add_argument("--max_abs_pitch", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    result = run_checks(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
