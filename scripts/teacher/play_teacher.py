from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mujoco.viewer
import numpy as np

from robo_trot.sim.a1_teacher_env import A1TeacherEnv
from robo_trot.robot.a1 import Q_HOME
from robo_trot.data_pipeline.record_teacher_demos import episode_yaw_delta
from robo_trot.data_pipeline.record_teacher_demos import FOOTSPACE_TEACHER_PROFILES
from robo_trot.teachers.footspace_cpg_ik import FootspaceCPGIKTeacher


def scheduled_command(t: float) -> np.ndarray:
    if t < 5.0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    if t < 15.0:
        return np.array([0.3, 0.0, 0.0], dtype=np.float32)
    if t < 25.0:
        return np.array([0.6, 0.0, 0.0], dtype=np.float32)
    if t < 35.0:
        return np.array([0.3, 0.0, 0.35], dtype=np.float32)
    return np.array([0.3, 0.0, -0.35], dtype=np.float32)


def summarize_rollout(states: list[dict], seconds: float, done_reason: str) -> dict:
    if not states:
        return {
            "survived": False,
            "survival_seconds": 0.0,
            "done_reason": done_reason or "no_steps",
            "forward_progress": 0.0,
            "min_base_height": 0.0,
            "max_abs_roll": 0.0,
            "max_abs_pitch": 0.0,
            "yaw_delta": 0.0,
        }
    base_pos = np.asarray([state["base_pos"] for state in states], dtype=np.float32)
    episode = {"base_quat": np.asarray([state["base_quat"] for state in states], dtype=np.float32)}
    return {
        "survived": done_reason == "",
        "survival_seconds": float(seconds),
        "done_reason": done_reason,
        "forward_progress": float(base_pos[-1, 0] - base_pos[0, 0]),
        "min_base_height": float(base_pos[:, 2].min()),
        "max_abs_roll": float(max(abs(float(state["roll"])) for state in states)),
        "max_abs_pitch": float(max(abs(float(state["pitch"])) for state in states)),
        "yaw_delta": float(episode_yaw_delta(episode)),
    }


def summary_line(states: list[dict], seconds: float, done_reason: str) -> str:
    return "summary: " + json.dumps(summarize_rollout(states, seconds, done_reason), sort_keys=True)


def run(args: argparse.Namespace) -> None:
    env = A1TeacherEnv(args.xml_path, {"use_contacts": args.use_contacts, "episode_seconds": args.seconds + 1.0})
    teacher = FootspaceCPGIKTeacher(
        xml_path=args.xml_path,
        policy_dt=env.policy_dt,
        **FOOTSPACE_TEACHER_PROFILES[args.teacher_profile],
    )
    rng = np.random.default_rng(args.seed)
    env.reset(seed=args.seed)
    teacher.reset(rng)
    steps = int(args.seconds / env.policy_dt)
    viewer = None
    if not args.no_viewer:
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
    try:
        last_print = -1
        states: list[dict] = []
        done_reason = ""
        for step in range(steps):
            t = step * env.policy_dt
            command = scheduled_command(t)
            state = env.get_state()
            if args.mode == "home":
                q_des = Q_HOME
                phase = 0.0
            else:
                output = teacher.compute(state, command)
                q_des = output["q_teacher"]
                phase = output["phase"]
            reward, done, info = env.step_q_des(q_des)
            states.append(env.get_state())
            if viewer is not None:
                viewer.sync()
                time.sleep(env.policy_dt)
            sec = int(t)
            if sec != last_print:
                last_print = sec
                s = env.get_state()
                print(
                    f"t={t:05.2f} cmd={command.tolist()} phase={phase:05.2f} "
                    f"base={s['base_pos'].round(3).tolist()} roll={s['roll']:.3f} "
                    f"pitch={s['pitch']:.3f} reward={reward:.3f} done={done}"
                )
            if done:
                done_reason = str(info.get("done_reason", "done"))
                print(summary_line(states, (step + 1) * env.policy_dt, done_reason))
                raise SystemExit(f"terminated at t={t:.2f}: {info.get('done_reason')}")
        print(summary_line(states, args.seconds, done_reason))
    finally:
        if viewer is not None:
            viewer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_path", default="assets/mujoco_menagerie/unitree_a1/scene.xml")
    parser.add_argument("--mode", choices=("home", "trot"), default="trot")
    parser.add_argument("--seconds", type=float, default=45.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_viewer", action="store_true")
    parser.add_argument("--use_contacts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--teacher", choices=("footspace",), default="footspace")
    parser.add_argument("--teacher_profile", choices=tuple(FOOTSPACE_TEACHER_PROFILES), default="strict_walk")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
