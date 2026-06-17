import subprocess
import sys

from scripts.sanity_check_teacher import evaluate_rollout


def test_evaluate_rollout_accepts_survived_rollout_with_required_progress():
    summary = {
        "survived": True,
        "survival_seconds": 20.0,
        "forward_progress": 2.0,
        "min_base_height": 0.24,
        "max_abs_roll": 0.1,
        "max_abs_pitch": 0.2,
    }

    result = evaluate_rollout(
        summary,
        min_seconds=20.0,
        min_forward_progress=1.0,
        min_base_height=0.18,
        max_abs_roll=0.9,
        max_abs_pitch=0.9,
    )

    assert result == {"ok": True, "reasons": []}


def test_evaluate_rollout_rejects_short_or_unstable_rollout():
    summary = {
        "survived": False,
        "survival_seconds": 8.0,
        "forward_progress": 0.1,
        "min_base_height": 0.16,
        "max_abs_roll": 1.1,
        "max_abs_pitch": 0.2,
        "done_reason": "roll",
    }

    result = evaluate_rollout(
        summary,
        min_seconds=20.0,
        min_forward_progress=1.0,
        min_base_height=0.18,
        max_abs_roll=0.9,
        max_abs_pitch=0.9,
    )

    assert not result["ok"]
    assert result["reasons"] == [
        "terminated: roll",
        "survival_seconds 8.000 < 20.000",
        "forward_progress 0.100 < 1.000",
        "min_base_height 0.160 < 0.180",
        "max_abs_roll 1.100 > 0.900",
    ]


def test_sanity_check_teacher_can_run_as_direct_script():
    result = subprocess.run(
        [sys.executable, "scripts/sanity_check_teacher.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--stand_seconds" in result.stdout
