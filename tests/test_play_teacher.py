import numpy as np
import pytest

from scripts.play_teacher import summary_line, summarize_rollout


def test_summarize_rollout_reports_core_sanity_metrics():
    states = [
        {
            "base_pos": np.array([0.0, 0.0, 0.25], dtype=np.float32),
            "base_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "roll": 0.1,
            "pitch": -0.1,
        },
        {
            "base_pos": np.array([1.2, 0.0, 0.24], dtype=np.float32),
            "base_quat": np.array([0.9800666, 0.0, 0.0, 0.1986693], dtype=np.float32),
            "roll": 0.2,
            "pitch": 0.15,
        },
    ]

    summary = summarize_rollout(states, seconds=20.0, done_reason="")

    assert summary["survived"] is True
    assert summary["survival_seconds"] == 20.0
    assert summary["forward_progress"] == np.float32(1.2)
    assert summary["min_base_height"] == np.float32(0.24)
    assert summary["max_abs_roll"] == 0.2
    assert summary["max_abs_pitch"] == 0.15
    assert summary["yaw_delta"] == pytest.approx(0.4, abs=1e-5)


def test_summary_line_marks_early_termination_as_not_survived():
    states = [
        {
            "base_pos": np.array([0.0, 0.0, 0.17], dtype=np.float32),
            "base_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "roll": 0.0,
            "pitch": 0.0,
        }
    ]

    line = summary_line(states, seconds=3.5, done_reason="base_height")

    assert line.startswith("summary: ")
    assert '"survived": false' in line
    assert '"done_reason": "base_height"' in line
    assert '"survival_seconds": 3.5' in line
