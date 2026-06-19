# Agent Notes

## Debug-Only Scripts

Files under `scripts/debug/` and files matching `scripts/**/*_debug.py` are local review/probe helpers and are intentionally ignored by git. Do not rely on them for the committed core pipeline.

Local debug workflow details live in `docs/debug_review_helpers.md`; that file is also intentionally ignored by git.

Current debug-only helpers:
- `scripts/debug/make_gif_contact_sheets_debug.py`
- `scripts/debug/probe_middle_yaw_debug.py`
- `scripts/debug/sweep_teacher_speeds_debug.py`
- `scripts/debug/sweep_teacher_yaw_debug.py`
- `data/record_balanced_teacher_demos_debug.py`
- `tests/*_debug.py`

Core scripts that should remain commit candidates include:
- `data/record_teacher_demos.py`
- `scripts/assets/fetch_menagerie_a1.py`
- `scripts/robot/inspect_a1_model.py`
- `scripts/data/inspect_dataset.py`
- `scripts/data/validate_dataset.py`
- `scripts/teacher/play_teacher.py`
- `scripts/teacher/sanity_check_teacher.py`
- `scripts/policy/play_random_policy.py`
- `scripts/policy/sanity_check_random_policy.py`
- `scripts/policy/audit_action_mapping.py`

## Package Layout

Keep implementation code under `robo_trot/`:
- `robo_trot/robot/` for A1 constants, kinematics, and model metadata.
- `robo_trot/sim/` for MuJoCo environment wrappers.
- `robo_trot/teachers/` for teacher controller code.
- `robo_trot/data_pipeline/` for rollout recording, dataset writing, sharding, manifests, and validation.
- `robo_trot/policies/` for policy implementations and action conversion.
- `robo_trot/training/` for policy rollout harnesses, contract checks, and training utilities.

Keep `data/*.py` as command-line compatibility wrappers. Keep runnable scripts in the relevant `scripts/<purpose>/` subfolder; the root `scripts/` directory should only contain package markers and subdirectories.

## Docstring Gate

`tests/test_docstring_coverage.py` is a production lint gate. Every committed class, function, and method under `robo_trot/`, `data/`, and `scripts/` needs at least two non-empty docstring lines, excluding ignored debug helpers.

Math-heavy helpers detected by trig, quaternion, yaw, or IK usage need longer docstrings with equation, unit, or frame detail. Do not bypass this with generic one-liners.
