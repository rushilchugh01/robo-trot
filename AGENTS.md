# Agent Notes

## Debug-Only Scripts

Files matching `scripts/*_debug.py` or `scripts/**/*_debug.py` are local review/probe helpers and are intentionally ignored by git. Do not rely on them for the committed core pipeline.

Local debug workflow details live in `docs/debug_review_helpers.md`; that file is also intentionally ignored by git.

Current debug-only helpers:
- `scripts/make_gif_contact_sheets_debug.py`
- `scripts/probe_middle_yaw_debug.py`
- `scripts/sweep_teacher_speeds_debug.py`
- `scripts/sweep_teacher_yaw_debug.py`
- `data/record_balanced_teacher_demos_debug.py`
- `tests/*_debug.py`

Core scripts that should remain commit candidates include:
- `data/record_teacher_demos.py`
- `scripts/fetch_menagerie_a1.py`
- `scripts/inspect_a1_model.py`
- `scripts/inspect_dataset.py`
- `scripts/play_teacher.py`
- `scripts/play_random_policy.py`
- `scripts/audit_action_mapping.py`
- `scripts/sanity_check_teacher.py`
- `scripts/sanity_check_random_policy.py`
- `scripts/validate_dataset.py`
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

Keep `data/*.py` and root `scripts/*.py` as command-line compatibility wrappers unless a script is explicitly debug-only. Place new script implementations in the relevant `scripts/<purpose>/` subfolder.
