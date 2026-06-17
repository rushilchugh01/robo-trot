# Agent Notes

## Debug-Only Scripts

Files matching `scripts/*_debug.py` are local review/probe helpers and are intentionally ignored by git. Do not rely on them for the committed core pipeline.

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
- `scripts/sanity_check_teacher.py`
- `scripts/validate_dataset.py`
