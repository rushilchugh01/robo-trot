# robo-trot

MuJoCo Unitree A1 teacher-controller rollout pipeline for flat-ground behavior cloning demonstrations.

The recorder stores teacher desired joint targets (`q_teacher`) and normalized action labels (`action_label`), not just actual joint positions.

## Quick Commands

```bash
python scripts/fetch_menagerie_a1.py --out_dir assets/mujoco_menagerie
python scripts/inspect_a1_model.py --xml_path assets/mujoco_menagerie/unitree_a1/scene.xml
python scripts/play_teacher.py --mode home --seconds 30 --no_viewer
python scripts/play_teacher.py --teacher footspace --teacher_profile strict_walk --mode trot --seconds 20 --no_viewer
python scripts/sanity_check_teacher.py --stand_seconds 30 --walk_seconds 20 --walk_vx 0.5 --teacher_profile strict_walk
python data/record_teacher_demos.py \
  --teacher footspace \
  --teacher_profile strict_walk \
  --command_profile default \
  --xml_path assets/mujoco_menagerie/unitree_a1/scene.xml \
  --out_dir datasets/a1_teacher_flat_v001 \
  --target_steps 10000 \
  --gif_every 100 \
  --seed 0
```

Run `scripts/validate_dataset.py` on any dataset before using it for cloning. The validator checks array shapes, `action_label == clip((q_teacher - q_home) / action_scale)`, episode reset/done boundaries, that the `sin(phase), cos(phase)` fields inside `obs` match the saved `phase` column, and that the observation state slices match the saved raw state arrays for that same timestep. Older review artifacts generated before these alignment checks are useful only for visual comparison; regenerate them before using their `.npz` episodes as training data.

The default observation includes foot contacts and has `obs_dim=56`. For the fallback observation without contact features, add `--no-use_contacts`; the episode still stores `foot_contacts` and `foot_pos` as debug arrays, but the actor observation has `obs_dim=52`.

## Acceptance Gates

Episodes are rejected if the robot falls, is too short, clips too many action labels, makes too little commanded forward progress, has too little foot clearance, or shows excessive contacted-foot slip. Sliding is a hard failure: `record_teacher_demos.py` reports it as `foot_sliding`, and `inspect_dataset.py` reports aggregate contact-slip metrics from accepted episodes.

The slip gate is not zero-slip. A small amount of contacted-foot motion is allowed because perfect sticking is not realistic in MuJoCo or hardware. Defaults:

- `--min_foot_clearance 0.025`
- `--max_contact_slip_mean 0.25`
- `--max_contact_slip_p95 1.0`

These values are written to `metadata.json` under `acceptance`.

With the current `strict_walk` footspace profile and default strict slip gate, fixed `vx=0.5`, `vx=0.7`, and `vx=0.9` smoke runs pass. `vx=0.9` passes with bounded nonzero slip, but tracks below the requested speed, so treat it as a review/probe speed rather than the main dataset envelope until the gait is tuned further.

## Profiles

The recorder and viewer support explicit profiles:

- `--teacher_profile strict_walk`: default strict-slip tuned gait, `max_freq=2.8`, `step_length_max=0.18`.
- `--teacher_profile cruise_walk`: slightly longer stride for experiments, `max_freq=2.8`, `step_length_max=0.20`.
- `--teacher_profile turn_walk`: yaw-authority profile with differential left/right stride and swing-weighted yaw foot placement.
- `--command_profile default`: normal collection envelope.

The chosen profiles and resolved configs are written to `metadata.json`.
