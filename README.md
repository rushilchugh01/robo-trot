# robo-trot

MuJoCo Unitree A1 teacher-controller rollout pipeline for flat-ground behavior cloning demonstrations.

The recorder stores teacher desired joint targets (`q_teacher`) and normalized action labels (`action_label`), not just actual joint positions.

## Code Layout

The implementation lives under `robo_trot/` by domain:

- `robo_trot/robot/`: A1 constants, kinematics, and MuJoCo model inspection helpers.
- `robo_trot/sim/`: MuJoCo simulation wrappers.
- `robo_trot/teachers/`: teacher interfaces and teacher controllers.
- `robo_trot/data_pipeline/`: behavior-cloning data generation, dataset writing, sharding, manifests, inspection, and validation.
- `robo_trot/policies/`: policy implementations.
- `robo_trot/training/`: training entry points and utilities.

Scripts are grouped by purpose under `scripts/`; the root `scripts/` directory only contains package metadata and subdirectories:

- `scripts/assets/`: asset fetch/setup.
- `scripts/robot/`: MuJoCo model and A1 mapping inspection.
- `scripts/teacher/`: teacher playback and teacher sanity checks.
- `scripts/policy/`: policy harness playback, policy sanity checks, and action mapping audits.
- `scripts/data/`: dataset inspection and validation wrappers.

The `data/*.py` files are CLI wrappers around `robo_trot.data_pipeline.*`, kept so existing data-generation commands continue to work.

## Quick Commands

```bash
python scripts/assets/fetch_menagerie_a1.py --out_dir assets/mujoco_menagerie
python scripts/robot/inspect_a1_model.py --xml_path assets/mujoco_menagerie/unitree_a1/scene.xml
python scripts/teacher/play_teacher.py --mode home --seconds 30 --no_viewer
python scripts/teacher/play_teacher.py --teacher footspace --teacher_profile strict_walk --mode trot --seconds 20 --no_viewer
python scripts/teacher/sanity_check_teacher.py --stand_seconds 30 --walk_seconds 20 --walk_vx 0.5 --teacher_profile strict_walk
python scripts/policy/sanity_check_random_policy.py \
  --xml_path assets/mujoco_menagerie/unitree_a1/scene.xml \
  --dataset_metadata datasets/a1_teacher_flat_7m_v001_main/shards/shard_00_forward/metadata.json
python scripts/policy/audit_action_mapping.py \
  --xml_path assets/mujoco_menagerie/unitree_a1/scene.xml \
  --dataset_metadata datasets/a1_teacher_flat_7m_v001_main/shards/shard_00_forward/metadata.json
python scripts/policy/play_random_policy.py \
  --xml_path assets/mujoco_menagerie/unitree_a1/scene.xml \
  --dataset_metadata datasets/a1_teacher_flat_7m_v001_main/shards/shard_00_forward/metadata.json \
  --seconds 20 \
  --action_limit 0.25
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

Run `scripts/data/validate_dataset.py` on any dataset before using it for cloning. The validator checks array shapes, `action_label == clip((q_teacher - q_home) / action_scale)`, episode reset/done boundaries, that the `sin(phase), cos(phase)` fields inside `obs` match the saved `phase` column, and that the observation state slices match the saved raw state arrays for that same timestep. Older review artifacts generated before these alignment checks are useful only for visual comparison; regenerate them before using their `.npz` episodes as training data.

The default observation includes foot contacts and has `obs_dim=56`. For the fallback observation without contact features, add `--no-use_contacts`; the episode still stores `foot_contacts` and `foot_pos` as debug arrays, but the actor observation has `obs_dim=52`.

## Random Policy Harness

The random policy harness verifies the future policy loop without training. It reads actor observations, emits normalized 12D action labels, converts labels with `q_des = q_home + action_scale * action_label`, and applies those raw joint targets through the MuJoCo environment.

Always pass dataset metadata when testing against imitation-learning data. The harness checks joint names, actuator names, `q_home`, `action_scale`, observation dimension, and action dimension before stepping the robot.

Run `scripts/policy/audit_action_mapping.py` when changing the environment, model XML, action adapter, or policy harness. It probes each normalized action index and reports the expected q-target delta, the MuJoCo control-slot delta, the observed joint delta, and the dominant moving joint. All 12 rows must pass before trusting a policy rollout against a dataset.

Use `--no_viewer` for headless checks and omit it to watch the robot move live in the MuJoCo viewer.

For visual joint-order debugging, use `--policy_mode joint_probe` to sweep one selected joint, or `--policy_mode joint_scan` to sweep through all 12 joints one at a time. For stronger coherent whole-body motion, use `--policy_mode flail`; lower `--flail_amplitude` if the robot falls too quickly.

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

## Sharded 5M Generation

For large no-GIF generation, use the sharded launcher. It runs independent shard processes so workers never share one `metadata.json`.

```bash
python data/launch_5m_shards.py \
  --out_dir datasets/a1_teacher_flat_5m_v001 \
  --workers 8 \
  --total_steps 5000000
python data/build_5m_manifest.py datasets/a1_teacher_flat_5m_v001
python scripts/data/inspect_dataset.py datasets/a1_teacher_flat_5m_v001
```

The default 5M composition is:

- forward: 3.00M transitions
- turn: 1.00M transitions
- slow: 0.75M transitions
- fast_probe: 0.25M transitions

Media export is disabled in the shard launcher.

For a 7M run with the same composition ratios:

```bash
python data/launch_5m_shards.py \
  --out_dir datasets/a1_teacher_flat_7m_v001 \
  --workers 8 \
  --total_steps 7000000
```
