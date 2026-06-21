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

Install the training extra before running BC trainers:

```bash
pip install -e ".[train]"
pip install -e ".[ray]"  # only needed for --ray / JOVA mode
```

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
python scripts/policy/parallel_train_bc.py \
  --dataset_dir datasets/a1_teacher_flat_7m_v001_main \
  --out_dir runs/bc_compare_v001 \
  --mlp_workers 4 \
  --txl_workers 4 \
  --eval_workers 1 \
  --mlp_cores 0,1 \
  --txl_cores 2,3 \
  --batch_size 4096 \
  --sequence_length 64 \
  --txl_memory_seconds 20.0 \
  --lr 3e-4 \
  --max_updates 200000 \
  --metrics_every 100 \
  --checkpoint_every 1000 \
  --eval_every 1000 \
  --gif_every_eval 1 \
  --dashboard \
  --dashboard_host 0.0.0.0 \
  --dashboard_port 8002
python scripts/policy/serve_training_dashboard.py \
  --run_dir runs/bc_compare_v001 \
  --host 0.0.0.0 \
  --port 8002
scripts/policy/launch_bc_training.sh start
scripts/policy/launch_bc_training.sh status
scripts/policy/launch_bc_training.sh tail
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

## Quality Gates

The normal test suite includes docstring linting:

```bash
pytest tests/test_docstring_coverage.py
```

Every production class, function, and method under `robo_trot/`, `data/`, and committed `scripts/` must have at least two non-empty docstring lines. Math-heavy helpers detected by trig, quaternion, yaw, or IK usage must have longer docstrings with equation, unit, or frame detail.

Run `scripts/data/validate_dataset.py` on any dataset before using it for cloning. The validator checks array shapes, `action_label == clip((q_teacher - q_home) / action_scale)`, episode reset/done boundaries, that the `sin(phase), cos(phase)` fields inside `obs` match the saved `phase` column, and that the observation state slices match the saved raw state arrays for that same timestep. Older review artifacts generated before these alignment checks are useful only for visual comparison; regenerate them before using their `.npz` episodes as training data.

The default observation includes foot contacts and has `obs_dim=56`. For the fallback observation without contact features, add `--no-use_contacts`; the episode still stores `foot_contacts` and `foot_pos` as debug arrays, but the actor observation has `obs_dim=52`.

## Random Policy Harness

The random policy harness verifies the future policy loop without training. It reads actor observations, emits normalized 12D action labels, converts labels with `q_des = q_home + action_scale * action_label`, and applies those raw joint targets through the MuJoCo environment.

Always pass dataset metadata when testing against imitation-learning data. The harness checks joint names, actuator names, `q_home`, `action_scale`, observation dimension, and action dimension before stepping the robot.

Run `scripts/policy/audit_action_mapping.py` when changing the environment, model XML, action adapter, or policy harness. It probes each normalized action index and reports the expected q-target delta, the MuJoCo control-slot delta, the observed joint delta, and the dominant moving joint. All 12 rows must pass before trusting a policy rollout against a dataset.

Use `--no_viewer` for headless checks and omit it to watch the robot move live in the MuJoCo viewer.

For visual joint-order debugging, use `--policy_mode joint_probe` to sweep one selected joint, or `--policy_mode joint_scan` to sweep through all 12 joints one at a time. For stronger coherent whole-body motion, use `--policy_mode flail`; lower `--flail_amplitude` if the robot falls too quickly.

## Behavior Cloning Comparison

`scripts/policy/parallel_train_bc.py` trains MLP and TXL policies concurrently from the teacher dataset. The default local layout starts four MLP workers pinned to cores `0,1`, four TXL workers pinned to cores `2,3`, one evaluator process, and an optional dashboard at `0.0.0.0:8002`.

For normal local operation, use `scripts/policy/launch_bc_training.sh start`. It starts or connects to Ray, launches the all-in-one orchestrator in the background with the default MLP group, TXL group, evaluator process, and dashboard, writes logs to `runs/bc_compare_v001/logs/`, and passes `--resume` so rerunning it continues from latest complete checkpoints when present. The launcher owns runtime control; the Python trainer owns Ray scheduling with two `num_cpus=2` training tasks and a separate MuJoCo evaluator task.

The launcher defaults to `USE_RAY=1`. If `ray status` cannot reach a cluster and `RAY_START_LOCAL=1`, it starts a local four-CPU Ray head with `ray start --head --block`, records `ray_head.pid`, and keeps that process alive for the run. Set `RAY_ADDRESS=auto` for the normal JOVA/Ray resolver path, or set `USE_RAY=0` to use the multiprocessing fallback.

The MLP uses transition batches: `obs_t -> action_label_t`. The TXL trains on streamed contiguous episode chunks, carries detached Transformer-XL memory from chunk to chunk, and resets memory only at true episode boundaries. The dataset `reset_flag` remains available as trial metadata, but it does not clear TXL memory by itself. Sequences never cross episode file boundaries, padded tail tokens are masked out of loss and memory, and per-stream cache validity prevents memory leakage when one stream is replaced. By default `--txl_memory_seconds 20.0` with `--policy_dt 0.02` creates a 1000-token memory, so rollout/eval memory spans 20 seconds at 50 Hz. `--sequence_length` still controls the supervised chunk length used for each BC update.

`--batch_size` is interpreted as the training-group sample/token budget and is sharded across worker processes. For example, `--batch_size 4096 --txl_workers 8 --sequence_length 64` gives each TXL worker 512 supervised tokens, or eight 64-step streams, per asynchronous update. This keeps all workers active while leaving memory headroom for the MuJoCo evaluator.

Checkpoints are written atomically under `runs/<name>/{mlp,txl}/checkpoints/step_000001000/`, with `latest`, `best_val_loss`, and `best_eval_reward` aliases. Evaluation watches complete checkpoints, runs fixed MuJoCo commands, writes reward metrics under `eval/metrics.jsonl`, saves browser-compatible H.264 MP4 previews under `eval/media/step_*/`, records the first 10 seconds of each command by default, and logs held-out dataset action-label diagnostics with `dataset_eval_action_mse`, `dataset_eval_action_l1`, and `dataset_eval_split`. Legacy GIF rows under `eval/gifs/step_*/` are still read for existing runs.

BC optimizes supervised action MSE only. Reward is computed during checkpoint evaluation so model selection compares validation loss, MuJoCo reward, survival, yaw response, velocity tracking, smoothness, foot slip, and rollout media quality instead of trusting loss alone.

The dashboard is served without browser auto-refresh; reload from the client when needed. The backend exposes `/api/summary`, `/api/train-metrics`, `/api/eval-metrics`, and `/api/gifs`, while the browser renders separate MLP/TXL panels, Plotly zoom/pan loss curves, MuJoCo reward curves, held-out action MSE curves, latest eval media, and model-separated historical media galleries for every evaluated command. MP4/WebM clips render as autoplaying muted looping `<video>` elements; legacy GIF/WebP rows render as image fallbacks.

Standalone checkpoint evaluation:

```bash
python scripts/policy/evaluate_checkpoint.py \
  --checkpoint runs/bc_compare_v001/mlp/checkpoints/latest \
  --model mlp \
  --xml_path assets/mujoco_menagerie/unitree_a1/scene.xml \
  --dataset_metadata datasets/a1_teacher_flat_7m_v001_main/shards/shard_00_forward/metadata.json \
  --dataset_dir datasets/a1_teacher_flat_7m_v001_main \
  --dataset_eval_split test \
  --seconds 20 \
  --command 0.5 0.0 0.0 \
  --save_media runs/bc_compare_v001/eval/manual_mlp_vx05.mp4

# Legacy .gif names and --save_gif are accepted, but output is redirected to .mp4.

python scripts/policy/evaluate_checkpoint.py \
  --checkpoint runs/bc_compare_v001/txl/checkpoints/latest \
  --model txl \
  --viewer \
  --seconds 20
```

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
