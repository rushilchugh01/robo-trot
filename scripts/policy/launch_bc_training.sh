#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DATASET_DIR="${DATASET_DIR:-datasets/a1_teacher_flat_7m_v001_main}"
OUT_DIR="${OUT_DIR:-runs/bc_compare_v001}"
XML_PATH="${XML_PATH:-assets/mujoco_menagerie/unitree_a1/scene.xml}"
DATASET_METADATA="${DATASET_METADATA:-${DATASET_DIR}/shards/shard_00_forward/metadata.json}"

MODELS="${MODELS:-mlp,txl}"
MLP_WORKERS="${MLP_WORKERS:-4}"
TXL_WORKERS="${TXL_WORKERS:-4}"
EVAL_WORKERS="${EVAL_WORKERS:-1}"
MLP_CORES="${MLP_CORES:-0,1}"
TXL_CORES="${TXL_CORES:-2,3}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-64}"
TXL_MEMORY_SECONDS="${TXL_MEMORY_SECONDS:-20.0}"
LR="${LR:-3e-4}"
MAX_UPDATES="${MAX_UPDATES:-200000}"
METRICS_EVERY="${METRICS_EVERY:-100}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
GIF_EVERY_EVAL="${GIF_EVERY_EVAL:-1}"
EVAL_GIF_FPS="${EVAL_GIF_FPS:-30}"
EVAL_GIF_SECONDS="${EVAL_GIF_SECONDS:-10.0}"
EVAL_GIF_WIDTH="${EVAL_GIF_WIDTH:-320}"
EVAL_GIF_HEIGHT="${EVAL_GIF_HEIGHT:-180}"
EVAL_SECONDS="${EVAL_SECONDS:-20}"
DATASET_EVAL_SPLIT="${DATASET_EVAL_SPLIT:-test}"
DATASET_EVAL_BATCH_SIZE="${DATASET_EVAL_BATCH_SIZE:-4096}"
DATASET_EVAL_MAX_BATCHES="${DATASET_EVAL_MAX_BATCHES:-16}"
DASHBOARD_HOST="${DASHBOARD_HOST:-0.0.0.0}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8002}"
SEED="${SEED:-0}"
USE_RAY="${USE_RAY:-1}"
RAY_ADDRESS="${RAY_ADDRESS:-auto}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-4}"
RAY_START_LOCAL="${RAY_START_LOCAL:-1}"
RAY_PORT="${RAY_PORT:-0}"

RUN_DIR="$OUT_DIR"
LOG_DIR="${RUN_DIR}/logs"
PID_FILE="${RUN_DIR}/parallel_train_bc.pid"
LOG_FILE="${LOG_DIR}/parallel_train_bc.log"
RAY_PID_FILE="${RUN_DIR}/ray_head.pid"
RAY_LOG_FILE="${LOG_DIR}/ray_head.log"
DASHBOARD_PID_FILE="${RUN_DIR}/dashboard.pid"
DASHBOARD_LOG_FILE="${LOG_DIR}/dashboard.log"

usage() {
  cat <<'USAGE'
Usage: scripts/policy/launch_bc_training.sh [start|stop|restart|status|tail]

Environment overrides:
  DATASET_DIR OUT_DIR XML_PATH DATASET_METADATA
  MODELS
  MLP_WORKERS TXL_WORKERS EVAL_WORKERS MLP_CORES TXL_CORES
  BATCH_SIZE SEQUENCE_LENGTH TXL_MEMORY_SECONDS LR MAX_UPDATES
  METRICS_EVERY CHECKPOINT_EVERY EVAL_EVERY GIF_EVERY_EVAL
  EVAL_GIF_FPS EVAL_GIF_SECONDS EVAL_GIF_WIDTH EVAL_GIF_HEIGHT EVAL_SECONDS
  DATASET_EVAL_SPLIT DATASET_EVAL_BATCH_SIZE DATASET_EVAL_MAX_BATCHES
  DASHBOARD_HOST DASHBOARD_PORT SEED
  USE_RAY RAY_ADDRESS RAY_NUM_CPUS RAY_START_LOCAL RAY_PORT

The start command launches the single all-in-one orchestrator:
  - MLP trainer group
  - TXL trainer group
  - checkpoint evaluator process
  - dashboard server

By default it connects to Ray with --ray_address auto. If no Ray cluster is
available and RAY_START_LOCAL=1, it starts a local four-CPU Ray head first.
RAY_PORT=0 lets Ray choose a free GCS port, avoiding local Redis on 6379.

It passes --resume, so rerunning start against an existing run directory resumes
from latest complete checkpoints when present.
USAGE
}

is_running() {
  [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

stop_standalone_dashboard() {
  if [[ -f "$DASHBOARD_PID_FILE" ]] && kill -0 "$(cat "$DASHBOARD_PID_FILE")" 2>/dev/null; then
    dashboard_pid="$(cat "$DASHBOARD_PID_FILE")"
    echo "Stopping standalone dashboard process $dashboard_pid..."
    kill -TERM "$dashboard_pid" 2>/dev/null || true
    for _ in {1..10}; do
      if ! kill -0 "$dashboard_pid" 2>/dev/null; then
        rm -f "$DASHBOARD_PID_FILE"
        return 0
      fi
      sleep 1
    done
    if kill -0 "$dashboard_pid" 2>/dev/null; then
      echo "Standalone dashboard did not stop after 10s; sending SIGKILL."
      kill -KILL "$dashboard_pid" 2>/dev/null || true
    fi
  fi
  rm -f "$DASHBOARD_PID_FILE"
}

ensure_ray() {
  if [[ "$USE_RAY" != "1" ]]; then
    return 0
  fi
  if ! python - <<'PY' >/dev/null 2>&1
import ray
PY
  then
    echo "Ray is not installed in this Python environment. Install ray or set USE_RAY=0." >&2
    exit 1
  fi
  if timeout 15 ray status >/dev/null 2>&1; then
    return 0
  fi
  if [[ "$RAY_START_LOCAL" != "1" ]]; then
    echo "Ray cluster is not reachable and RAY_START_LOCAL=0." >&2
    exit 1
  fi
  if [[ -f "$RAY_PID_FILE" ]] && kill -0 "$(cat "$RAY_PID_FILE")" 2>/dev/null; then
    echo "Ray head pid $(cat "$RAY_PID_FILE") is running, waiting for status..."
  else
    echo "Starting local Ray head with ${RAY_NUM_CPUS} CPUs on port ${RAY_PORT}..."
    setsid bash -c 'exec "$@"' launch-ray-head \
      ray start --head \
        --port="$RAY_PORT" \
        --num-cpus="$RAY_NUM_CPUS" \
        --include-dashboard=false \
        --disable-usage-stats \
        --block \
        >"$RAY_LOG_FILE" 2>&1 &
    echo "$!" >"$RAY_PID_FILE"
  fi
  for _ in {1..30}; do
    if timeout 10 ray status >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "Ray head did not become reachable. Recent Ray log:" >&2
  tail -80 "$RAY_LOG_FILE" >&2 || true
  exit 1
}

start() {
  mkdir -p "$LOG_DIR"
  if is_running; then
    echo "Already running: pid $(cat "$PID_FILE")"
    status
    return 0
  fi
  stop_standalone_dashboard
  if [[ ! -d "$DATASET_DIR" ]]; then
    echo "Missing dataset directory: $DATASET_DIR" >&2
    exit 1
  fi
  if [[ ! -f "$XML_PATH" ]]; then
    echo "Missing MuJoCo XML: $XML_PATH" >&2
    exit 1
  fi
  if [[ ! -f "$DATASET_METADATA" ]]; then
    echo "Missing dataset metadata: $DATASET_METADATA" >&2
    exit 1
  fi
  ensure_ray
  ray_flags=()
  if [[ "$USE_RAY" == "1" ]]; then
    ray_flags+=(--ray --ray_address "$RAY_ADDRESS")
  fi

  echo "Launching BC trainer/evaluator/dashboard..."
  echo "Run dir: $RUN_DIR"
  echo "Dashboard: http://${DASHBOARD_HOST}:${DASHBOARD_PORT}"
  if [[ "$USE_RAY" == "1" ]]; then
    echo "Ray: enabled, address ${RAY_ADDRESS}, trainer CPU reservation ${RAY_NUM_CPUS}"
  fi
  setsid bash -c 'exec "$@"' launch-bc-training \
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python scripts/policy/parallel_train_bc.py \
      --dataset_dir "$DATASET_DIR" \
      --out_dir "$OUT_DIR" \
      --models "$MODELS" \
      --mlp_workers "$MLP_WORKERS" \
      --txl_workers "$TXL_WORKERS" \
      --eval_workers "$EVAL_WORKERS" \
      --mlp_cores "$MLP_CORES" \
      --txl_cores "$TXL_CORES" \
      --batch_size "$BATCH_SIZE" \
      --sequence_length "$SEQUENCE_LENGTH" \
      --txl_memory_seconds "$TXL_MEMORY_SECONDS" \
      --lr "$LR" \
      --max_updates "$MAX_UPDATES" \
      --metrics_every "$METRICS_EVERY" \
      --checkpoint_every "$CHECKPOINT_EVERY" \
      --eval_every "$EVAL_EVERY" \
      --gif_every_eval "$GIF_EVERY_EVAL" \
      --eval_gif_fps "$EVAL_GIF_FPS" \
      --eval_gif_seconds "$EVAL_GIF_SECONDS" \
      --eval_gif_width "$EVAL_GIF_WIDTH" \
      --eval_gif_height "$EVAL_GIF_HEIGHT" \
      --eval_seconds "$EVAL_SECONDS" \
      --dataset_eval_split "$DATASET_EVAL_SPLIT" \
      --dataset_eval_batch_size "$DATASET_EVAL_BATCH_SIZE" \
      --dataset_eval_max_batches "$DATASET_EVAL_MAX_BATCHES" \
      --dashboard \
      --dashboard_host "$DASHBOARD_HOST" \
      --dashboard_port "$DASHBOARD_PORT" \
      --xml_path "$XML_PATH" \
      --dataset_metadata "$DATASET_METADATA" \
      --seed "$SEED" \
      --resume \
      "${ray_flags[@]}" \
      >>"$LOG_FILE" 2>&1 &
  echo "$!" >"$PID_FILE"
  sleep 1
  status
}

stop() {
  stop_standalone_dashboard
  if is_running; then
    pid="$(cat "$PID_FILE")"
    echo "Stopping trainer process group $pid..."
    kill -TERM -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    for _ in {1..20}; do
      if ! kill -0 "$pid" 2>/dev/null; then
        rm -f "$PID_FILE"
        break
      fi
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      echo "Trainer did not stop after 20s; sending SIGKILL to process group."
      kill -KILL -- "-$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
      rm -f "$PID_FILE"
    fi
  else
    echo "Trainer is not running."
  fi
  if [[ -f "$RAY_PID_FILE" ]] && kill -0 "$(cat "$RAY_PID_FILE")" 2>/dev/null; then
    ray_pid="$(cat "$RAY_PID_FILE")"
    echo "Stopping managed Ray head process group $ray_pid..."
    kill -TERM -- "-$ray_pid" 2>/dev/null || kill "$ray_pid" 2>/dev/null || true
    for _ in {1..20}; do
      if ! kill -0 "$ray_pid" 2>/dev/null; then
        rm -f "$RAY_PID_FILE"
        break
      fi
      sleep 1
    done
    if kill -0 "$ray_pid" 2>/dev/null; then
      echo "Ray head did not stop after 20s; sending SIGKILL."
      kill -KILL -- "-$ray_pid" 2>/dev/null || kill -9 "$ray_pid" 2>/dev/null || true
      rm -f "$RAY_PID_FILE"
    fi
  fi
  echo "Stopped."
}

status() {
  echo "Run dir: $RUN_DIR"
  if is_running; then
    echo "Status: running, pid $(cat "$PID_FILE")"
  else
    echo "Status: not running"
  fi
  for metrics in "$RUN_DIR"/mlp/metrics.jsonl "$RUN_DIR"/txl/metrics.jsonl "$RUN_DIR"/eval/metrics.jsonl; do
    echo "== ${metrics} =="
    tail -n 3 "$metrics" 2>/dev/null || echo "no metrics yet"
  done
  echo "== checkpoints =="
  find "$RUN_DIR" -maxdepth 4 \( -name '_SUCCESS' -o -name 'latest' -o -name 'best_val_loss' -o -name 'best_eval_reward' \) 2>/dev/null | sort | tail -40 || true
  echo "Dashboard URL: http://${DASHBOARD_HOST}:${DASHBOARD_PORT}"
  if [[ -f "$DASHBOARD_PID_FILE" ]]; then
    dashboard_pid="$(cat "$DASHBOARD_PID_FILE")"
    if kill -0 "$dashboard_pid" 2>/dev/null; then
      echo "Standalone dashboard pid: $dashboard_pid"
      echo "Dashboard log file: $DASHBOARD_LOG_FILE"
    else
      echo "Standalone dashboard pid file is stale: $dashboard_pid"
    fi
  fi
  echo "Log file: $LOG_FILE"
  if [[ -f "$RAY_PID_FILE" ]]; then
    echo "Ray head pid: $(cat "$RAY_PID_FILE")"
    echo "Ray log file: $RAY_LOG_FILE"
  fi
}

tail_logs() {
  mkdir -p "$LOG_DIR"
  touch "$LOG_FILE"
  tail -n 120 -f "$LOG_FILE"
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  tail) tail_logs ;;
  -h|--help|help) usage ;;
  *) usage; exit 2 ;;
esac
