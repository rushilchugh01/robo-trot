#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DATASET_DIR="${DATASET_DIR:-datasets/a1_teacher_flat_7m_v001_main}"
OUT_DIR="${OUT_DIR:-runs/bc_compare_v001}"
XML_PATH="${XML_PATH:-assets/mujoco_menagerie/unitree_a1/scene.xml}"
DATASET_METADATA="${DATASET_METADATA:-${DATASET_DIR}/shards/shard_00_forward/metadata.json}"

TXL_WORKERS="${TXL_WORKERS:-8}"
TXL_CORES="${TXL_CORES:-0,1,2,3}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-64}"
TXL_MEMORY_SECONDS="${TXL_MEMORY_SECONDS:-20.0}"
LR="${LR:-3e-4}"
MAX_UPDATES="${MAX_UPDATES:-200000}"
METRICS_EVERY="${METRICS_EVERY:-100}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-20}"
SEED="${SEED:-0}"

LOG_DIR="${OUT_DIR}/logs"
PID_FILE="${LOG_DIR}/txl_train_only.pid"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
SUP_LOG="${LOG_DIR}/txl_train_only_supervisor_${STAMP}.log"
TRAIN_LOG="${LOG_DIR}/txl_train_only_${STAMP}.log"

usage() {
  cat <<'USAGE'
Usage: scripts/policy/launch_txl_training_only.sh [start|run|stop|status|tail]

Environment overrides:
  DATASET_DIR OUT_DIR XML_PATH DATASET_METADATA
  TXL_WORKERS TXL_CORES BATCH_SIZE SEQUENCE_LENGTH TXL_MEMORY_SECONDS
  LR MAX_UPDATES METRICS_EVERY CHECKPOINT_EVERY HEARTBEAT_SECONDS SEED

This is a recovery/operations launcher for TXL-only BC training. It does not
start MLP, evaluator, dashboard, or Ray. The dashboard can keep running as a
separate process against the same OUT_DIR.
USAGE
}

is_running() {
  [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

run_supervisor() {
  mkdir -p "$LOG_DIR"
  echo "$$" > "$PID_FILE"
  echo "[txl-train] stamp=${STAMP} supervisor=$$ workers=${TXL_WORKERS} cores=${TXL_CORES}" >> "$SUP_LOG"

  env PYTHONUNBUFFERED=1 LP_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python scripts/policy/parallel_train_bc.py \
      --dataset_dir "$DATASET_DIR" \
      --out_dir "$OUT_DIR" \
      --models txl \
      --mlp_workers 0 \
      --txl_workers "$TXL_WORKERS" \
      --eval_workers 0 \
      --mlp_cores 0,1 \
      --txl_cores "$TXL_CORES" \
      --batch_size "$BATCH_SIZE" \
      --sequence_length "$SEQUENCE_LENGTH" \
      --txl_memory_seconds "$TXL_MEMORY_SECONDS" \
      --lr "$LR" \
      --max_updates "$MAX_UPDATES" \
      --metrics_every "$METRICS_EVERY" \
      --checkpoint_every "$CHECKPOINT_EVERY" \
      --eval_every 1000 \
      --gif_every_eval 1 \
      --eval_gif_fps 30 \
      --eval_gif_seconds 10.0 \
      --eval_gif_width 320 \
      --eval_gif_height 180 \
      --eval_seconds 20 \
      --dataset_eval_split test \
      --dataset_eval_batch_size 4096 \
      --dataset_eval_max_batches 16 \
      --xml_path "$XML_PATH" \
      --dataset_metadata "$DATASET_METADATA" \
      --seed "$SEED" \
      --resume \
      >> "$TRAIN_LOG" 2>&1 &
  child="$!"
  echo "[txl-train] child=${child} train_log=${TRAIN_LOG}" >> "$SUP_LOG"

  cleanup() {
    echo "[txl-train] stopping child=${child}" >> "$SUP_LOG"
    kill -TERM "$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
  }
  trap cleanup TERM INT

  while kill -0 "$child" 2>/dev/null; do
    {
      echo "[txl-train] $(date -u +%Y-%m-%dT%H:%M:%SZ) alive child=${child}"
      tail -n 1 "${OUT_DIR}/txl/metrics.jsonl" 2>/dev/null || true
      ps -o pid,ppid,pgid,sid,stat,etime,psr,pcpu,pmem,rss,cmd -p "$child" 2>/dev/null || true
      pgrep -P "$child" -a 2>/dev/null || true
    } >> "$SUP_LOG"
    sleep "$HEARTBEAT_SECONDS"
  done

  wait "$child"
  status="$?"
  echo "[txl-train] child exited status=${status}" >> "$SUP_LOG"
  exit "$status"
}

start() {
  mkdir -p "$LOG_DIR"
  if is_running; then
    echo "TXL trainer already running: pid $(cat "$PID_FILE")"
    status
    return 0
  fi
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
  echo "Launching TXL-only trainer with ${TXL_WORKERS} workers on cores ${TXL_CORES}..."
  setsid bash "$0" run >/dev/null 2>&1 &
  echo "$!" > "$PID_FILE"
  sleep 2
  status
}

stop() {
  if is_running; then
    pid="$(cat "$PID_FILE")"
    echo "Stopping TXL trainer process group ${pid}..."
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in {1..20}; do
      if ! kill -0 "$pid" 2>/dev/null; then
        rm -f "$PID_FILE"
        echo "Stopped."
        return 0
      fi
      sleep 1
    done
    echo "TXL trainer did not stop after 20s; sending SIGKILL."
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    rm -f "$PID_FILE"
  else
    echo "TXL trainer is not running."
  fi
}

status() {
  echo "Run dir: $OUT_DIR"
  if is_running; then
    echo "Status: running, pid $(cat "$PID_FILE")"
  else
    echo "Status: not running"
  fi
  echo "Latest TXL metrics:"
  tail -n 5 "${OUT_DIR}/txl/metrics.jsonl" 2>/dev/null || echo "no metrics yet"
  echo "Latest supervisor log:"
  ls -1t "${LOG_DIR}"/txl_train_only_supervisor_*.log 2>/dev/null | head -1 || true
  echo "Latest train log:"
  ls -1t "${LOG_DIR}"/txl_train_only_*.log 2>/dev/null | head -1 || true
}

tail_logs() {
  log="$(ls -1t "${LOG_DIR}"/txl_train_only_supervisor_*.log 2>/dev/null | head -1 || true)"
  if [[ -z "$log" ]]; then
    echo "No supervisor log yet." >&2
    exit 1
  fi
  tail -n 120 -f "$log"
}

case "${1:-start}" in
  start) start ;;
  run) run_supervisor ;;
  stop) stop ;;
  status) status ;;
  tail) tail_logs ;;
  -h|--help|help) usage ;;
  *) usage; exit 2 ;;
esac
