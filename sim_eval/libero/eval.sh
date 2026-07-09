#!/bin/bash
# Full LIBERO eval driver: 4 suites × 10 tasks × N episodes × N seeds.
#
# Usage:
#   bash sim_eval/libero/eval.sh [overlap_steps] [seeds...]
#
# Optional positional:
#   overlap_steps  AsyncStreamingActionManager.overlap_steps (default 1). 0 = sync (each
#                  chunk boundary blocks on inference). >0 = async (next chunk
#                  inference launches `overlap_steps` action-steps before
#                  current chunk ends). policy.compile_model is auto-set to
#                  true iff overlap_steps>0 (cuda graph required for non-
#                  blocking dispatch).
#
# Optional positional:
#   seeds...       Whitespace-separated seed list. Default: 1000 2000 3000 4000 5000.
#
# Environment overrides:
#   POLICY_PATH    Default: z-lab/flashvla-pi05-libero (released checkpoint)
#   OUTPUT_ROOT    Default: /tmp/libero_eval
#   SUITES         Default: "libero_spatial libero_object libero_goal libero_10"
#   N_EPISODES     Default: 50 (per task)
#   N_ACTION_STEPS Default: 10
#   EPISODE_LENGTH Optional override for env.episode_length, applied to every
#                  suite. Unset → per-suite defaults from TASK_SUITE_MAX_STEPS
#                  (spatial/object 280, goal 300, libero_10 520).
#   GPUS           Space-separated GPU index list. Each (seed, suite) job runs on
#                  one GPU; with N GPUs, up to N jobs run concurrently. Default: "1".
#                  Examples: GPUS="0 1 2 3", GPUS="1".
#
# Behavior:
#   - One subprocess per (seed, suite). EGL contexts cap at ~30 per process,
#     so a single python invocation cannot construct >~3 suites worth of
#     LIBERO envs (40 envs total) at once. Per-suite isolation also lets a
#     failure abort only that one subprocess.
#   - Multi-GPU: one worker per GPU, all workers pull (seed, suite) jobs from a
#     shared queue (flock-atomic pop). This is pull-based so a slow suite
#     (libero_10) doesn't leave faster GPUs idle waiting on barrier.
#   - Resume support: skips any (seed, suite) whose eval_info.json already
#     exists. Re-runs with the same OUTPUT_ROOT pick up where they left off.
#   - Each job writes:
#       $OUTPUT_ROOT/ov{ov}/seed{s}/{suite}/eval_info.json
#       $OUTPUT_ROOT/ov{ov}/seed{s}/{suite}.log
#   - After all jobs complete, prints a summary table grouped by seed.
#
# Examples:
#   bash sim_eval/libero/eval.sh 2                             # 5 seeds, GPU 1
#   GPUS="0 1 2 3" bash sim_eval/libero/eval.sh 2              # 5 seeds, 4-way parallel
#   bash sim_eval/libero/eval.sh 0 1000                        # 1 seed
#   N_EPISODES=10 bash sim_eval/libero/eval.sh 2 1000 2000     # smoke
#   POLICY_PATH=./some_other_ckpt bash sim_eval/libero/eval.sh 1

set -u

# ---- args ----
# overlap_steps is optional (default 1 = async, 1-step delay). Pass 0 for sync.
OVERLAP_STEPS="${1:-1}"
[ "$#" -gt 0 ] && shift 1

if [ "$#" -gt 0 ]; then
  SEEDS=("$@")
else
  SEEDS=(1000 2000 3000 4000 5000)
fi

# ---- defaults ----
: "${POLICY_PATH:=z-lab/flashvla-pi05-libero}"
: "${OUTPUT_ROOT:=/tmp/libero_eval}"
: "${SUITES:=libero_spatial libero_object libero_goal libero_10}"
: "${N_EPISODES:=50}"
: "${N_ACTION_STEPS:=10}"
: "${GPUS:=1}"
# EPISODE_LENGTH: optional override for env.episode_length. Default unset
# → uses lerobot's TASK_SUITE_MAX_STEPS (280/280/300/520 for spatial/object/
# goal/10). Set e.g. EPISODE_LENGTH=500 to give every suite the same cap.
: "${EPISODE_LENGTH:=}"

# overlap=0 → sync; overlap>0 → must compile for cuda graph (non-blocking dispatch).
# COMPILE_MODEL can be preset in the env to force compile on/off (e.g. compile sync
# runs too — compile doesn't change success rate, only speed; matches §11 methodology).
if [ -z "${COMPILE_MODEL:-}" ]; then
  if [ "$OVERLAP_STEPS" -gt 0 ]; then
    COMPILE_MODEL=true
  else
    COMPILE_MODEL=false
  fi
fi

# fuse_qkv/fuse_gate_up: concatenate the QKV and gate/up projections into single
# matmuls — accuracy-neutral, speed only. Defaults to match COMPILE_MODEL (the
# compiled fast path used by the §11 reference is compile+fuse). Override with
# FUSE=true/false.
: "${FUSE:=$COMPILE_MODEL}"


RUN_ROOT="$OUTPUT_ROOT/ov${OVERLAP_STEPS}"
mkdir -p "$RUN_ROOT"

# Repo root (the script's parent dir's parent)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Eval implementation lives in the flashvla package (flashvla/eval.py).
if ! python -c "import flashvla" 2>/dev/null; then
  echo "ERROR: cannot import flashvla — did you 'pip install -e .' ?" >&2
  exit 1
fi
cd "$REPO_DIR"

GPU_ARR=($GPUS)
N_GPUS=${#GPU_ARR[@]}

echo "============================================================"
echo " libero eval"
echo "   overlap_steps:  $OVERLAP_STEPS  $([ "$OVERLAP_STEPS" -gt 0 ] && echo "(async)" || echo "(sync)")"
echo "   compile_model:  $COMPILE_MODEL"
echo "   fuse_qkv/gate:  $FUSE"
echo "   n_action_steps: $N_ACTION_STEPS"
echo "   n_episodes:     $N_EPISODES per task"
echo "   episode_length: ${EPISODE_LENGTH:-default per-suite}"
echo "   suites:         $SUITES"
echo "   seeds:          ${SEEDS[*]}"
echo "   policy:         $POLICY_PATH"
echo "   output:         $RUN_ROOT"
echo "   GPUs:           ${GPU_ARR[*]}  (parallel = $N_GPUS)"
echo "   started:        $(date)"
echo "============================================================"

# ---- build job queue ----
# Pre-construct the (seed, suite) list, then workers pop atomically from a
# shared file via flock. This is pull-based, so a slow job (e.g. libero_10
# can be 50% longer than libero_goal) doesn't leave faster GPUs idle.
QUEUE_FILE=$(mktemp -t libero_eval_queue.XXXXXX)
QUEUE_LOCK="${QUEUE_FILE}.lock"
touch "$QUEUE_LOCK"
trap 'rm -f "$QUEUE_FILE" "$QUEUE_LOCK"' EXIT

for SEED in "${SEEDS[@]}"; do
  for SUITE in $SUITES; do
    OUT="$RUN_ROOT/seed${SEED}/${SUITE}"
    if [ -f "$OUT/eval_info.json" ]; then
      echo "[skip] seed=$SEED suite=$SUITE — eval_info.json already exists"
      continue
    fi
    echo "$SEED $SUITE" >> "$QUEUE_FILE"
  done
done

TOTAL_JOBS=$(wc -l < "$QUEUE_FILE" | tr -d ' ')
if [ "$TOTAL_JOBS" -eq 0 ]; then
  echo ""
  echo "All jobs already complete. Skipping to SUMMARY."
else
  echo ""
  echo "queued $TOTAL_JOBS jobs across $N_GPUS GPU worker(s)"
fi

# ---- worker: pop from queue, run jobs sequentially on assigned GPU ----
worker() {
  local gpu=$1
  while :; do
    # Atomic pop: take first line under exclusive flock, leave rest in the file.
    local line=""
    {
      flock -x 9
      line=$(head -n 1 "$QUEUE_FILE")
      if [ -n "$line" ]; then
        # sed -i creates a temp file but is atomic enough under flock
        tail -n +2 "$QUEUE_FILE" > "$QUEUE_FILE.tmp" && mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"
      fi
    } 9>"$QUEUE_LOCK"
    [ -z "$line" ] && break

    local seed suite
    read -r seed suite <<<"$line"
    local OUT="$RUN_ROOT/seed${seed}/${suite}"
    local LOG="$RUN_ROOT/seed${seed}/${suite}.log"
    mkdir -p "$(dirname "$OUT")"
    rm -rf "$OUT"

    echo ""
    echo "------------------------------------------------------------"
    echo " [gpu=$gpu] seed=$seed suite=$suite  START: $(date '+%H:%M:%S')"
    echo "------------------------------------------------------------"

    local episode_length_arg=()
    if [ -n "$EPISODE_LENGTH" ]; then
      episode_length_arg=(--env.episode_length="$EPISODE_LENGTH")
    fi

    CUDA_VISIBLE_DEVICES=$gpu python sim_eval/libero/eval.py \
      --policy.path="$POLICY_PATH" \
      --policy.compile_model="$COMPILE_MODEL" \
      --policy.fuse_qkv="$FUSE" \
      --policy.fuse_gate_up="$FUSE" \
      --policy.n_action_steps="$N_ACTION_STEPS" \
      --policy.device=cuda \
      --env.type=libero \
      --env.task="$suite" \
      "${episode_length_arg[@]}" \
      --env.max_parallel_tasks=1 \
      --eval.batch_size=1 \
      --eval.n_episodes="$N_EPISODES" \
      --inference_overlap_steps="$OVERLAP_STEPS" \
      --seed="$seed" \
      --output_dir="$OUT" \
      > "$LOG" 2>&1
    local rc=$?

    if [ -f "$OUT/eval_info.json" ]; then
      python3 - "$OUT/eval_info.json" "$gpu" "$seed" "$suite" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))["overall"]
print(f"  [gpu={sys.argv[2]}] seed={sys.argv[3]} suite={sys.argv[4]}  "
      f"DONE: {d['pc_success']:5.1f}%  ({d['n_episodes']} ep, {d['eval_s']/60:.1f} min)")
PY
    else
      echo "  [gpu=$gpu] seed=$seed suite=$suite  FAILED (rc=$rc) — see $LOG"
    fi
  done
}

# Spawn one worker per GPU
PIDS=()
for gpu in "${GPU_ARR[@]}"; do
  worker "$gpu" &
  PIDS+=($!)
done

# Wait for all workers
for pid in "${PIDS[@]}"; do
  wait "$pid"
done

echo ""
echo "============================================================"
echo " SUMMARY  ($RUN_ROOT)"
echo "============================================================"
python3 - "$RUN_ROOT" <<'PY'
import json, glob, os, sys
from collections import defaultdict
root = sys.argv[1]

# rows: (seed, suite) -> dict with pc, n, eval_s
table = {}
for f in sorted(glob.glob(f"{root}/seed*/*/eval_info.json")):
    seed = int(os.path.basename(os.path.dirname(os.path.dirname(f)))[4:])
    suite = os.path.basename(os.path.dirname(f))
    o = json.load(open(f))["overall"]
    table[(seed, suite)] = (o["pc_success"], o["n_episodes"], o["eval_s"])

if not table:
    print("(no completed jobs)")
    sys.exit(0)

seeds = sorted({s for s, _ in table})
suites = sorted({su for _, su in table})

# per-seed table
print(f"\n{'seed':>6s}  ", end="")
for su in suites:
    print(f"{su:>16s}  ", end="")
print(f"{'AVG':>8s}")

per_suite_succ = defaultdict(list)
for s in seeds:
    print(f"{s:>6d}  ", end="")
    succ_total, n_total = 0.0, 0
    for su in suites:
        if (s, su) in table:
            pc, n, _ = table[(s, su)]
            print(f"{pc:>14.1f}%   ", end="")
            succ_total += pc * n / 100
            n_total += n
            per_suite_succ[su].append(pc)
        else:
            print(f"{'-':>14s}   ", end="")
    print(f"{100 * succ_total / max(n_total, 1):>7.1f}%")

# per-suite mean ± std across seeds
import statistics
print(f"\n{'mean':>6s}  ", end="")
overall_pcs = []
for su in suites:
    pcs = per_suite_succ[su]
    if len(pcs) >= 2:
        mean = statistics.mean(pcs); std = statistics.stdev(pcs)
        print(f"{mean:>9.1f}±{std:>3.1f}%  ", end="")
    elif len(pcs) == 1:
        print(f"{pcs[0]:>14.1f}%   ", end="")
    else:
        print(f"{'-':>14s}   ", end="")
    overall_pcs.extend(pcs)
if len(overall_pcs) >= 2:
    print(f"{statistics.mean(overall_pcs):>4.1f}±{statistics.stdev(overall_pcs):.1f}%")
elif overall_pcs:
    print(f"{overall_pcs[0]:>7.1f}%")
else:
    print()

# total wall-clock
total_s = sum(v[2] for v in table.values())
print(f"\n  total wall-clock across all completed jobs: {total_s/3600:.1f} h "
      f"({len(table)}/{len(seeds)*len(suites)} jobs)")
PY

echo ""
echo "done: $(date)"
