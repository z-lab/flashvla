#!/usr/bin/env bash
# RoboTwin 2.0 -> LeRobot v3 dataset pipeline (one-click).
#
# Downloads the official RoboTwin 2.0 raw dataset, converts it to the LeRobot v3
# layout FlashVLA trains on, and computes the exact normalization statistics.
# Runs locally in the activated `flashvla` conda env — no GPU, no RoboTwin sim
# env, no SLURM. (~273 GB raw download; the converted tree is much smaller.)
#
# Usage:
#   conda activate flashvla
#   bash run_pipeline.sh [DATA_DIR] [PARALLELISM]
#
#     DATA_DIR     where raw + converted data live   (default: ./robotwin_data)
#     PARALLELISM  concurrent task conversions        (default: half the CPU cores)
#
# Optional env overrides:
#   SETTINGS="aloha-agilex_clean_50"   # space-separated; default = clean_50 + randomized_500
#   SKIP_DOWNLOAD=1                    # raw data already extracted under $DATA_DIR/raw/dataset
#
# Idempotent: re-running skips already-converted task/settings (.CONVERT_DONE marker),
# so it is safe to resume after an interruption.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${1:-${DATA_DIR:-./robotwin_data}}"
PAR="${2:-${PAR:-$(( ($(nproc) + 1) / 2 ))}}"
# shellcheck disable=SC2206
SETTINGS=(${SETTINGS:-aloha-agilex_clean_50 aloha-agilex_randomized_500})

RAW="$DATA_DIR/raw"                       # hf download target -> $RAW/dataset/<task>/<setting>.zip
DATASET="$RAW/dataset"                    # extracted raw tree   -> <task>/<setting>/data/*.hdf5
OUT="$DATA_DIR/RoboTwin-LeRobot-v3.0"     # converted LeRobot v3 tree
SCRIPTS="$HERE/scripts"
LOGDIR="$DATA_DIR/logs"
mkdir -p "$RAW" "$OUT" "$OUT/_pooled_stats" "$LOGDIR"

echo "=== RoboTwin pipeline | DATA_DIR=$DATA_DIR | PAR=$PAR | settings=${SETTINGS[*]} ==="

# --- 1) Download raw RoboTwin 2.0 (aloha-agilex clean_50 + randomized_500 zips) ---
if [ "${SKIP_DOWNLOAD:-0}" != "1" ]; then
  echo "=== [1/5] Download TianxingChen/RoboTwin2.0 -> $RAW ==="
  if command -v hf >/dev/null 2>&1; then HF=(hf download); else HF=(huggingface-cli download); fi
  "${HF[@]}" TianxingChen/RoboTwin2.0 --repo-type dataset \
    --include "dataset/*/aloha-agilex_*.zip" \
    --local-dir "$RAW" --max-workers 4
else
  echo "=== [1/5] SKIP_DOWNLOAD=1 (using existing $DATASET) ==="
fi

# --- 2) Unzip (idempotent: skip if the extracted dir is already present) ---
echo "=== [2/5] Unzip ==="
find "$DATASET" -name "*.zip" | while read -r z; do
  d="$(dirname "$z")"; base="$(basename "$z" .zip)"
  [ -d "$d/$base" ] || printf '%s\0' "$z"
done | xargs -0 -r -P 8 -I{} sh -c 'd="$(dirname "$1")"; unzip -q -o "$1" -d "$d"' _ {}
# Tip: once conversion succeeds you can reclaim space with:
#   find "$DATASET" -name "*.zip" -delete

# --- 3) Convert each setting to LeRobot v3 (parallel, idempotent) ---
export OUT SCRIPTS DATASET LOGDIR
for SETTING in "${SETTINGS[@]}"; do
  echo "=== [3/5] Convert $SETTING (PAR=$PAR) ==="
  export SETTING
  ls "$DATASET" | xargs -P "$PAR" -I {} bash -c '
    task="$1"
    marker="$OUT/$task/$SETTING/.CONVERT_DONE"
    [ -f "$marker" ] && { echo "SKIP $task"; exit 0; }
    if python "$SCRIPTS/convert_robotwin_to_lerobot.py" \
         --task "$task" --setting "$SETTING" \
         --raw-root "$DATASET" --out-root "$OUT" --image-writer-threads 4 \
         > "$LOGDIR/convert_${task}_${SETTING}.log" 2>&1; then
      touch "$marker"; echo "OK   $task"
    else
      echo "FAIL $task (see $LOGDIR/convert_${task}_${SETTING}.log)"
    fi
  ' _ {}
done

# --- 4) Per-task exact quantile stats (q01/q99; lerobot only stores min/max/mean/std) ---
echo "=== [4/5] Augment per-task quantile stats ==="
python "$SCRIPTS/augment_robotwin_quantile_stats.py" --root "$OUT" --overwrite

# --- 5) Exact pooled stats over the union (for merged multi-task training) ---
echo "=== [5/5] Compute pooled stats ==="
python "$SCRIPTS/compute_robotwin_pooled_stats.py" \
  --root "$OUT" --subdirs "${SETTINGS[@]}" \
  --output "$OUT/_pooled_stats/pooled_stats.json"

echo "=== DONE: $(find "$OUT" -name .CONVERT_DONE | wc -l) task/settings converted -> $OUT ==="
echo "Train on it by setting in your config:"
echo "    robotwin_multitask.root=$OUT"
echo "    robotwin_multitask.stats_path=$OUT/_pooled_stats/pooled_stats.json"
