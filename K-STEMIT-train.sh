#!/bin/bash

# Sample tmux launcher for one configurable K-STEMIT training run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-myenv}"
CODE_DIR="${CODE_DIR:-$SCRIPT_DIR/Internal_Layer_Shallow_To_Deep/K-STEMIT-Code-Clean}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/Internal_Layer_Shallow_To_Deep/K-STEMIT-Train-Results}"
RUN_PREFIX="${RUN_PREFIX:-$(date +%Y%m%d_%H%M)}"
SESSION_NAME="${SESSION_NAME:-kstemit_train_$(date +%s)}"
AUTO_ATTACH="${AUTO_ATTACH:-True}"

MODEL="${MODEL:-Ablation1}"
BATCH="${BATCH:-1}"
EPOCH="${EPOCH:-450}"
ADAPTIVE="${ADAPTIVE:-False}"
ABLATION="${ABLATION:-True}"
FEATURE_ABLATION="${FEATURE_ABLATION:-0101100}"

LR="${LR:-5e-3}"
SCHEDULER="${SCHEDULER:-cosine}"
SCHEDULERARGS="${SCHEDULERARGS:-450}"
ETA_MIN="${ETA_MIN:-1e-7}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"

RUNS_DIR="${OUTPUT_DIR}/runs"
LOGS_DIR="${OUTPUT_DIR}/logs"
RUN_NAME="${RUN_PREFIX}_${MODEL}_fa_${FEATURE_ABLATION}_lr_${LR}_${SCHEDULER}_${SCHEDULERARGS}_etamin_${ETA_MIN}_wd_${WEIGHT_DECAY}"
FOLDER="${RUNS_DIR}/${RUN_NAME}"
LOGFILE="${LOGS_DIR}/${RUN_NAME}.log"

usage() {
    cat <<'EOF'
Usage:
  ./K-STEMIT-train.sh
  AUTO_ATTACH=False ./K-STEMIT-train.sh
  SESSION_NAME=my_run ./K-STEMIT-train.sh

This sample script launches one K-STEMIT training run with:
  - model=Ablation1
  - batch=1
  - epoch=450
  - adaptive=False
  - ablation=True
  - featureablation=0101100
  - lr=5e-3
  - scheduler=cosine
  - schedulerargs=450
  - eta_min=1e-7
  - weight_decay=1e-5

Useful overrides:
  CONDA_ENV_NAME
  CODE_DIR
  OUTPUT_DIR
  RUN_PREFIX
  SESSION_NAME
  AUTO_ATTACH
  CUDA_VISIBLE_DEVICES
EOF
}

case "${1:-}" in
    "" )
        ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        printf 'Unknown argument: %s\n\n' "${1}" >&2
        usage
        exit 1
        ;;
esac

if [ ! -d "$CODE_DIR" ]; then
    printf 'Code directory not found: %s\n' "$CODE_DIR" >&2
    exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    printf 'tmux session already exists: %s\n' "$SESSION_NAME" >&2
    printf 'Attach with: tmux attach -t %s\n' "$SESSION_NAME" >&2
    exit 1
fi

mkdir -p "$RUNS_DIR" "$LOGS_DIR"

tmux new-session -d -s "$SESSION_NAME"
tmux send-keys -t "$SESSION_NAME" "source \"$HOME/anaconda3/etc/profile.d/conda.sh\"" C-m
tmux send-keys -t "$SESSION_NAME" "conda activate ${CONDA_ENV_NAME}" C-m
tmux send-keys -t "$SESSION_NAME" "cd \"${CODE_DIR}\"" C-m

CMD="accelerate launch run_new_acc.py --model ${MODEL} --batch ${BATCH} --epoch ${EPOCH} --adaptive ${ADAPTIVE} --ablation ${ABLATION} --lr ${LR} --scheduler ${SCHEDULER} --schedulerargs ${SCHEDULERARGS} --eta_min ${ETA_MIN} --weight_decay ${WEIGHT_DECAY} --featureablation ${FEATURE_ABLATION} --folder \"${FOLDER}\" > \"${LOGFILE}\" 2>&1"

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    CMD="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${CMD}"
fi

printf 'Session: %s\n' "$SESSION_NAME"
printf 'Code dir: %s\n' "$CODE_DIR"
printf 'Output dir: %s\n' "$OUTPUT_DIR"
printf 'Run folder: %s\n' "$FOLDER"
printf 'Log file: %s\n' "$LOGFILE"
printf 'Training setup: model=%s batch=%s epoch=%s adaptive=%s ablation=%s featureablation=%s\n' \
    "$MODEL" "$BATCH" "$EPOCH" "$ADAPTIVE" "$ABLATION" "$FEATURE_ABLATION"
printf 'Optimizer setup: lr=%s scheduler=%s schedulerargs=%s eta_min=%s weight_decay=%s\n' \
    "$LR" "$SCHEDULER" "$SCHEDULERARGS" "$ETA_MIN" "$WEIGHT_DECAY"

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
fi

tmux send-keys -t "$SESSION_NAME" "$CMD" C-m

if [ "$AUTO_ATTACH" = "True" ]; then
    tmux attach -t "$SESSION_NAME"
else
    printf 'Detached session: %s\n' "$SESSION_NAME"
fi
