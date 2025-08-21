#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage:
  bash $0 +primitive=<name> +exp_name=<name> +checkpoint=<path> [extra hydra overrides]

Examples:
  bash $0 +primitive=pass_from_left +exp_name=ft/pass_from_left +checkpoint=training_results/<pretrain_exp_name>/last.ckpt
  bash $0 +primitive=follow +exp_name=ft/follow +checkpoint=training_results/<pretrain_exp_name>/last.ckpt compile=true
EOF
}

# Show usage if no args or help flag
[[ $# -eq 0 || "${1:-}" =~ ^(-h|--help)$ ]] && { usage; exit 0; }

# Parse args
primitive=""
exp_name=""
checkpoint=""
args=()

for arg in "$@"; do
  case "$arg" in
    +primitive=*)  primitive="${arg#+primitive=}";;
    +exp_name=*)   exp_name="${arg#+exp_name=}";;
    +checkpoint=*) checkpoint="${arg#+checkpoint=}";;
  esac
  args+=("$arg")
done

# Validation
: "${primitive:?Missing required +primitive=<name>}"
: "${exp_name:?Missing required +exp_name=<name>}"
: "${checkpoint:?Missing required +checkpoint=<path>}"

# Display
echo "──────────────────────────────────────────"
echo " Running RL finetuning"
echo " Primitive   : ${primitive}"
echo " Experiment  : ${exp_name}"
echo " Checkpoint  : ${checkpoint}"
echo "──────────────────────────────────────────"

# Execute
export HYDRA_FULL_ERROR=1
exec python composablenav/train/rl_finetuning.py \
  +exps=rl_finetuning \
  "${args[@]}"
