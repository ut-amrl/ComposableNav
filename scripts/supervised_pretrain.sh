#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage:
  bash $0 +data_path=<path> +exp_name=<name> [extra hydra overrides]

Examples:
  bash $0 +data_path=generated_data/pretrain_<datetime> +exp_name=supervised_pretrain
  bash $0 +data_path=generated_data/pretrain_<datetime> +exp_name=supervised_pretrain trainer.devices=2
EOF
}

# Show usage if no args or help flag
[[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]] && { usage; exit 0; }

# Parse arguments
data_path=""
exp_name=""
args=()

for arg in "$@"; do
  case "$arg" in
    +data_path=*)       data_path="${arg#+data_path=}";;
    +exp_name=*) exp_name="${arg#+exp_name=}";;
  esac
  args+=("$arg")
done

# Validation
: "${data_path:?Missing required +data_path=<path>}"
: "${exp_name:?Missing required +exp_name=<name>}"

# Display
echo "──────────────────────────────────────────"
echo " Running supervised pretrain"
echo " Data path        : ${data_path}"
echo " Experiment name  : ${exp_name}"
echo "──────────────────────────────────────────"

# Execute
exec python composablenav/train/supervised_pretraining.py \
  +exps=supervised_pretraining \
  "${args[@]}"
