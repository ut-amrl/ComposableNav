#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [optional hydra overrides]"
  echo "Example: $0 dataset.generation.num_dynamic=40000"
  echo "Example: $0 dataset.generation.num_proc=128"
}

# Show help if requested
[[ "${1:-}" =~ ^(-h|--help)$ ]] && { usage; exit 0; }

echo "── Generating dataset ─────────────────────"
[[ $# -gt 0 ]] && echo "Args : $*" || echo "Args : (none, using defaults)"
echo "──────────────────────────────────────────"

exec python composablenav/datasets/generate_data.py +exps=generate_data "$@"
