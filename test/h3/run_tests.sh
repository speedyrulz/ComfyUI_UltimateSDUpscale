#!/usr/bin/env bash
# CPU-only tests for the MiniMax H3 tiling nodes. Run inside WSL (needs /home/jeff/ComfyUI and its venv).
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES=
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
cd /home/jeff/ComfyUI
exec /home/jeff/ComfyUI/venv/bin/python "$HERE/h3_tiled_tests.py" "$@"
