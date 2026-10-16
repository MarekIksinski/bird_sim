#!/usr/bin/env bash
# bird_sim launcher — recreates the venv automatically if it's missing.
set -euo pipefail
cd "$(dirname "$0")"

# CuPy needs CUDA headers for JIT kernels; this box has them in /usr/include
export CUDA_PATH="${CUDA_PATH:-/usr}"

if [ ! -x .venv/bin/python ]; then
    echo "[setup] venv missing — creating with Python 3.12..."
    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python -r requirements.txt
fi

exec .venv/bin/python -u main.py
