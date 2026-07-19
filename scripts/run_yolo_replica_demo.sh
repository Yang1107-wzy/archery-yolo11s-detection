#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON_BIN" -m streamlit run app/yolo_replica_demo.py \
  --server.headless true \
  --server.address "::" \
  --server.port 8504 \
  "$@"
