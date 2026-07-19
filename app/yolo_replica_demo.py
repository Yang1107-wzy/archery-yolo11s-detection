from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archery_ml.demo.yolo_replica_streamlit import render_yolo_replica_app


if __name__ == "__main__":
    render_yolo_replica_app()

