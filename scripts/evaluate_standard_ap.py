from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from archery_ml.evaluation.standard_ap import run_standard_ap_test


def main() -> int:
    parser = argparse.ArgumentParser(description="Run standard low-confidence AP evaluation on the public test split.")
    parser.add_argument("--model", default="models/yolo11s-target-v6/best.pt")
    parser.add_argument("--data", default="data/target-arrow-detection-v6/data.yaml")
    parser.add_argument("--output", default="evaluation/standard_ap_test.json")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    payload = run_standard_ap_test(
        args.model,
        args.data,
        args.output,
        device=args.device,
        imgsz=args.imgsz,
        batch=args.batch,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
