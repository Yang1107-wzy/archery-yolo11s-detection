#!/usr/bin/env python3
"""Generate the public repository's source-backed visual evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from archery_ml.viz.repository_visuals import copy_local_inference_asset, generate_all


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--local-inference-source", type=Path)
    args = parser.parse_args()
    for output in generate_all(args.root):
        print(output)
    if args.local_inference_source:
        destination = args.root / "docs/assets/visuals/local_model_inference_streamlit.png"
        print(copy_local_inference_asset(args.local_inference_source, destination))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
