from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.release_tools import build_file_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a deterministic SHA-256 manifest for the release tree.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("release/manifest.json"))
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    payload = build_file_manifest(root, manifest_path=output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {payload['file_count']} entries to {output.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
