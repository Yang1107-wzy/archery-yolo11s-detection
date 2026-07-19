from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from archery_ml.evaluation.threshold_sweep import run_public_test_from_validation, run_validation_sweep
from archery_ml.models.predict_score import predict_scores
from archery_ml.models.train_yolo import train_yolo


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _add_eval_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m archery_ml.cli")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Train YOLO11s from a reviewed YAML configuration.")
    train.add_argument("--config", required=True)
    modes = train.add_mutually_exclusive_group()
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--sanity", action="store_true")

    infer = commands.add_parser("infer", help="Run local inference and export JSON plus annotated images.")
    infer.add_argument("--model", required=True)
    infer.add_argument("--source", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--confidence", type=float, default=0.25)
    infer.add_argument("--iou", type=float, default=0.7)
    infer.add_argument("--device", default="cpu")

    validation = commands.add_parser(
        "evaluate-validation",
        help="Select confidence and NMS IoU using only the validation split.",
    )
    _add_eval_arguments(validation)

    public_test = commands.add_parser(
        "evaluate-test",
        help="Evaluate the public test split once using a locked validation selection.",
    )
    _add_eval_arguments(public_test)
    public_test.add_argument("--validation", required=True)
    return parser


def _print(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = project_root()
    if args.command == "train":
        if not args.smoke and not args.sanity and os.environ.get("ARCHERY_RUN_FULL") != "1":
            raise SystemExit("full training blocked: set ARCHERY_RUN_FULL=1 explicitly")
        _print(
            train_yolo(
                args.config,
                smoke=bool(args.smoke),
                sanity=bool(args.sanity),
                project_root=root,
            )
        )
        return 0
    if args.command == "infer":
        payloads = predict_scores(
            args.model,
            args.source,
            args.output,
            confidence=args.confidence,
            iou=args.iou,
            device=args.device,
        )
        _print({"images": len(payloads), "output": str(Path(args.output).resolve())})
        return 0
    if args.command == "evaluate-validation":
        _print(
            run_validation_sweep(
                args.model,
                args.data,
                args.output,
                device=args.device,
                imgsz=args.imgsz,
                batch=args.batch,
            )
        )
        return 0
    if args.command == "evaluate-test":
        _print(
            run_public_test_from_validation(
                args.model,
                args.data,
                args.validation,
                args.output,
                device=args.device,
                imgsz=args.imgsz,
                batch=args.batch,
            )
        )
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
