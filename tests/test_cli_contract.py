from __future__ import annotations

from archery_ml.cli import build_parser


def test_train_cli_keeps_config_and_training_modes() -> None:
    parser = build_parser()

    args = parser.parse_args(["train", "--config", "configs/train/run.yaml", "--smoke"])

    assert args.command == "train"
    assert args.config == "configs/train/run.yaml"
    assert args.smoke is True
    assert args.sanity is False


def test_infer_cli_keeps_model_source_output_and_thresholds() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "infer",
            "--model",
            "models/best.pt",
            "--source",
            "image.jpg",
            "--output",
            "outputs/demo",
        ]
    )

    assert args.command == "infer"
    assert args.confidence == 0.25
    assert args.iou == 0.7
    assert args.device == "cpu"


def test_evaluation_cli_separates_validation_selection_from_public_test() -> None:
    parser = build_parser()

    validation = parser.parse_args(
        [
            "evaluate-validation",
            "--model",
            "models/best.pt",
            "--data",
            "data/data.yaml",
            "--output",
            "evaluation/validation.json",
        ]
    )
    test = parser.parse_args(
        [
            "evaluate-test",
            "--model",
            "models/best.pt",
            "--data",
            "data/data.yaml",
            "--validation",
            "evaluation/validation.json",
            "--output",
            "evaluation/test.json",
        ]
    )

    assert validation.command == "evaluate-validation"
    assert test.command == "evaluate-test"
    assert test.validation == "evaluation/validation.json"
