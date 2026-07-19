from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from archery_ml.evaluation.standard_ap import run_standard_ap_test


class FakeModel:
    def __init__(self) -> None:
        self.kwargs = None
        self.names = {0: "0", 11: "target"}

    def val(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            results_dict={
                "metrics/precision(B)": 0.78,
                "metrics/recall(B)": 0.67,
                "metrics/mAP50(B)": 0.759,
                "metrics/mAP50-95(B)": 0.413,
            }
        )


def test_standard_ap_uses_low_confidence_and_public_test_split(tmp_path: Path) -> None:
    model = FakeModel()
    output = tmp_path / "standard_ap.json"

    payload = run_standard_ap_test(
        "best.pt",
        "data.yaml",
        output,
        device="cpu",
        model=model,
    )

    assert model.kwargs["split"] == "test"
    assert model.kwargs["conf"] == 0.001
    assert model.kwargs["iou"] == 0.7
    assert payload["protocol"] == "standard_ap_low_confidence_sweep"
    assert payload["metrics"]["map50"] == 0.759
    assert output.is_file()
