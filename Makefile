PYTHON ?= python3
PYTHONPATH := src:.
MODEL := models/yolo11s-target-v6/best.pt
EXAMPLE_IMAGE := $(firstword $(wildcard data/target-arrow-detection-v6/test/images/*))

.PHONY: test verify manifest infer-example demo

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q

verify:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m tools.verify_release --root .

manifest:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m tools.build_manifest --root . --output release/manifest.json

infer-example:
	test -n "$(EXAMPLE_IMAGE)"
	rm -rf outputs/infer-example
	PYTHONPATH=src $(PYTHON) -m archery_ml.cli infer --model $(MODEL) --source "$(EXAMPLE_IMAGE)" --output outputs/infer-example --device cpu

demo:
	./scripts/run_yolo_replica_demo.sh
