# Model Card: YOLO11s Target v6

## Model details

- Architecture: Ultralytics YOLO11s object detector
- Task: 12-class target/score-region detection
- Input: RGB images, evaluated at \(640\times640\)
- Training: 150 epochs, batch 8, seed 42
- Selection: highest validation mAP50–95, Epoch 142
- Authors: Zhengyang Wang and Jiacheng Yao, BNBU

`best.pt` SHA-256: `699235268b229cbb5e401d4fb0559d788630de04267605d2927aad60aa262b20`

`last.pt` SHA-256: `6e1db278f809bfd9474c8a0873adb10dd2e34e5e1cab82055e8c32ff91ab0384`

## Intended use

Research, teaching, qualitative visualization and non-safety-critical prototyping on imagery similar to Target v6. The model may help localize target regions and approximate arrow-region detections.

## Out-of-scope use

Do not use this checkpoint as an official competition judge, safety system, or a calibrated physical impact-point estimator. Predictions use bounding-box centers; they are not physical arrow-impact annotations.

## Performance and risks

The selected checkpoint reaches validation mAP50–95 43.65%; independent standard-AP test mAP50–95 is 41.31%. See [RESULTS.md](RESULTS.md) for protocols. Possible related-image leakage, class imbalance, domain shift, occlusion and small-object localization remain material risks.

## License

The fine-tuned weights are distributed under AGPL-3.0 in accordance with this open-source release path. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

