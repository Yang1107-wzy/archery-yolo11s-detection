# v1.0.0 — Reproducible YOLO11s 150-Epoch Release / 双语可复现发布

This first public release contains the complete extracted Target v6 dataset, portable training/evaluation/inference code, Streamlit demo, `best.pt` and `last.pt`, the complete 150-epoch metric history, figures, standard-AP test evidence, locked-threshold test evidence and a SHA-256 release manifest.

Main independently reported test result: mAP50 75.90% and mAP50–95 41.31% under standard AP evaluation. Best validation at Epoch 142 was mAP50 80.12% and mAP50–95 43.65%; validation and test are deliberately reported separately.

首次公开版本包含 Target v6 完整解压数据、可移植训练/评估/推理代码、Streamlit demo、`best.pt`、`last.pt`、完整 150-epoch 指标、图表、标准 AP test、锁定阈值 test 和 SHA-256 发布清单。

独立标准 AP test 为 mAP50 75.90%、mAP50–95 41.31%。Epoch 142 最佳 validation 为 mAP50 80.12%、mAP50–95 43.65%，二者严格分开报告。

Important limitations / 重要限制：the release is an independent reproduction, not the original Roboflow checkpoint; public splits may contain visually related variants; bounding-box centers are not physically annotated arrow-impact points. 本项目是独立复现而非 Roboflow 原作者权重；公开 split 可能含视觉近似版本；检测框中心不等于物理标注撞击点。
