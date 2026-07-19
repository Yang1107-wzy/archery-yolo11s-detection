# 可复现性说明

仓库保留完整解压数据、可移植配置、连续 150 个 epoch 记录、恢复训练历史、优化器摘要、图表、`best.pt`、`last.pt`、来源清单、路径改写记录和文件哈希清单；排除重复原始 ZIP 与 150 个中间 checkpoint。

训练使用 YOLO11s、150 epochs、batch 8、输入 640、seed 42。`optimizer=auto` 选择 AdamW，有效初始学习率为 0.000625；配置值 `lr0=0.01`、`lrf=0.01`、beta 0.9/0.999、weight decay 0.0005 单独记录。Epoch 80 与 81 之间从模型和优化器状态恢复。发布配置记录实际生效的 `patience=0`；最初请求曾设置 patience 30，之后改为完整不早停训练。

`make test` 运行单元与发布契约测试；`make verify` 检查数据、隐私、checkpoint 和 manifest；`make infer-example` 在 CPU 上加载真实权重并输出单图 JSON 与标注图。GitHub Actions 明确使用 `lfs: false`，只运行轻量测试，避免自动消耗模型和数据的 LFS 带宽。

依赖版本见 `requirements-*.lock`。硬件和浮点实现差异可能导致重新训练无法得到逐位相同权重，因此本项目的复现目标是数据、配置、代码、来源证据完整，并能得到协议一致的指标，而不是保证权重逐位一致。

