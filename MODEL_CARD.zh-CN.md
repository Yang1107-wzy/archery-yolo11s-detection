# 模型卡：YOLO11s Target v6

模型为 12 类 Ultralytics YOLO11s 目标检测器，输入评估尺寸为 \(640\times640\)，训练 150 epochs、batch 8、seed 42，由 validation mAP50–95 最高的 Epoch 142 选择 `best.pt`。作者为 Zhengyang Wang、Jiacheng Yao（BNBU）。

- `best.pt` SHA-256：`699235268b229cbb5e401d4fb0559d788630de04267605d2927aad60aa262b20`
- `last.pt` SHA-256：`6e1db278f809bfd9474c8a0873adb10dd2e34e5e1cab82055e8c32ff91ab0384`

适用范围是研究、教学、可视化与非安全关键原型。不应作为正式比赛裁判、安全系统或经过标定的真实撞击点估计器。模型用检测框中心表示位置。validation mAP50–95 为 43.65%，独立标准 AP test mAP50–95 为 41.31%；详细口径见 [RESULTS.zh-CN.md](RESULTS.zh-CN.md)。数据近似样本跨 split、类别不平衡、域偏移、遮挡和小目标定位均是剩余风险。权重采用 AGPL-3.0。

