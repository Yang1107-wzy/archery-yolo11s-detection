# YOLO11s Target v6 本地复刻正式报告

## 数据与协议

- 数据集：Roboflow `Target and Arrow detection` v6，YOLOv11 导出。
- split：train 1482、valid 98、public test 65，共 1645 张。
- 数据 fingerprint：`b99178e6c76f9fe94f2b1d961de4209c9e5eeb062e33cd2ded1cb17798367a8f`。
- ZIP SHA-256：`d81c8389e9bf1698b8bf86d76471024882c45cf090685f3704c0132357be53bd`。
- 类别：`0, 1, 10, 2, 3, 4, 5, 6, 7, 8, 9, target`。
- 选模与阈值只使用 valid；public test 在 A/B 与阈值锁定后只正式评测一次。

## A/B 正式训练

### A：default_aug

- run_id：`replica_20260719_002149_184e8ce2_full`
- 训练：YOLO11s、640、batch 8、seed 42、MPS，实际完成 150 epoch。
- 中断说明：第 80 epoch 后工作区被 iCloud 移动，随后从 `last.pt` 恢复至 150 epoch；恢复保留模型、EMA、优化器与 epoch 状态，但不声明与完全不中断轨迹位级相同。
- best epoch：142。
- best validation：P 0.73944、R 0.78580、mAP50 0.80121、mAP50-95 0.43646。
- checkpoint：`artifacts/models/replica_target_v6/default_aug_seed42/best.pt`。

### B：offline_only

- run_id：`replica_20260719_074851_46375c29_full`
- 训练：与 A 相同，但关闭 YOLO 在线几何、颜色、翻转与 mosaic 等增强。
- patience=30 在 44 epoch 正常早停；best epoch 14。
- best validation：P 0.68557、R 0.54748、mAP50 0.58656、mAP50-95 0.33329。
- checkpoint：`artifacts/models/replica_target_v6/offline_only_seed42/best.pt`。

## Valid 阈值扫描与选模

每个模型扫描 confidence `{0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50}` 与 NMS IoU `{0.30,0.40,0.50,0.60,0.70}`，按同一工作点的 F1 选择，并以 mAP50 破同分。

- A：conf 0.25、IoU 0.50，P 0.75990、R 0.80369、F1 0.78118、mAP50 0.74383、mAP50-95 0.40743。
- B：conf 0.30、IoU 0.50，P 0.69985、R 0.54753、F1 0.61439、mAP50 0.46939、mAP50-95 0.27927。
- 胜者：A `default_aug`。

## Official public test（首次且唯一一次正式评测）

锁定 A、conf 0.25、IoU 0.50 后，在 65 张 official test 上得到：

- P：0.78164
- R：0.67233
- F1：0.72287
- mAP50：0.67503
- mAP50-95：0.37737
- score classes macro-F1：0.69209
- target F1：0.98462

`target` 类单独为 P 0.98462、R 0.98462、AP50 0.98069、AP50-95 0.93743。12 类 aggregate 低于网页参考 P 0.985、R 0.873、mAP50 0.754，不能宣称整体达到网页指标。

## Claim boundary

- 未取得原作者私有 `best.pt`、私有训练参数或网页评测实现。
- 本项目完成的是官方 v6 数据与 YOLO11s 架构/训练流程的本地复刻，不是原权重的位级复现。
- sanity 与 smoke 结果不属于 full-training 指标。
- valid 用于选模与阈值；上面的 65 张结果才是 held-out public test。
- 官方 split 存在相似图与同源命名候选，public test 仍按官方 split 报告，但其独立性风险需保留。
