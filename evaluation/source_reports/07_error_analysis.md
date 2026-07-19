# Official public test 误差分析

分析对象：A `default_aug` 的 `best.pt`，locked conf 0.25、IoU 0.50，official test 65 张。

## 计数与匹配诊断

- 图片数：65。
- score 检测数量 MAE：0.36923。
- score 数量完全一致率：0.72308。
- IoU>=0.5 贪心诊断匹配：TP 161、FP 28、FN 21、wrong-ring 18。
- target miss 图片：0。

## Public-test per-class

| class | P | R | AP50 | AP50-95 |
|---|---:|---:|---:|---:|
| 0 | 0.78489 | 0.66667 | 0.66500 | 0.41988 |
| 1 | 0.64207 | 0.54545 | 0.45700 | 0.27620 |
| 10 | 0.82112 | 0.46337 | 0.53667 | 0.25400 |
| 2 | 0.67512 | 0.66667 | 0.70852 | 0.34965 |
| 3 | 1.00000 | 0.80325 | 0.87500 | 0.39925 |
| 4 | 0.85546 | 0.75000 | 0.81423 | 0.40903 |
| 5 | 0.70226 | 0.86667 | 0.70281 | 0.45440 |
| 6 | 0.68522 | 0.46667 | 0.39567 | 0.18215 |
| 7 | 0.85619 | 0.79419 | 0.76393 | 0.27411 |
| 8 | 0.58517 | 0.50410 | 0.60833 | 0.28290 |
| 9 | 0.78751 | 0.55629 | 0.59250 | 0.28949 |
| target | 0.98462 | 0.98462 | 0.98069 | 0.93743 |

严格 IoU 的主要薄弱类别是 6、10、7、1、8、9；这解释了 aggregate mAP50-95 明显低于 mAP50。`target` 类已很强，主要瓶颈是 score 类分类与精确框定位。

## 可视化

- 全量预测 JSON：`artifacts/predictions/replica_target_v6_test/`
- 分类 gallery：`artifacts/visualizations/replica_target_v6_test/`
- gallery 图片数：TP 65、FP 19、FN 14、wrong-ring 13、cluster-overlap 12、worst-20 20。

## Claim boundary

诊断使用 IoU>=0.5 的贪心匹配，不等同于 Ultralytics COCO-style AP 计算。score bbox center 不是物理箭孔落点 GT；旧孔、环线边界等原因必须人工看图后才能定性。
