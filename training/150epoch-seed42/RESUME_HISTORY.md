# Resume history / 恢复训练历史

The run completed all 150 epochs. It was resumed after the workspace was moved, without reinitializing model or optimizer state. The final effective run arguments therefore record `patience: 0` and a resume checkpoint even though the initial requested configuration used `patience: 30`.

本次训练完成了全部 150 个 epoch。工作区移动后通过 checkpoint 恢复训练，没有重新初始化模型或优化器。因此，最终有效参数中为 `patience: 0` 并记录了 resume checkpoint，而最初请求配置为 `patience: 30`。

The `time` column resets between epochs 80 and 81 because the resumed process started a new wall-clock timer. Epoch numbering, weights, and optimizer continuity remain intact. Consequently, the two time segments must not be summed or compared directly with the Roboflow screenshot as a single uninterrupted timer.

`time` 列在 epoch 80 和 81 之间归零，原因是恢复后的进程重新开始计时。Epoch 序号、模型权重和优化器状态仍然连续。因此，不应把两段时间当作单次连续计时，也不应直接与 Roboflow 截图比较训练效率。
