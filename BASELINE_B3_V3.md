# B3 DINOv3 + LoRA + 原始 BCIA

B3 从 DINOv3 ViT-L/16 预训练权重和 seed 42 原始 G2 初始化构建。沿用 G2 的 DSM Stem、Segmenter Head 和第 18 个 Block 执行结束后的单次 BCIA 注入。24 个 Block 的 Q/V 各使用 B2 的 `LoRALinear`，rank 4、alpha 4、dropout 0、B 矩阵全零、A 矩阵沿用 Linear 默认初始化。原始 DINO 权重冻结。B3 按 G2 构造后当前 RNG 状态初始化 A，B2 按 G1 构造后的 RNG 状态初始化 A。两组采用相同初始化规则和种子。零 B 使两组分别保持各自的原始模型初始前向。

| 组别 | 总参数 | 可训练参数 |
| --- | ---: | ---: |
| G1 | 331,702,252 | 28,572,652 |
| G2 | 331,702,768 | 28,573,168 |
| B2 | 332,095,468 | 28,965,868 |
| B3 | 332,095,984 | 28,966,384 |

B3 相对 G2 新增 393,216 个可训练 LoRA 参数，相对 B2 多 516 个可训练参数，差额来自融合模块。分割训练只使用有效像素 CE。两方向辅助 MSE 系数均为 0。

## 冻结协议

[configs/lora_b3.json](configs/lora_b3.json) 复用 [configs/formal_g2.json](configs/formal_g2.json) 的训练协议与 [B2 的 LoRA 配置](configs/lora_b2.json)。Potsdam v3 的 24/14 场景分别有 3,456/2,016 个窗口，使用 Eroded 标签及 ignore 255。seed 42、batch 1、workers 0、FP32，启用确定性算法并关闭 TF32。统一 AdamW 初始 LR 1e-4、weight decay 0.05，warmup 1,500 步、起始比例 0.1，衰减 PolyLR power 1，共 40,000 次优化器更新。每 2,000 步保存及全量验证，mIoU5 选 Best，无 TTA。

数据、主干与非 LoRA 模块初始化指纹必须与原 G2 正式记录匹配。预检及正式运行从同一初始化开始，分别存放于 `work_dirs/baselines/b3_lora_bcia_v3_preflight` 与 `work_dirs/baselines/b3_lora_bcia_v3_seed42`。设置 `BCIA_DATA_ROOT` 与 `BCIA_BACKBONE_CHECKPOINT` 后，用 `python -m lora_b3 --preflight` 预检。发布并固定训练提交后用 `python -m lora_b3 --commit <训练提交 SHA>` 正式运行。中断后在同一提交以相同参数追加 `--resume`，从正式目录的 Last 读取模型、AdamW、调度器及随机数状态。

比较以各组按 mIoU5 选择的 Best 为准。报告 B3 − G2、B3 − B2，以及 mIoU5 的 `(B3 − G2) − (B2 − G1)`。交互差值描述 seed 42 的四组运行。

## 训练前验收

RTX 3070 Ti 8 GB，PyTorch 2.1.1+cu121，CUDA 12.1，FP32。38 项单测通过。使用实际训练 DataLoader 的前两批进行 AdamW 更新。加入 LoRA 前后初始 logits 逐元素相同。DSM Stem、BCIA、Head 的初始化 SHA-256 与 G2 原记录匹配，48 个 Q/V LoRA 模块均已注入。原始 DINO 参数无梯度，更新前后与源权重逐元素相同。预检 Last 通过严格加载并恢复第 2 步、AdamW 状态和调度器。

| 步 | 样本 ID | CE | LR | A 非零梯度 | B 非零梯度 | 峰值 allocated 字节 | 峰值 reserved 字节 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4_10_0_3584_512_4096 | 2.10227132 | 1e-5 | 0/48 | 48/48 | 4,530,654,208 | 4,718,592,000 |
| 2 | 6_8_2048_2048_2560_2560 | 2.09609723 | 1.006e-5 | 48/48 | 48/48 | 4,761,654,272 | 4,968,153,088 |

首步 48 个 B 矩阵都已更新。第二步 DSM Stem 梯度范数为 1.0627e-4，BCIA 五个上游子模块的范数在 5.9567e-5 至 6.9706e-4 之间。首次预检在首步后触发了原 G2 的全主干冻结诊断断言；B3 改为单独检查适配器梯度，复用训练步中针对实际冻结权重的无梯度断言。两步通过记录来自修正后的预检，训练配置保持原值。

| 已核对文件或初始状态 | SHA-256 |
| --- | --- |
| Potsdam v3 metadata.json | `c5b1b911d5de65ff2a58581add340f5bb4c89d2832fdabdd900caabe542236c8` |
| normalization.json | `02da0e6cec2ffbe240fed78eb1bc0f70484147e8da8be7a73fd9b9ad8dcd1c3b` |
| manifests/train.jsonl | `378dda5f53bc4a0c45991f0eb96677c3208d5532e3a85aacf670cf691cd25bf6` |
| manifests/val.jsonl | `3b4e7819b542eff1f584b39ba96b33f63cc4a71df2c69c39f14b71501595811a` |
| DINOv3 ViT-L/16 源权重 | `f7224cd45820d310d8ba9777b5c4379589d6d3e44d525596fff5805223bcba9e` |
| DSM Stem 初始 | `f0e86886fa660c8184625ca351b336317130cbd88e15785157287a71fe2f87a0` |
| 原始 BCIA 初始 | `2a33f9b33d91920d3374f04319e74ac677bc814b6c34b29f7d922bea22382e0f` |
| Segmenter Head 初始 | `a5c9a386c0de58d48a1a2b664ad9d8071ec13dd03596e0b8095c3bcca011183a` |
| B3 LoRA 初始 | `951fa3467891fddc4bedf6e7094fc16935ddd198278e7e16c3d873f844f75c08` |
