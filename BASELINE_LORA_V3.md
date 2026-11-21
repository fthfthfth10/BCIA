# B2 DINOv3 + LoRA + 常规 RGB/DSM 融合

基准提交 fc2f3ecdeb6d0c68e936f13c2a0309cb9e475604。独立入口 lora_b2.py 从相同初始化构建 G1，再在 24 个 ViT-L/16 Block 的 q_proj、v_proj 上包装冻结 Linear。前向为 W₀x+b₀+(α/r)BAx，r=4、α=4、dropout=0，无额外偏置。A 使用 Linear 默认初始化，B 为零。其余 DINOv3 参数冻结，LoRA A/B、原 G1 DSM Stem、Block18 常规融合和 Segmenter Head 训练。G1 训练产物与源码沿用原路径。

| 模型 | 总参数 | 可训练参数 |
| --- | ---: | ---: |
| G1 | 331,702,252 | 28,572,652 |
| G2 | 331,702,768 | 28,573,168 |
| B2 | 332,095,468 | 28,965,868 |

LoRA 新增 48×(1024×4+4×1024)=393,216 个参数。B2 比 G2 多 392,700 个可训练参数，约 1.374%。B2 使用 G1 的常规融合、有效像素 CE 与六类 Segmenter。

配置 configs/lora_b2.json 与正式 G1 配置逐项核对。Potsdam v3 3,456/2,016、seed 42、batch 1、FP32、AdamW lr=1e-4、wd=0.05、40,000 步、warmup 1,500 起始比例 0.1、PolyLR power=1、每 2,000 步无 TTA 全量验证并保存，mIoU5 选 Best。辅助损失为零。沿用 BCIA 的确定性设置、CPU 分割损失和数据加载。

RTX 3070 Ti 上首个真实 batch 4_10_0_3584_512_4096 验收通过。FP32 初始 logits 与 G1 逐元素相同，非 LoRA 模块初始化指纹相同。首步 CE 为 2.10227132，LR 为 1e-5。48 个 LoRA B 矩阵均有非零梯度并更新，A 首步任务梯度符合零 B 初始化的预期。原始主干无梯度，所有源参数和缓冲值在更新后逐项匹配。含完整 AdamW 更新的峰值 allocated 4,529,199,104 字节、reserved 4,718,592,000 字节。验收 Last 严格恢复第 1 步及 AdamW、调度器和 RNG。

入口为 python -m lora_b2。设置 BCIA_DATA_ROOT 与 BCIA_BACKBONE_CHECKPOINT，传入 --work-dir，预检附 --preflight，正式运行附 --commit 和固定源码提交。正式输出目录为 work_dirs/baselines/lora_v3_seed42。
