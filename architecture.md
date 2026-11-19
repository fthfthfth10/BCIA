# BCIA-PEFT 架构

## 目录

| 组件 | 文件 |
| --- | --- |
| 冻结 DINOv3 接口，固定 Block 插入点，Segmenter Head | bcia_peft/model.py、bcia_peft/_dinov3.py |
| DSM 卷积、BCIA、独立预测辅助损失 | bcia_peft/adapter.py |
| 预训练 checkpoint 只读审计和严格加载 | bcia_peft/checkpoints.py |
| 切片预处理、单窗口验证与指标 | bcia_peft/data.py、bcia_peft/validation.py、configs/potsdam.json |
| 单元测试及 batch1 前反向 | tests/test_components.py、tests/smoke.py |

## 前向张量（batch 1）

| 位置 | 张量 | 形状 |
| --- | --- | --- |
| 输入 | 已归一化 RGB / DSM / valid | [1,3,512,512] / [1,1,512,512] / [1,1,512,512] |
| DINO embeddings → Blocks 1–18 | CLS + 4 Registers + 空间 tokens | [1,1029,1024] |
| **Block 18 输出（索引 17）** | 分离 5 个前缀 token 和 1024 个空间 token | [1,5,1024]、[1,1024,1024] |
| DSM stem | [masked DSM, valid] → 2→16→32→64；kernel/stride 4/2/2 | [1,2,512,512] → [1,64,32,32] |
| 覆盖率 q | 对 valid ∧ isfinite(DSM) 做 16×16 平均池化 | [1,1,32,32] |
| BCIA | zR = Linear(1024,64) 后恢复空间布局；zH = DSM stem | 各 [1,64,32,32] |
| 预测与残差 | 两个独立的 3×3 卷积；uH = zH − gR2H(zR)，uR = zR − gH2R(zH) | 各 [1,64,32,32] |
| 路由 | concat(zR,zH,uR,uH,q) → 1×1(257,2) → 独立 Sigmoid | wR、wH 各 [1,1,32,32] |
| 融合 | concat(wR·uR,wH·uH) → 1×1(128,64) → 展平 → zero Linear(64,1024) | 残差 [1,1024,1024] |
| **Block 19 输入（索引 18）** | 5 个前缀 token 原样拼回；只给空间 token 加残差 | [1,1029,1024] |
| Blocks 19–24 → 末端 LayerNorm | 移除前缀并恢复 patch 网格 | [1,1024,32,32] |
| MMSeg Segmenter | 原 2 层 Mask Transformer、6 类；单窗口双线性上采样 | [1,6,32,32] → [1,6,512,512] |

除 bool 的 valid 和 int64 标签，验收张量均为 FP32，参数和输入在同一设备。
无效或非有限 DSM 像素入 stem 前替换为 0。有效 patch 使用原预测残差；q=0 时清零 zH、uH、wH，并屏蔽 gH2R 的局部输出，避免 3×3 邻域与 bias 在该位置注入 DSM，保留 uR=zR 的 RGB 路由。部分有效 patch 的 q 作为路由输入，而不是强制归一化或额外缩放两个权重。严格零贡献是 **BCIA 注入点的局部约束**，不禁止后续全局自注意力传播来自其他有效 patch 的信息。

DINO 始终 eval，所有参数冻结；后六层仍参与 autograd，不能断开 BCIA 到分割头的梯度。仅 BCIA 最终投影的 weight 和 bias 特意清零；第一步分割损失的梯度可到达该投影，但其上游梯度为零。DSM stem、BCIA、Segmenter 都可训练。prediction_losses 单独提供以 q 加权的双向 MSE，目标特征 detach；不纳入默认分割前向及验收反向。

验证保留 impervious_surface / building / low_vegetation / tree / car / clutter，原标签 0→255，1…6→0…5。使用 MMSeg 上采样和 IoU 交并计数，按旧验证口径输出百分数：前五类 mIoU5、六类 mIoU6、OA；子集缺类且交并比未定义时输出 null。
