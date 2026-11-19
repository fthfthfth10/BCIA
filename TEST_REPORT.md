# 本地验收 — 2026-11-19

## G2 真实数据 Pilot

低值损失的独立复现和同 logits 归约比较见 [Pilot 标签审计](PILOT_LABEL_AUDIT.md)。64 批重放只执行推理。

- 起始本地和远端均为 882fbceddd3829035b8b51d32d3916f36d4cc19f。上一个请求使用的 64 步总预算和 8 步 warmup 已产生完整 64 条本地日志，目录保持隔离。以下结果来自修正配置的独立运行。
- Windows、Python 3.8.20、PyTorch 2.1.1+cu121、CUDA 12.1、RTX 3070 Ti、驱动 616.92、CPU 计算线程 4。FP32，关闭 TF32，确定性算法开启。使用已审计 Potsdam Eroded/noBoundary 训练切片及固定同步增强。
- G2 完成 64/64 步，seed=42、batch=1、workers=0、AdamW lr=1e-4、wd=0.05、warmup=1500、起始 LR 比例 0.1、PolyLR power=1、总调度预算 40,000、每 16 步保存 Last。辅助 MSE 关闭。64 条日志连续且样本 ID 唯一，逐步 LR 与公式一致。首步/末步实际 LR 为 1e-5 / 1.378e-5。分割损失首步 0.91692090、末步 1.27954471、范围 0.00099059–2.26320529。
- CUDA 同步后的 64 步训练计算和检查合计 13.808 s，中位数 0.1990 s/step，约 4.64 batch/s。显存峰值 allocated 2644.24 MiB、reserved 2802.00 MiB。运行中未触发 OOM、非有限值或冻结梯度检查。
- 首步 BCIA 输出投影 L2 梯度 0.31824347，上游各组件为零。第 2 步 RGB 投影、RGB→DSM、DSM→RGB、路由、混合和输出投影的梯度范数分别为 4.6036e-4、1.6582e-4、3.9339e-5、9.6632e-5、1.2091e-4、0.41551828，DSM stem 为 7.0212e-5。每步原始记录留在本地 work_dirs/pilot/g2_seed42_schedule40000_cap64/steps。
- Last 严格读取并以原命令恢复至 step=64、scheduler step=64，执行上限使这次恢复增加 **0** 个更新。DINO 全部状态与输入权重逐元素相等。DSM stem、BCIA、head 分别有 10,976、214,784、28,347,184 个状态元素与初始值不同。Last 为 1,555,633,214 字节，Best 为空，日志无验证指标。相同大小下正式运行的 Last + Best 约 2.90 GiB，更新交接时按至多约 5.80 GiB 预留。18 项单测通过，覆盖 mIoU5 选优、调度上限、Best 清理顺序和合成恢复。

### 固定真实 Batch 诊断

使用首批样本 4_10_0_3584_512_4096、相同初始权重与增强，平均有效覆盖率 0.48345947。模型以 eval 生成一次共享 logits。各路径 12 次计算分割损失及 logits 梯度，计入设备转移和同步。

| 路径 | 分割损失 | batch/s | 重复结果 |
| --- | ---: | ---: | --- |
| CPU | 0.8888877630 | 48.72 | 12 次逐位相同 |
| CUDA | 0.8888877630 | 33.95 | 12 次逐位相同，严格确定性模式通过 |

CPU/GPU 损失差为 0，logits 梯度最大绝对差 1.572e-9、相对 L2 差 4.768e-7。吞吐是单 batch 损失与反向子路径实测，正式入口继续使用 CPU 分割损失。

| 辅助方向 | G3/G4 MSE | G3 预测器梯度 L2 | G4 预测器梯度 L2 | G4 输入侧梯度 L2 |
| --- | ---: | ---: | ---: | --- |
| RGB→DSM | 4.33984804 | 18.60138 | 18.60138 | RGB 投影 31.30133 |
| DSM→RGB | 12.39626980 | 10.05000 | 10.05000 | DSM stem 9.75548 |

G3 的辅助梯度仅进入对应预测器。两方向的 MSE 在 G3/G4 逐元素一致。相同 batch 的纯分割梯度 L2 为 head 42.99179、输出投影 0.30896，上游为零。诊断使用未加权的单向损失，仅求梯度。

## G0–G4 机制框架

- 13 项自动测试覆盖组别初始化、组间初始 logits、主干与 G0 冻结、MSE 数值/梯度、无效区、新息屏蔽、同步缩放/裁剪/翻转、固定采样、预算校验以及恢复后的损失、学习率和权重逐元素一致性。恢复测试使用小型合成模型。
- RTX 3070 Ti 上以实际预训练 backbone、Segmenter 和合成输入逐组验证。G0–G4 初始 logits 逐元素一致，G2–G4 三个可训练模块的初始 SHA-256 完全相同。可训练参数分别为 28,347,404 / 28,572,652 / 28,573,168 / 28,573,168 / 28,573,168。
- 实际训练步函数在同一合成 batch 上对 G3/G4 各执行一次 CUDA AdamW 更新。两组分割损失均为 1.96950579，两个 MSE 分别为 3.95397806、9.74754524。G3 的 DSM stem 首步梯度为零，G4 非零，两组 DINO 梯度均为空。
- 只读核对训练/验证清单为 3,456/2,016 张切片，场景为 24/14。训练 DataLoader 读取一个 batch，RGB 形状为 [2,3,512,512]，有效掩码为 bool。

## 第二阶段

- 单测 **8/8 通过**。非零残差回归：固定 RGB 和有效掩码，单独修改被屏蔽的 DSM 值时所有注入残差逐元素不变；改变邻近有效 DSM 时，q=0 patch 的 RGB 权重和非零 RGB 残差逐元素不变，而邻近有效 patch 的残差确实变化。gH2R 输出在无效中心被屏蔽，uR=zR、路由的 DSM 通道为零；没有发现局部泄漏，未修改 BCIA 算子。后六层和 head 的全局注意力仍可传播其他区域的合法信息，不承诺最终 logits 的区域隔离。
- 非零输出投影时，受控残差反向可达 DSM stem、RGB 投影、两个预测器、路由器、融合层和输出投影，梯度均有限且各组至少有一项非零。预训练零投影 smoke 中，DINO 实际为 eval、requires_grad=False 且无梯度，其他模块为 train、requires_grad=True；head 和输出投影有非零梯度，BCIA 上游与 DSM stem 首步梯度为零。
- **预训练 GPU smoke PASS**：严格加载 416 个张量，以合成 batch 1 执行 FP32（无 autocast、TF32 关闭）分割前反向。初始恒等、CLS/Register 保持、覆盖率门控及 shape/dtype/device 检查均通过。loss **2.0023036003112793**，优化器更新 **0**。参数量见下方原表，未改变。
- RTX 3070 Ti：前向前 allocated **1282.73 MiB**，前反向峰值 allocated **2387.11 MiB**、reserved **2508.00 MiB**。测量方法与第一阶段相同。
- 原生公式对照通过：RoPE 2×2 / 32×32 逐元素相同，fused QKV/SDPA 逐层对照全部 24 层及最终归一化；最大最终 patch 误差 **2.59e-6**。权重来源、SHA-256 和具体映射见 [WEIGHT_AUDIT.md](WEIGHT_AUDIT.md)。
- 未运行真实全量验证、正式训练或超参数搜索；无优化器更新。

## 第一阶段结果（历史）

~~~text
test_checkpoint_gate ................................... ok
test_dsm_invalid_patch_and_partial_coverage ............ ok
test_model_requires_explicit_init ...................... ok
test_router_predictors_and_isolated_dsm_gate ........... ok
test_spatial_only_insertion_with_nonzero_residual ....... ok
test_validation_tile_loader_and_metrics ............... ok
Ran 6 tests — OK

GPU integration smoke: PASS (random synthetic only)
batch=1, RGB/DSM=FP32, valid=bool, TF32/autocast disabled
segmentation loss=1.925273060798645; optimizer updates=0
~~~

- 确认各阶段 shape/dtype/device、仅第 18 层之后插入一次、24 个 Block 均执行；零残差和启用/旁路的末端特征逐元素相等。单测另用非零残差核对 CLS/Register 原样保留。
- 无效/非有限 DSM 不传播 NaN/Inf；覆盖率 0、0.5、1 和 255/256 均正确，q=0 的 DSM 路由及融合输入严格为零，RGB 路由非零。两个预测器与路由器都收到非零输入；辅助 MSE 单独反向可到两个预测器。
- MMSeg 六类交叉熵一次反向：所有冻结参数 grad=None，输出投影及分割头有有限非零梯度，BCIA 上游和 DSM stem 有梯度张量但本步值为零。MMSeg 对 avg_non_ignore 默认值的提示不影响 ignore=255 的屏蔽；保留原 loss 默认设置。
- 数据：已有 val 清单共 2,016 张；索引 0、420（含 512 个无效边界像素）、2015 的 RGB/DSM/valid/标签与旧 loader 逐元素一致。合成切片通过验证指标接口，未运行真实全验证。

## 参数及显存

| 模块 | 参数 | 可训练 |
| --- | ---: | ---: |
| DINOv3 ViT-L/16 | 303,129,600 | 0 |
| DSM stem | 10,976 | 10,976 |
| BCIA | 214,788 | 214,788 |
| Segmenter Head | 28,347,404 | 28,347,404 |
| **总计** | **331,702,768** | **28,573,168** |

RTX 3070 Ti（8191.5 MiB，驱动 616.92，PyTorch CUDA 12.1）：前向前 allocated **1282.73 MiB**，前反向峰值 allocated **2387.11 MiB**、reserved **2508.00 MiB**。PyTorch allocator 统计，计入常驻模型，不是系统级显存；无优化器状态。

## 第一阶段只读 checkpoint 审计（历史）

| 文件 | SHA-256 | 检查结果 |
| --- | --- | --- |
| dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth | 8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035 | Meta layout 368/368，键、形状和 dtype 符合预期、全部有限；49 个 LayerNorm scales 均非全零。末端 weight/bias 均 FP32 [1024]，max abs 分别为 9.531106、2.632558。不是 MsRE 布局。 |
| dinov3-vitl16-pretrain-lvd1689m-hf.pth | 23bd8a1ae9ca54f39749af838a7c951d4363edb2edf90a9d050bc57b42966452 | MsRE layout 413/415，仅缺 norm.weight、norm.bias；无额外键、形状或 dtype 冲突，无非有限参数；已有 48 个 LayerNorm scales 均非全零。严格加载拒绝，审计 CLI 按预期退出 1。 |

没有修补、转换、下载或替换 checkpoint。未用预训练权重运行模型或真实全验证；未启动正式训练、超参数搜索或 40k 实验。
