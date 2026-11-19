# 本地验收 — 2026-11-19

## 结果

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

## 只读 checkpoint 审计

| 文件 | SHA-256 | 检查结果 |
| --- | --- | --- |
| dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth | 8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035 | Meta layout 368/368，键、形状和 dtype 符合预期、全部有限；49 个 LayerNorm scales 均非全零。末端 weight/bias 均 FP32 [1024]，max abs 分别为 9.531106、2.632558。不是 MsRE 布局。 |
| dinov3-vitl16-pretrain-lvd1689m-hf.pth | 23bd8a1ae9ca54f39749af838a7c951d4363edb2edf90a9d050bc57b42966452 | MsRE layout 413/415，仅缺 norm.weight、norm.bias；无额外键、形状或 dtype 冲突，无非有限参数；已有 48 个 LayerNorm scales 均非全零。严格加载拒绝，审计 CLI 按预期退出 1。 |

没有修补、转换、下载或替换 checkpoint。未用预训练权重运行模型或真实全验证；未启动正式训练、超参数搜索或 40k 实验。
