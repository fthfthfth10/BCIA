# B1 MoBaNet v3

## 正式训练结果

训练提交 617ca8f6b6a287d6977e1ae624b0005c8d4dfce1。RTX 3070 Ti 上完成 40,000 步、20 次全量验证。训练阶段含验证、日志和 checkpoint 写入的时间为 21654.55 秒。峰值 allocated / reserved 为 1774.94 / 1998.00 MiB。RGB / DSM 遮挡命中分别 5045 / 4992 次，总触发比例 25.093%。

| Best 步数 | mIoU5 | mIoU6 | OA | 不透水面 | 建筑 | 低植被 | 树 | 车 | 杂类 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40000 | 79.8609 | 72.0384 | 88.4879 | 83.6990 | 92.7180 | 73.8026 | 73.3847 | 75.7002 | 32.9260 |

Best 与 Last 均严格加载，共享 DINOv2 张量与原权重逐项一致。Best 在相同 2,016 个窗口复算的指标与原验证记录一致。文件 SHA-256。

- last.pth 4c69bdb844781e2d9bfd32bc55ccc71f312d444b854749fdffd139706075262c
- best-00040000.pth 4c69bdb844781e2d9bfd32bc55ccc71f312d444b854749fdffd139706075262c

[20 次验证及对应训练记录](BASELINE_MOBANET_V3.csv)。逐步 CE、辅助损失、学习率及 checkpoint 保存在本地。

## Batch 1 续接协议与验收

基准提交 5966b1d3a6db3f7cdfc641de382db6b39603f01b。正式入口使用独立适配层、已核验的本地官方 MoBaNet 树和原 Meta 权重。仅 PPM 1×1 池化分支的 BatchNorm 固定使用 eval 统计，保留 affine 梯度，其他 BN 正常训练。MCRC 每样本独立 Bernoulli(0.25)。命中后等概率选择一个模态，调用原官方局部矩形遮挡函数（1 块、面积比例 0.1–0.3、长宽比 0.5–2）。推理无 MCRC。主分支用有效像素平均 CE，两个辅助头只对主分支误分的非忽略像素各算 CE，总权重 0.01。空监督集为零。按原始 512 窗口计算损失及验证，RGB/DSM 右下补零到 518，输出裁回 512，DSM 的无效或非有限像素在输入处置零。

冻结配置为 configs/mobanet_b1.json，Potsdam v3 的 3,456 / 2,016 个窗口、seed 42、batch 1、workers 0、FP32、AdamW lr=1e-4、wd=0.05、1,500 步 warmup 且起始比例 0.1、40,000 步多项式衰减 power=1、每 2,000 步验证与保存、mIoU5 选 Best、无 TTA。保留官方核心网络和专用辅助损失。

Python 3.8 无法导入官方 PEP 604 类型注解。续接环境为 Windows、Python 3.10.22、PyTorch 2.1.0+cu118、torchvision 0.16.0+cu118、MMSeg 1.2.2、MMCV 2.1.0、MMEngine 0.10.7。本地额外依赖为 einops 0.8.1、timm 0.9.16、safetensors 0.5.3。严格 CUDA 算子确定性模式拒绝官方 PPM adaptive pooling 反向。因此 B1 独立设定 cuDNN deterministic=True、关闭 benchmark 及 TF32，并将全局非确定算子检查设为 warn_only，保留官方算子。CUDA adaptive pooling 和 memory-efficient attention 反向存在非确定性警告，固定 seed 不保证训练逐位重现。这是 B1 与 BCIA 的运行时差异。

真实首批次 4_10_0_3584_512_4096 预检通过，AdamW 实际更新一次。主 CE 1.82354248，RGB/DSM 辅助 CE 1.80981970 / 1.85582483，总损失 1.86019897。该步未选中 MCRC。冻结主干无梯度，1×1 BN 的 running mean/var 不变且 affine 两参数梯度非零。峰值 allocated 1,805,293,056 字节，reserved 2,080,374,784 字节，含创建 AdamW 状态。Last 加载成功，恢复第 1 步的模型、AdamW、RNG 和调度器。单窗验证输出六类。33 项测试通过，涵盖全局 BN 训练反向、保留其他 BN 模式、MCRC 概率和 seed 回放、忽略标签、补齐与裁切。

运行方式为 python -m mobanet_b1，设置 BCIA_DATA_ROOT、BCIA_MOBANET_SOURCE、BCIA_MOBANET_WEIGHTS，传入 --work-dir；预检附 --preflight，正式运行附 --commit 和已推送的训练提交 SHA。官方源码、权重依赖和输出均保留本地。

## 原始预检与来源

- BCIA 起始提交 df30bfeab551c59d6e952a08409650f631b80041。本机 MoBaNet 快照无 .git，无法归属到确定的提交。对比发现其 Net.py 增加了 DINOv3 支持，DINOv2 构建文件也有本地差异。本机与官方 hubconf.py、patch_embed.py 的 blob 相同。预检使用原样官方树 77b59b9ff2763ddc8d07065bb7547f23d292d391，在本地忽略目录运行，未迁入第三方源码。该树未找到 LICENSE、COPYING 或 NOTICE，论文的 arXiv 条款也没有为仓库源码指定许可证。[官方提交](https://github.com/sauryeo/MoBaNet/commit/77b59b9ff2763ddc8d07065bb7547f23d292d391)
- 无 register 的蒸馏 DINOv2 ViT-B/14 backbone 权重来自 Meta，SHA-256 为 0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73。175 个张量的键和值逐项相同，严格加载报告零缺失、额外键或形状丢弃。Meta 为常规 DINOv2 源码与权重注明 Apache-2.0。[预训练来源](https://github.com/facebookresearch/dinov2#pretrained-models)
- 保留官方 CPIA、DGFM、UPerNet 和两个 MCRC 辅助头。RGB 和 DSM 两路共享 DINOv2，DSM 先广播到三通道并使用由 RGB patch embed 初始化的独立可训练 patch embed。零基抽取层为 2、5、8、11。核对了共享主干全部参数冻结。DSM patch embed、CPIA、DGFM、融合归一化、投影和各解码头参与训练。参数总量 92,759,936，可训练 6,179,456。[官方模型](https://github.com/sauryeo/MoBaNet/blob/77b59b9ff2763ddc8d07065bb7547f23d292d391/Net.py)
- 论文称训练遮挡机制为 MCRM，发布代码称 MCRC。官方默认启用，ratio=0.25，两个辅助头在主预测误分的像素上计算 CE，原 ignore 像素保留忽略。总损失为主 CE 加 0.01 倍两个辅助 CE 之和。batch 1 时遮挡选择数 int(1×0.25)=0，实测两个模态张量未变化。若进入训练损失计算，辅助头和困难像素监督仍启用。[论文训练目标](https://arxiv.org/html/2603.17705v1#S3.SS4) · [官方训练代码](https://github.com/sauryeo/MoBaNet/blob/77b59b9ff2763ddc8d07065bb7547f23d292d391/train.py)
- v3 训练和验证清单指纹分别为 378dda5f53bc4a0c45991f0eb96677c3208d5532e3a85aacf670cf691cd25bf6 和 3b4e7819b542eff1f584b39ba96b33f63cc4a71df2c69c39f14b71501595811a，与冻结协议一致。归一化 SHA-256 为 02da0e6cec2ffbe240fed78eb1bc0f70484147e8da8be7a73fd9b9ad8dcd1c3b。源码类别顺序与 BCIA 对应：不透水面、建筑、低植被、树、车、杂类，未定义类映射到 ignore=255。[官方类别定义](https://github.com/sauryeo/MoBaNet/blob/77b59b9ff2763ddc8d07065bb7547f23d292d391/utils.py)

计划中的 v3 baseline 使用 24/14 场景和 3,456/2,016 窗口、v3 同步增强和 DSM 统计、有效像素主 CE、seed 42、batch 1、FP32、40,000 步、每 2,000 步完整验证、mIoU5 选 Best、无 TTA。BCIA AdamW 和 PolyLR 参数沿用正式协议。与作者论文不同的项目包括 v3 切片及标签制备、512 窗口、DSM 归一化、训练预算和调度。论文使用 256 裁剪、ViT-B batch 12、AdamW lr=3e-4 和 wd=0.01、50×1,000 iterations、5 epoch warmup 加 cosine、逐图 DSM min–max。该计划尚未通过预检，未形成已执行的训练协议。[论文实现细节](https://arxiv.org/html/2603.17705v1#S4.SS2) · [BCIA 正式协议](EXPERIMENT_PROTOCOL.md)

## 实测与停止条件

| 检查 | 结果 |
| --- | --- |
| 原版训练模式直接使用 512×512 | PatchEmbed 断言失败，512 非 14 的整数倍 |
| 作者式右下补零至 518×518，eval 后裁回原窗口 | 官方权重输出 [1,6,512,512] FP32，有穷。峰值 allocated 482.32 MiB、reserved 534 MiB，仅为无梯度推理 |
| 同样的补齐与权重，train 模式 | UPerNet PPM 的 1×1 池化分支送入 BatchNorm2d 的形状为 [1,64,1,1]，报 Expected more than 1 value per channel when training |
| 官方 MCRC，batch 1，ratio=0.25 | RGB 和 DSM 均未发生遮挡 |
| 反向、优化器更新、训练 checkpoint 与正式评估 | 在训练前向失败处停止，未得到这些验收结果 |

环境为 Windows、Python 3.11.11、PyTorch 2.7.1+cu118、RTX 3070 Ti 8GB、CUDA 11.8。关闭 TF32，使用官方仓库自带 DINOv2 hub。已通过原模型训练模式复现结构错误，未更改归一化层、MCRC 采样、批量、精度、超参数或骨干。由于反向未运行，推理显存不能用于判定训练显存能否满足 8GB。

复现入口为 python -m tests.mobanet_b1_preflight，传入 --source 和 --weights。该入口严格加载、检查冻结并验证已知的 batch 1 错误。30 项 BCIA 单测通过，其中新增测试覆盖补齐、原窗口保留和无效 DSM 置零。
