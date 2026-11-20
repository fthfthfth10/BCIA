# B1 MoBaNet v3 预检

结论：正式训练未启动。官方 ViT-B/14 经作者的补齐规则处理后，在 RTX 3070 Ti 上仍无法完成 FP32 batch 1 的训练前向。失败点为 UPerNet 的 PPM 全局池化分支 BatchNorm。正式训练结果和验证 CSV 尚无数据。

## 来源与协议

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
