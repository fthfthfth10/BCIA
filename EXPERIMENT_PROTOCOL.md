# G0–G4 机制实验协议

| 组 | 分割路径和辅助监督 | 优化器组及可训练参数 |
| --- | --- | --- |
| G0 | RGB-only Segmenter | Head 28,347,404 |
| G1 | RGB+DSM 低秩拼接融合 | Stem 10,976 + Fusion 214,272 + Head 28,347,404 = 28,572,652 |
| G2 | BCIA，分割监督 | Stem 10,976 + BCIA 214,788 + Head 28,347,404 = 28,573,168 |
| G3 | BCIA，共享预测器的辅助输入和目标 detach | 同 G2 |
| G4 | BCIA，直接用分割前向预测做辅助 MSE，目标 detach | 同 G2 |

每组冻结 DINO 的 303,129,600 个参数。G1 的适配器比 BCIA 少 516 个参数，对应 BCIA 路由器。G1 和 BCIA 都使用 rank 64、两次 3×3 卷积、128→64 混合、零初始化输出投影和同一个 DSM stem。G1 的空间卷积分别处理各自模态。两种融合都将无效中心的 DSM 分支置零。

相同 seed 下五组共享同一初始 head 和 DSM stem。G2–G4 的完整初始权重一致。初始化后重新设定训练随机状态，使 G1 专有层的随机初始化不改变后续随机序列。保持跨组的输入文件指纹、seed、batch、调度参数和验证间隔一致。G3/G4 使用相同的两个显式 MSE 系数。G2 阶段无辅助损失，诊断时可离线计算其前向 MSE。

## 数据

使用 MsRE 已制备的固定 24/14 场景、3,456/2,016 个 512×512 Eroded/noBoundary 切片。训练数据沿用 [MsRE 同步几何](https://github.com/fthfthfth10/msre/blob/75d235a2435fe36f124780f7bf5d1be25df0ca5b/msre/datasets/potsdam_rgb_dsm.py) 的 0.5–2.0 缩放、0.75 类别占比约束裁剪、0.5 水平翻转及 RGB 光度扰动。标签最近邻插值。DSM 按有效双线性支持插值，只有全部双线性支持有效的像素保留有效标记。训练集既有 mean/std 给 DSM 做 z-score。归一化后右下补齐 512，RGB/DSM/valid/label 分别填 0/0/false/255。验证读取原切片，按完整集 mIoU5 选择最佳。

## 复现与恢复

- configs/mechanisms.json 的 steps、validate_every、save_every 必须显式填写。MSE 两个方向分别配置，模板默认 0。调度按完成的优化器步数进行线性 warmup 与多项式衰减。
- 每个 epoch 独立生成固定 permutation。每个采样位置携带确定的 NumPy 增强种子。尾部不足一个 batch 的样本本 epoch 丢弃。恢复直接从已完成的 step 定位同一批索引和增强种子。
- 使用单进程训练、同设备与同软件栈。启用 PyTorch 确定性算法并关闭 TF32 和 cuDNN benchmark。分割 logits 上采样及 MMSeg 损失在 CPU 计算以保留确定性反传，梯度传回模型所在设备。
- run.json 和 --describe 报告各组的参数量、初始 tensor SHA-256、backbone 文件哈希、设备、配置、场景和清单/归一化文件哈希。每步 JSON 记录样本 ID、实际学习率、分割和辅助损失。验证时追加六类指标。
- last.pth 原子替换，包含模型、AdamW、调度器、已完成步数、最佳记录、Python/NumPy/Torch/CUDA RNG 和运行身份。仍按完整验证集 mIoU5 选择 Best，继续报告 mIoU6、OA 和六类 IoU。并列保留首次。发布顺序为新 best-步数.pth、指向它的 last.pth、删除上一个 Best。中断可能留下额外快照，清理时应先读取 last.pth 的 best 字段并保留所指文件。正常保留 Last 和当前 Best，发布期间为两个 Best、Last 和临时文件预留空间。恢复以 last.pth 为准，其后的步骤日志可被重跑覆盖。

## G2 Pilot

使用 configs/g2_pilot.json 和独立 work-dir，传入 --pilot-steps 64。上限计入恢复前已完成的步骤，调度仍由 steps=40000 和 warmup_steps=1500 计算。Pilot 记录每次优化的样本、分割损失、学习率、各模块梯度范数、同步设备后的耗时和显存峰值。耗时覆盖训练计算及诊断检查，排除数据加载、日志和 checkpoint 写入。Pilot 跳过验证与 Best 选择。运行身份绑定模式和执行上限，恢复使用相同模式、配置及目录。已有缺少模式字段的 checkpoint 会被身份校验拒绝。
