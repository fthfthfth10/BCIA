# BCIA-PEFT

RGB–DSM 语义分割与机制实验。冻结 DINOv3 ViT-L/16，在第 18 个 Block 后插入 rank-64 适配器，复用 MMSegmentation Segmenter Head，输出 Potsdam 六类 logits。

- [架构与项目结构](architecture.md)
- [源码来源、版本及许可证](SOURCES.md)
- [验收结果与显存记录](TEST_REPORT.md)
- [权重审计与转换映射](WEIGHT_AUDIT.md)
- [G0–G4 数据、监督与恢复协议](EXPERIMENT_PROTOCOL.md)

## 本机运行

使用已验证的 Conda 环境，而不是默认 Python；无需安装或升级。依赖参见 [requirements.txt](requirements.txt)。

~~~powershell
conda activate msre
python -m unittest discover -s tests -v
python -m tests.smoke --device cuda
~~~

无 GPU 时可用 --device cpu。smoke 显式使用随机权重，不执行优化器更新。模型 API：BCIAPeft(checkpoint=...) 接收严格匹配 MsRE 接口的 backbone state dict；round-trip 及 smoke 可显式使用 BCIAPeft(random_init=True)。两种模式互斥，不会静默回退。RGB/DSM 必须已归一化；forward 返回 [B,6,32,32]，logits_for_tile 上采样到原切片大小。

## G0–G4 入口

在本机设置 BCIA_DATA_ROOT 和 BCIA_BACKBONE_CHECKPOINT，分别指向已审计切片目录和 BCIA 布局的 DINO 权重。以 configs/mechanisms.json 为模板创建受忽略的 configs/mechanisms.local.json。明确填写 group、steps、validate_every、save_every。G3/G4 需要显式填写两个正 MSE 系数。

~~~powershell
python -m bcia_peft.train --config configs/mechanisms.local.json --work-dir work_dirs/G2 --device cuda --describe
~~~

上面的命令只输出数据指纹、初始权重哈希和优化器参数组。正式实验使用明确的预算，移除 --describe 即开始运行。追加 --resume 从同一目录的 last.pth 恢复，配置、设备和输入指纹必须匹配。详见 [实验协议](EXPERIMENT_PROTOCOL.md)。

G2 诊断可使用 configs/g2_pilot.json，指定独立目录和 --pilot-steps 64。该上限保留原长程调度。正式验证以 mIoU5 选择 Best。固定批次的损失与辅助梯度诊断可用 python -m tests.g2_diagnostics 复现。

## 权重转换、验收和验证接口

~~~powershell
New-Item -ItemType Directory -Force checkpoints
$env:BCIA_BACKBONE_CHECKPOINT = "checkpoints/dinov3_vitl16_lvd1689m_bcia.pth"
python -m bcia_peft.convert_checkpoint --meta $env:BCIA_META_CHECKPOINT --msre $env:BCIA_MSRE_CHECKPOINT --output $env:BCIA_BACKBONE_CHECKPOINT
python -m tests.native_parity --meta $env:BCIA_META_CHECKPOINT --converted $env:BCIA_BACKBONE_CHECKPOINT
python -m tests.smoke --device cuda --checkpoint $env:BCIA_BACKBONE_CHECKPOINT
python -m bcia_peft.validation --config configs/potsdam.json --checkpoint $env:BCIA_BACKBONE_CHECKPOINT --data-root $env:BCIA_DATA_ROOT --device cuda --limit 1
~~~

在本机设置环境变量，或将路径写入被忽略的 configs/potsdam.local.json 并传给 --config；配置内相对路径以配置文件所在目录为基准。数据根目录应指向既有的 Potsdam 已验证切片清单。不要提交个人路径、数据或权重。

转换仅接受 [WEIGHT_AUDIT.md](WEIGHT_AUDIT.md) 指定的 Meta 原件，并逐张量核对既有 MsRE 权重。它只写不存在的输出文件，不覆盖原件；输出保留原末端 LayerNorm 和 RoPE 周期。旧权重和原生格式都不直接加载。验证 CLI 的 Segmenter 仍未训练，指标仅用于验证流程检查，不是模型性能。
