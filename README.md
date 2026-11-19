# BCIA-PEFT

独立的最小 RGB–DSM 语义分割项目：冻结 DINOv3 ViT-L/16，在第 18 个 Block 后插入一次 rank-64 BCIA，输出 Potsdam 六类 logits。复用 MsRE 的 DINO 接口和 MMSegmentation Segmenter Head；不含历史网络、训练 Runner、优化器或训练入口。

- [架构与项目结构](architecture.md)
- [源码来源、版本及许可证](SOURCES.md)
- [验收结果与显存记录](TEST_REPORT.md)

## 本机运行

使用已验证的 Conda 环境，而不是默认 Python；无需安装或升级。依赖参见 [requirements.txt](requirements.txt)。

~~~powershell
conda activate msre
python -m unittest discover -s tests -v
python -m tests.smoke --device cuda
~~~

无 GPU 时可用 --device cpu。smoke 显式使用随机权重，不执行优化器更新。模型 API：BCIAPeft(checkpoint=...) 接收严格匹配 MsRE 接口的 backbone state dict；round-trip 及 smoke 可显式使用 BCIAPeft(random_init=True)。两种模式互斥，不会静默回退。RGB/DSM 必须已归一化；forward 返回 [B,6,32,32]，logits_for_tile 上采样到原切片大小。

## 权重审计和验证接口

~~~powershell
python -m bcia_peft.checkpoints --meta $env:BCIA_META_CHECKPOINT --msre $env:BCIA_MSRE_CHECKPOINT
python -m bcia_peft.validation --config configs/potsdam.json --checkpoint $env:BCIA_BACKBONE_CHECKPOINT --data-root $env:BCIA_DATA_ROOT --device cuda --limit 1
~~~

在本机设置环境变量，或将路径写入被忽略的 configs/potsdam.local.json 并传给 --config；配置内相对路径以配置文件所在目录为基准。数据根目录应指向既有的 Potsdam 已验证切片清单。不要提交个人路径、数据或权重。

**当前预训练验证阻塞**：既有 MsRE 转换权重缺少末端 norm.weight / norm.bias；原 Meta 文件是不同键布局。两者均不直接加载到此 MsRE 接口。本项目不转换、不修补、不替换已有权重。验证 CLI 即使使用兼容 backbone，Segmenter 仍未训练，指标仅用于验证流程检查，不是模型性能。
