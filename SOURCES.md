# 源码、Git SHA 与许可证

| 来源 | 固定 GitHub commit | 用途 | 许可证 |
| --- | --- | --- | --- |
| fthfthfth10/msre | [75d235a2435fe36f124780f7bf5d1be25df0ca5b](https://github.com/fthfthfth10/msre/commit/75d235a2435fe36f124780f7bf5d1be25df0ca5b) | 复用现有 DINOv3 Python 接口，移植 DSM encoder 卷积 stem、验证预处理子集 | [Apache-2.0](https://github.com/fthfthfth10/msre/blob/75d235a2435fe36f124780f7bf5d1be25df0ca5b/LICENSE)，文本见 [LICENSE](LICENSE) |
| open-mmlab/mmsegmentation | [b040e147adfa027bbc071b624bedf0ae84dfc922](https://github.com/open-mmlab/mmsegmentation/commit/b040e147adfa027bbc071b624bedf0ae84dfc922) | 复用本机 1.2.2 editable 实例；直接导入 SegmenterMaskTransformerHead、PotsdamDataset.METAINFO、IoUMetric，不复制源码 | [Apache-2.0](https://github.com/open-mmlab/mmsegmentation/blob/b040e147adfa027bbc071b624bedf0ae84dfc922/LICENSE) |
| facebookresearch/dinov3 | [6876159a11b4df116f30f667f8c9888617df0751](https://github.com/facebookresearch/dinov3/commit/6876159a11b4df116f30f667f8c9888617df0751) | 核对 ViT-L/16、CLS/storage token 与原始权重格式；不 vendoring Meta 仓库、不装原版包 | [Meta DINOv3 License](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/LICENSE.md)；权重不随项目分发 |
| huggingface/transformers | [47b0e478f324b54f177ea7998a0791870fdd0324](https://github.com/huggingface/transformers/commit/47b0e478f324b54f177ea7998a0791870fdd0324) | MsRE vendored DINO 接口本身注明的上游；未安装 Transformers | [Apache-2.0](https://github.com/huggingface/transformers/blob/47b0e478f324b54f177ea7998a0791870fdd0324/LICENSE)；原实现标明 Meta AI 与 The HuggingFace Inc. team (2025) |

## 迁移边界

- MsRE 本机 HEAD：7b159fb3d084289a2c60ec5bdfb23c0dd79949ff。独立项目中的 _dinov3.py 来自本机运行版本 msre/models/backbone/dinov3_vision_transfromer.py（blob 709437cc1b38abc2c58bd2c838cd11b9ce6d8540；相对本机 HEAD 增加 Python 3.8 延迟注解和空行；GitHub 快照同文件带延迟注解，blob 7708a2f574d184630e3f901de199e0bc04615024）。迁移仅整理行末空白；在独立 model.py 组合调用已有 embeddings、RoPE、Block 和 norm，不改 DINO 算子。
- msre/m1/model.py 的 LightweightDSMEncoder 仅迁移 2→16→32→64 卷积序列（blob 3ab58c239b7e203fb19b06c97aad717aed98c6f7）；未迁移旧 projection、网络或 Runner。msre/datasets/potsdam_rgb_dsm.py 仅适配确定性验证部分（blob 252d1d638e99615b493c47d4ee02bddf2f5c9c66）；训练增强与几何同步不在本轮范围。两个 blob 均核对到上述 GitHub 快照。
- Meta 和 MsRE 都有 ViT-L/16 的 24 层、16 个注意力头、1024 维及 4 个附加 tokens，但 Meta 原生实现使用 native 键、fused QKV、RoPE periods；MsRE 使用独立 Q/K/V 和派生 RoPE 频率。Meta 上游 Conda 文件以 Python 3.11 为基准，不将其替换进现有 Python 3.8 环境；未声明 MsRE 与原生实现数值等价。
- 本机 Python 3.8.20 / Windows、torch 2.1.1+cu121、torchvision 0.16.1+cu121、MMCV 2.1.0、MMEngine 0.9.1；本地 MMSegmentation 源码 HEAD 与上述固定 SHA 一致，导入和 GPU forward/backward 通过。[requirements.txt](requirements.txt) 记录本项目主要及已核对的运行依赖；不要求在现有环境重装。pip check 无破损依赖。
