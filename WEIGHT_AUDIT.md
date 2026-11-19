# DINOv3 ViT-L/16 权重审计 — 2026-11-19

审计对象是 Meta LVD1689M、patch 16、24 Blocks、1024 维、16 heads、1 CLS + 4 storage tokens；不扩展到 SAT493M 等其他变体。参考固定在 [Meta DINOv3 6876159](https://github.com/facebookresearch/dinov3/commit/6876159a11b4df116f30f667f8c9888617df0751)，重点核对 [ViT-L 构造](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/hub/backbones.py#L316)、[注意力](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/layers/attention.py) 和 [RoPE](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/layers/rope_position_encoding.py)。不分发任何权重。

| 文件（本次审计） | SHA-256 | 结果 |
| --- | --- | --- |
| Meta 原件 dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth | 8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035 | native 368/368；所有张量有限，49 个 LayerNorm scale 均非全零 |
| MsRE 既有 dinov3-vitl16-pretrain-lvd1689m-hf.pth | 23bd8a1ae9ca54f39749af838a7c951d4363edb2edf90a9d050bc57b42966452 | 413 个键逐张量精确等于原始映射的公共部分；旧布局缺最终 LayerNorm 和 RoPE 周期 |
| 新本地 dinov3_vitl16_lvd1689m_bcia.pth | f7224cd45820d310d8ba9777b5c4379589d6d3e44d525596fff5805223bcba9e | 416/416，零缺失/额外/形状不符/dtype 不符/非有限；严格加载成功 |

## 逐项映射

| Native → BCIA | 校验与形状 |
| --- | --- |
| cls_token / storage_tokens → embeddings.cls_token / register_tokens | FP32 [1,1,1024] / [1,4,1024]，按值复制 |
| mask_token → embeddings.mask_token | FP32 [1,1024] → [1,1,1024] |
| patch_embed.proj → embeddings.patch_embeddings | FP32 [1024,3,16,16] weight 与 [1024] bias |
| blocks.i.norm1 / norm2 → layer.i.norm1 / norm2 | weight、bias 皆 FP32 [1024]，eps=1e-5 |
| attn.qkv → attention.q_proj / k_proj / v_proj | FP32 [3072,1024] 沿输出维切为三个 [1024,1024]；先按 native bias_mask 逐元素乘 FP32 [3072] bias，再切分；有效 K bias 必须全零才可省略 |
| attn.proj → attention.o_proj | FP32 [1024,1024] weight 与 [1024] bias |
| ls1 / ls2.gamma → layer_scale1 / layer_scale2.lambda1 | FP32 [1024]，原值复制 |
| mlp.fc1 / fc2 → mlp.up_proj / down_proj | FP32 [4096,1024] 和 [1024,4096]；bias 分别 [4096]、[1024] |
| norm.weight / norm.bias → norm.weight / norm.bias | FP32 [1024]；max abs 9.5311059952、2.6325578690，均来自原件 |
| rope_embed.periods → rope_embeddings.periods | 持久缓冲区 BF16 [16] → FP32 [16]（精确提升），用实际 period 做 FP32 除法 |

实测 24 层的 QKV bias 和 bias mask 均全零，未用假定的默认 mask 覆盖。BCIA 保存的是掩码作用后的 Q/V bias；K bias 为零才可忽略，不再需要 native mask。旧 RoPE 派生周期与原件最大相差 0.080349，不能用默认值。转换入口校验原件完整 SHA-256、全部源键及其 shape/dtype/有限性、公共 413 张量值和目标全部 416 键；独占创建新文件，不修改原件或旧转换文件。加载端严格要求周期等全部键，不使用缺省初始化兜底。

## 数值校验

在现有环境用原始权重按官方 eval 公式构造 CPU 参考：fused QKV 和 bias mask、空间 RoPE、SDPA、LayerScale、MLP、末端 LayerNorm；不导入原生 Python 包。2×2、32×32 的 RoPE 和 2×2 的 embeddings 逐元素一致。2×2 token 序列对比全部 24 个 Block（同输入）及整链路末端 patch：单层最大绝对误差 1.2207e-4（符合 atol=1e-4、rtol=2e-5），末端 2.5928e-6（符合 atol=3e-4、rtol=5e-5）。MsRE 分拆线性层和 eager attention 不要求与 fused/SDPA 逐位相同；另在 32×32 patch 网格运行严格预训练的 GPU 前反向。
