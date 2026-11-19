# Potsdam v3 正式数据验收

从 f1567d34c31a7f1d3a1a25dfd06b3e83b0288ef9 制备。v3 同级复用 v2 相对路径资产，包括 4_12 修复。v1 和 v2 保留。v3 只写 6_7 的 144 张标签、清单、归一化副本及元数据。

- 用参与者 Eroded 成员 SHA-256 7bd08018f36a1836c0de7ecb256f7fd677a1ba1e7a27440fdcbad64ce80e383e 精确解码七色，6000×6000。参与者 Full SHA-256 522a36d1196f041bb8d260e0c189f459639e590ad60659e70c81ca4829184670，全部 33,625,472 个候选有效像素原位匹配。共同有效的旧/新 33,441,748 个像素类别一致。
- 旧 TIFF SHA-256 efacadba7e790d9cf2b84bdb679f98a9e5f4deccef37817bebeccf23fcf12e25。183,600 个 (252,255,0) 旧像素按历史转换为 ignore。旧转新差异 183,776 个源像素，其中 183,724 个 ignore 变为低植被 27、树 44、车 183,653。52 个原有效像素变为 ignore，包括不透水面 2、低植被 43、树 7。
- 独立 verify 通过全部 5,472 条记录的坐标、标签精确裁切、RGB/DSM/valid/标签指纹、尺寸、类型与归一化。24/14 场景无交叉，3456/2016 样本，均无全忽略片。训练有效率 min/P50 为 82.2773%/92.9863%，验证为 83.4686%/92.6884%。训练切片累加的有效像素增加 209,189，验证六类与 ignore 计数与 v2 逐项相等。
- v3 SHA-256。Train manifest 378dda5f53bc4a0c45991f0eb96677c3208d5532e3a85aacf670cf691cd25bf6。Val manifest 3b4e7819b542eff1f584b39ba96b33f63cc4a71df2c69c39f14b71501595811a。Normalization 02da0e6cec2ffbe240fed78eb1bc0f70484147e8da8be7a73fd9b9ad8dcd1c3b。
- 26 项单测通过。RTX 3070 Ti 上三个 describe 输出 Head 初始指纹一致，G1/G2 DSM stem 初始指纹一致，数据、backbone、调度及设备身份通过交叉比较。预训练权重 SHA-256 f7224cd45820d310d8ba9777b5c4379589d6d3e44d525596fff5805223bcba9e。每组使用独立子目录。预检 E 盘空闲约 1.99 TB，GPU 空闲 7343 MiB。

使用 python -m potsdam_v3 build 或 verify，传入 --v2、--v3、--eroded-archive、--full-archive 和 --legacy-tiff。原始运行证据保存在本地忽略目录。
