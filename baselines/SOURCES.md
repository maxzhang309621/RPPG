# 基础模型来源

本机直连 `github.com:443` 失败。下列仓库经公开代理从 GitHub 原仓库浅克隆，远程地址已改回官方仓库。提交号是下载当时的 `HEAD`。

| 本地路径 | 官方仓库 | 提交 | 内容 |
|----------|----------|------|------|
| `baselines/classical/iphys-toolbox` | https://github.com/danmcduff/iphys-toolbox | `65deeb4` | MATLAB：POS、CHROM、GREEN、ICA |
| `baselines/rPPG-Toolbox` | https://github.com/ubicomplab/rPPG-Toolbox | `b7500b8` | Python 统一实现：传统方法与 DeepPhys、PhysNet、PhysFormer 等 |
| `baselines/neural/PhysNet` | https://github.com/ZitongYu/PhysNet | `e78d387` | 作者原版网络与负 Pearson 损失 |
| `baselines/neural/PhysFormer` | https://github.com/ZitongYu/PhysFormer | `094c694` | 作者原版训练与推理脚本 |
| `baselines/neural/RhythmMamba` | https://github.com/zizheng-guo/RhythmMamba | `1533ad2` | AAAI 2025 作者代码，含其改过的工具箱 |

论文 PDF 在 `papers/methods/`，索引见 `papers/catalog.md`。CHROM 2013 年 IEEE 原文没有公开全文，算法实现见 `iphys-toolbox/CHROM_DEHAAN.m` 与 `rPPG-Toolbox/unsupervised_methods/methods/CHROME_DEHAAN.py`。
