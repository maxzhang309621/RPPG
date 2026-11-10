# 传统基线

`iphys-toolbox` 是 Daniel McDuff 的 MATLAB 实现，入口与论文对应关系：

| 文件 | 方法 |
|------|------|
| `GREEN_VERKRUYSSE.m` | 绿色通道空间平均 |
| `ICA_POH.m` | Poh 的 ICA |
| `CHROM_DEHAAN.m` | CHROM |
| `POS_WANG.m` | POS |

Python 版在 `baselines/rPPG-Toolbox/unsupervised_methods/methods/`：`GREEN.py`、`ICA_POH.py`、`CHROME_DEHAAN.py`、`POS_WANG.py`。无监督评测配置示例为该仓库的 `configs/infer_configs/UBFC-rPPG_UNSUPERVISED.yaml`。
