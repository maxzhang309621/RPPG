# baselines：基础模型代码

这里存放对照论文实现的基础模型。实验程序只调用它们，不在这里改研究想法。

| 路径 | 内容 |
|------|------|
| `classical/iphys-toolbox` | MATLAB 传统基线：`POS_WANG.m`、`CHROM_DEHAAN.m`、`GREEN_VERKRUYSSE.m`、`ICA_POH.m` |
| `rPPG-Toolbox` | Python 统一工具箱。传统方法在 `unsupervised_methods/methods/`，网络在 `neural_methods/model/` |
| `neural/PhysNet` | 作者原版，`PhysNetED_BMVC.py`、`NegPearsonLoss.py` |
| `neural/PhysFormer` | 作者原版，`model/Physformer.py` |
| `neural/RhythmMamba` | 作者原版，`neural_methods/model/RhythmMamba.py` |

来源与提交号见 `SOURCES.md`。

## 修改规则

- 与论文不一致的实现错误，在对应仓库里修正，并在该模型目录记一笔。上游仓库优先保持可拉取，研究改动不要直接堆在上游文件上。
- 新损失、新融合、新数据划分，写到 `experiments/`，通过导入这里的模型完成。
- 不把某一次实验的配置、日志、人脸数据放进这里。
