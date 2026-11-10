# 神经基线

| 目录 | 论文编号 | 入口 |
|------|----------|------|
| `PhysNet` | M03 | `PhysNetED_BMVC.py` |
| `PhysFormer` | M04 | `model/Physformer.py`，训练脚本 `train_Physformer_160_VIPL.py` |
| `RhythmMamba` | M06 | `neural_methods/model/RhythmMamba.py`，用 `configs/train_configs` 调用 `main.py` |

DeepPhys、TS-CAN、EfficientPhys、PhysMamba、RhythmFormer 的 Python 实现在 `baselines/rPPG-Toolbox/neural_methods/model/`，不另拷一份。DeepPhys 没有单独的作者 Python 仓库，原实现在 MATLAB 工具箱之外，以 rPPG-Toolbox 这份为准。
