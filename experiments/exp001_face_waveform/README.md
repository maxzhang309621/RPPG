# exp001 阶段 0：脸上的一条波形

本机没有 PURE / UBFC 视频。这一步用一段已知心率的合成肤色图像，确认两件基线还能用。

- POS 来自 `baselines/rPPG-Toolbox`，不复制源码。心率应离设定值不超过 2 bpm。
- PhysNet 来自同一工具箱。在这一段上做短训练，负相关损失要下降，输出长度等于帧数。
- 不估计手掌，不估计传播时间。

```text
python experiments/exp001_face_waveform/eval.py
```
