# exp002 阶段 1：脸和手掌分开的输出头

共用 PhysNet 的前面几层。脸和手掌各有自己的后几层，训练时互不可见。

手掌片段的中心是真脉搏，整幅还有一个亮度闪烁。这一步检查三件事：两个输出头的参数互不改动，手掌头的损失下降，手掌头和 POS 都读到中心脉搏的心率。

合成片段里脉搏仍然够强，脸上的输出头也会读到同一个心率。这不能代替真实手掌上「适配器好过 POS」的验收。Face-Hand 视频放到 `data/` 之后再报那一项。

```text
C:\Users\max\.conda\envs\gs310\python.exe experiments\exp002_site_adapter\eval.py
```
