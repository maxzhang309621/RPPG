# experiments：实验研究程序

每一次基于基础模型的研究都是一个独立目录，从 `_template` 复制，编号递增：

```text
experiments/exp001_pos_pure/
  README.md       假设、使用的基线、数据、指标、结论
  config.yaml     只写这一次实验相对基线的差异
  train.py        需要训练时才有
  eval.py         评测入口
  outputs/        日志与表格，不进版本库
```

## 和 baselines 的关系

- `eval.py` / `train.py` 导入 `baselines/` 中的模型，不复制一份模型源码。
- 实验可以包一层新模块（融合头、质量门控、新损失），这个新模块只放在该实验目录内。
- 若连续多个实验共用同一改动，再把该改动提成新的基线目录，不在实验之间互相复制。

## 命名

`exp` + 三位序号 + 下划线 + 短名。短名用小写英文，标明基线和数据或问题，例如 `exp001_pos_pure`、`exp004_rgb_nir_fusion`。
