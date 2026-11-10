"""各 rPPG 模型共用的训练记录和曲线。

CSV 列覆盖论文和 rPPG-Toolbox 里常见的量：学习率、训练/验证损失、
心率 MAE、RMSE、误差标准差、MAPE、Pearson r，以及波形 SNR、MACC。
缺的列留空，出图时跳过。compare_runs 把多次训练画在同一张图上。
"""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np

FIELDS = (
    "model",
    "dataset",
    "phase",
    "epoch",
    "lr",
    "train_loss",
    "valid_loss",
    "test_mae",
    "test_rmse",
    "test_std",
    "test_mape",
    "test_pearson",
    "test_snr",
    "test_macc",
)

ALIASES = {
    "train_mse": "train_loss",
    "test_mae": "test_mae",
    "mae": "test_mae",
    "rmse": "test_rmse",
    "std": "test_std",
    "mape": "test_mape",
    "pearson": "test_pearson",
    "snr": "test_snr",
    "macc": "test_macc",
}

PANELS = (
    ("train_loss", "训练损失", "损失"),
    ("valid_loss", "验证损失", "损失"),
    ("lr", "学习率", "lr"),
    ("test_mae", "心率 MAE", "bpm"),
    ("test_rmse", "心率 RMSE", "bpm"),
    ("test_std", "心率误差 SD", "bpm"),
    ("test_mape", "心率 MAPE", "%"),
    ("test_pearson", "心率 Pearson r", "r"),
    ("test_snr", "波形 SNR", "dB"),
    ("test_macc", "波形 MACC", "相关"),
)

COMPARE_KEYS = (
    "test_mae",
    "test_rmse",
    "test_std",
    "test_mape",
    "test_pearson",
    "test_snr",
    "test_macc",
    "train_loss",
    "valid_loss",
)


def summarize_hr(pred, label, snr=None, macc=None):
    """由逐段心率（以及可选的逐段 SNR、MACC）汇总论文常用指标。"""
    pred = np.asarray(pred, dtype=np.float64)
    label = np.asarray(label, dtype=np.float64)
    mask = np.isfinite(pred) & np.isfinite(label)
    pred = pred[mask]
    label = label[mask]
    error = pred - label
    result = {
        "mae": None,
        "rmse": None,
        "std": None,
        "mape": None,
        "pearson": None,
        "snr": None,
        "macc": None,
        "n": int(len(error)),
    }
    if len(error):
        result["mae"] = float(np.mean(np.abs(error)))
        result["rmse"] = float(np.sqrt(np.mean(error ** 2)))
        result["std"] = float(np.std(error))
        usable = np.abs(label) > 1e-6
        if np.any(usable):
            result["mape"] = float(np.mean(np.abs(error[usable] / label[usable])) * 100.0)
    if len(error) >= 2 and np.std(pred) > 0 and np.std(label) > 0:
        result["pearson"] = float(np.corrcoef(pred, label)[0, 1])
    if snr is not None:
        values = np.asarray(snr, dtype=np.float64)
        values = values[np.isfinite(values)]
        if len(values):
            result["snr"] = float(np.mean(values))
    if macc is not None:
        values = np.asarray(macc, dtype=np.float64)
        values = values[np.isfinite(values)]
        if len(values):
            result["macc"] = float(np.mean(values))
    return result


def _blank(value):
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(number):
        return ""
    return number


class TrainRecorder:
    def __init__(self, path, model, dataset):
        self.path = path
        self.model = model
        self.dataset = dataset
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, FIELDS).writeheader()

    def add(self, epoch, phase="finetune", lr=None, train_loss=None, valid_loss=None, metrics=None):
        metrics = metrics or {}
        row = {name: "" for name in FIELDS}
        row.update({
            "model": self.model,
            "dataset": self.dataset,
            "phase": phase,
            "epoch": int(epoch),
            "lr": _blank(lr),
            "train_loss": _blank(train_loss),
            "valid_loss": _blank(valid_loss),
            "test_mae": _blank(metrics.get("mae")),
            "test_rmse": _blank(metrics.get("rmse")),
            "test_std": _blank(metrics.get("std")),
            "test_mape": _blank(metrics.get("mape")),
            "test_pearson": _blank(metrics.get("pearson")),
            "test_snr": _blank(metrics.get("snr")),
            "test_macc": _blank(metrics.get("macc")),
        })
        with open(self.path, "a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, FIELDS).writerow(row)

    def plot(self, image_path, title=None):
        plot_run(read_log(self.path), image_path, title or f"{self.model} / {self.dataset}")


def read_log(path, model=None, dataset=None):
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row = {name: "" for name in FIELDS}
            for key, value in raw.items():
                target = ALIASES.get(key, key)
                if target in row and row[target] == "":
                    row[target] = value
            if model and not row["model"]:
                row["model"] = model
            if dataset and not row["dataset"]:
                row["dataset"] = dataset
            row["epoch"] = int(float(row["epoch"])) if str(row["epoch"]).strip() else 0
            rows.append(row)
    return rows


def _series(rows, key, phase=None):
    xs, ys = [], []
    for row in rows:
        if phase is not None and row.get("phase") not in ("", phase):
            continue
        value = _blank(row.get(key))
        if value == "":
            continue
        xs.append(int(row["epoch"]))
        ys.append(value)
    return xs, ys


def _style():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_run(rows, image_path, title):
    _style()
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.2))
    axes = axes.ravel()
    for axis, (key, name, unit) in zip(axes, PANELS):
        if key == "train_loss":
            pre_x, pre_y = _series(rows, key, "pretrain")
            fine_x, fine_y = _series(rows, key, "finetune")
            other_x, other_y = _series([row for row in rows if row.get("phase") not in ("pretrain", "finetune")], key)
            if pre_x:
                axis.plot(pre_x, pre_y, color="#1f4e79", marker="o", markersize=2.5, label="预训练")
            if fine_x:
                axis.plot(fine_x, fine_y, color="#b85c38", marker="o", markersize=2.5, label="微调/训练")
            if other_x:
                axis.plot(other_x, other_y, color="#b85c38", marker="o", markersize=2.5)
            if pre_x or (pre_x and fine_x):
                axis.legend(fontsize=8)
            has = bool(pre_x or fine_x or other_x)
        else:
            phase = "finetune" if any(row.get("phase") == "finetune" for row in rows) else None
            xs, ys = _series(rows, key, phase)
            has = bool(xs)
            if has:
                axis.plot(xs, ys, color="#1f4e79", marker="o", markersize=2.5)
        axis.set_title(name, fontsize=11)
        axis.set_xlabel("epoch")
        axis.set_ylabel(unit)
        axis.grid(True, alpha=0.3)
        if not has:
            axis.text(0.5, 0.5, "无记录", ha="center", va="center", transform=axis.transAxes, color="#888888")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(image_path) or ".", exist_ok=True)
    fig.savefig(image_path, dpi=150)
    plt.close(fig)


def compare_runs(named_rows, image_path, title="训练结果对比"):
    """named_rows: [(标签, rows), ...]。只画各次训练都可能有的心率与损失曲线。"""
    _style()
    colors = ["#1f4e79", "#b85c38", "#2e7d32", "#6a1b9a", "#ef6c00", "#00838f"]
    present = []
    for key in COMPARE_KEYS:
        if any(_series(rows, key)[0] for _, rows in named_rows):
            present.append(key)
    if not present:
        return
    columns = 3
    rows_n = int(np.ceil(len(present) / columns))
    fig, axes = plt.subplots(rows_n, columns, figsize=(4.6 * columns, 3.3 * rows_n))
    axes = np.atleast_1d(axes).ravel()
    titles = {key: name for key, name, _unit in PANELS}
    units = {key: unit for key, _name, unit in PANELS}
    for axis, key in zip(axes, present):
        for index, (label, rows) in enumerate(named_rows):
            phase = "finetune" if key != "train_loss" and any(row.get("phase") == "finetune" for row in rows) else None
            if key == "train_loss" and any(row.get("phase") == "finetune" for row in rows):
                phase = "finetune"
            xs, ys = _series(rows, key, phase)
            if not xs:
                continue
            axis.plot(xs, ys, color=colors[index % len(colors)], marker="o", markersize=2.2, label=label)
        axis.set_title(titles[key])
        axis.set_xlabel("epoch")
        axis.set_ylabel("损失（各模型定义不同）" if key in ("train_loss", "valid_loss") else units[key])
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    for axis in axes[len(present):]:
        axis.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(image_path) or ".", exist_ok=True)
    fig.savefig(image_path, dpi=150)
    plt.close(fig)


def render_saved_logs(root):
    """用已经写好的训练 CSV 重新出单次曲线和对比图。"""
    jobs = (
        ("RADIANT", "UBFC", os.path.join(root, "RADIANT", "outputs", "ubfc_train_log.csv"),
         os.path.join(root, "RADIANT", "outputs", "ubfc_loss.png")),
        ("RADIANT", "PURE", os.path.join(root, "RADIANT", "outputs", "pure_train_log.csv"),
         os.path.join(root, "RADIANT", "outputs", "pure_loss.png")),
        ("CanonicalPhys", "UBFC", os.path.join(root, "CanonicalPhys", "outputs", "ubfc", "train_log.csv"),
         os.path.join(root, "CanonicalPhys", "outputs", "ubfc", "loss.png")),
        ("CanonicalPhys", "PURE", os.path.join(root, "CanonicalPhys", "outputs", "pure", "train_log.csv"),
         os.path.join(root, "CanonicalPhys", "outputs", "pure", "loss.png")),
    )
    named = []
    for model, dataset, csv_path, image_path in jobs:
        if not os.path.exists(csv_path):
            continue
        rows = read_log(csv_path, model=model, dataset=dataset)
        plot_run(rows, image_path, f"{model} / {dataset}")
        named.append((f"{model}-{dataset}", rows))
        print("wrote", image_path, flush=True)
    if named:
        compare_path = os.path.join(root, "compare", "training_metrics.png")
        compare_runs(named, compare_path)
        print("wrote", compare_path, flush=True)


if __name__ == "__main__":
    render_saved_logs(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
