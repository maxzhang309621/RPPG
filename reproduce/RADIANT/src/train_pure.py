"""用预处理后的 PURE h5 训练 RADIANT。

划分按受试者：01–07 训练，08–10 测试。同一人的 6 个任务不会同时出现在两边。
网络、损失、学习率和 UBFC 那次相同。每个任务单独计算 20 秒心率，再汇总。
"""

import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from train_log import TrainLog
from train_ubfc import RADIANT, WindowDataset, _run_epoch, _stack_subjects, evaluate, pretrain
from ubfc_signals import iter_subject_windows


def load_pure(root):
    names = sorted(name for name in os.listdir(root) if name.endswith(".h5"))
    sessions = []
    for name in names:
        subject_text, _session = os.path.splitext(name)[0].split("-")
        signals, pulses = iter_subject_windows(os.path.join(root, name))
        sessions.append({
            "id": name[:-3],
            "subject": int(subject_text),
            "signals": signals,
            "pulses": pulses,
        })
    return sessions


def split_by_subject(sessions, test_subjects):
    train, test = [], []
    for item in sessions:
        if item["subject"] in test_subjects:
            test.append(item)
        else:
            train.append(item)
    return train, test


def main():
    data = r"D:\maxzhang\python files\datasets\pure_h5"
    out = os.path.join(os.path.dirname(SRC), "outputs")
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sessions = load_pure(data)
    test_subjects = {8, 9, 10}
    train_sessions, test_sessions = split_by_subject(sessions, test_subjects)
    print("train subjects", sorted({item["subject"] for item in train_sessions}), flush=True)
    print("test sessions", [item["id"] for item in test_sessions], flush=True)
    print("windows", sum(len(item["signals"]) for item in train_sessions), flush=True)

    model = RADIANT().to(device)
    logger = TrainLog(os.path.join(out, "pure_train_log.csv"), model="RADIANT", dataset="PURE")
    pretrain(model, device, epochs=20, batch_size=32, n_samples=5000, logger=logger)
    signals, pulses = _stack_subjects(train_sessions)
    loader = DataLoader(WindowDataset(signals, pulses, augment=True), batch_size=4, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    best = None
    for epoch in range(50):
        loss = _run_epoch(model, loader, optimizer, device)
        metrics = evaluate(model, test_sessions, device)
        mae = metrics["mae"]
        logger.add(
            "finetune", epoch + 1, loss, mae, metrics["rmse"], metrics["std"],
            lr=optimizer.param_groups[0]["lr"], mape=metrics["mape"], pearson=metrics["pearson"],
            snr=metrics["snr"], macc=metrics["macc"],
        )
        print(
            f"finetune {epoch + 1}/50 mse {loss:.4f} test MAE {mae} RMSE {metrics['rmse']} std {metrics['std']}",
            flush=True,
        )
        if mae is not None and (best is None or mae < best):
            best = mae
            torch.save(model.state_dict(), os.path.join(out, "radiant_pure.pt"))
            with open(os.path.join(out, "pure_metrics.json"), "w", encoding="utf-8") as handle:
                metrics["best_mae"] = best
                metrics["test_subjects"] = sorted(test_subjects)
                json.dump(metrics, handle, indent=2)
    loss_png = os.path.join(out, "pure_loss.png")
    logger.plot(loss_png)
    print("best MAE", best, flush=True)
    print("wrote", loss_png, flush=True)


if __name__ == "__main__":
    main()
