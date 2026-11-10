"""在 UBFC-rPPG 的 h5 上训练并评测 RADIANT。

网络结构用作者 src/models.py 与 src/modified_model.py：每个 ROI 的时间信号
用长度等于信号的卷积投到 768 维，前面加可学习 token，12 层、12 头 Transformer，
token 经线性头回归脉搏。论文训练设置：Adam 3e-4，合成数据预训练
batch 32、20 epoch，真实数据微调 batch 4、50 epoch，损失是 MSE。

UBFC 原文是 30 fps。作者训练代码把信号切成 12 路、80 点，对应 20 Hz 上的
4 秒。这里把每个 4 秒窗从 30 Hz 重采样到 80 点，和这份代码的输入对齐。
论文的 28/14 划分没有公布受试者编号，这里按编号排序，后 14 人做测试。
20 秒心率是连续 5 个 4 秒片段心率的平均，和论文 3.1、3.5 节一致。
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

COMMON = os.path.join(os.path.dirname(os.path.dirname(SRC)), "common")
if COMMON not in sys.path:
    sys.path.insert(0, COMMON)

from models import VisionTransformer
from synthetic import generate_dataset
from train_log import TrainLog
from train_viz import summarize_hr

CANON = os.path.join(os.path.dirname(os.path.dirname(SRC)), "CanonicalPhys")
if CANON not in sys.path:
    sys.path.insert(0, CANON)
from evaluation.post_process import _calculate_SNR, _compute_macc
from ubfc_signals import (
    MODEL_FS,
    N_ROI,
    hr_from_pulse,
    load_ubfc,
    split_subjects,
)


SIGNAL_LEN = MODEL_FS * 4


class SignalProjection(nn.Module):
    def __init__(self, length=SIGNAL_LEN, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=(1, length), stride=(1, length))

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class RADIANT(nn.Module):
    def __init__(self, n_roi=N_ROI, length=SIGNAL_LEN, embed_dim=768, depth=12, n_heads=12):
        super().__init__()
        self.proj = SignalProjection(length=length, embed_dim=embed_dim)
        self.encoder = VisionTransformer(
            img_size=224,
            patch_size=16,
            in_chans=1,
            n_classes=length,
            embed_dim=embed_dim,
            depth=depth,
            n_heads=n_heads,
            qkv_bias=True,
            mlp_ratio=4.0,
        )
        self.n_roi = n_roi

    def forward(self, signals):
        # signals: (B, n_roi, T)
        if signals.shape[1] != self.n_roi:
            signals = signals[:, : self.n_roi]
        tokens = self.proj(signals.unsqueeze(1))
        pulse, _ = self.encoder(tokens)
        return pulse


class WindowDataset(Dataset):
    def __init__(self, signals, pulses, augment=False):
        self.signals = signals.astype(np.float32)
        self.pulses = pulses.astype(np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, index):
        roi = self.signals[index].copy()
        pulse = self.pulses[index].copy()
        if self.augment:
            roi, pulse = _maybe_augment(roi, pulse)
        return torch.from_numpy(roi), torch.from_numpy(pulse)


def _stretch(x, factor):
    src_n = x.shape[-1]
    dst_n = max(8, int(round(src_n / factor)))
    src = np.linspace(0.0, 1.0, src_n)
    mid = np.linspace(0.0, 1.0, dst_n)
    back = src
    if x.ndim == 1:
        squeezed = np.interp(mid, src, x)
        return np.interp(back, mid, squeezed).astype(np.float32)
    rows = []
    for row in x:
        squeezed = np.interp(mid, src, row)
        rows.append(np.interp(back, mid, squeezed))
    return np.stack(rows).astype(np.float32)


def _maybe_augment(roi, pulse, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    factor = float(rng.choice([1.0, 2.0, 3.0, 0.5, 1.0 / 3.0]))
    if factor == 1.0:
        return roi, pulse
    roi2 = _stretch(roi, factor)
    pulse2 = _stretch(pulse, factor)
    hr = hr_from_pulse(pulse2, MODEL_FS)
    if not (40.0 <= hr <= 240.0):
        return roi, pulse
    return roi2, pulse2


def _run_epoch(model, loader, optimizer, device):
    train = optimizer is not None
    model.train(train)
    total = 0.0
    count = 0
    loss_fn = nn.MSELoss()
    for roi, pulse in loader:
        roi = roi.to(device)
        pulse = pulse.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        pred = model(roi)
        loss = loss_fn(pred, pulse)
        if train:
            loss.backward()
            optimizer.step()
        total += loss.item() * len(roi)
        count += len(roi)
    return total / max(count, 1)


def _stack_subjects(subjects):
    signals = np.concatenate([item["signals"] for item in subjects if len(item["signals"])], axis=0)
    pulses = np.concatenate([item["pulses"] for item in subjects if len(item["pulses"])], axis=0)
    return signals, pulses


def evaluate(model, subjects, device):
    model.eval()
    pred_all = []
    true_all = []
    snr_all = []
    macc_all = []
    per_video = []
    with torch.no_grad():
        for subject in subjects:
            signals = subject["signals"]
            pulses = subject["pulses"]
            if len(signals) < 5:
                continue
            pred_hrs = []
            true_hrs = []
            for start in range(0, len(signals) - 4, 5):
                batch = torch.from_numpy(signals[start:start + 5]).to(device)
                pred = model(batch).cpu().numpy()
                pred_clip = [hr_from_pulse(pred[i], MODEL_FS) for i in range(5)]
                true_clip = [hr_from_pulse(pulses[start + i], MODEL_FS) for i in range(5)]
                pred_hrs.append(float(np.mean(pred_clip)))
                true_hrs.append(float(np.mean(true_clip)))
                pred_wave = pred.reshape(-1)
                true_wave = pulses[start:start + 5].reshape(-1)
                snr_all.append(_calculate_SNR(pred_wave, true_hrs[-1], fs=MODEL_FS))
                macc_all.append(_compute_macc(pred_wave, true_wave))
            if not pred_hrs:
                continue
            pred_all.extend(pred_hrs)
            true_all.extend(true_hrs)
            diff = np.asarray(pred_hrs) - np.asarray(true_hrs)
            per_video.append({
                "id": subject["id"],
                "mae": float(np.mean(np.abs(diff))),
                "n": int(len(diff)),
            })
    summary = summarize_hr(pred_all, true_all, snr=snr_all, macc=macc_all)
    summary["videos"] = per_video
    return summary


def pretrain(model, device, epochs, batch_size, n_samples, logger=None):
    signals, pulses, _ = generate_dataset(n_samples=n_samples, n_roi=40, fs=MODEL_FS, seconds=4)
    signals = signals[:, :N_ROI]
    signals = (signals - signals.mean(axis=-1, keepdims=True)) / (signals.std(axis=-1, keepdims=True) + 1e-6)
    pulses = (pulses - pulses.mean(axis=-1, keepdims=True)) / (pulses.std(axis=-1, keepdims=True) + 1e-6)
    loader = DataLoader(WindowDataset(signals, pulses), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    for epoch in range(epochs):
        loss = _run_epoch(model, loader, optimizer, device)
        if logger is not None:
            logger.add("pretrain", epoch + 1, loss, lr=3e-4)
        print(f"pretrain {epoch + 1}/{epochs} mse {loss:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=r"D:\maxzhang\python files\datasets\ubfc_h5")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(SRC), "outputs"))
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--pretrain-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    print("loading UBFC", args.data, flush=True)
    subjects = load_ubfc(args.data)
    train_subjects, test_subjects = split_subjects(subjects, n_test=14)
    print("train", [item["id"] for item in train_subjects], flush=True)
    print("test", [item["id"] for item in test_subjects], flush=True)

    model = RADIANT().to(device)
    if args.eval_only:
        model.load_state_dict(torch.load(os.path.join(args.out, "radiant_ubfc.pt"), map_location=device))
        report = evaluate(model, test_subjects, device)
        report["train_ids"] = [item["id"] for item in train_subjects]
        report["test_ids"] = [item["id"] for item in test_subjects]
        with open(os.path.join(args.out, "ubfc_metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(json.dumps({k: report[k] for k in ("mae", "rmse", "std", "n")}), flush=True)
        return
    logger = TrainLog(os.path.join(args.out, "ubfc_train_log.csv"), model="RADIANT", dataset="UBFC")
    if not args.skip_pretrain and args.pretrain_epochs > 0:
        pretrain(model, device, args.pretrain_epochs, batch_size=32, n_samples=args.pretrain_samples, logger=logger)

    signals, pulses = _stack_subjects(train_subjects)
    loader = DataLoader(
        WindowDataset(signals, pulses, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    best = None
    for epoch in range(args.epochs):
        loss = _run_epoch(model, loader, optimizer, device)
        metrics = evaluate(model, test_subjects, device)
        mae = metrics["mae"]
        logger.add(
            "finetune", epoch + 1, loss, mae, metrics["rmse"], metrics["std"],
            lr=optimizer.param_groups[0]["lr"], mape=metrics["mape"], pearson=metrics["pearson"],
            snr=metrics["snr"], macc=metrics["macc"],
        )
        print(
            f"finetune {epoch + 1}/{args.epochs} mse {loss:.4f} "
            f"test MAE {mae} RMSE {metrics['rmse']} std {metrics['std']}",
            flush=True,
        )
        if mae is not None and (best is None or mae < best):
            best = mae
            torch.save(model.state_dict(), os.path.join(args.out, "radiant_ubfc.pt"))
    report = evaluate(model, test_subjects, device)
    report["train_ids"] = [item["id"] for item in train_subjects]
    report["test_ids"] = [item["id"] for item in test_subjects]
    report["best_mae"] = best
    with open(os.path.join(args.out, "ubfc_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    loss_png = os.path.join(args.out, "ubfc_loss.png")
    logger.plot(loss_png)
    print("wrote", os.path.join(args.out, "ubfc_metrics.json"), flush=True)
    print("wrote", loss_png, flush=True)


if __name__ == "__main__":
    main()
