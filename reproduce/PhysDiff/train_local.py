"""用本地 PURE / UBFC 的人脸 h5 训练官方 PhysDiff。

官方仓库只有去噪器，没有数据入口。这里补的是论文写明的部分：
6 通道 MSTmap（RGB+YUV，7×9=63 块，300 帧）、Adam 1e-3、50 epoch、
负 Pearson 加 0.1 的趋势/幅度一致性，推理时 10 条假设、5 步 DDIM，
再用相邻片段功率谱差选出最周期的一条。
"""

import argparse
import os
import random
import sys

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

SRC = os.path.dirname(os.path.abspath(__file__))
REPRO = os.path.dirname(SRC)
sys.path.insert(0, SRC)
sys.path.insert(0, REPRO)
sys.path.insert(0, os.path.join(REPRO, "CanonicalPhys"))

import config
from common.train_viz import TrainRecorder, summarize_hr
from evaluation.post_process import calculate_metric_per_video
from model.physdiff import distance, norm_diff, physdiff

UBFC_ROOT = r"D:\maxzhang\python files\datasets\ubfc_h5"
PURE_ROOT = r"D:\maxzhang\python files\datasets\pure_h5"
ROWS, COLS = 7, 9
CROP = 126


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _session_paths(root):
    paths = []
    for name in os.listdir(root):
        if name.endswith(".h5"):
            paths.append(os.path.join(root, name))
    paths.sort()
    return paths


def split_ubfc(root):
    paths = _session_paths(root)
    paths.sort(key=lambda path: int(os.path.splitext(os.path.basename(path))[0]))
    test = paths[-14:]
    rest = paths[:-14]
    n_valid = max(1, int(round(len(rest) * 0.2)))
    return rest[:-n_valid], rest[-n_valid:], test


def split_pure(root):
    groups = {"train": [], "valid": [], "test": []}
    for path in _session_paths(root):
        with h5py.File(path, "r") as handle:
            subject = int(handle.attrs["subject"])
        if subject <= 6:
            groups["train"].append(path)
        elif subject == 7:
            groups["valid"].append(path)
        else:
            groups["test"].append(path)
    return groups["train"], groups["valid"], groups["test"]


def video_fs(path):
    with h5py.File(path, "r") as handle:
        if "fs" in handle.attrs:
            return float(handle.attrs["fs"])
    return 30.0


def mstmap_from_frames(frames):
    """uint8 RGB (T,H,W,3) -> float32 MSTmap (6, 63, T)。"""
    if frames.shape[1] != 128 or frames.shape[2] != 128:
        resized = np.empty((frames.shape[0], 128, 128, 3), dtype=np.uint8)
        for index, frame in enumerate(frames):
            resized[index] = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_AREA)
        frames = resized
    offset = (128 - CROP) // 2
    frames = frames[:, offset:offset + CROP, offset:offset + CROP]
    yuv = np.empty_like(frames)
    for index, frame in enumerate(frames):
        yuv[index] = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)
    def pool(video):
        time, _, _, channels = video.shape
        blocks = video.reshape(time, ROWS, CROP // ROWS, COLS, CROP // COLS, channels)
        return blocks.mean(axis=(2, 4)).reshape(time, ROWS * COLS, channels)
    rgb = pool(frames.astype(np.float32))
    yuv = pool(yuv.astype(np.float32))
    stacked = np.concatenate([rgb, yuv], axis=-1) / 255.0
    return np.transpose(stacked, (2, 1, 0)).astype(np.float32)


def zscore(values):
    values = np.asarray(values, dtype=np.float32)
    return (values - values.mean()) / (values.std() + 1e-6)


class WindowDataset(Dataset):
    def __init__(self, paths, train):
        self.train = train
        self.videos = []
        self.index = []
        for path in paths:
            with h5py.File(path, "r") as handle:
                length = int(handle["imgs"].shape[0])
            if length < config.FRAMES:
                continue
            video_index = len(self.videos)
            self.videos.append((path, length, video_fs(path)))
            for start in range(0, length - config.FRAMES + 1, config.FRAMES):
                self.index.append((video_index, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        video_index, start = self.index[item]
        path, length, fs = self.videos[video_index]
        if self.train:
            start = random.randrange(0, length - config.FRAMES + 1)
        with h5py.File(path, "r") as handle:
            frames = handle["imgs"][start:start + config.FRAMES]
            pulse = handle["bvp"][start:start + config.FRAMES]
        return mstmap_from_frames(frames), zscore(pulse), np.float32(fs)


def neg_pearson(pred, target):
    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    cov = (pred * target).sum(dim=-1)
    denom = pred.norm(dim=-1) * target.norm(dim=-1)
    return (1 - cov / denom.clamp_min(1e-6)).mean()


def total_loss(pred, target):
    consistency = torch.nn.functional.mse_loss(distance(pred), distance(target))
    consistency = consistency + torch.nn.functional.mse_loss(norm_diff(pred), norm_diff(target))
    return neg_pearson(pred, target) + config.LAMBDA_DIS * consistency


def select_hypothesis(hypotheses):
    """score = 1 - 相邻 3 段功率谱的 L2 差，取得分最高的一条。"""
    hypotheses = np.asarray(hypotheses, dtype=np.float64)
    segment = hypotheses.shape[-1] // 3
    best_score = None
    best = hypotheses[0]
    for wave in hypotheses:
        spectra = []
        for index in range(3):
            clip = wave[index * segment:(index + 1) * segment]
            clip = clip - clip.mean()
            spectra.append(np.abs(np.fft.rfft(clip)) ** 2)
        score = 1.0 - sum(np.linalg.norm(spectra[index + 1] - spectra[index]) for index in range(2))
        if best_score is None or score > best_score:
            best_score = score
            best = wave
    return best.astype(np.float32)


@torch.no_grad()
def predict_windows(model, mst):
    model.is_test = True
    expanded = mst.expand(config.H, -1, -1, -1)
    hypotheses = model(expanded, None)
    model.is_test = False
    return select_hypothesis(hypotheses.detach().cpu().numpy())


@torch.no_grad()
def evaluate(model, paths, device):
    model.eval()
    pred_hr, label_hr, snr, macc = [], [], [], []
    for path in paths:
        fs = video_fs(path)
        with h5py.File(path, "r") as handle:
            length = int(handle["imgs"].shape[0])
            chunks_pred, chunks_label = [], []
            for start in range(0, length - config.FRAMES + 1, config.FRAMES):
                frames = handle["imgs"][start:start + config.FRAMES]
                pulse = np.asarray(handle["bvp"][start:start + config.FRAMES], dtype=np.float32)
                mst = torch.from_numpy(mstmap_from_frames(frames)).unsqueeze(0).to(device)
                chunks_pred.append(predict_windows(model, mst))
                chunks_label.append(pulse)
        if not chunks_pred:
            continue
        hr_label, hr_pred, signal_snr, signal_macc = calculate_metric_per_video(
            np.concatenate(chunks_pred),
            np.concatenate(chunks_label),
            fs=fs,
            diff_flag=False,
        )
        pred_hr.append(hr_pred)
        label_hr.append(hr_label)
        snr.append(signal_snr)
        macc.append(signal_macc)
    return summarize_hr(pred_hr, label_hr, snr, macc)


@torch.no_grad()
def valid_objective(model, paths, device, batch_size):
    model.eval()
    model.is_test = False
    loader = DataLoader(WindowDataset(paths, train=False), batch_size=batch_size, shuffle=False)
    losses = []
    for mst, pulse, _fs in loader:
        pred = model(mst.to(device), pulse.to(device))
        losses.append(float(total_loss(pred, pulse.to(device))))
    if not losses:
        return None
    return float(np.mean(losses))


def run(dataset, data_root, batch_size):
    set_seed(0)
    if dataset == "UBFC":
        train_paths, valid_paths, test_paths = split_ubfc(data_root or UBFC_ROOT)
    else:
        train_paths, valid_paths, test_paths = split_pure(data_root or PURE_ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = physdiff(
        roi_num=config.ROIS,
        dim=config.DIM,
        mlp_dim=config.DIM * 2,
        depth=config.DEPTH,
        heads=8,
        channels=9,
    ).to(device)
    model.is_test = False
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    loader = DataLoader(
        WindowDataset(train_paths, train=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    out_dir = os.path.join(SRC, "outputs", dataset.lower())
    os.makedirs(out_dir, exist_ok=True)
    recorder = TrainRecorder(os.path.join(out_dir, "train_log.csv"), "PhysDiff", dataset)
    best_valid = None
    print(
        f"PhysDiff {dataset} device={device} train={len(train_paths)} "
        f"valid={len(valid_paths)} test={len(test_paths)} clips={len(loader.dataset)}",
        flush=True,
    )
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        model.is_test = False
        losses = []
        for mst, pulse, _fs in loader:
            mst = mst.to(device)
            pulse = pulse.to(device)
            pred = model(mst, pulse)
            loss = total_loss(pred, pulse)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        train_loss = float(np.mean(losses))
        if device.type == "cuda":
            torch.cuda.empty_cache()
        valid_loss = valid_objective(model, valid_paths, device, batch_size)
        test_metrics = evaluate(model, test_paths, device)
        recorder.add(
            epoch,
            phase="finetune",
            lr=config.LR,
            train_loss=train_loss,
            valid_loss=valid_loss,
            metrics=test_metrics,
        )
        print(
            f"epoch {epoch} loss {train_loss:.4f} valid {valid_loss:.4f} "
            f"test_mae {test_metrics['mae']:.3f} rmse {test_metrics['rmse']:.3f} "
            f"r {test_metrics['pearson']:.3f} snr {test_metrics['snr']:.3f}",
            flush=True,
        )
        if best_valid is None or valid_loss < best_valid:
            best_valid = valid_loss
            torch.save(model.state_dict(), os.path.join(out_dir, "physdiff.pt"))
    recorder.plot(os.path.join(out_dir, "loss.png"), f"PhysDiff / {dataset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("UBFC", "PURE"), required=True)
    parser.add_argument("--data-root", default="")
    parser.add_argument("--batch", type=int, default=config.BATCH)
    args = parser.parse_args()
    run(args.dataset, args.data_root, args.batch)
