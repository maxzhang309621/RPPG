"""从已裁好的 UBFC-rPPG h5 提取 RADIANT 的时间信号。

论文用 OpenFace 68 点在原始视频上圈脸颊，再切皮肤方块。当前数据只有
128×128 人脸图和 BVP，没有原始帧，也没有 landmark。这里用第一帧的
YCbCr 皮肤掩膜，把脸分成不重叠方块，只保留皮肤占比足够的方块，再按
论文的质量分 1/σ 取标准差最小的 12 路。色度投影用 CHROM。
"""

import os

import h5py
import numpy as np
from scipy import signal


UBFC_FS = 30
MODEL_FS = 20
CLIP_SECONDS = 4
N_ROI = 12
BLOCK = 16


def _skin_mask(rgb_uint8):
    r = rgb_uint8[..., 0].astype(np.float32)
    g = rgb_uint8[..., 1].astype(np.float32)
    b = rgb_uint8[..., 2].astype(np.float32)
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return (cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173)


def _bandpass(x, fs, low=0.7, high=4.2, order=4):
    nyq = 0.5 * fs
    high = min(high, nyq - 0.05)
    b, a = signal.butter(order, [low / nyq, high / nyq], btype="band")
    if x.shape[-1] < 3 * max(len(a), len(b)):
        return x - np.mean(x, axis=-1, keepdims=True)
    return signal.filtfilt(b, a, x, axis=-1).astype(np.float32)


def _chrom(rgb, fs):
    """rgb: (T, 3)。先按均值归一化，带通，再做 CHROM，最后去趋势。"""
    rgb = rgb.astype(np.float32)
    rgb = rgb / (np.mean(rgb, axis=0, keepdims=True) + 1e-6)
    filtered = np.stack([_bandpass(rgb[:, c], fs) for c in range(3)], axis=1)
    xs = 3.0 * filtered[:, 0] - 2.0 * filtered[:, 1]
    ys = 1.5 * filtered[:, 0] + filtered[:, 1] - 1.5 * filtered[:, 2]
    alpha = np.std(xs) / (np.std(ys) + 1e-8)
    out = xs - alpha * ys
    return signal.detrend(out).astype(np.float32)


def _resample(x, src_len, dst_len):
    if src_len == dst_len:
        return x
    src = np.linspace(0.0, 1.0, src_len)
    dst = np.linspace(0.0, 1.0, dst_len)
    if x.ndim == 1:
        return np.interp(dst, src, x).astype(np.float32)
    return np.stack([np.interp(dst, src, row) for row in x]).astype(np.float32)


def _zscore(x, axis=-1):
    mu = np.mean(x, axis=axis, keepdims=True)
    sd = np.std(x, axis=axis, keepdims=True)
    return ((x - mu) / (sd + 1e-6)).astype(np.float32)


def roi_rgb(frames):
    """frames (T,H,W,3) -> list of (T, 3) 方块平均 RGB。皮肤方块不足时退回全部方块。"""
    mask = _skin_mask(frames[0])
    height, width = frames.shape[1], frames.shape[2]
    flat = frames.astype(np.float32)
    skin_rgb = []
    all_rgb = []
    for y0 in range(0, height - BLOCK + 1, BLOCK):
        for x0 in range(0, width - BLOCK + 1, BLOCK):
            patch = flat[:, y0:y0 + BLOCK, x0:x0 + BLOCK]
            all_rgb.append(patch.mean(axis=(1, 2)))
            skin = mask[y0:y0 + BLOCK, x0:x0 + BLOCK]
            if skin.mean() >= 0.5:
                skin_rgb.append(patch[:, skin].mean(axis=1))
    return skin_rgb if len(skin_rgb) >= N_ROI else all_rgb


def select_rois(traces, n_roi=N_ROI):
    scored = []
    for trace in traces:
        sigma = float(np.std(trace))
        quality = 0.0 if sigma < 1e-4 else 1.0 / sigma
        scored.append((quality, trace))
    scored.sort(key=lambda item: item[0], reverse=True)
    chosen = [trace for _, trace in scored[:n_roi]]
    while len(chosen) < n_roi:
        chosen.append(chosen[-1] if chosen else np.zeros(1, dtype=np.float32))
    return chosen[:n_roi]


def hr_from_pulse(pulse, fs, low=0.7, high=4.2):
    pulse = np.asarray(pulse, dtype=np.float64)
    pulse = pulse - pulse.mean()
    spec = np.abs(np.fft.rfft(pulse))
    freqs = np.fft.rfftfreq(len(pulse), d=1.0 / fs)
    band = (freqs >= low) & (freqs <= high)
    if not np.any(band):
        return float("nan")
    return float(freqs[band][np.argmax(spec[band])] * 60.0)


def iter_subject_windows(path, fs=UBFC_FS, model_fs=MODEL_FS, seconds=CLIP_SECONDS):
    src_len = int(fs * seconds)
    dst_len = int(model_fs * seconds)
    with h5py.File(path, "r") as handle:
        frames = handle["imgs"][:]
        bvp = handle["bvp"][:].astype(np.float32)
        if "fs" in handle.attrs:
            fs = float(handle.attrs["fs"])
    rgb_rois = roi_rgb(frames)
    usable = min(len(frames), len(bvp))
    n_win = usable // src_len
    signals = []
    pulses = []
    for index in range(n_win):
        start = index * src_len
        end = start + src_len
        traces = [_chrom(rgb[start:end], fs) for rgb in rgb_rois]
        chosen = select_rois(traces)
        roi = _resample(np.stack(chosen), src_len, dst_len)
        pulse = _resample(bvp[start:end], src_len, dst_len)
        pulse = _bandpass(pulse, model_fs)
        pulse = signal.detrend(pulse).astype(np.float32)
        signals.append(_zscore(roi))
        pulses.append(_zscore(pulse))
    if not signals:
        return np.zeros((0, N_ROI, dst_len), np.float32), np.zeros((0, dst_len), np.float32)
    return np.stack(signals), np.stack(pulses)


def load_ubfc(root):
    names = [name for name in os.listdir(root) if name.endswith(".h5")]
    names.sort(key=lambda name: int(os.path.splitext(name)[0]))
    subjects = []
    for name in names:
        subject_id = int(os.path.splitext(name)[0])
        signals, pulses = iter_subject_windows(os.path.join(root, name))
        subjects.append({"id": subject_id, "signals": signals, "pulses": pulses})
    return subjects


def split_subjects(subjects, n_test=14):
    ordered = sorted(subjects, key=lambda item: item["id"])
    if len(ordered) <= n_test:
        raise ValueError(f"受试者只有 {len(ordered)} 个，无法留出 {n_test} 个测试")
    return ordered[:-n_test], ordered[-n_test:]
