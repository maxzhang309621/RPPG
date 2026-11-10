"""在已有的脸部裁剪 h5 上训练 CanonicalPhys。

官方 train.py 需要 RetinaFace 裁剪后的原始视频 LMDB。这里的入口直接读
datasets 下的 h5（imgs, bvp），把脸缩到 72x72，用 MediaPipe 取 68 点
和面部变换，再走论文里的四锚点单应、Lambert 权重、跨区域一致性和 POS 蒸馏。

输入是原始 RGB，时间差分在模型内部完成，标签用一阶差分，和 FactorizePhys
骨干的 torch.diff 对齐。划分按受试者，不把同一个人分到训练和测试两侧。
"""

import argparse
import json
import os
import sys

COMMON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common")
if COMMON not in sys.path:
    sys.path.insert(0, COMMON)

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation.post_process import calculate_metric_per_video
from train_viz import TrainRecorder, summarize_hr
from losses.neg_pearson import Neg_Pearson
from losses.pos_distillation import POSDistillationLoss
from losses.spatial_consistency import SpatialConsistencyLoss
from models.canonical_phys import CanonicalPhys
from utils.mediapipe_landmarks import DLIB68_FROM_MP468

UBFC_ROOT = r"D:\maxzhang\python files\datasets\ubfc_h5"
PURE_ROOT = r"D:\maxzhang\python files\datasets\pure_h5"
SIZE = 72
CHUNK = 128
TASK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "mediapipe", "face_landmarker.task")


def _stems(root):
    names = [os.path.splitext(name)[0] for name in os.listdir(root) if name.endswith(".h5")]
    return sorted(names, key=lambda item: [int(part) for part in item.split("-")])


def split_ids(dataset, stems):
    if dataset == "ubfc":
        test = stems[-14:]
        rest = stems[:-14]
        n_valid = max(1, int(round(len(rest) * 0.2)))
        return rest[:-n_valid], rest[-n_valid:], test
    train = [item for item in stems if int(item.split("-")[0]) <= 6]
    valid = [item for item in stems if int(item.split("-")[0]) == 7]
    test = [item for item in stems if int(item.split("-")[0]) >= 8]
    return train, valid, test


def _diff_label(label):
    diff = np.diff(label.astype(np.float64))
    std = np.std(diff)
    if std > 0:
        diff = diff / std
    diff = np.append(diff, 0.0).astype(np.float32)
    diff[np.isnan(diff)] = 0
    return diff


def _landmarker():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=TASK),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        output_facial_transformation_matrixes=True,
    )
    return mp, mp_vision.FaceLandmarker.create_from_options(options)


def extract_geometry(frames_rgb, cache_path):
    """frames_rgb: (T, 72, 72, 3) uint8. 缓存 68 点、4x4 变换和有效帧。"""
    if os.path.exists(cache_path):
        with np.load(cache_path) as packed:
            if packed["landmarks"].shape[0] == len(frames_rgb):
                return packed["landmarks"], packed["transforms"], packed["valid"]

    mp, landmarker = _landmarker()
    count = len(frames_rgb)
    landmarks = np.zeros((count, 68, 2), np.float32)
    transforms = np.tile(np.eye(4, dtype=np.float32), (count, 1, 1))
    valid = np.zeros(count, np.float32)
    last_lm = np.zeros((68, 2), np.float32)
    last_tf = np.eye(4, dtype=np.float32)
    try:
        for index, frame in enumerate(frames_rgb):
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
            result = landmarker.detect_for_video(image, (index + 1) * 33)
            if not result.face_landmarks:
                landmarks[index] = last_lm
                transforms[index] = last_tf
                continue
            points = result.face_landmarks[0]
            coords = np.array([[point.x, point.y] for point in points], np.float32)
            landmarks[index] = coords[DLIB68_FROM_MP468]
            if result.facial_transformation_matrixes:
                transforms[index] = np.asarray(result.facial_transformation_matrixes[0], np.float32)
            else:
                transforms[index] = last_tf
            valid[index] = 1.0
            last_lm = landmarks[index]
            last_tf = transforms[index]
    finally:
        landmarker.close()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, landmarks=landmarks, transforms=transforms, valid=valid)
    return landmarks, transforms, valid


def load_video(path, cache_path):
    with h5py.File(path, "r") as handle:
        frames = handle["imgs"][:]
        pulse = handle["bvp"][:].astype(np.float32)
        fs = float(handle.attrs["fs"]) if "fs" in handle.attrs else 30.0
    usable = min(len(frames), len(pulse))
    frames = frames[:usable]
    pulse = pulse[:usable]
    resized = np.stack([cv2.resize(frame, (SIZE, SIZE), interpolation=cv2.INTER_AREA) for frame in frames])
    if resized.shape[-1] != 3:
        raise RuntimeError(f"期望 RGB 帧: {path}")
    print(f"几何 {os.path.basename(path)} 帧数 {len(resized)}", flush=True)
    landmarks, transforms, valid = extract_geometry(resized, cache_path)
    print(f"  有效帧 {int(valid.sum())}/{len(valid)}", flush=True)
    return resized, pulse, landmarks, transforms, valid, fs


class ClipDataset(Dataset):
    def __init__(self, root, cache_root, stems):
        self.samples = []
        self.videos = {}
        for stem in stems:
            frames, pulse, landmarks, transforms, valid, fs = load_video(
                os.path.join(root, stem + ".h5"),
                os.path.join(cache_root, stem + ".npz"),
            )
            self.videos[stem] = {
                "frames": frames,
                "pulse": pulse,
                "landmarks": landmarks,
                "transforms": transforms,
                "valid": valid,
                "fs": fs,
            }
            n_clip = len(frames) // CHUNK
            for index in range(n_clip):
                self.samples.append((stem, index * CHUNK))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        stem, start = self.samples[index]
        video = self.videos[stem]
        end = start + CHUNK
        frames = video["frames"][start:end].astype(np.float32)
        data = np.transpose(frames, (3, 0, 1, 2))
        label = _diff_label(video["pulse"][start:end])
        return (
            torch.from_numpy(data),
            torch.from_numpy(label),
            torch.from_numpy(video["landmarks"][start:end].copy()),
            torch.from_numpy(video["valid"][start:end].copy()),
            torch.from_numpy(video["transforms"][start:end].copy()),
            stem,
            index,
        )


def collate(batch):
    return (
        torch.stack([item[0] for item in batch]),
        torch.stack([item[1] for item in batch]),
        torch.stack([item[2] for item in batch]),
        torch.stack([item[3] for item in batch]),
        torch.stack([item[4] for item in batch]),
        [item[5] for item in batch],
        [item[6] for item in batch],
    )


def build_model(device):
    md_config = {
        "FRAME_NUM": CHUNK,
        "MD_FSAM": True,
        "MD_TYPE": "NMF",
        "MD_TRANSFORM": "T_KAB",
        "MD_S": 1,
        "MD_R": 1,
        "MD_STEPS": 4,
        "MD_INFERENCE": True,
        "MD_RESIDUAL": True,
        "INV_T": 1,
        "ETA": 0.9,
        "RAND_INIT": True,
        "align_channels": 8,
    }
    canonical = {
        "padding_mode": "zeros",
        "stabilize_warp": True,
        "adaptive_stabilization": True,
        "motion_threshold": 0.08,
        "physics_weight": True,
        "physics_sharpness": 8.0,
        "physics_bias": 0.55,
        "canonical_normals_path": os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "weights", "mediapipe", "canonical_normals_dlib68.npy"
        ),
        "canonical_landmarks_2d_path": os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "weights", "mediapipe", "canonical_landmarks_2d.npy"
        ),
    }
    model = CanonicalPhys(
        frames=CHUNK, md_config=md_config, in_channels=3, dropout=0.1, device=device, canonical_config=canonical
    )
    return model.to(device)


def _append_last(data):
    return torch.cat((data, data[:, :, -1:]), dim=2)


def run_epoch(model, loader, device, optimizer, scheduler, loss_bvp, loss_cons, loss_distill, train):
    model.train(train)
    totals = []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for data, labels, landmarks, valid, transforms, _, _ in loader:
            data = _append_last(data.to(device))
            labels = labels.to(device)
            labels = (labels - labels.mean(dim=-1, keepdim=True)) / (labels.std(dim=-1, keepdim=True) + 1e-8)
            landmarks = landmarks.to(device)
            valid = valid.to(device)
            transforms = transforms.to(device)
            if train:
                optimizer.zero_grad()
            pred, voxels, _, _ = model(data, landmarks=landmarks, valid_mask=valid, transforms=transforms)
            pred_n = (pred - pred.mean(dim=-1, keepdim=True)) / (pred.std(dim=-1, keepdim=True) + 1e-8)
            loss = loss_bvp(pred_n, labels)
            if train:
                loss = loss + 0.1 * loss_cons(voxels, valid_mask=valid)
                loss = loss + 0.2 * loss_distill(model._canonical_frames, pred_n)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()
            totals.append(float(loss.detach()))
    return float(np.mean(totals)) if totals else float("nan")


def evaluate(model, dataset, device):
    model.eval()
    by_video = {}
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate)
    with torch.no_grad():
        for data, labels, landmarks, valid, transforms, stems, _ in loader:
            data = _append_last(data.to(device))
            pred, _, _, _ = model(
                data, landmarks=landmarks.to(device), valid_mask=valid.to(device), transforms=transforms.to(device)
            )
            pred = pred.cpu().numpy()
            labels = labels.numpy()
            for row, stem in enumerate(stems):
                by_video.setdefault(stem, {"pred": [], "label": [], "fs": dataset.videos[stem]["fs"]})
                by_video[stem]["pred"].append(pred[row])
                by_video[stem]["label"].append(labels[row])
    pred_hr = []
    true_hr = []
    snr = []
    macc = []
    videos = []
    for stem, packed in by_video.items():
        prediction = np.concatenate(packed["pred"])
        label = np.concatenate(packed["label"])
        fs = packed["fs"]
        hr_label, hr_pred, snr_value, macc_value = calculate_metric_per_video(
            prediction, label, fs=fs, diff_flag=True, hr_method="FFT"
        )
        if not np.isfinite(hr_label) or not np.isfinite(hr_pred):
            continue
        pred_hr.append(hr_pred)
        true_hr.append(hr_label)
        snr.append(snr_value)
        macc.append(macc_value)
        videos.append({"id": stem, "pred": float(hr_pred), "label": float(hr_label), "abs": float(abs(hr_pred - hr_label))})
    summary = summarize_hr(pred_hr, true_hr, snr=snr, macc=macc)
    summary["videos"] = videos
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ubfc", "pure"), default="ubfc")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = UBFC_ROOT if args.dataset == "ubfc" else PURE_ROOT
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", args.dataset)
    cache_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    train_ids, valid_ids, test_ids = split_ids(args.dataset, _stems(root))
    print(f"device={device} train={len(train_ids)} valid={len(valid_ids)} test={len(test_ids)}", flush=True)
    train_set = ClipDataset(root, cache_root, train_ids)
    valid_set = ClipDataset(root, cache_root, valid_ids)
    test_set = ClipDataset(root, cache_root, test_ids)
    print(f"clips train={len(train_set)} valid={len(valid_set)} test={len(test_set)}", flush=True)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    model = build_model(device)
    n_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"trainable_params={n_params}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, epochs=args.epochs, steps_per_epoch=max(len(train_loader), 1)
    )
    loss_bvp = Neg_Pearson()
    loss_cons = SpatialConsistencyLoss(feature_size=13, temporal_diff=True).to(device)
    loss_distill = POSDistillationLoss(fps=30.0, spatial_size=SIZE).to(device)

    log_path = os.path.join(out_dir, "train_log.csv")
    recorder = TrainRecorder(log_path, model="CanonicalPhys", dataset=args.dataset.upper())

    best = None
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model, train_loader, device, optimizer, scheduler, loss_bvp, loss_cons, loss_distill, True
        )
        valid_loss = run_epoch(
            model, valid_loader, device, None, None, loss_bvp, loss_cons, loss_distill, False
        )
        metrics = evaluate(model, test_set, device)
        print(
            f"epoch {epoch} train {train_loss:.4f} valid {valid_loss:.4f} "
            f"mae {metrics['mae']:.3f} rmse {metrics['rmse']:.3f} "
            f"mape {metrics['mape']:.3f} r {metrics['pearson']:.3f} snr {metrics['snr']:.3f}",
            flush=True,
        )
        recorder.add(
            epoch,
            lr=optimizer.param_groups[0]["lr"],
            train_loss=train_loss,
            valid_loss=valid_loss,
            metrics=metrics,
        )
        if best is None or valid_loss < best:
            best = valid_loss
            torch.save({"epoch": epoch, "model": model.state_dict(), "metrics": metrics}, os.path.join(out_dir, "canonicalphys.pt"))
            with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as handle:
                json.dump({"best_epoch": epoch, "valid_loss": valid_loss, **metrics}, handle, ensure_ascii=False, indent=2)
    recorder.plot(os.path.join(out_dir, "loss.png"))


if __name__ == "__main__":
    main()
