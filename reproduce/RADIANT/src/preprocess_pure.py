"""把 PURE 原始 zip 预处理成和 UBFC h5 相同的人脸片段。

每个 `NN-MM.zip` 里是约 30 Hz 的 PNG 和约 60 Hz 的指夹脉搏 JSON。
脚本按 JSON 时间戳对齐波形，用 YuNet 检测人脸，裁成 128×128，写入

    imgs (T, 128, 128, 3) uint8
    bvp  (T,) float32
    attrs: fs, subject, session

输出文件名是 `01-01.h5` 这种会话名。同一受试者的 6 个任务不要拆进训练和测试。
"""

import argparse
import json
import os
import re
import zipfile

import cv2
import h5py
import numpy as np

SESSION_NAME = re.compile(r"^(\d{2})-(\d{2})\.zip$")
SIZE = 128


def _session_zips(root):
    names = []
    for name in os.listdir(root):
        match = SESSION_NAME.match(name)
        if match:
            names.append((int(match.group(1)), int(match.group(2)), name))
    names.sort()
    return names


def _load_labels(blob):
    data = json.loads(blob)
    frames = data["/Image"]
    pulse = data["/FullPackage"]
    frame_t = np.array([item["Timestamp"] for item in frames], dtype=np.float64)
    pulse_t = np.array([item["Timestamp"] for item in pulse], dtype=np.float64)
    pulse_v = np.array([item["Value"]["waveform"] for item in pulse], dtype=np.float32)
    order = np.argsort(frame_t)
    frame_t = frame_t[order]
    pulse_order = np.argsort(pulse_t)
    pulse_t = pulse_t[pulse_order]
    pulse_v = pulse_v[pulse_order]
    bvp = np.interp(frame_t, pulse_t, pulse_v).astype(np.float32)
    duration = (frame_t[-1] - frame_t[0]) / 1e9
    fs = (len(frame_t) - 1) / duration if duration > 0 else 30.0
    return frame_t, bvp, float(fs)


def _png_by_timestamp(archive):
    found = {}
    for name in archive.namelist():
        base = os.path.basename(name)
        if not base.startswith("Image") or not base.endswith(".png"):
            continue
        stamp = base[len("Image"):-len(".png")]
        found[int(stamp)] = name
    return found


def _square(box, width, height, scale=1.4):
    x, y, w, h = box
    cx = x + w / 2.0
    cy = y + h / 2.0
    side = max(w, h) * scale
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    x1 = int(round(cx + side / 2.0))
    y1 = int(round(cy + side / 2.0))
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(width, x1)
    y1 = min(height, y1)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    return x0, y0, x1, y1


def _detect(detector, image):
    height, width = image.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(image)
    if faces is None or len(faces) == 0:
        return None
    best = max(faces, key=lambda row: row[2] * row[3])
    return _square(best[:4], width, height)


def _crop(image, box):
    x0, y0, x1, y1 = box
    face = image[y0:y1, x0:x1]
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return cv2.resize(face, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def process_zip(path, detector, out_path):
    with zipfile.ZipFile(path) as archive:
        label_name = next(name for name in archive.namelist() if name.endswith(".json"))
        frame_t, bvp, fs = _load_labels(archive.read(label_name))
        pngs = _png_by_timestamp(archive)
        stamps = np.array(sorted(pngs), dtype=np.int64)
        images = []
        aligned_bvp = []
        box = None
        for timestamp, sample in zip(frame_t, bvp):
            nearest = int(stamps[np.argmin(np.abs(stamps - timestamp))])
            if abs(nearest - timestamp) > 50_000_000:
                continue
            raw = cv2.imdecode(np.frombuffer(archive.read(pngs[nearest]), np.uint8), cv2.IMREAD_COLOR)
            if raw is None:
                continue
            found = _detect(detector, raw)
            if found is not None:
                box = found
            if box is None:
                height, width = raw.shape[:2]
                side = int(min(width, height) * 0.6)
                box = _square((width // 2 - side // 2, height // 3, side, side), width, height, scale=1.0)
            images.append(_crop(raw, box))
            aligned_bvp.append(sample)
    if not images:
        raise RuntimeError(f"没有解出帧: {path}")
    frames = np.stack(images).astype(np.uint8)
    wave = np.asarray(aligned_bvp, dtype=np.float32)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as handle:
        handle.create_dataset("imgs", data=frames)
        handle.create_dataset("bvp", data=wave)
        handle.attrs["fs"] = fs
        stem = os.path.splitext(os.path.basename(path))[0]
        subject, session = stem.split("-")
        handle.attrs["subject"] = int(subject)
        handle.attrs["session"] = int(session)
    return frames.shape[0], fs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=r"D:\maxzhang\python files\datasets\PURE")
    parser.add_argument("--out", default=r"D:\maxzhang\python files\datasets\pure_h5")
    parser.add_argument(
        "--detector",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "face_detection_yunet_2023mar.onnx"),
    )
    args = parser.parse_args()
    detector = cv2.FaceDetectorYN.create(args.detector, "", (640, 480), 0.6, 0.3, 5000)
    jobs = _session_zips(args.src)
    if not jobs:
        raise SystemExit(f"没有找到 NN-MM.zip: {args.src}")
    for subject, session, name in jobs:
        dest = os.path.join(args.out, f"{subject:02d}-{session:02d}.h5")
        if os.path.exists(dest):
            print(f"skip {name}", flush=True)
            continue
        count, fs = process_zip(os.path.join(args.src, name), detector, dest)
        print(f"wrote {dest} frames {count} fs {fs:.2f}", flush=True)


if __name__ == "__main__":
    main()
