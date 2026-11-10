#!/usr/bin/env python
"""Preprocess rPPG datasets and store in LMDB format.

Stores face-cropped, resized uint8 frames and raw BVP signals per subject
in a single LMDB database. Transforms (DiffNormalized, etc.) and chunking
are applied at load time by LMDBDataset, allowing different models to use
different chunk lengths from the same preprocessed data.

Usage:
    python preprocess_lmdb.py --data_config configs/datasets/ubfc.yaml
    python preprocess_lmdb.py --data_config configs/datasets/pure.yaml
    python preprocess_lmdb.py --data_config configs/datasets/obf.yaml
    python preprocess_lmdb.py --data_config configs/datasets/mmpd.yaml
"""

import argparse
import io
import json
import logging
import os
import sys
from math import ceil

import cv2
import lmdb
import numpy as np
import torch
from tqdm import tqdm

from utils.config import load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Face detection and cropping (reused from BaseLoader)
# ---------------------------------------------------------------------------

_retinaface_detector = None


def get_retinaface_detector(model_path):
    """Lazily initialize a shared RetinaFace detector."""
    global _retinaface_detector
    if _retinaface_detector is None:
        from utils.retinaface import FaceDetector
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _retinaface_detector = FaceDetector(
            model_path=model_path, network='resnet50', device=device)
    return _retinaface_detector


def face_detection_rf(frame, model_path, use_larger_box=False, larger_box_coef=1.0):
    """Detect face using RetinaFace, return [x, y, w, h]."""
    detector = get_retinaface_detector(model_path)
    frame_bgr = cv2.cvtColor(frame[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
    detections = detector.detect(frame_bgr)
    if not detections:
        logger.warning("No Face Detected")
        return [0, 0, frame.shape[1], frame.shape[0]]
    best = max(detections, key=lambda d: d.confidence)
    x1, y1, x2, y2 = best.bbox
    face_box = np.array([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
    if use_larger_box:
        face_box[0] = max(0, face_box[0] - (larger_box_coef - 1.0) / 2 * face_box[2])
        face_box[1] = max(0, face_box[1] - (larger_box_coef - 1.0) / 2 * face_box[3])
        face_box[2] = larger_box_coef * face_box[2]
        face_box[3] = larger_box_coef * face_box[3]
    return face_box.astype(int)


def crop_face_resize(frames, config_preprocess):
    """Detect face on first frame, crop and resize all frames to (H, W)."""
    crop_cfg = config_preprocess.CROP_FACE
    resize_cfg = config_preprocess.RESIZE
    width, height = resize_cfg.W, resize_cfg.H

    if not crop_cfg.DO_CROP_FACE:
        total_frames = frames.shape[0]
        resized = np.zeros((total_frames, height, width, frames.shape[3]),
                           dtype=np.uint8)
        for i in range(total_frames):
            resized[i] = cv2.resize(frames[i], (width, height),
                                    interpolation=cv2.INTER_AREA)
        return resized

    # Detect face(s) — use dynamic detection if configured
    do_dynamic = getattr(crop_cfg.DETECTION, 'DO_DYNAMIC_DETECTION', False)
    det_freq = getattr(crop_cfg.DETECTION, 'DYNAMIC_DETECTION_FREQUENCY', 32)
    use_median = getattr(crop_cfg.DETECTION, 'USE_MEDIAN_FACE_BOX', False)

    if do_dynamic:
        num_det = ceil(frames.shape[0] / det_freq)
    else:
        num_det = 1

    face_regions = []
    for idx in range(num_det):
        frame_idx = det_freq * idx
        face_box = face_detection_rf(
            frames[frame_idx], crop_cfg.RETINAFACE_MODEL_PATH,
            crop_cfg.USE_LARGE_FACE_BOX, crop_cfg.LARGE_BOX_COEF)
        face_regions.append(face_box)
    face_regions = np.array(face_regions, dtype=int)

    if use_median:
        face_region_median = np.median(face_regions, axis=0).astype(int)

    total_frames = frames.shape[0]
    resized = np.zeros((total_frames, height, width, frames.shape[3]),
                       dtype=np.uint8)
    for i in range(total_frames):
        frame = frames[i]
        ref_idx = i // det_freq if do_dynamic else 0
        if use_median:
            fr = face_region_median
        else:
            fr = face_regions[ref_idx]
        cropped = frame[max(fr[1], 0):min(fr[1] + fr[3], frame.shape[0]),
                        max(fr[0], 0):min(fr[0] + fr[2], frame.shape[1])]
        if cropped.size == 0:
            cropped = frame
        resized[i] = cv2.resize(cropped, (width, height),
                                interpolation=cv2.INTER_AREA)
    return resized


# ---------------------------------------------------------------------------
#  Dataset-specific raw data readers
# ---------------------------------------------------------------------------

def _read_video_pyav(vid_path):
    """Read video frames using PyAV (avoids cv2.VideoCapture segfault).

    cv2.VideoCapture.read() causes 'corrupted double-linked list' SIGABRT on
    UBFC .avi files inside the LUMI Singularity container. PyAV uses its own
    bundled ffmpeg and bypasses the problematic cv2 codepath entirely.
    """
    import av
    container = av.open(vid_path)
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format='rgb24'))
    container.close()
    return np.array(frames)


def read_ubfc_subjects(data_path):
    """Yield (subject_id, frames_uint8, bvps_float64) for UBFC-rPPG."""
    import glob
    import re
    subject_dirs = sorted(glob.glob(os.path.join(data_path, "subject*")))
    logger.info(f"UBFC: found {len(subject_dirs)} subjects in {data_path}")
    for sdir in subject_dirs:
        sid = re.search(r'subject(\d+)', sdir).group(0)
        vid_path = os.path.join(sdir, "vid.avi")
        frames = _read_video_pyav(vid_path)
        # Read BVP
        with open(os.path.join(sdir, "ground_truth.txt"), "r") as f:
            bvp_str = f.read().split("\n")[0].split()
            bvps = np.array([float(x) for x in bvp_str])
        yield sid, frames, bvps


def read_pure_subjects(data_path):
    """Yield (subject_id, frames_uint8, bvps_float64) for PURE."""
    import glob
    import json as json_mod
    trial_dirs = sorted([d for d in glob.glob(os.path.join(data_path, "*-*"))
                         if os.path.isdir(d)])
    for trial_dir in trial_dirs:
        trial_name = os.path.basename(trial_dir)
        sid = trial_name.replace('-', '')
        # Read frames
        img_dir = os.path.join(trial_dir, "")
        all_png = sorted(glob.glob(img_dir + '*.png'))
        frames = []
        for p in all_png:
            img = cv2.imread(p)
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        frames = np.array(frames)
        # Read BVP
        json_path = os.path.join(os.path.dirname(trial_dir),
                                 f"{trial_name}.json")
        if not os.path.exists(json_path):
            json_path = os.path.join(trial_dir + ".json")
        with open(json_path, "r") as f:
            labels = json_mod.load(f)
            bvps = np.array([l["Value"]["waveform"]
                             for l in labels["/FullPackage"]])
        # Resample BVP to match frame count
        bvps = np.interp(
            np.linspace(1, len(bvps), frames.shape[0]),
            np.linspace(1, len(bvps), len(bvps)), bvps)
        yield sid, frames, bvps


def read_obf_subjects(data_path, target_fs=30):
    """Yield (subject_id, frames_uint8, bvps_float64) for OBF."""
    import glob
    import re
    import scipy.io as sio

    video_dir = os.path.join(data_path, "OBF_video")
    mat_dir = os.path.join(data_path, "OBF OY Bio")
    video_files = sorted(glob.glob(os.path.join(video_dir, "*_RGB_*.avi")))

    # Build mat lookup
    mat_lookup = {}
    for mf in os.listdir(mat_dir):
        if not mf.endswith('.mat'):
            continue
        parts = mf.replace('.mat', '').split('_')
        subj_id, session = parts[0], parts[-1]
        mat_lookup[(subj_id, session)] = os.path.join(mat_dir, mf)

    for vpath in video_files:
        vname = os.path.basename(vpath).replace('.avi', '')
        m = re.match(r'(\d+)_RGB_(\d+)', vname)
        if not m:
            continue
        subj_id, session = m.group(1), m.group(2)
        mat_path = mat_lookup.get((subj_id, session))
        if mat_path is None:
            logger.warning(f"No .mat for {subj_id} session {session}")
            continue

        sid = f"{subj_id}_{session}"

        # Read video with temporal downsampling
        cap = cv2.VideoCapture(vpath)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 60
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        max_dim = 480
        scale = min(max_dim / orig_w, max_dim / orig_h)
        do_resize = scale < 1.0
        new_w = int(orig_w * scale) if do_resize else orig_w
        new_h = int(orig_h * scale) if do_resize else orig_h
        skip_ratio = video_fps / target_fs

        frames = []
        frame_idx = 0
        next_target = 0.0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx >= next_target:
                if do_resize:
                    frame = cv2.resize(frame, (new_w, new_h),
                                       interpolation=cv2.INTER_AREA)
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                next_target += skip_ratio
            frame_idx += 1
        cap.release()
        frames = np.array(frames)

        # Read BVP from .mat
        mat_data = sio.loadmat(mat_path)
        bvp_256 = mat_data['Biosignal_Cropped'][:, 2]
        bvp_128 = bvp_256[::2]
        bvps = np.interp(
            np.linspace(1, len(bvp_128), frames.shape[0]),
            np.linspace(1, len(bvp_128), len(bvp_128)), bvp_128)

        yield sid, frames, bvps


def _load_mat_from_bytes(mat_bytes):
    """Parse .mat bytes into (frames_uint8, bvps_float64) or None on error."""
    import scipy.io as sio
    try:
        mat_data = sio.loadmat(io.BytesIO(mat_bytes))
    except Exception as e:
        logger.warning(f"Failed to parse .mat bytes: {e}")
        return None

    video = mat_data.get('video')
    gt_ppg = mat_data.get('GT_ppg')
    if video is None or gt_ppg is None:
        return None

    # Video: (T, H, W, C) float [0,1] → uint8 [0,255]
    frames = video
    if frames.dtype != np.uint8:
        frames = (frames * 255).clip(0, 255).astype(np.uint8)

    # BVP: (1, T) or (T,) → float64
    bvps = gt_ppg.flatten().astype(np.float64)

    # Resample BVP if lengths don't match
    if len(bvps) != frames.shape[0]:
        bvps = np.interp(
            np.linspace(1, len(bvps), frames.shape[0]),
            np.linspace(1, len(bvps), len(bvps)), bvps)

    return frames, bvps


def read_mmpd_subjects(data_path, target_fs=30):
    """Yield (subject_id, frames_uint8, bvps_float64) for MMPD.

    Supports two input modes:
    1. Extracted directory: data_path points to a folder with subject*/p*.mat
    2. Zip files (no extraction needed): data_path points to a folder
       containing MMPD*.zip files. Individual .mat files are streamed from
       the zip via 'unzip -p' (handles Deflate64 compression that Python's
       zipfile module does not support).
    """
    import subprocess
    import zipfile

    # Detect mode: check for zip files vs extracted directories
    zip_files = sorted([f for f in os.listdir(data_path)
                        if f.endswith('.zip') and 'MMPD' in f])
    subject_dirs = sorted([d for d in os.listdir(data_path)
                           if os.path.isdir(os.path.join(data_path, d))
                           and d.startswith("subject")])

    if zip_files and not subject_dirs:
        # --- Mode: read directly from zip files (no extraction) ---
        logger.info(f"MMPD: Reading from {len(zip_files)} zip file(s) "
                     f"without extraction")
        for zname in zip_files:
            zpath = os.path.join(data_path, zname)
            logger.info(f"Scanning {zname}...")

            # List .mat members using Python zipfile (listing works fine,
            # only decompression of Deflate64 is unsupported)
            with zipfile.ZipFile(zpath, 'r') as zf:
                mat_members = sorted([n for n in zf.namelist()
                                      if n.endswith('.mat')])

            logger.info(f"  Found {len(mat_members)} recordings")
            for member in mat_members:
                sid = os.path.basename(member).replace('.mat', '')
                logger.info(f"  Extracting {member} via unzip -p ...")
                try:
                    proc = subprocess.run(
                        ['unzip', '-p', zpath, member],
                        capture_output=True, timeout=600)
                    if proc.returncode != 0:
                        logger.warning(f"  unzip failed for {member}: "
                                       f"{proc.stderr[:200]}")
                        continue
                except subprocess.TimeoutExpired:
                    logger.warning(f"  Timeout extracting {member}")
                    continue

                result = _load_mat_from_bytes(proc.stdout)
                if result is None:
                    logger.warning(f"  Missing video/GT_ppg in {member}")
                    continue
                frames, bvps = result
                yield sid, frames, bvps

    elif subject_dirs:
        # --- Mode: read from extracted directories ---
        import scipy.io as sio
        logger.info(f"MMPD: Reading from {len(subject_dirs)} extracted "
                     f"subject directories")
        for sdir_name in subject_dirs:
            sdir = os.path.join(data_path, sdir_name)
            mat_files = sorted([f for f in os.listdir(sdir)
                                if f.endswith('.mat')])
            for mat_name in mat_files:
                mat_path = os.path.join(sdir, mat_name)
                sid = mat_name.replace('.mat', '')
                try:
                    with open(mat_path, 'rb') as f:
                        mat_bytes = f.read()
                except Exception as e:
                    logger.warning(f"Failed to read {mat_path}: {e}")
                    continue

                result = _load_mat_from_bytes(mat_bytes)
                if result is None:
                    logger.warning(f"Missing video/GT_ppg in {mat_path}")
                    continue
                frames, bvps = result
                yield sid, frames, bvps
    else:
        raise ValueError(
            f"MMPD: No zip files or subject directories found in {data_path}")


DATASET_READERS = {
    "UBFC-rPPG": read_ubfc_subjects,
    "PURE": read_pure_subjects,
    "OBF": read_obf_subjects,
    "MMPD": read_mmpd_subjects,
}


# ---------------------------------------------------------------------------
#  LMDB writing
# ---------------------------------------------------------------------------

def numpy_to_bytes(arr):
    """Serialize numpy array to bytes using np.save into a buffer."""
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def write_to_lmdb(subjects_iter, lmdb_path, dataset_name,
                   config_preprocess, map_size=1 << 40):
    """Write preprocessed subjects to LMDB.

    Args:
        subjects_iter: Iterator yielding (subject_id, frames_uint8, bvps)
        lmdb_path: Output LMDB database path
        dataset_name: Name of the dataset
        config_preprocess: Preprocessing config (for face crop, resize)
        map_size: Max LMDB database size in bytes (default 1TB)
    """
    os.makedirs(lmdb_path, exist_ok=True)
    env = lmdb.open(lmdb_path, map_size=map_size)

    subject_ids = []
    subject_nframes = {}

    for sid, frames, bvps in subjects_iter:
        logger.info(f"Processing {sid}: {frames.shape[0]} frames")

        # Face crop and resize
        frames_cropped = crop_face_resize(frames, config_preprocess)

        # Store uint8 frames and raw BVP
        n_frames = frames_cropped.shape[0]
        subject_ids.append(sid)
        subject_nframes[sid] = n_frames

        with env.begin(write=True) as txn:
            txn.put(f"{sid}_frames".encode(), numpy_to_bytes(frames_cropped))
            txn.put(f"{sid}_labels".encode(), numpy_to_bytes(bvps.astype(np.float64)))
            txn.put(f"{sid}_nframes".encode(), str(n_frames).encode())

        logger.info(f"  Stored {sid}: {n_frames} frames, "
                     f"shape {frames_cropped.shape}")

    # Write metadata
    with env.begin(write=True) as txn:
        txn.put(b"__subjects__", json.dumps(subject_ids).encode())
        txn.put(b"__dataset__", dataset_name.encode())
        txn.put(b"__nframes__", json.dumps(subject_nframes).encode())

    env.close()

    # Validate: reopen read-only and check metadata exists
    env_check = lmdb.open(lmdb_path, readonly=True, lock=False)
    with env_check.begin(buffers=True) as txn:
        subj_check = txn.get(b"__subjects__")
        nf_check = txn.get(b"__nframes__")
    env_check.close()
    if subj_check is None or nf_check is None:
        raise RuntimeError(
            f"LMDB validation failed: metadata missing after write "
            f"(__subjects__={subj_check is not None}, "
            f"__nframes__={nf_check is not None})")

    logger.info(f"LMDB created at {lmdb_path}: {len(subject_ids)} subjects")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess rPPG dataset to LMDB format")
    parser.add_argument('--data_config', type=str, required=True,
                        help='Path to dataset config YAML')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

    config = load_config(args.data_config)
    # Use TRAIN section config for preprocessing
    data_cfg = config.TRAIN.DATA
    dataset_name = data_cfg.DATASET
    data_path = data_cfg.DATA_PATH
    config_preprocess = data_cfg.PREPROCESS

    lmdb_path = getattr(data_cfg, 'LMDB_PATH', None)
    if lmdb_path is None:
        # Default: datasets/{dataset}/lmdb
        lmdb_path = os.path.join("datasets", dataset_name.replace("-", "_"), "lmdb")

    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Raw data: {data_path}")
    logger.info(f"LMDB output: {lmdb_path}")

    # Skip if LMDB already has valid metadata
    if os.path.exists(os.path.join(lmdb_path, "data.mdb")):
        try:
            env_check = lmdb.open(lmdb_path, readonly=True, lock=False)
            valid = False
            with env_check.begin(buffers=True) as txn:
                subj_raw = txn.get(b"__subjects__")
                nf_raw = txn.get(b"__nframes__")
                if subj_raw is not None and nf_raw is not None:
                    subjs = json.loads(bytes(subj_raw))
                    valid = True
            env_check.close()
            if valid:
                logger.info(f"LMDB already valid with {len(subjs)} subjects, skipping.")
                return
        except Exception:
            logger.info("Existing LMDB is invalid, will recreate.")

    reader_fn = DATASET_READERS.get(dataset_name)
    if reader_fn is None:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # OBF and MMPD readers take target_fs parameter
    if dataset_name in ("OBF", "MMPD"):
        subjects_iter = reader_fn(data_path, target_fs=data_cfg.FS)
    else:
        subjects_iter = reader_fn(data_path)

    # Stream directly into LMDB (avoids loading all subjects into memory,
    # critical for large datasets like MMPD where each recording is ~450MB)
    logger.info(f"Streaming {dataset_name} into LMDB...")
    write_to_lmdb(subjects_iter, lmdb_path, dataset_name,
                   config_preprocess)

    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    main()
