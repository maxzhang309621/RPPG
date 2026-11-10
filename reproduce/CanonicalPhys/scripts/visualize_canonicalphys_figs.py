"""Generate paper visualization figures for CanonicalPhys.

Produces three CVPR-style PDF figures from MMPD-trained weights:

    figs/canonicalization.pdf   -- canonical warp grid overlay before/after
    figs/physics_weight.pdf     -- per-pixel Lambertian weight across yaw
    figs/waveforms.pdf          -- CanonicalPhys vs FactorizePhys BVP

The script loads a CanonicalPhys (full recipe) checkpoint and a
FactorizePhys baseline, runs both over the MMPD test split, and renders
the three figures with matplotlib.

Usage (run from project root):
    python scripts/visualize_canonicalphys_figs.py \
        --cp_ckpt  runs/train/exp_20260418_125444_17607291/checkpoints/canonicalphys_best.pth \
        --fp_ckpt  archive/baselines/factorizephys/MMPD/factorizephys_best.pth
"""

import argparse
import io
import logging
import os
import sys

# Allow `python scripts/visualize_canonicalphys_figs.py` from the project root.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import lmdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import butter, filtfilt
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.lmdb_loader import LMDBDataset
from evaluation.post_process import calculate_metric_per_video
from models.canonical_phys import CanonicalPhys
from models.canonical_warp import (
    CanonicalWarp, ANCHOR_INDICES, CANONICAL_ANCHORS,
)
from models.factorizephys import FactorizePhys
from models.physics_weight import PhysicsWeight
from utils.config import deep_merge, load_config, sync_chunk_length

logger = logging.getLogger(__name__)

# Yaw bins in degrees (match the paper's pose-stratified table).
YAW_BINS = [(0, 15), (15, 30), (30, 45), (45, 90)]
YAW_BIN_LABELS = ["Frontal", "Mild", "Moderate", "Large"]
YAW_BIN_RANGES = [
    r"$<\!15^\circ$",
    r"$15\!-\!30^\circ$",
    r"$30\!-\!45^\circ$",
    r"$\geq\!45^\circ$",
]

# CVPR/NeurIPS-friendly accent palette (ColorBrewer-inspired).
COLOR_GT      = "#1a1a1a"
COLOR_CP      = "#1f4e79"   # deep blue
COLOR_FP      = "#b2182b"   # deep red
COLOR_ANCHOR  = "#d97706"   # amber
COLOR_TARGET  = "#1f4e79"
COLOR_ROI     = "#b2182b"

# Canonical ROIs from the paper (§3.4, §3.5).
CANONICAL_ROIS = [
    ('forehead', (0.50, 0.20)),
    ('L cheek',  (0.28, 0.58)),
    ('R cheek',  (0.72, 0.58)),
]


# --------------------------------------------------------------------------- #
#  Style
# --------------------------------------------------------------------------- #

def setup_cvpr_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'mathtext.fontset': 'dejavusans',
        'font.size': 8,
        'axes.titlesize': 8,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 8,
        'figure.titlesize': 9,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.linewidth': 0.6,
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#1a1a1a',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'lines.linewidth': 1.0,
        'patch.linewidth': 0.5,
        'legend.frameon': False,
        'legend.borderpad': 0.2,
        'legend.columnspacing': 1.0,
        'legend.handlelength': 1.6,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'savefig.dpi': 300,
    })


# --------------------------------------------------------------------------- #
#  Model construction
# --------------------------------------------------------------------------- #

def _md_config_from(lp_cfg):
    return {
        "FRAME_NUM": int(lp_cfg.FRAME_NUM),
        "MD_FSAM": bool(lp_cfg.MD_FSAM),
        "MD_TYPE": lp_cfg.MD_TYPE,
        "MD_TRANSFORM": lp_cfg.MD_TRANSFORM,
        "MD_S": int(lp_cfg.MD_S),
        "MD_R": int(lp_cfg.MD_R),
        "MD_STEPS": int(lp_cfg.MD_STEPS),
        "MD_INFERENCE": bool(lp_cfg.MD_INFERENCE),
        "MD_RESIDUAL": bool(lp_cfg.MD_RESIDUAL),
        "INV_T": float(lp_cfg.INV_T),
        "ETA": float(lp_cfg.ETA),
        "RAND_INIT": bool(lp_cfg.RAND_INIT),
        "align_channels": int(lp_cfg.ALIGN_CHANNELS),
    }


def build_canonicalphys(config, device):
    lp = config.MODEL.LANDMARKPHYS
    canonical_config = {
        'padding_mode':           str(getattr(lp, 'PADDING_MODE', 'zeros')),
        'stabilize_warp':         bool(getattr(lp, 'STABILIZE_WARP', True)),
        'adaptive_stabilization': bool(getattr(lp, 'ADAPTIVE_STABILIZATION', True)),
        'motion_threshold':       float(getattr(lp, 'MOTION_THRESHOLD', 0.08)),
        'physics_weight':         bool(getattr(lp, 'PHYSICS_WEIGHT', True)),
        'physics_sharpness':      float(getattr(lp, 'PHYSICS_SHARPNESS', 8.0)),
        'physics_bias':           float(getattr(lp, 'PHYSICS_BIAS', 0.55)),
    }
    drop_rate = float(getattr(config.MODEL, 'DROP_RATE', 0.1))
    in_channels = int(getattr(lp, 'IN_CHANNELS', 3))
    return CanonicalPhys(
        frames=int(lp.FRAME_NUM),
        md_config=_md_config_from(lp),
        in_channels=in_channels,
        dropout=drop_rate,
        device=device,
        canonical_config=canonical_config).to(device)


def build_factorizephys(config, device):
    fp = config.MODEL.FACTORIZEPHYS
    md_config = {
        "FRAME_NUM": int(fp.FRAME_NUM),
        "MD_FSAM": bool(fp.MD_FSAM),
        "MD_TYPE": fp.MD_TYPE,
        "MD_TRANSFORM": fp.MD_TRANSFORM,
        "MD_S": int(fp.MD_S),
        "MD_R": int(fp.MD_R),
        "MD_STEPS": int(fp.MD_STEPS),
        "MD_INFERENCE": bool(fp.MD_INFERENCE),
        "MD_RESIDUAL": bool(fp.MD_RESIDUAL),
        "INV_T": float(fp.INV_T),
        "ETA": float(fp.ETA),
        "RAND_INIT": bool(fp.RAND_INIT),
        "align_channels": int(fp.ALIGN_CHANNELS),
    }
    drop_rate = float(getattr(config.MODEL, 'DROP_RATE', 0.1))
    in_channels = int(getattr(fp, 'IN_CHANNELS', 3))
    return FactorizePhys(
        frames=int(fp.FRAME_NUM),
        md_config=md_config,
        in_channels=in_channels,
        dropout=drop_rate,
        device=device).to(device)


def append_last_frame(data):
    return torch.cat([data, data[:, :, -1:, :, :]], dim=2)


# --------------------------------------------------------------------------- #
#  Data helpers
# --------------------------------------------------------------------------- #

def load_raw_frames(lmdb_path, sid):
    env = lmdb.open(lmdb_path, readonly=True, lock=False,
                    readahead=False, meminit=False)
    with env.begin() as txn:
        buf = txn.get(f"{sid}_frames".encode())
    env.close()
    return np.load(io.BytesIO(bytes(buf)))


def _landmark_validity(landmarks_norm):
    """Reject frames with collapsed or hallucinated landmarks."""
    in_bounds = ((landmarks_norm >= 0.0) & (landmarks_norm <= 1.0)).all(axis=(2, 3))
    px = landmarks_norm * 71.0
    bbox_w = px[..., 0].max(axis=-1) - px[..., 0].min(axis=-1)
    bbox_h = px[..., 1].max(axis=-1) - px[..., 1].min(axis=-1)
    big_enough = (bbox_w >= 30.0) & (bbox_h >= 30.0)
    cx = px[..., 0].mean(axis=-1)
    centroid_ok = (cx >= 0.15 * 71.0) & (cx <= 0.85 * 71.0)
    return in_bounds & big_enough & centroid_ok


# --------------------------------------------------------------------------- #
#  Inference pass
# --------------------------------------------------------------------------- #

def collect_test_data(canonicalphys, factorizephys, dataset, batch_size,
                      device, max_chunks=-1):
    """Run both models over the test set. Returns stacked arrays for all chunks."""
    canonicalphys.eval()
    factorizephys.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=8, pin_memory=True)

    sids, chunk_ids = [], []
    poses_l, lm_l, tf_l = [], [], []
    cp_l, fp_l, lab_l = [], [], []

    n = 0
    pbar = tqdm(loader, desc="inference", ncols=80,
                disable=not sys.stdout.isatty())
    with torch.no_grad():
        for batch in pbar:
            data = batch[0].to(torch.float32).to(device)
            label = batch[1].to(torch.float32)
            sids_b = batch[2]
            chunk_ids_b = batch[3]
            landmarks = batch[4].to(torch.float32).to(device)
            poses = batch[5].to(torch.float32).to(device)
            if len(batch) >= 8:
                transforms = batch[7].to(torch.float32).to(device)
            else:
                raise RuntimeError(
                    "Expected 8-tuple from LMDBDataset (with transforms).")

            data_aug = append_last_frame(data)

            cp_out = canonicalphys(data_aug, landmarks=landmarks, poses=poses,
                                   transforms=transforms)
            cp_pred = cp_out[0]

            fp_out = factorizephys(data_aug)
            fp_pred = fp_out[0]

            B = data.shape[0]
            for i in range(B):
                sids.append(str(sids_b[i]))
                chunk_ids.append(int(chunk_ids_b[i]))
                poses_l.append(poses[i].cpu().numpy())
                lm_l.append(landmarks[i].cpu().numpy())
                tf_l.append(transforms[i].cpu().numpy())
                cp_l.append(cp_pred[i].cpu().numpy())
                fp_l.append(fp_pred[i].cpu().numpy())
                lab_l.append(label[i].cpu().numpy())

            n += B
            if 0 < max_chunks <= n:
                break

    return {
        'sids':        sids,
        'chunk_ids':   chunk_ids,
        'poses':       np.stack(poses_l).astype(np.float32),
        'landmarks':   np.stack(lm_l).astype(np.float32),
        'transforms':  np.stack(tf_l).astype(np.float32),
        'cp_pred':     np.stack(cp_l).astype(np.float32),
        'fp_pred':     np.stack(fp_l).astype(np.float32),
        'label':       np.stack(lab_l).astype(np.float32),
    }


def _pick_yaw_bin_frames(data):
    """Return list of (chunk_idx, frame_idx) for one frame per yaw bin."""
    abs_yaws = np.abs(data['poses'][..., 0])
    valid = _landmark_validity(data['landmarks'])
    selected = []
    for low, high in YAW_BINS:
        in_bin = (abs_yaws >= low) & (abs_yaws < high)
        candidates = in_bin & valid
        if candidates.any():
            center = (low + high) / 2
            diff = np.where(candidates, np.abs(abs_yaws - center), np.inf)
            idx_flat = int(np.argmin(diff))
        elif in_bin.any():
            logger.warning(
                f"No valid-landmark frame in yaw bin [{low}, {high}); "
                f"using closest with possibly noisy landmarks.")
            center = (low + high) / 2
            diff = np.where(in_bin, np.abs(abs_yaws - center), np.inf)
            idx_flat = int(np.argmin(diff))
        else:
            logger.warning(
                f"Yaw bin [{low}, {high}) entirely empty; "
                f"falling back to largest available valid yaw.")
            yaws_valid = np.where(valid, abs_yaws, -np.inf)
            idx_flat = int(np.argmax(yaws_valid))
        ci, fi = np.unravel_index(idx_flat, abs_yaws.shape)
        selected.append((int(ci), int(fi)))
    return selected


# --------------------------------------------------------------------------- #
#  Figure 1: canonicalization overlay (before / after)
# --------------------------------------------------------------------------- #

def _draw_canonical_overlay(ax, frame_uint8, landmarks_norm):
    """Overlay the four anchor landmarks and the three ROI markers."""
    H, W = frame_uint8.shape[:2]
    ax.imshow(frame_uint8)

    anchors_px = landmarks_norm[ANCHOR_INDICES] * (W - 1)
    canon_px = CANONICAL_ANCHORS.numpy() * (W - 1)

    ax.plot(anchors_px[:, 0], anchors_px[:, 1],
            'o', markersize=5.0, markerfacecolor='none',
            markeredgecolor=COLOR_ANCHOR, markeredgewidth=1.3, zorder=3)
    ax.plot(canon_px[:, 0], canon_px[:, 1],
            'x', markersize=4.5, markeredgecolor=COLOR_TARGET,
            markeredgewidth=1.2, zorder=4)

    for _, (cx, cy) in CANONICAL_ROIS:
        ax.plot(cx * (W - 1), cy * (H - 1), '+',
                markersize=6.0, markeredgecolor=COLOR_ROI,
                markeredgewidth=1.3, zorder=5)

    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_edgecolor('#555555')
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)


def make_figure_canonicalization(data, lmdb_path, chunk_length, device, out_path):
    """2 x 4 grid: top row raw frames, bottom row canonicalized frames.

    In each cell we overlay (i) the four anchor landmarks, (ii) their fixed
    canonical target positions, and (iii) the three canonical ROI markers.
    After canonicalization the anchor rings should coincide with the
    canonical crosses and the ROI markers should sit on forehead and cheek.
    """
    selected = _pick_yaw_bin_frames(data)
    warp = CanonicalWarp().to(device)

    fig, axes = plt.subplots(2, 4, figsize=(8.2, 4.6))

    for col, (ci, fi) in enumerate(selected):
        sid = data['sids'][ci]
        cid = data['chunk_ids'][ci]
        global_idx = cid * chunk_length + fi

        raw_all = load_raw_frames(lmdb_path, sid)
        raw_frame = raw_all[global_idx]

        lm_norm = data['landmarks'][ci, fi]
        abs_yaw = float(np.abs(data['poses'][ci, fi, 0]))

        frame_t = (torch.from_numpy(raw_frame).to(device, torch.float32)
                   .permute(2, 0, 1).unsqueeze(0).unsqueeze(2))
        lm_t = torch.from_numpy(lm_norm).to(device, torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            warped = warp(frame_t, lm_t)

        warped_img = warped.squeeze().permute(1, 2, 0).cpu().numpy()
        warped_img = np.clip(warped_img, 0, 255).astype(np.uint8)

        warped_lm = lm_norm.copy()
        warped_lm[ANCHOR_INDICES] = CANONICAL_ANCHORS.numpy()

        _draw_canonical_overlay(axes[0, col], raw_frame, lm_norm)
        axes[0, col].set_title(
            f"{YAW_BIN_LABELS[col]} ({YAW_BIN_RANGES[col]})",
            fontsize=9, pad=3)

        _draw_canonical_overlay(axes[1, col], warped_img, warped_lm)

    axes[0, 0].set_ylabel('Raw', fontsize=10, labelpad=4)
    axes[1, 0].set_ylabel('Canonical', fontsize=10, labelpad=4)

    legend_handles = [
        plt.Line2D([0], [0], marker='o', linestyle='', markersize=6,
                   markerfacecolor='none', markeredgecolor=COLOR_ANCHOR,
                   markeredgewidth=1.2, label='Observed anchors'),
        plt.Line2D([0], [0], marker='x', linestyle='', markersize=6,
                   markeredgecolor=COLOR_TARGET, markeredgewidth=1.2,
                   label='Canonical targets'),
        plt.Line2D([0], [0], marker='+', linestyle='', markersize=7,
                   markeredgecolor=COLOR_ROI, markeredgewidth=1.2,
                   label='ROI centers'),
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False,
               fontsize=9)
    fig.tight_layout(h_pad=0.6, w_pad=0.5, rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, format='pdf')
    plt.close(fig)
    logger.info(f"Saved {out_path}")


# --------------------------------------------------------------------------- #
#  Figure 2: physics weight on canonical frames
# --------------------------------------------------------------------------- #

def make_figure_physics_weight(data, lmdb_path, chunk_length, device,
                               out_path, weight_module):
    """4 x 3 grid: rows = yaw bins, cols = canonical frame, w^phys, weighted."""
    selected = _pick_yaw_bin_frames(data)
    warp = CanonicalWarp().to(device)

    fig, axes = plt.subplots(len(selected), 3, figsize=(5.8, 7.0))

    col_titles = ['Canonical', r'$w^{\mathrm{phys}}$', 'Weighted']

    rendered = []
    for (ci, fi) in selected:
        sid = data['sids'][ci]
        cid = data['chunk_ids'][ci]
        global_idx = cid * chunk_length + fi

        raw_all = load_raw_frames(lmdb_path, sid)
        raw_frame = raw_all[global_idx]
        lm_norm = data['landmarks'][ci, fi]
        tf = data['transforms'][ci, fi]
        abs_yaw = float(np.abs(data['poses'][ci, fi, 0]))

        frame_t = (torch.from_numpy(raw_frame).to(device, torch.float32)
                   .permute(2, 0, 1).unsqueeze(0).unsqueeze(2))
        lm_t = torch.from_numpy(lm_norm).to(device, torch.float32).unsqueeze(0).unsqueeze(0)
        tf_t = torch.from_numpy(tf).to(device, torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            warped = warp(frame_t, lm_t)
            w = weight_module(tf_t)

        canonical_img = np.clip(
            warped.squeeze().permute(1, 2, 0).cpu().numpy(), 0, 255
        ).astype(np.uint8)
        w_map = w.squeeze().cpu().numpy()
        weighted = (canonical_img.astype(np.float32)
                    * w_map[..., None]).clip(0, 255).astype(np.uint8)
        rendered.append((canonical_img, w_map, weighted, abs_yaw))

    for row, ((canonical_img, w_map, weighted, abs_yaw), label) in enumerate(
            zip(rendered, YAW_BIN_LABELS)):
        axes[row, 0].imshow(canonical_img)
        im = axes[row, 1].imshow(w_map, cmap='magma', vmin=0.0, vmax=1.0)
        axes[row, 2].imshow(weighted)
        axes[row, 0].set_ylabel(
            f"{label}\n$|\\mathrm{{yaw}}|={abs_yaw:.0f}^\\circ$",
            fontsize=9, labelpad=4)
        for c in range(3):
            axes[row, c].set_xticks([])
            axes[row, c].set_yticks([])
            for spine in axes[row, c].spines.values():
                spine.set_linewidth(0.5)
                spine.set_edgecolor('#555555')

    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=10, pad=4)

    fig.subplots_adjust(left=0.10, right=0.86, top=0.95, bottom=0.03,
                        hspace=0.12, wspace=0.08)
    cbar_ax = fig.add_axes([0.88, 0.10, 0.022, 0.78])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=7, width=0.5, length=2.5)
    cbar.outline.set_linewidth(0.5)
    cbar.set_label(r'$w^{\mathrm{phys}}$', fontsize=10, labelpad=6)
    fig.savefig(out_path, format='pdf')
    plt.close(fig)
    logger.info(f"Saved {out_path}")


# --------------------------------------------------------------------------- #
#  Combined pipeline figure (merges canonicalization + physics weight)
# --------------------------------------------------------------------------- #

def _pick_frames_near_yaws(data, targets, max_yaw=58.0):
    """Pick the valid-landmark frame closest to each target |yaw|, excluding
    the extreme regime (|yaw| >= max_yaw) where the 4-point homography becomes
    near-singular and the warp degenerates (see the >60 deg failure mode)."""
    abs_yaws = np.abs(data['poses'][..., 0])
    valid = _landmark_validity(data['landmarks']) & (abs_yaws < max_yaw)
    yy = np.where(valid, abs_yaws, 1e9)
    selected = []
    for t in targets:
        idx = int(np.argmin(np.abs(yy - t)))
        ci, fi = np.unravel_index(idx, abs_yaws.shape)
        selected.append((int(ci), int(fi)))
    return selected


def make_figure_pipeline(data, lmdb_path, chunk_length, device, out_path,
                         weight_module):
    """Single per-yaw pipeline figure. Columns are yaw bins; rows are the four
    pipeline stages (Raw, Canonical, w^phys, Weighted). The canonical frame
    appears exactly once, so this replaces the separate canonicalization and
    physics-weight figures without repeating content."""
    selected = _pick_frames_near_yaws(data, targets=[7.0, 22.0, 38.0, 52.0])
    warp = CanonicalWarp().to(device)
    ncols = len(selected)

    fig, axes = plt.subplots(4, ncols, figsize=(2.05 * ncols, 8.4))

    im = None
    for col, (ci, fi) in enumerate(selected):
        sid = data['sids'][ci]
        cid = data['chunk_ids'][ci]
        global_idx = cid * chunk_length + fi

        raw_all = load_raw_frames(lmdb_path, sid)
        raw_frame = raw_all[global_idx]
        lm_norm = data['landmarks'][ci, fi]
        tf = data['transforms'][ci, fi]
        abs_yaw = float(np.abs(data['poses'][ci, fi, 0]))

        frame_t = (torch.from_numpy(raw_frame).to(device, torch.float32)
                   .permute(2, 0, 1).unsqueeze(0).unsqueeze(2))
        lm_t = torch.from_numpy(lm_norm).to(device, torch.float32).unsqueeze(0).unsqueeze(0)
        tf_t = torch.from_numpy(tf).to(device, torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            warped = warp(frame_t, lm_t)
            w = weight_module(tf_t)

        warped_img = np.clip(
            warped.squeeze().permute(1, 2, 0).cpu().numpy(), 0, 255).astype(np.uint8)
        w_map = w.squeeze().cpu().numpy()
        weighted = (warped_img.astype(np.float32)
                    * w_map[..., None]).clip(0, 255).astype(np.uint8)
        warped_lm = lm_norm.copy()
        warped_lm[ANCHOR_INDICES] = CANONICAL_ANCHORS.numpy()

        # Row 0: raw frame with anchor/ROI overlay.
        _draw_canonical_overlay(axes[0, col], raw_frame, lm_norm)
        axes[0, col].set_title(
            r'$|\mathrm{{yaw}}|={:.0f}^\circ$'.format(abs_yaw), fontsize=9, pad=4)
        # Row 1: canonicalized frame with overlay.
        _draw_canonical_overlay(axes[1, col], warped_img, warped_lm)
        # Row 2: per-pixel Lambertian weight.
        im = axes[2, col].imshow(w_map, cmap='magma', vmin=0.0, vmax=1.0)
        # Row 3: weighted canonical frame.
        axes[3, col].imshow(weighted)
        for r in (2, 3):
            axes[r, col].set_xticks([]); axes[r, col].set_yticks([])
            for spine in axes[r, col].spines.values():
                spine.set_linewidth(0.5); spine.set_edgecolor('#555555')

    for r, lab in enumerate(['Raw', 'Canonical', r'$w^{\mathrm{phys}}$', 'Weighted']):
        axes[r, 0].set_ylabel(lab, fontsize=10, labelpad=4)

    legend_handles = [
        plt.Line2D([0], [0], marker='o', linestyle='', markersize=6,
                   markerfacecolor='none', markeredgecolor=COLOR_ANCHOR,
                   markeredgewidth=1.2, label='Observed anchors'),
        plt.Line2D([0], [0], marker='x', linestyle='', markersize=6,
                   markeredgecolor=COLOR_TARGET, markeredgewidth=1.2,
                   label='Canonical targets'),
        plt.Line2D([0], [0], marker='+', linestyle='', markersize=7,
                   markeredgecolor=COLOR_ROI, markeredgewidth=1.2,
                   label='ROI centers'),
    ]
    fig.legend(handles=legend_handles, loc='upper center',
               bbox_to_anchor=(0.49, 0.995), ncol=3, frameon=False, fontsize=8)

    fig.subplots_adjust(left=0.08, right=0.90, top=0.88, bottom=0.05,
                        hspace=0.10, wspace=0.06)
    cbar_ax = fig.add_axes([0.915, 0.06, 0.014, 0.80])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=7, width=0.5, length=2.5)
    cbar.outline.set_linewidth(0.5)
    cbar.set_label(r'$w^{\mathrm{phys}}$', fontsize=10, labelpad=6)
    fig.savefig(out_path, format='pdf')
    plt.close(fig)
    logger.info(f"Saved {out_path}")


# --------------------------------------------------------------------------- #
#  Figure 3: BVP waveforms on side-face clips
# --------------------------------------------------------------------------- #

def _bandpass(signal, fs):
    b, a = butter(1, [0.6 / fs * 2, 3.3 / fs * 2], btype='bandpass')
    return filtfilt(b, a, np.double(signal))


def _zscore(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


def _corr(a, b):
    a = _zscore(a)
    b = _zscore(b)
    return float(np.clip(np.mean(a * b), -1.0, 1.0))


def make_figure_waveforms(data, fs, out_path, side_yaw_threshold=30.0,
                          n_examples=4):
    """Wide strip of side-face clips where CanonicalPhys most improves BVP
    waveform fidelity over FactorizePhys. Clips are ranked by the gain in
    Pearson correlation to ground truth (continuous, unlike FFT-quantized HR
    error), and laid out as a single row for a full-width figure."""
    abs_yaws = np.abs(data['poses'][..., 0])
    median_abs_yaw = np.median(abs_yaws, axis=1)

    N = len(data['cp_pred'])
    cp_corr = np.full(N, -np.inf, dtype=np.float64)
    fp_corr = np.full(N, -np.inf, dtype=np.float64)
    for i in range(N):
        try:
            gt = _bandpass(np.cumsum(data['label'][i]), fs)
            cp_corr[i] = _corr(_bandpass(np.cumsum(data['cp_pred'][i]), fs), gt)
            fp_corr[i] = _corr(_bandpass(np.cumsum(data['fp_pred'][i]), fs), gt)
        except Exception as e:
            logger.debug(f"corr calc failed for chunk {i}: {e}")

    side_mask = median_abs_yaw >= side_yaw_threshold
    if int(side_mask.sum()) < n_examples:
        thresh = np.percentile(median_abs_yaw, 75)
        side_mask = median_abs_yaw >= thresh
        logger.warning(
            f"Few clips above {side_yaw_threshold:.0f} deg yaw; "
            f"falling back to top quartile (>={thresh:.1f} deg).")

    candidates = np.where(side_mask
                          & np.isfinite(cp_corr)
                          & np.isfinite(fp_corr))[0]
    if len(candidates) == 0:
        raise RuntimeError("No valid side-face chunks for waveform figure.")

    gain = cp_corr[candidates] - fp_corr[candidates]
    order = np.argsort(-gain)
    actual = min(n_examples, len(candidates))
    top = candidates[order[:actual]]

    fig, axes = plt.subplots(1, actual, figsize=(3.25 * actual, 2.7),
                             sharey=True, squeeze=False)
    axes_flat = axes.ravel()

    for col, idx in enumerate(top):
        gt = _zscore(_bandpass(np.cumsum(data['label'][idx]), fs))
        cp = _zscore(_bandpass(np.cumsum(data['cp_pred'][idx]), fs))
        fp = _zscore(_bandpass(np.cumsum(data['fp_pred'][idx]), fs))
        t = np.arange(len(gt)) / fs

        ax = axes_flat[col]
        ax.plot(t, gt, color=COLOR_GT, linewidth=1.4,
                label='Ground truth', zorder=3)
        ax.plot(t, fp, color=COLOR_FP, linewidth=1.1, alpha=0.9,
                label='FactorizePhys', zorder=2)
        ax.plot(t, cp, color=COLOR_CP, linewidth=1.1, alpha=0.9,
                label='CanonicalPhys', zorder=2)

        ax.set_xlabel('Time (s)', fontsize=8)
        if col == 0:
            ax.set_ylabel('BVP (z-score)', fontsize=8)
        ax.set_title(
            r'$|\mathrm{{yaw}}|={:.0f}^\circ$'.format(median_abs_yaw[idx])
            + '\n'
            + r'$r$: FP$=${:.2f}, CP$=${:.2f}'.format(fp_corr[idx], cp_corr[idx]),
            fontsize=8, pad=3)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(-2.8, 2.8)
        ax.tick_params(axis='both', labelsize=7)
        ax.grid(True, which='major', axis='both',
                color='#cccccc', linewidth=0.35, alpha=0.6)
        ax.set_axisbelow(True)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        for spine in ('left', 'bottom'):
            ax.spines[spine].set_linewidth(0.6)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',
               bbox_to_anchor=(0.5, 1.04), ncol=3, fontsize=9,
               frameon=False, handlelength=1.8, columnspacing=1.8)
    fig.tight_layout(rect=[0, 0, 1, 0.86], w_pad=1.2)
    fig.savefig(out_path, format='pdf')
    plt.close(fig)
    logger.info(f"Saved {out_path}")


# --------------------------------------------------------------------------- #
#  Main entry point
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Generate CanonicalPhys paper figures.")
    parser.add_argument(
        '--cp_ckpt', type=str,
        default='runs/train/exp_20260418_125444_17607291/checkpoints/canonicalphys_best.pth')
    parser.add_argument(
        '--fp_ckpt', type=str,
        default='archive/baselines/factorizephys/MMPD/factorizephys_best.pth')
    parser.add_argument('--data_config', type=str,
                        default='configs/datasets/mmpd.yaml')
    parser.add_argument('--cp_model_config', type=str,
                        default='configs/models/canonicalphys.yaml')
    parser.add_argument('--fp_model_config', type=str,
                        default='configs/models/factorizephys.yaml')
    parser.add_argument('--out_dir', type=str, default='figs')
    parser.add_argument('--max_chunks', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)])
    setup_cvpr_style()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # ---- Load configs --------------------------------------------------- #
    data_cfg = load_config(args.data_config)
    cp_cfg = deep_merge(data_cfg, load_config(args.cp_model_config))
    fp_cfg = deep_merge(data_cfg, load_config(args.fp_model_config))
    sync_chunk_length(cp_cfg)
    sync_chunk_length(fp_cfg)

    lmdb_path = cp_cfg.TEST.DATA.LMDB_PATH
    ds_dir = os.path.dirname(lmdb_path)
    for k, sub in [('LANDMARKS_PATH', 'landmarks'),
                   ('POSES_PATH',     'poses'),
                   ('VALID_MASK_PATH', 'valid_mask'),
                   ('TRANSFORMS_PATH', 'transforms'),
                   ('HOMOGRAPHY_PATH', 'homographies')]:
        if not getattr(cp_cfg.TEST.DATA, k, None):
            setattr(cp_cfg.TEST.DATA, k, os.path.join(ds_dir, sub))

    # ---- Build models --------------------------------------------------- #
    logger.info("Building CanonicalPhys ...")
    canonicalphys = build_canonicalphys(cp_cfg, device)
    logger.info(f"Loading CP weights: {args.cp_ckpt}")
    sd = torch.load(args.cp_ckpt, map_location=device, weights_only=True)
    canonicalphys.load_state_dict(sd)
    canonicalphys.eval()

    logger.info("Building FactorizePhys ...")
    factorizephys = build_factorizephys(fp_cfg, device)
    logger.info(f"Loading FP weights: {args.fp_ckpt}")
    sd = torch.load(args.fp_ckpt, map_location=device, weights_only=True)
    factorizephys.load_state_dict(sd)
    factorizephys.eval()

    # ---- Test dataset --------------------------------------------------- #
    logger.info("Loading test dataset ...")
    test_dataset = LMDBDataset(
        cp_cfg.TEST.DATA.DATASET,
        cp_cfg.TEST.DATA.LMDB_PATH,
        cp_cfg.TEST.DATA)
    logger.info(f"Test dataset: {len(test_dataset)} chunks")

    # ---- Inference ------------------------------------------------------ #
    logger.info("Running inference on test set ...")
    data = collect_test_data(
        canonicalphys, factorizephys, test_dataset,
        args.batch_size, device, max_chunks=args.max_chunks)
    logger.info(f"Collected data for {len(data['cp_pred'])} chunks")

    fs = int(cp_cfg.TEST.DATA.FS)
    chunk_length = int(test_dataset.chunk_length)

    # ---- Physics weight module for figure 2 ----------------------------- #
    # Access the live module from the CP model for consistent buffers.
    weight_module = canonicalphys.physics_weight

    # ---- Figures -------------------------------------------------------- #
    logger.info("Generating Figure: waveforms.pdf")
    make_figure_waveforms(
        data, fs=fs,
        out_path=os.path.join(args.out_dir, 'waveforms.pdf'),
        side_yaw_threshold=30.0, n_examples=4)

    try:
        logger.info("Generating Figure: canonical_pipeline.pdf (merged)")
        make_figure_pipeline(
            data, lmdb_path, chunk_length=chunk_length, device=device,
            out_path=os.path.join(args.out_dir, 'canonical_pipeline.pdf'),
            weight_module=weight_module)
    except Exception as e:
        logger.warning(f"Skipped canonical_pipeline.pdf: {e}")

    logger.info("Done. Figures written to %s", args.out_dir)


if __name__ == '__main__':
    main()
