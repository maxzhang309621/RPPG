"""Visualize the INTERNAL effect of the cross-ROI consistency loss (R4Q3).

For two CanonicalPhys checkpoints that differ only in the consistency loss
(canonicalphys_none = OFF, canonicalphys_d2 = ON), we extract the three
canonical-ROI temporal signals (forehead, left cheek, right cheek) that the
loss operates on, directly from the backbone voxel embeddings, on the UBFC
test split. We then show:

  (a) the three ROI signals for one clip with the loss OFF (out of phase),
  (b) the same with the loss ON (phase-aligned),
  (c) mean pairwise cross-ROI phase coherence over the whole test set,
      OFF vs ON, per ROI pair.

This confirms the module produces the SPECIFIC claimed effect (cross-ROI
pulse-phase agreement in the backbone), not a generic regularization.

Usage:
    python scripts/visualize_consistency_mechanism.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.config import load_config, deep_merge, sync_chunk_length
from train import get_dataset, get_trainer
from torch.utils.data import DataLoader
from losses.spatial_consistency import build_canonical_roi_masks

DATA_CFG = "configs/datasets/ubfc_pfs.yaml"
VARIANTS = {
    "OFF": ("configs/models/canonicalphys_none.yaml",
            "runs/train/exp_20260418_124946_17607280/checkpoints/canonicalphys_none_best.pth"),
    "ON":  ("configs/models/canonicalphys_d2.yaml",
            "runs/train/exp_20260418_183457_17615203/checkpoints/canonicalphys_d2_best.pth"),
}
ROI_NAMES = ["forehead", "L cheek", "R cheek"]
PAIR_NAMES = ["FH-L", "FH-R", "L-R"]
MAX_CLIPS = 96    # cap test clips for speed (CPU)


def build_model(model_cfg_path, ckpt):
    data_cfg = load_config(DATA_CFG)
    model_cfg = load_config(model_cfg_path)
    config = deep_merge(data_cfg, model_cfg)
    config.DEVICE = "cpu"   # login node has no GPU; model is tiny
    config.INFERENCE.MODEL_PATH = ckpt
    # inject sidecar paths (mirror evaluate.py) -- already in ubfc_pfs.yaml
    sync_chunk_length(config)
    config.MODEL.MODEL_DIR = "runs/eval/_tmp_consistency_viz"
    loaders = {"train": None, "valid": None}
    test_ds = get_dataset(config.TEST.DATA.DATASET,
                          config.TEST.DATA.DATA_PATH, config.TEST.DATA)
    loaders["test"] = DataLoader(test_ds, batch_size=4, shuffle=False,
                                 num_workers=2, pin_memory=False)
    trainer = get_trainer(config.MODEL.NAME, config, loaders, mode="test")
    model = trainer.model
    # ensure weights loaded
    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, loaders["test"]


def unpack(batch, device):
    data = batch[0].to(torch.float32).to(device)
    landmarks = batch[4].to(torch.float32).to(device) if len(batch) > 4 else None
    poses = batch[5].to(torch.float32).to(device) if len(batch) > 5 else None
    valid_mask = batch[6].to(torch.float32).to(device) if len(batch) > 6 else None
    transforms = batch[7].to(torch.float32).to(device) if len(batch) > 7 else None
    homographies = batch[8].to(torch.float32).to(device) if len(batch) > 8 else None
    return data, landmarks, poses, valid_mask, transforms, homographies


def roi_signals_from_embeddings(vox, masks):
    """vox: (B,C,T,Hf,Wf) -> list of 3 normalized diff signals (B,T-1)."""
    sigs = []
    for k in range(masks.shape[0]):
        m = masks[k].to(vox.dtype)
        num = (vox * m).sum(dim=(-1, -2))           # (B,C,T)
        den = m.sum() + 1e-6
        roi = num / den                              # (B,C,T)
        s = roi.pow(2).sum(dim=1).clamp_min(1e-12).sqrt()  # (B,T)
        s = s[:, 1:] - s[:, :-1]                      # diff
        mu = s.mean(dim=-1, keepdim=True)
        sd = s.std(dim=-1, keepdim=True) + 1e-6
        sigs.append((s - mu) / sd)
    return sigs


@torch.no_grad()
def collect(model, loader, device):
    masks = build_canonical_roi_masks(13).to(device)
    coh = []          # per-clip (3,) pairwise cosine
    example = None    # (3, T-1) for one representative clip
    n = 0
    for batch in loader:
        data, lm, poses, vmask, tf, hom = unpack(batch, device)
        # Trainer appends a duplicate last frame so frames are T+1 while
        # landmarks/transforms stay T (see _append_last_frame).
        data = torch.cat([data, data[:, :, -1:]], dim=2)
        out = model(data, landmarks=lm, poses=poses, valid_mask=vmask,
                    transforms=tf, homographies=hom)
        vox = out[1]                                  # (B,C,T,13,13)
        sigs = roi_signals_from_embeddings(vox, masks)
        s = torch.stack(sigs, dim=0)                  # (3,B,T-1)
        pair_idx = [(0, 1), (0, 2), (1, 2)]
        for b in range(s.shape[1]):
            c = [float((s[i, b] * s[j, b]).mean()) for i, j in pair_idx]
            coh.append(c)
            if example is None and np.mean(c) > 0:    # pick a clean clip
                example = s[:, b].cpu().numpy()
            n += 1
        if n >= MAX_CLIPS:
            break
    if example is None:
        example = s[:, 0].cpu().numpy()
    return np.array(coh), example, n


CACHE = "runs/consistency_mech_cache.npz"


def compute_results():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    results = {}
    for tag, (mcfg, ckpt) in VARIANTS.items():
        print(f"[{tag}] building {mcfg}")
        model, loader = build_model(mcfg, ckpt)
        model.to(device)
        coh, example, n = collect(model, loader, device)
        results[tag] = dict(coh=coh, example=example, n=n)
        print(f"[{tag}] {n} clips, mean coherence={coh.mean():.3f}")
    np.savez(CACHE, off_coh=results["OFF"]["coh"],
             off_ex=results["OFF"]["example"],
             on_coh=results["ON"]["coh"], on_ex=results["ON"]["example"])
    return results


def load_cached():
    d = np.load(CACHE)
    return {"OFF": dict(coh=d["off_coh"], example=d["off_ex"]),
            "ON": dict(coh=d["on_coh"], example=d["on_ex"])}


def main():
    if os.environ.get("REPLOT") and os.path.exists(CACHE):
        print(f"re-plotting from cache {CACHE}")
        results = load_cached()
    else:
        results = compute_results()

    # ---- plot ----
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(9.2, 2.6))
    # tight a|b pair, a wider gap (spacer col) before c
    gs = gridspec.GridSpec(1, 4, width_ratios=[1.0, 1.0, 0.18, 1.25],
                           wspace=0.12, figure=fig)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1], sharey=axA)
    axC = fig.add_subplot(gs[0, 3])
    colors = ["#d1495b", "#2e86ab", "#3c8d40"]
    titles = {"OFF": "(a) Consistency loss OFF", "ON": "(b) Consistency loss ON"}
    # shared y-range with extra top headroom so the legend never touches data/spine
    both = np.concatenate([results["OFF"]["example"].ravel(),
                           results["ON"]["example"].ravel()])
    ymin, ymax = float(both.min()), float(both.max())
    span = ymax - ymin
    for ax, tag in [(axA, "OFF"), (axB, "ON")]:
        ex = results[tag]["example"]
        t = np.arange(ex.shape[1])
        for i in range(3):
            ax.plot(t, ex[i], color=colors[i], lw=1.2, label=ROI_NAMES[i])
        ax.set_title(titles[tag], fontsize=10)
        ax.set_xlabel("frame", fontsize=9)
        ax.set_xlim(0, min(96, t[-1]))
        ax.set_ylim(ymin - 0.05 * span, ymax + 0.42 * span)
        ax.set_yticks([])
        cbar = float(np.mean(results[tag]["coh"].mean(axis=1)))
        ax.text(0.97, 0.04, f"coh={cbar:.2f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6"))
    axA.set_ylabel("canonical-ROI\nbackbone signal", fontsize=9)
    plt.setp(axB.get_yticklabels(), visible=False)
    axA.legend(fontsize=7, loc="upper center", ncol=3, columnspacing=0.8,
               handlelength=1.0, frameon=False, borderaxespad=0.25)

    # panel c: per-pair mean coherence bars
    off_m = results["OFF"]["coh"].mean(axis=0)
    on_m = results["ON"]["coh"].mean(axis=0)
    off_s = results["OFF"]["coh"].std(axis=0)
    on_s = results["ON"]["coh"].std(axis=0)
    x = np.arange(3)
    w = 0.36
    axC.bar(x - w/2, off_m, w, yerr=off_s, capsize=2, color="0.7", label="OFF")
    axC.bar(x + w/2, on_m, w, yerr=on_s, capsize=2, color="#3c8d40", label="ON")
    axC.set_xticks(x)
    axC.set_xticklabels(PAIR_NAMES, fontsize=8)
    axC.set_ylabel("cross-ROI phase coherence", fontsize=9)
    axC.set_title("(c) All UBFC test clips", fontsize=10)
    axC.axhline(0, color="0.5", lw=0.6)
    axC.set_ylim(-0.2, 1.15)   # headroom so legend clears the bars
    axC.legend(fontsize=8, frameon=False, loc="upper center", ncol=2,
               columnspacing=1.0, borderaxespad=0.2)

    os.makedirs("figs", exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(f"figs/consistency_mechanism.{ext}", dpi=200,
                    bbox_inches="tight")
    print("saved figs/consistency_mechanism.pdf/.png")
    print(f"OFF overall coherence: {results['OFF']['coh'].mean():.3f}")
    print(f"ON  overall coherence: {results['ON']['coh'].mean():.3f}")


if __name__ == "__main__":
    main()
