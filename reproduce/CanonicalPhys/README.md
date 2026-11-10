# CanonicalPhys: Pose-Robust Remote Photoplethysmography via Canonical-Space Priors

Official implementation of **CanonicalPhys** (IJCB 2026).

CanonicalPhys treats head pose as a **coordinate-structural** nuisance rather than a
data-augmentation problem: it prepends a differentiable four-point homography that maps
every frame into a **canonical face coordinate system**, in which three classical rPPG
priors become **zero-added-parameter** components on top of an unmodified
[FactorizePhys](https://github.com/PhysiologicAILab/FactorizePhys) backbone. At an
identical parameter count, it cuts MMPD's frontal-to-large-yaw MAE degradation from
1.60x to 1.33x and improves cross-dataset generalization on pose-rich targets.

> **Paths are relative to the repo root** — run all commands from there. Raw-dataset
> locations are `PATH/TO/RAW_DATASETS/...` placeholders in `configs/datasets/*.yaml`;
> edit the `DATA_PATH` fields to point at your local copies.

## Method

A four-stage pipeline around a frozen-capacity backbone; **none of the components adds
trainable parameters**:

1. **Canonicalization** — parameter-free 4-point homography on eye/mouth landmarks (`models/canonical_warp.py`).
2. **Per-pixel Lambertian weighting** — physics-informed weight from canonical normals + facial rotation (`models/physics_weight.py`).
3. **Cross-ROI consistency loss** — pulse-phase agreement across canonical forehead/cheek features (`losses/spatial_consistency.py`).
4. **POS distillation** — from a handcrafted POS pseudo-BVP on canonical forehead pixels (`losses/pos_distillation.py`).

Full model: `models/canonical_phys.py`.

## Installation

```bash
pip install -r requirements.txt   # Python 3.10+, PyTorch 2.x
```

## Data preparation

Obtain the four public datasets (**UBFC-rPPG**, **PURE**, **OBF**, **MMPD**) and set
`DATA_PATH` in each `configs/datasets/<name>.yaml`.

Download the RetinaFace face-detector weights
([Google Drive](https://drive.google.com/file/d/1P6r36A6DYdk1IKLFnXQWbyf0J0pdmfNL/view?usp=drive_link))
to `weights/retinaface/RetinaFace_Resnet50_Final.pth` (used only for face cropping). The MediaPipe
landmarker and the canonical buffers are already bundled under `weights/`.

Preprocess **per dataset** (example: MMPD):

```bash
python preprocess_lmdb.py             --data_config configs/datasets/mmpd.yaml  # face-crop -> LMDB
python extract_landmarks_mediapipe.py --data_config configs/datasets/mmpd.yaml  # landmark sidecar
python compute_poses_mediapipe.py     --data_config configs/datasets/mmpd.yaml  # pose + transform sidecars
python build_pose_splits.py --model_config configs/models/canonicalphys.yaml \
    --data_config configs/datasets/mmpd.yaml                                     # pose-stratified eval only
```

This writes `datasets/MMPD/{lmdb, landmarks, poses, transforms, valid_mask}`.

## Pretrained models

Three ready-to-use checkpoints ship in `checkpoints/` (each < 300 KB) — evaluate directly,
no training required:

| Checkpoint | Model | Trained on |
|---|---|---|
| `canonicalphys_ubfc.pth` | CanonicalPhys | UBFC-rPPG (headline cross-dataset) |
| `canonicalphys_mmpd.pth` | CanonicalPhys | MMPD (in-domain / pose-stratified) |
| `factorizephys_ubfc.pth` | FactorizePhys | UBFC-rPPG (matched baseline) |

Pass one via `--model_path` with the **matching** `--model_config`
(`configs/models/canonicalphys.yaml`, or `factorizephys_fn128.yaml` for the baseline);
see Evaluation below.

## Training

```bash
python train.py --model_config configs/models/canonicalphys.yaml \
    --data_config configs/datasets/ubfc.yaml --seed 0
```

Swap `--data_config` for other sources; use `factorizephys_fn128.yaml` for the matched
128-frame baseline. Reported numbers average seeds 0/1/2. Checkpoints land in `runs/train/`.

## Evaluation

```bash
# Cross-dataset (train one dataset, test another): MAE / RMSE / MAPE / Pearson r / SNR
python evaluate.py --model_config configs/models/canonicalphys.yaml \
    --data_config configs/datasets/mmpd.yaml --model_path checkpoints/canonicalphys_ubfc.pth

# Pose-stratified (per-|yaw| bin on MMPD; needs build_pose_splits above)
python evaluate_pose_stratified.py --model_config configs/models/canonicalphys.yaml \
    --data_config configs/datasets/mmpd.yaml --model_path checkpoints/canonicalphys_mmpd.pth \
    --pose_splits datasets/MMPD/pose_splits.json
```

Aggregate seeds with `scripts/aggregate_seed_results.py`, run the paired test with
`scripts/paired_test_mmpd.py`, and regenerate the paper figures with
`scripts/visualize_canonicalphys_figs.py`.

<!-- ## Citation

```bibtex
@inproceedings{canonicalphys2026,
  title     = {CanonicalPhys: Pose-Robust Remote Photoplethysmography via Canonical-Space Priors},
  author    = {TODO: author list},
  booktitle = {IEEE International Joint Conference on Biometrics},
  year      = {2026}
}
``` -->

## Acknowledgements

Builds on the FactorizePhys backbone and [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox)
preprocessing; uses [MediaPipe](https://github.com/google-ai-edge/mediapipe) Face Mesh
(Apache-2.0; the bundled `face_landmarker.task` is redistributed under that license) and
RetinaFace. Thanks to the UBFC-rPPG, PURE, OBF, and MMPD dataset authors.
