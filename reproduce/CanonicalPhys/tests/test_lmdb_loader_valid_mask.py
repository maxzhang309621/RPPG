"""Verify LMDBDataset returns valid_mask as the 7th tuple element when configured."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from data.lmdb_loader import LMDBDataset
from utils.config import load_config, deep_merge


def test_loader_returns_valid_mask():
    data_cfg = load_config("configs/datasets/ubfc.yaml")
    model_cfg = load_config("configs/models/factorizephys.yaml")
    config = deep_merge(data_cfg, model_cfg)

    train_data_cfg = config.TRAIN.DATA
    # Inject sidecar paths the same way the trainer does at runtime.
    project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    lmdb_path = train_data_cfg.LMDB_PATH
    if not os.path.isabs(lmdb_path):
        lmdb_path = os.path.join(project_dir, lmdb_path)
    ds_dir = os.path.dirname(lmdb_path)
    train_data_cfg.LANDMARKS_PATH = os.path.join(ds_dir, "landmarks")
    train_data_cfg.POSES_PATH = os.path.join(ds_dir, "poses")
    train_data_cfg.VALID_MASK_PATH = os.path.join(ds_dir, "valid_mask")

    ds = LMDBDataset(train_data_cfg.DATASET, lmdb_path, train_data_cfg)
    sample = ds[0]
    print(f"sample length: {len(sample)}")
    assert len(sample) == 7, f"Expected 7-tuple, got {len(sample)}"
    data, labels, sid, chunk_id, lm, pose, vm = sample
    assert lm.shape[1:] == (68, 2)
    assert pose.shape[1:] == (3,)
    assert vm.ndim == 1
    assert vm.dtype == np.float32
    assert (vm >= 0).all() and (vm <= 1).all()
    print(f"valid_mask: shape={vm.shape} mean={vm.mean():.3f}")
    print("OK")


def _run_all():
    test_loader_returns_valid_mask()


if __name__ == "__main__":
    _run_all()
