"""Verify LMDBDataset returns transforms as the 8th tuple element when configured."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from data.lmdb_loader import LMDBDataset
from utils.config import load_config, deep_merge


def test_loader_returns_transforms():
    data_cfg = load_config("configs/datasets/ubfc.yaml")
    model_cfg = load_config("configs/models/factorizephys.yaml")
    config = deep_merge(data_cfg, model_cfg)

    train_data_cfg = config.TRAIN.DATA
    project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    lmdb_path = train_data_cfg.LMDB_PATH
    if not os.path.isabs(lmdb_path):
        lmdb_path = os.path.join(project_dir, lmdb_path)
    ds_dir = os.path.dirname(lmdb_path)
    train_data_cfg.LANDMARKS_PATH = os.path.join(ds_dir, "landmarks")
    train_data_cfg.POSES_PATH = os.path.join(ds_dir, "poses")
    train_data_cfg.VALID_MASK_PATH = os.path.join(ds_dir, "valid_mask")
    train_data_cfg.TRANSFORMS_PATH = os.path.join(ds_dir, "transforms")

    ds = LMDBDataset(train_data_cfg.DATASET, lmdb_path, train_data_cfg)
    sample = ds[0]
    assert len(sample) == 8, f"Expected 8-tuple, got {len(sample)}"
    data, labels, sid, chunk_id, lm, pose, vm, tf = sample
    assert tf.shape[1:] == (4, 4)
    assert tf.dtype == np.float32
    print(f"transforms: shape={tf.shape} dtype={tf.dtype}")
    print("OK")


if __name__ == "__main__":
    test_loader_returns_transforms()
