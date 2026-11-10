"""Pose-stratified evaluation entry point.

Runs standard evaluation and additionally reports metrics per pose bin
(frontal, mild, moderate, large yaw angle).

Usage:
    python evaluate_pose_stratified.py \
        --model_config configs/models/factorizephys.yaml \
        --data_config configs/datasets/pure.yaml \
        --model_path runs/train/exp_xxx/checkpoints/factorizephys_best.pth \
        --pose_splits datasets/PURE/pose_splits_cl160.json
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import torch
import yaml
from torch.utils.data import DataLoader

from utils.config import load_config, deep_merge, sync_chunk_length
from data.lmdb_loader import LMDBDataset
from data.ubfc_loader import UBFCrPPGLoader
from data.pure_loader import PURELoader
from data.obf_loader import OBFLoader
from data.mmpd_loader import MMPDLoader
from trainers.physnet_trainer import PhysnetTrainer
from trainers.factorizephys_trainer import FactorizePhysTrainer
from trainers.efficientphys_trainer import EfficientPhysTrainer
from trainers.landmarkphys_trainer import LandmarkPhysTrainer
from evaluation.pose_stratified import calculate_pose_stratified_metrics


def get_dataset(dataset_name, data_path, config_data):
    """Create a dataset based on the dataset name."""
    lmdb_path = getattr(config_data, 'LMDB_PATH', '')
    if lmdb_path and os.path.exists(lmdb_path):
        return LMDBDataset(dataset_name, lmdb_path, config_data)

    if dataset_name == "UBFC-rPPG":
        return UBFCrPPGLoader(dataset_name, data_path, config_data)
    elif dataset_name == "PURE":
        return PURELoader(dataset_name, data_path, config_data)
    elif dataset_name == "OBF":
        return OBFLoader(dataset_name, data_path, config_data)
    elif dataset_name == "MMPD":
        return MMPDLoader(dataset_name, data_path, config_data)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_trainer(model_name, config, data_loader, mode="test"):
    """Create a trainer based on the model name."""
    if model_name == "Physnet":
        return PhysnetTrainer(config, data_loader, mode=mode)
    elif model_name == "FactorizePhys":
        return FactorizePhysTrainer(config, data_loader, mode=mode)
    elif model_name == "EfficientPhys":
        return EfficientPhysTrainer(config, data_loader, mode=mode)
    elif model_name in ("LandmarkPhys", "CanonicalPhys"):
        return LandmarkPhysTrainer(config, data_loader, mode=mode)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Pose-stratified evaluation of rPPG model")
    parser.add_argument('--model_config', type=str, required=True)
    parser.add_argument('--data_config', type=str, required=True)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--pose_splits', type=str, required=True,
                        help='Path to pose_splits_clXXX.json')
    args = parser.parse_args()

    data_cfg = load_config(args.data_config)
    model_cfg = load_config(args.model_config)
    config = deep_merge(data_cfg, model_cfg)
    sync_chunk_length(config)

    if args.model_path:
        config.INFERENCE.MODEL_PATH = args.model_path

    # Auto-inject landmark/pose/transforms paths
    if config.MODEL.NAME in ("LandmarkPhys", "CanonicalPhys"):
        for split in ['TRAIN', 'VALID', 'TEST']:
            if hasattr(config, split) and hasattr(getattr(config, split), 'DATA'):
                split_data = getattr(config, split).DATA
                lmdb_path = getattr(split_data, 'LMDB_PATH', '')
                if lmdb_path:
                    ds_dir = os.path.dirname(lmdb_path)
                    if not getattr(split_data, 'LANDMARKS_PATH', None):
                        split_data.LANDMARKS_PATH = os.path.join(
                            ds_dir, "landmarks")
                    if not getattr(split_data, 'POSES_PATH', None):
                        split_data.POSES_PATH = os.path.join(ds_dir, "poses")
                    if not getattr(split_data, 'VALID_MASK_PATH', None):
                        split_data.VALID_MASK_PATH = os.path.join(
                            ds_dir, "valid_mask")
                    if not getattr(split_data, 'TRANSFORMS_PATH', None):
                        split_data.TRANSFORMS_PATH = os.path.join(
                            ds_dir, "transforms")
                    if not getattr(split_data, 'HOMOGRAPHIES_PATH', None):
                        hom_path = os.path.join(ds_dir, "homographies")
                        if os.path.isdir(hom_path):
                            split_data.HOMOGRAPHIES_PATH = hom_path

    job_id = os.environ.get('SLURM_JOB_ID', str(os.getpid()))
    exp_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{job_id}"
    exp_dir = os.path.join("runs", "eval", exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    log_format = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format,
                        handlers=[logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger(__name__)
    logger.info(f"Pose-stratified evaluation directory: {exp_dir}")

    config.MODEL.MODEL_DIR = os.path.join(exp_dir, "checkpoints")

    data_loader = {"train": None, "valid": None}
    logger.info("Loading test data...")
    test_dataset = get_dataset(
        config.TEST.DATA.DATASET,
        config.TEST.DATA.DATA_PATH,
        config.TEST.DATA)
    data_loader["test"] = DataLoader(
        test_dataset,
        batch_size=config.INFERENCE.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True)

    # Detect training dataset
    train_dataset = "Unknown"
    model_path = config.INFERENCE.MODEL_PATH
    if model_path:
        train_exp_dir = os.path.dirname(os.path.dirname(model_path))
        train_config_path = os.path.join(train_exp_dir, "config.yaml")
        if os.path.exists(train_config_path):
            with open(train_config_path, 'r') as f:
                train_cfg = yaml.safe_load(f)
            try:
                train_dataset = train_cfg["TRAIN"]["DATA"]["DATASET"]
            except (KeyError, TypeError):
                pass

    # Run standard evaluation (also returns predictions for pose analysis)
    model_name = config.MODEL.NAME
    logger.info(f"Model: {model_name}")
    trainer = get_trainer(model_name, config, data_loader, mode="test")
    predictions, labels = trainer.test(
        data_loader, save_dir=exp_dir, train_dataset=train_dataset)

    # Run pose-stratified evaluation
    logger.info("Running pose-stratified analysis...")
    calculate_pose_stratified_metrics(
        predictions, labels, config,
        pose_splits_path=args.pose_splits,
        save_dir=exp_dir,
        train_dataset=train_dataset)

    logger.info(f"Evaluation complete. Results saved to {exp_dir}")


if __name__ == "__main__":
    main()
