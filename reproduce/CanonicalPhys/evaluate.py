"""Evaluation entry point (test a trained model)."""

import argparse
import logging
import os
import sys
from datetime import datetime

import torch
import yaml
from torch.utils.data import DataLoader

from utils.config import load_config, deep_merge, sync_chunk_length
from data.ubfc_loader import UBFCrPPGLoader
from data.pure_loader import PURELoader
from data.obf_loader import OBFLoader
from data.mmpd_loader import MMPDLoader
from data.lmdb_loader import LMDBDataset
from trainers.physnet_trainer import PhysnetTrainer
from trainers.factorizephys_trainer import FactorizePhysTrainer
from trainers.efficientphys_trainer import EfficientPhysTrainer
from trainers.landmarkphys_trainer import LandmarkPhysTrainer


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
    parser = argparse.ArgumentParser(description="Evaluate rPPG model")
    parser.add_argument('--model_config', type=str, required=True,
                        help='Path to model config YAML')
    parser.add_argument('--data_config', type=str, required=True,
                        help='Path to dataset config YAML')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model .pth file (overrides INFERENCE.MODEL_PATH in config)')
    parser.add_argument('--ablation', type=str, default=None,
                        help='Ablation variant: no_pose_gating, no_spatial_prior, '
                             'no_temporal_conv, no_fsam')
    args = parser.parse_args()

    data_cfg = load_config(args.data_config)
    model_cfg = load_config(args.model_config)
    config = deep_merge(data_cfg, model_cfg)
    sync_chunk_length(config)

    # CLI --model_path overrides config
    if args.model_path:
        config.INFERENCE.MODEL_PATH = args.model_path

    # CLI --ablation sets ablation flags on LandmarkPhys config
    if args.ablation and config.MODEL.NAME == "LandmarkPhys":
        lp = config.MODEL.LANDMARKPHYS
        ablation_map = {
            'no_pose_gating':   {'USE_POSE_GATING': False},
            'no_spatial_prior': {'USE_SPATIAL_PRIOR': False},
            'no_temporal_conv': {'USE_TEMPORAL_CONV': False},
            'no_fsam':          {'MD_FSAM': False, 'MD_INFERENCE': False},
        }
        if args.ablation not in ablation_map:
            raise ValueError(f"Unknown ablation: {args.ablation}")
        for key, val in ablation_map[args.ablation].items():
            setattr(lp, key, val)

    # Auto-inject landmark/pose paths for LandmarkPhys and CanonicalPhys
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

    # Create experiment directory (use SLURM_JOB_ID or PID to avoid collisions)
    job_id = os.environ.get('SLURM_JOB_ID', str(os.getpid()))
    exp_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{job_id}"
    exp_dir = os.path.join("runs", "eval", exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Set up logging (stdout only; SLURM captures to logs/)
    log_format = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format,
                        handlers=[logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger(__name__)

    logger.info(f"Evaluation directory: {exp_dir}")

    # Set model dir (not used for saving in eval mode, but needed by trainer)
    config.MODEL.MODEL_DIR = os.path.join(exp_dir, "checkpoints")

    # Create data loaders (only test is needed)
    data_loader = dict()
    data_loader["train"] = None
    data_loader["valid"] = None

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

    # Try to detect model's training dataset from training config
    train_dataset = "Unknown"
    model_path = config.INFERENCE.MODEL_PATH
    if model_path:
        # Model path is like: runs/train/exp_xxx/checkpoints/physnet_best.pth
        # Training config is at: runs/train/exp_xxx/config.yaml
        train_exp_dir = os.path.dirname(os.path.dirname(model_path))
        train_config_path = os.path.join(train_exp_dir, "config.yaml")
        if os.path.exists(train_config_path):
            with open(train_config_path, 'r') as f:
                train_cfg = yaml.safe_load(f)
            try:
                train_dataset = train_cfg["TRAIN"]["DATA"]["DATASET"]
            except (KeyError, TypeError):
                pass
        if train_dataset == "Unknown":
            logger.warning(
                f"Could not find training dataset info. "
                f"Looked for {train_config_path}")

    # Create trainer in test mode and run
    model_name = config.MODEL.NAME
    logger.info(f"Model: {model_name}")
    trainer = get_trainer(model_name, config, data_loader, mode="test")
    trainer.test(data_loader, save_dir=exp_dir, train_dataset=train_dataset)

    logger.info(f"Evaluation complete. Results saved to {exp_dir}")


if __name__ == "__main__":
    main()
