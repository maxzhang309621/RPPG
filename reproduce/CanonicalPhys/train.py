"""Training entry point."""

import argparse
import logging
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


def set_seed(seed):
    """Set random seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

from utils.config import load_config, deep_merge, config_to_dict, sync_chunk_length
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
    """Create a dataset based on the dataset name.

    Automatically uses LMDB if LMDB_PATH exists on disk.
    Falls back to .npy-based loaders otherwise.
    """
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


def get_trainer(model_name, config, data_loader, mode="train"):
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
    parser = argparse.ArgumentParser(description="Train rPPG model")
    parser.add_argument('--model_config', type=str, required=True,
                        help='Path to model config YAML')
    parser.add_argument('--data_config', type=str, required=True,
                        help='Path to dataset config YAML')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained backbone .pth (sets PRETRAINED_BACKBONE)')
    parser.add_argument('--ablation', type=str, default=None,
                        help='Ablation variant: no_pose_gating, no_spatial_prior, '
                             'no_temporal_conv, no_fsam')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility (default 0)')
    args = parser.parse_args()

    set_seed(args.seed)

    data_cfg = load_config(args.data_config)
    model_cfg = load_config(args.model_config)
    config = deep_merge(data_cfg, model_cfg)
    sync_chunk_length(config)
    config.SEED = args.seed

    # CLI --pretrained injects PRETRAINED_BACKBONE into model config
    if args.pretrained:
        model_section = getattr(config.MODEL, config.MODEL.NAME.upper(), None)
        if model_section is not None:
            model_section.PRETRAINED_BACKBONE = args.pretrained

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
            raise ValueError(f"Unknown ablation: {args.ablation}. "
                             f"Options: {list(ablation_map.keys())}")
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
    exp_dir = os.path.join("runs", "train", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    config.MODEL.MODEL_DIR = os.path.join(exp_dir, "checkpoints")

    # Set up logging (stdout only; SLURM captures to logs/)
    log_format = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format,
                        handlers=[logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger(__name__)

    logger.info(f"Experiment directory: {exp_dir}")

    # Create data loaders
    data_loader = dict()

    logger.info("Loading training data...")
    train_dataset = get_dataset(
        config.TRAIN.DATA.DATASET,
        config.TRAIN.DATA.DATA_PATH,
        config.TRAIN.DATA)
    data_loader["train"] = DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True)

    if hasattr(config, 'VALID') and hasattr(config.VALID, 'DATA'):
        logger.info("Loading validation data...")
        valid_dataset = get_dataset(
            config.VALID.DATA.DATASET,
            config.VALID.DATA.DATA_PATH,
            config.VALID.DATA)
        data_loader["valid"] = DataLoader(
            valid_dataset,
            batch_size=config.TRAIN.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True)
    else:
        logger.info("No VALID config found, skipping validation.")
        data_loader["valid"] = None

    if hasattr(config, 'TEST') and hasattr(config.TEST, 'DATA'):
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
    else:
        logger.info("No TEST config found, skipping test.")
        data_loader["test"] = None

    # Save configuration for reproducibility
    config_save_path = os.path.join(exp_dir, "config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config_to_dict(config), f, default_flow_style=False, sort_keys=False)
    logger.info(f"Configuration saved to {config_save_path}")

    # Create trainer and run
    model_name = config.MODEL.NAME
    logger.info(f"Model: {model_name}")
    trainer = get_trainer(model_name, config, data_loader, mode="train")
    trainer.train(data_loader)
    if data_loader.get("test") is not None:
        trainer.test(data_loader)
    else:
        logger.info("Skipping test (no test data configured).")

    logger.info(f"Training complete. Results saved to {exp_dir}")


if __name__ == "__main__":
    main()
