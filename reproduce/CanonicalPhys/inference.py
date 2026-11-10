"""Inference entry point (predict on data, optionally with labels)."""

import argparse
import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.config import load_config, deep_merge, sync_chunk_length
from data.ubfc_loader import UBFCrPPGLoader
from data.pure_loader import PURELoader
from data.obf_loader import OBFLoader
from data.mmpd_loader import MMPDLoader
from data.lmdb_loader import LMDBDataset
from models.physnet import PhysNet_padding_Encoder_Decoder_MAX
from models.factorizephys import FactorizePhys
from models.efficientphys import EfficientPhys
from evaluation.metrics import calculate_metrics


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


def main():
    parser = argparse.ArgumentParser(description="rPPG Inference")
    parser.add_argument('--model_config', type=str, required=True,
                        help='Path to model config YAML')
    parser.add_argument('--data_config', type=str, required=True,
                        help='Path to dataset config YAML')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model .pth file (overrides INFERENCE.MODEL_PATH in config)')
    args = parser.parse_args()

    data_cfg = load_config(args.data_config)
    model_cfg = load_config(args.model_config)
    config = deep_merge(data_cfg, model_cfg)
    sync_chunk_length(config)

    # CLI --model_path overrides config
    if args.model_path:
        config.INFERENCE.MODEL_PATH = args.model_path

    # Create experiment directory (use SLURM_JOB_ID or PID to avoid collisions)
    job_id = os.environ.get('SLURM_JOB_ID', str(os.getpid()))
    exp_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{job_id}"
    exp_dir = os.path.join("runs", "inference", exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Set up logging (stdout only; SLURM captures to logs/)
    log_format = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format,
                        handlers=[logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger(__name__)

    logger.info(f"Inference directory: {exp_dir}")

    device = torch.device(config.DEVICE)

    # Load model
    model_name = config.MODEL.NAME
    logger.info(f"Model: {model_name}")
    drop_rate = float(getattr(config.MODEL, 'DROP_RATE', 0.0))
    if model_name == "Physnet":
        model = PhysNet_padding_Encoder_Decoder_MAX(
            frames=config.MODEL.PHYSNET.FRAME_NUM,
            drop_rate=drop_rate).to(device)
    elif model_name == "FactorizePhys":
        fp_cfg = config.MODEL.FACTORIZEPHYS
        md_config = {
            "FRAME_NUM": fp_cfg.FRAME_NUM,
            "MD_FSAM": fp_cfg.MD_FSAM,
            "MD_TYPE": fp_cfg.MD_TYPE,
            "MD_TRANSFORM": fp_cfg.MD_TRANSFORM,
            "MD_S": fp_cfg.MD_S,
            "MD_R": fp_cfg.MD_R,
            "MD_STEPS": fp_cfg.MD_STEPS,
            "MD_INFERENCE": fp_cfg.MD_INFERENCE,
            "MD_RESIDUAL": fp_cfg.MD_RESIDUAL,
            "INV_T": fp_cfg.INV_T,
            "ETA": float(fp_cfg.ETA),
            "RAND_INIT": fp_cfg.RAND_INIT,
            "align_channels": fp_cfg.ALIGN_CHANNELS,
        }
        in_channels = getattr(fp_cfg, 'IN_CHANNELS', 3)
        model = FactorizePhys(
            frames=fp_cfg.FRAME_NUM,
            md_config=md_config,
            in_channels=in_channels,
            dropout=drop_rate,
            device=device).to(device)
    elif model_name == "EfficientPhys":
        ep_cfg = config.MODEL.EFFICIENTPHYS
        model = EfficientPhys(
            in_channels=3,
            frame_depth=ep_cfg.FRAME_DEPTH,
            img_size=getattr(ep_cfg, 'IMG_SIZE', 72),
            dropout_rate1=drop_rate,
        ).to(device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if not os.path.exists(config.INFERENCE.MODEL_PATH):
        raise ValueError(
            f"Model path does not exist: {config.INFERENCE.MODEL_PATH}")
    model.load_state_dict(
        torch.load(config.INFERENCE.MODEL_PATH, map_location=device,
                   weights_only=True))
    logger.info(f"Loaded model from: {config.INFERENCE.MODEL_PATH}")

    # Load data
    logger.info("Loading inference data...")
    test_dataset = get_dataset(
        config.TEST.DATA.DATASET,
        config.TEST.DATA.DATA_PATH,
        config.TEST.DATA)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.INFERENCE.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True)

    # Determine if model needs extra frame (FactorizePhys uses temporal diff)
    needs_extra_frame = (model_name == "FactorizePhys")
    is_efficientphys = (model_name == "EfficientPhys")
    # Determine if model returns multiple outputs
    use_fsam = (model_name == "FactorizePhys" and
                getattr(config.MODEL.FACTORIZEPHYS, 'MD_FSAM', False) and
                getattr(config.MODEL.FACTORIZEPHYS, 'MD_INFERENCE', False))

    # Run inference
    model.eval()
    predictions = dict()
    labels = dict()

    _tqdm_disabled = not sys.stdout.isatty()
    logger.info("Running inference...")
    with torch.no_grad():
        for _, test_batch in enumerate(tqdm(test_loader, ncols=80,
                                            disable=_tqdm_disabled)):
            batch_size = test_batch[0].shape[0]
            data = test_batch[0].to(torch.float32).to(device)
            label = test_batch[1].to(torch.float32).to(device)

            if is_efficientphys:
                # EfficientPhys: NDCHW → (N*D, C, H, W), add extra frame
                N, D, C, H, W = data.shape
                data_flat = data.reshape(N * D, C, H, W)
                frame_depth = config.MODEL.EFFICIENTPHYS.FRAME_DEPTH
                aligned_len = (N * D) // frame_depth * frame_depth
                data_flat = data_flat[:aligned_len]
                last_frame = data_flat[-1:].clone()
                data_flat = torch.cat([data_flat, last_frame], dim=0)

                output = model(data_flat)
                pred_ppg = output[:aligned_len].squeeze(-1)
                actual_samples = min(batch_size, aligned_len // D)
                pred_ppg = pred_ppg[:actual_samples * D].reshape(actual_samples, D)
                label = label[:actual_samples].cpu()
                pred_ppg = pred_ppg.cpu()
                batch_size = actual_samples
            else:
                if needs_extra_frame:
                    last_frame = data[:, :, -1:, :, :]
                    data = torch.cat((data, last_frame), dim=2)

                output = model(data)
                if isinstance(output, tuple):
                    pred_ppg = output[0]
                else:
                    pred_ppg = output

                label = label.cpu()
                pred_ppg = pred_ppg.cpu()

            for idx in range(batch_size):
                subj_index = test_batch[2][idx]
                sort_index = int(test_batch[3][idx])
                if subj_index not in predictions:
                    predictions[subj_index] = dict()
                    labels[subj_index] = dict()
                predictions[subj_index][sort_index] = pred_ppg[idx]
                labels[subj_index][sort_index] = label[idx]

    # Save predictions
    pred_save_path = os.path.join(exp_dir, "predictions.npy")
    np.save(pred_save_path, predictions, allow_pickle=True)
    logger.info(f"Predictions saved to: {pred_save_path}")

    # Try to detect model's training dataset from training config
    train_dataset = "Unknown"
    model_path_str = config.INFERENCE.MODEL_PATH
    if model_path_str:
        train_exp_dir = os.path.dirname(os.path.dirname(model_path_str))
        train_config_path = os.path.join(train_exp_dir, "config.yaml")
        if os.path.exists(train_config_path):
            with open(train_config_path, 'r') as f:
                train_cfg = yaml.safe_load(f)
            try:
                train_dataset = train_cfg["TRAIN"]["DATA"]["DATASET"]
            except (KeyError, TypeError):
                pass

    # Calculate metrics if labels are available
    logger.info("Calculating metrics...")
    calculate_metrics(predictions, labels, config, save_dir=exp_dir,
                      train_dataset=train_dataset)

    logger.info(f"Inference complete. Results saved to {exp_dir}")


if __name__ == "__main__":
    main()
