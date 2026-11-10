"""Trainer for landmark-augmented rPPG models.

Supports two model families that share the same forward interface
(frames + landmarks + poses + valid_mask + transforms):
    - LandmarkPhys: v1 LandmarkAttn or v2 VisibilityMaskAttn
    - CanonicalPhys: pose-canonicalizing homography warp + FactorizePhys

Dispatch is based on ``config.MODEL.NAME``.
"""

import logging
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from models.landmark_phys import LandmarkPhys
from models.canonical_phys import CanonicalPhys
from losses.neg_pearson import Neg_Pearson
from losses.pos_distillation import POSDistillationLoss
from losses.spatial_consistency import SpatialConsistencyLoss
from evaluation.metrics import calculate_metrics
from utils.wandb_logger import WandbLogger

logger = logging.getLogger(__name__)

_tqdm_disabled = not sys.stdout.isatty()


class LandmarkPhysTrainer:

    def __init__(self, config, data_loader, mode="train"):
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.config = config
        self.mode = mode
        self.min_valid_loss = None
        self.best_epoch = 0
        self.train_losses = []
        self.valid_losses = []

        # Build md_config from YAML (same as FactorizePhys)
        lp_cfg = config.MODEL.LANDMARKPHYS
        md_config = {
            "FRAME_NUM": lp_cfg.FRAME_NUM,
            "MD_FSAM": lp_cfg.MD_FSAM,
            "MD_TYPE": lp_cfg.MD_TYPE,
            "MD_TRANSFORM": lp_cfg.MD_TRANSFORM,
            "MD_S": lp_cfg.MD_S,
            "MD_R": lp_cfg.MD_R,
            "MD_STEPS": lp_cfg.MD_STEPS,
            "MD_INFERENCE": lp_cfg.MD_INFERENCE,
            "MD_RESIDUAL": lp_cfg.MD_RESIDUAL,
            "INV_T": lp_cfg.INV_T,
            "ETA": float(lp_cfg.ETA),
            "RAND_INIT": lp_cfg.RAND_INIT,
            "align_channels": lp_cfg.ALIGN_CHANNELS,
        }

        self.use_fsam = lp_cfg.MD_FSAM
        self.md_infer = lp_cfg.MD_INFERENCE

        drop_rate = float(getattr(config.MODEL, 'DROP_RATE', 0.1))
        in_channels = getattr(lp_cfg, 'IN_CHANNELS', 3)

        model_name = str(getattr(config.MODEL, 'NAME', 'LandmarkPhys'))
        self.model_name = model_name

        # Build landmark_config. ATTN_TYPE ('v1'|'v2') selects the attention
        # variant; v1 uses LandmarkAttn (Gaussian prior + pose gating),
        # v2 uses VisibilityMaskAttn (3D visibility mask).
        attn_type = str(getattr(lp_cfg, 'ATTN_TYPE', 'v1')).lower()
        if attn_type == 'v1':
            landmark_config = {
                'attn_type': 'v1',
                'num_landmarks': int(getattr(lp_cfg, 'NUM_LANDMARKS', 68)),
                'landmark_dim': int(getattr(lp_cfg, 'LANDMARK_DIM', 2)),
                'spatial_size': int(getattr(lp_cfg, 'SPATIAL_SIZE', 13)),
                'embed_channels': int(getattr(lp_cfg, 'EMBED_CHANNELS', 16)),
                'hidden_dim': int(getattr(lp_cfg, 'LANDMARK_HIDDEN', 32)),
                'pose_dim': int(getattr(lp_cfg, 'POSE_DIM', 3)),
                'pose_hidden': int(getattr(lp_cfg, 'POSE_HIDDEN', 8)),
                'sigma_init': float(getattr(lp_cfg, 'SIGMA_INIT', 0.1)),
                'temporal_kernel': int(getattr(lp_cfg, 'TEMPORAL_KERNEL', 3)),
                'use_spatial_prior': bool(
                    getattr(lp_cfg, 'USE_SPATIAL_PRIOR', True)),
                'use_pose_gating': bool(
                    getattr(lp_cfg, 'USE_POSE_GATING', True)),
                'use_temporal_conv': bool(
                    getattr(lp_cfg, 'USE_TEMPORAL_CONV', True)),
                'landmark_noise_std': float(
                    getattr(lp_cfg, 'LANDMARK_NOISE_STD', 0.0)),
                'landmark_dropout_rate': float(
                    getattr(lp_cfg, 'LANDMARK_DROPOUT_RATE', 0.0)),
            }
        elif attn_type == 'v2':
            landmark_config = {
                'attn_type': 'v2',
                'spatial_size': int(getattr(lp_cfg, 'SPATIAL_SIZE', 13)),
                'alpha_init': float(getattr(lp_cfg, 'ALPHA_INIT', 0.3)),
                'visibility_sharpness': float(
                    getattr(lp_cfg, 'VISIBILITY_SHARPNESS', 8.0)),
                'visibility_bias': float(
                    getattr(lp_cfg, 'VISIBILITY_BIAS', 0.55)),
                'raster_temp': float(getattr(lp_cfg, 'RASTER_TEMP', 50.0)),
                'canonical_normals_path': getattr(
                    lp_cfg, 'CANONICAL_NORMALS_PATH',
                    'weights/mediapipe/canonical_normals_dlib68.npy'),
                'mask_dropout': float(getattr(lp_cfg, 'MASK_DROPOUT', 0.0)),
            }
        else:
            raise ValueError(
                f"Unknown ATTN_TYPE: {attn_type!r}. Expected 'v1' or 'v2'.")

        if model_name == "CanonicalPhys":
            canonical_config = {
                'padding_mode': str(getattr(lp_cfg, 'PADDING_MODE', 'zeros')),
                'stabilize_warp': bool(
                    getattr(lp_cfg, 'STABILIZE_WARP', True)),
                'adaptive_stabilization': bool(
                    getattr(lp_cfg, 'ADAPTIVE_STABILIZATION', False)),
                'motion_threshold': float(
                    getattr(lp_cfg, 'MOTION_THRESHOLD', 0.08)),
                'physics_weight': bool(
                    getattr(lp_cfg, 'PHYSICS_WEIGHT', False)),
                'physics_sharpness': float(
                    getattr(lp_cfg, 'PHYSICS_SHARPNESS', 8.0)),
                'physics_bias': float(
                    getattr(lp_cfg, 'PHYSICS_BIAS', 0.55)),
            }
            self.model = CanonicalPhys(
                frames=lp_cfg.FRAME_NUM,
                md_config=md_config,
                in_channels=in_channels,
                dropout=drop_rate,
                device=self.device,
                canonical_config=canonical_config).to(self.device)
        else:
            self.model = LandmarkPhys(
                frames=lp_cfg.FRAME_NUM,
                md_config=md_config,
                in_channels=in_channels,
                dropout=drop_rate,
                device=self.device,
                landmark_config=landmark_config).to(self.device)

        # Optionally load pretrained FactorizePhys weights for backbone
        pretrained_path = getattr(lp_cfg, 'PRETRAINED_BACKBONE', None)
        if pretrained_path and os.path.exists(pretrained_path):
            logger.info(f"Loading pretrained backbone from: {pretrained_path}")
            state_dict = torch.load(
                pretrained_path, map_location=self.device)
            missing, unexpected = self.model.load_state_dict(
                state_dict, strict=False)
            logger.info(f"  Loaded backbone. Missing keys: {len(missing)}, "
                        f"Unexpected keys: {len(unexpected)}")

        # W&B logging
        use_wandb = getattr(config, 'USE_WANDB', False)
        self.wb = WandbLogger(config, project="emergentphys",
                              name=f"LandmarkPhys_{self.model_file_name}",
                              enabled=(mode == "train" and use_wandb))

        # Spatial consistency loss (Direction 2) — only meaningful when
        # the model returns canonical frames (CanonicalPhys).
        self.consistency_weight = float(
            getattr(lp_cfg, 'CONSISTENCY_WEIGHT', 0.0))
        self.use_consistency = (
            self.consistency_weight > 0
            and model_name == "CanonicalPhys")
        if self.use_consistency:
            # Operate on backbone feature maps at 13x13 (FactorizePhys
            # voxel_embeddings spatial resolution) so gradients flow to
            # the backbone.
            self.consistency_loss = SpatialConsistencyLoss(
                feature_size=13, temporal_diff=True
            ).to(self.device)
        else:
            self.consistency_loss = None

        # POS distillation loss (Direction 3) — classical POS method
        # applied to canonical forehead pixels as auxiliary teacher.
        self.distill_weight = float(
            getattr(lp_cfg, 'DISTILL_WEIGHT', 0.0))
        self.use_distill = (
            self.distill_weight > 0
            and model_name == "CanonicalPhys")
        if self.use_distill:
            fps = float(getattr(lp_cfg, 'FPS', 30.0))
            self.distill_loss = POSDistillationLoss(
                fps=fps, spatial_size=72
            ).to(self.device)
        else:
            self.distill_loss = None

        if mode == "train":
            self.exp_dir = os.path.dirname(self.model_dir)
            self.num_train_batches = len(data_loader["train"])
            self.loss_model = Neg_Pearson()
            lr = float(config.TRAIN.LR)
            weight_decay = float(getattr(config.TRAIN, 'WEIGHT_DECAY', 0.0))
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=weight_decay)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=lr,
                epochs=config.TRAIN.EPOCHS,
                steps_per_epoch=self.num_train_batches)

            total_params = sum(p.numel() for p in self.model.parameters())
            if hasattr(self.model, 'landmark_attn'):
                extra = sum(p.numel()
                            for p in self.model.landmark_attn.parameters())
                branch_name = 'Landmark branch'
            elif hasattr(self.model, 'canonical_warp'):
                extra = sum(p.numel()
                            for p in self.model.canonical_warp.parameters())
                branch_name = 'Canonical warp'
            else:
                extra = 0
                branch_name = 'Extra'
            logger.info(f"Total params: {total_params:,}, "
                        f"{branch_name}: {extra:,} "
                        f"({100*extra/max(total_params, 1):.1f}%)")
        elif mode == "test":
            pass
        else:
            raise ValueError(
                f"LandmarkPhys trainer: unsupported mode '{mode}'")

    def _append_last_frame(self, data):
        """Append duplicate of last frame for temporal differencing."""
        last_frame = data[:, :, -1:, :, :]
        return torch.cat((data, last_frame), dim=2)

    def _extract_landmark_inputs(self, batch):
        """Extract landmarks, poses, valid_mask, transforms, and
        homographies from the batch tuple.

        Returns (landmarks, poses, valid_mask, transforms, homographies)
        tensors on device, or Nones if not available.
        """
        # Batch: [data, labels, sid, chunk_id, (landmarks), (poses),
        #         (valid_mask), (transforms), (homographies)]
        if len(batch) >= 6:
            landmarks = batch[4].to(torch.float32).to(self.device)
            poses = batch[5].to(torch.float32).to(self.device)
            valid_mask = None
            transforms = None
            homographies = None
            if len(batch) >= 7:
                valid_mask = batch[6].to(torch.float32).to(self.device)
            if len(batch) >= 8:
                transforms = batch[7].to(torch.float32).to(self.device)
            if len(batch) >= 9:
                homographies = batch[8].to(torch.float32).to(self.device)
            return landmarks, poses, valid_mask, transforms, homographies
        return None, None, None, None, None

    def train(self, data_loader):
        """Training routine."""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        for epoch in range(self.max_epoch_num):
            train_loss = []
            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80,
                        disable=_tqdm_disabled)
            for idx, batch in enumerate(tbar):
                tbar.set_description(f"Epoch {epoch}")

                data = batch[0].to(torch.float32).to(self.device)
                labels = batch[1].to(torch.float32).to(self.device)
                labels = (labels - torch.mean(labels)) / (torch.std(labels) + 1e-8)
                data = self._append_last_frame(data)

                landmarks, poses, valid_mask, transforms, homographies = \
                    self._extract_landmark_inputs(batch)

                self.optimizer.zero_grad()

                if self.use_fsam:
                    pred_ppg, vox_embed, factorized_embed, appx_error = \
                        self.model(data, landmarks=landmarks, poses=poses,
                                   valid_mask=valid_mask, transforms=transforms,
                                   homographies=homographies)
                else:
                    pred_ppg, vox_embed = self.model(
                        data, landmarks=landmarks, poses=poses,
                        valid_mask=valid_mask, transforms=transforms,
                        homographies=homographies)

                pred_ppg = (pred_ppg - torch.mean(pred_ppg)) / (torch.std(pred_ppg) + 1e-8)

                loss_bvp = self.loss_model(pred_ppg, labels)
                loss = loss_bvp

                # Cross-spatial consistency loss on backbone feature
                # maps. Gradients flow to the backbone since voxel_embed
                # is a learned representation.
                if self.use_consistency:
                    loss_cons = self.consistency_loss(
                        vox_embed, valid_mask=valid_mask)
                    loss = loss + self.consistency_weight * loss_cons

                # POS distillation loss — classical POS pseudo-BVP
                # from canonical forehead pixels as auxiliary teacher.
                if self.use_distill and self.model._canonical_frames is not None:
                    loss_distill = self.distill_loss(
                        self.model._canonical_frames, pred_ppg)
                    loss = loss + self.distill_weight * loss_distill

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                train_loss.append(loss.item())

                self.optimizer.step()
                self.scheduler.step()

                tbar.set_postfix(loss=loss.item())

            mean_train_loss = np.mean(train_loss)
            self.train_losses.append(mean_train_loss)
            self.wb.log({"train/loss": mean_train_loss, "epoch": epoch})

            self.save_checkpoint("last")
            if not self.config.TEST.USE_LAST_EPOCH:
                valid_loss, _, _ = self.valid(data_loader)
                self.valid_losses.append(valid_loss)
                self.wb.log({"valid/loss": valid_loss, "epoch": epoch})
                if self.min_valid_loss is None or valid_loss < self.min_valid_loss:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    self.save_checkpoint("best")
                logger.info(f"Epoch {epoch}: train_loss={mean_train_loss:.4f}, "
                            f"valid_loss={valid_loss:.4f}, best_epoch={self.best_epoch}")
            else:
                if self.min_valid_loss is None or mean_train_loss < self.min_valid_loss:
                    self.min_valid_loss = mean_train_loss
                    self.best_epoch = epoch
                    self.save_checkpoint("best")
                logger.info(f"Epoch {epoch}: train_loss={mean_train_loss:.4f}, "
                            f"best_epoch={self.best_epoch}")

        self._plot_loss_curves()
        logger.info(f"Training done. Best epoch: {self.best_epoch}, "
                    f"min_loss: {self.min_valid_loss:.4f}")
        self.wb.finish()

    def valid(self, data_loader):
        """Validation routine."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        logger.info("====Validating===")
        valid_loss = []
        example_preds = None
        example_labels = None
        self.model.eval()
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80,
                        disable=_tqdm_disabled)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")

                data = valid_batch[0].to(torch.float32).to(self.device)
                labels = valid_batch[1].to(torch.float32).to(self.device)
                labels = (labels - torch.mean(labels)) / (torch.std(labels) + 1e-8)
                data = self._append_last_frame(data)

                landmarks, poses, valid_mask, transforms, homographies = \
                    self._extract_landmark_inputs(valid_batch)

                if self.md_infer and self.use_fsam:
                    pred_ppg, vox_embed, factorized_embed, appx_error = \
                        self.model(data, landmarks=landmarks, poses=poses,
                                   valid_mask=valid_mask, transforms=transforms,
                                   homographies=homographies)
                else:
                    pred_ppg, vox_embed = self.model(
                        data, landmarks=landmarks, poses=poses,
                        valid_mask=valid_mask, transforms=transforms,
                                   homographies=homographies)

                pred_ppg = (pred_ppg - torch.mean(pred_ppg)) / (torch.std(pred_ppg) + 1e-8)
                loss = self.loss_model(pred_ppg, labels)

                valid_loss.append(loss.item())
                vbar.set_postfix(loss=loss.item())

                if valid_idx == 0:
                    example_preds = pred_ppg.cpu().numpy()
                    example_labels = labels.cpu().numpy()

            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss), example_preds, example_labels

    def test(self, data_loader, save_dir=None, train_dataset=None):
        """Testing routine."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        logger.info("===Testing===")
        predictions = dict()
        labels = dict()

        if self.mode == "test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError(
                    "Inference model path error! "
                    "Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(
                torch.load(self.config.INFERENCE.MODEL_PATH,
                           map_location=self.device))
            logger.info("Testing uses pretrained model!")
            logger.info(self.config.INFERENCE.MODEL_PATH)
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_model_path = os.path.join(
                    self.model_dir,
                    self.model_file_name + '_last.pth')
                logger.info("Testing uses last epoch model!")
                logger.info(last_model_path)
                self.model.load_state_dict(
                    torch.load(last_model_path, map_location=self.device))
            else:
                best_model_path = os.path.join(
                    self.model_dir,
                    self.model_file_name + '_best.pth')
                logger.info("Testing uses best epoch model!")
                logger.info(best_model_path)
                self.model.load_state_dict(
                    torch.load(best_model_path, map_location=self.device))

        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80,
                                                disable=_tqdm_disabled)):
                batch_size = test_batch[0].shape[0]
                data = test_batch[0].to(torch.float32).to(self.device)
                label = test_batch[1].to(torch.float32).to(self.device)
                data = self._append_last_frame(data)

                landmarks, poses, valid_mask, transforms, homographies = \
                    self._extract_landmark_inputs(test_batch)

                if self.md_infer and self.use_fsam:
                    pred_ppg_test, vox_embed, factorized_embed, appx_error = \
                        self.model(data, landmarks=landmarks, poses=poses,
                                   valid_mask=valid_mask, transforms=transforms,
                                   homographies=homographies)
                else:
                    pred_ppg_test, vox_embed = self.model(
                        data, landmarks=landmarks, poses=poses,
                        valid_mask=valid_mask, transforms=transforms,
                                   homographies=homographies)

                label = label.cpu()
                pred_ppg_test = pred_ppg_test.cpu()

                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx]
                    labels[subj_index][sort_index] = label[idx]

        calculate_metrics(predictions, labels, self.config,
                          save_dir=save_dir, train_dataset=train_dataset)
        return predictions, labels

    def _plot_loss_curves(self):
        """Plot training and validation loss curves."""
        fig, ax = plt.subplots(figsize=(8, 5))
        epochs = range(1, len(self.train_losses) + 1)
        ax.plot(epochs, self.train_losses, 'b-', label='Train Loss')
        if self.valid_losses:
            ax.plot(epochs, self.valid_losses, 'r-', label='Valid Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('LandmarkPhys Loss Curve')
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(self.exp_dir, 'loss_curve.png'), dpi=150)
        plt.close(fig)

    def save_checkpoint(self, tag):
        """Save model checkpoint."""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir,
            self.model_file_name + '_' + tag + '.pth')
        torch.save(self.model.state_dict(), model_path)
        logger.info(f'Saved Model Path: {model_path}')
