"""EfficientPhys Trainer: training, validation, and testing logic.

EfficientPhys operates on 2D frames with TSM for temporal modeling.
Data format: NDCHW (batch, time, channels, height, width).
The model takes (N*T, C, H, W) flattened frames and applies torch.diff
internally. An extra frame is appended to preserve the output length.
"""

import logging
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models.efficientphys import EfficientPhys
from losses.neg_pearson import Neg_Pearson
from evaluation.metrics import calculate_metrics

logger = logging.getLogger(__name__)
_tqdm_disabled = not sys.stdout.isatty()


class EfficientPhysTrainer:

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

        ep_cfg = config.MODEL.EFFICIENTPHYS
        self.frame_depth = ep_cfg.FRAME_DEPTH
        # Detect data format from config to handle NCDHW → NDCHW conversion
        self.data_format = getattr(config.TRAIN.DATA, 'DATA_FORMAT', 'NCDHW')

        drop_rate = float(getattr(config.MODEL, 'DROP_RATE', 0.25))
        img_size = getattr(ep_cfg, 'IMG_SIZE', 72)

        self.model = EfficientPhys(
            in_channels=3,
            nb_filters1=32,
            nb_filters2=64,
            kernel_size=3,
            dropout_rate1=drop_rate,
            dropout_rate2=0.5,
            pool_size=(2, 2),
            nb_dense=128,
            frame_depth=self.frame_depth,
            img_size=img_size,
        ).to(self.device)

        # Loss: use Neg_Pearson (consistent with other models in this codebase)
        # The original paper uses MSE but NegPearson works better for rPPG
        loss_type = getattr(ep_cfg, 'LOSS', 'NegPearson')
        if loss_type == 'MSE':
            self.loss_model = nn.MSELoss()
        else:
            self.loss_model = Neg_Pearson()
        self.loss_type = loss_type

        if mode == "train":
            self.exp_dir = os.path.dirname(self.model_dir)
            self.num_train_batches = len(data_loader["train"])
            lr = float(config.TRAIN.LR)
            weight_decay = float(getattr(config.TRAIN, 'WEIGHT_DECAY', 0.0))
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=weight_decay)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=lr,
                epochs=config.TRAIN.EPOCHS,
                steps_per_epoch=self.num_train_batches)

    def _to_ndchw(self, data):
        """Convert NCDHW (B, C, T, H, W) to NDCHW (B, T, C, H, W) if needed."""
        if self.data_format == 'NCDHW':
            return data.permute(0, 2, 1, 3, 4)
        return data

    def _reshape_data(self, data):
        """Reshape data to (N*D, C, H, W) for EfficientPhys.

        Handles both NCDHW and NDCHW input formats.
        Also appends an extra frame for torch.diff and trims to a multiple
        of frame_depth.

        Returns:
            reshaped: (aligned + 1, C, H, W) tensor
            aligned_len: number of frames before extra frame addition
        """
        data = self._to_ndchw(data)  # ensure (N, D, C, H, W)
        N, D, C, H, W = data.shape
        data = data.reshape(N * D, C, H, W)

        # Trim to multiple of frame_depth
        base_len = self.frame_depth
        aligned_len = (N * D) // base_len * base_len
        data = data[:aligned_len]

        # Append one extra frame for torch.diff
        last_frame = data[-1:].clone()
        data = torch.cat([data, last_frame], dim=0)

        return data, aligned_len

    def _reshape_labels(self, labels, aligned_len):
        """Reshape labels to match aligned data length.

        Args:
            labels: (N, D) tensor
            aligned_len: target length after trimming

        Returns:
            labels: (aligned_len,) tensor
        """
        labels = labels.reshape(-1)[:aligned_len]
        return labels

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

                data, aligned_len = self._reshape_data(data)
                labels = self._reshape_labels(labels, aligned_len)

                self.optimizer.zero_grad()

                pred = self.model(data)
                pred = pred[:aligned_len].squeeze(-1)  # (aligned_len,)

                # Z-score normalize for Neg_Pearson loss
                if self.loss_type != 'MSE':
                    pred = (pred - pred.mean()) / (pred.std() + 1e-8)
                    labels_norm = (labels - labels.mean()) / (labels.std() + 1e-8)
                else:
                    labels_norm = labels

                # Reshape to (N, D') for per-sample loss
                n_chunks = aligned_len // self.frame_depth
                pred_r = pred.reshape(n_chunks, self.frame_depth)
                labels_r = labels_norm.reshape(n_chunks, self.frame_depth)

                loss = self.loss_model(pred_r, labels_r)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                train_loss.append(loss.item())

                self.optimizer.step()
                self.scheduler.step()
                tbar.set_postfix(loss=loss.item())

            mean_train_loss = np.mean(train_loss)
            self.train_losses.append(mean_train_loss)

            self.save_checkpoint("last")
            if not self.config.TEST.USE_LAST_EPOCH:
                valid_loss, _, _ = self.valid(data_loader)
                self.valid_losses.append(valid_loss)
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

                data, aligned_len = self._reshape_data(data)
                labels = self._reshape_labels(labels, aligned_len)

                pred = self.model(data)
                pred = pred[:aligned_len].squeeze(-1)

                if self.loss_type != 'MSE':
                    pred = (pred - pred.mean()) / (pred.std() + 1e-8)
                    labels_norm = (labels - labels.mean()) / (labels.std() + 1e-8)
                else:
                    labels_norm = labels

                n_chunks = aligned_len // self.frame_depth
                pred_r = pred.reshape(n_chunks, self.frame_depth)
                labels_r = labels_norm.reshape(n_chunks, self.frame_depth)

                loss = self.loss_model(pred_r, labels_r)
                valid_loss.append(loss.item())
                vbar.set_postfix(loss=loss.item())

                if valid_idx == 0:
                    # Store first few samples for visualization
                    B = min(4, pred_r.shape[0])
                    example_preds = pred_r[:B].cpu().numpy()
                    example_labels = labels_r[:B].cpu().numpy()

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
                           map_location=self.device, weights_only=True))
            logger.info("Testing uses pretrained model!")
            logger.info(self.config.INFERENCE.MODEL_PATH)
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                model_path = os.path.join(
                    self.model_dir,
                    self.model_file_name + '_last.pth')
                logger.info(f"Testing uses last epoch model: {model_path}")
            else:
                model_path = os.path.join(
                    self.model_dir,
                    self.model_file_name + '_best.pth')
                logger.info(f"Testing uses best epoch model: {model_path}")
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device,
                           weights_only=True))

        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80,
                                                disable=_tqdm_disabled)):
                batch_size = test_batch[0].shape[0]
                data = test_batch[0].to(torch.float32).to(self.device)
                label = test_batch[1].to(torch.float32).to(self.device)

                data = self._to_ndchw(data)
                N, D, C, H, W = data.shape
                data_flat = data.reshape(N * D, C, H, W)

                # Trim and add extra frame
                base_len = self.frame_depth
                aligned_len = (N * D) // base_len * base_len
                data_flat = data_flat[:aligned_len]
                last_frame = data_flat[-1:].clone()
                data_flat = torch.cat([data_flat, last_frame], dim=0)

                pred = self.model(data_flat)
                pred = pred[:aligned_len].squeeze(-1)

                # Reshape back to per-sample predictions
                samples_per_batch = aligned_len // D if D > 0 else batch_size
                # Handle case where aligned_len < N*D
                actual_samples = min(batch_size, aligned_len // D) if D > 0 else batch_size
                pred = pred[:actual_samples * D].reshape(actual_samples, D)
                label = label[:actual_samples].cpu()
                pred = pred.cpu()

                for idx in range(actual_samples):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions:
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred[idx]
                    labels[subj_index][sort_index] = label[idx]

        calculate_metrics(predictions, labels, self.config,
                          save_dir=save_dir, train_dataset=train_dataset)
        return predictions, labels

    def _plot_loss_curves(self):
        """Plot training (and optionally validation) loss curves."""
        fig, ax = plt.subplots(figsize=(8, 5))
        epochs = range(1, len(self.train_losses) + 1)
        ax.plot(epochs, self.train_losses, 'b-', label='Train Loss')
        if self.valid_losses:
            ax.plot(epochs, self.valid_losses, 'r-', label='Valid Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curve')
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
