"""PhysNet Trainer: training, validation, and testing logic."""

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

from models.physnet import PhysNet_padding_Encoder_Decoder_MAX
from losses.neg_pearson import Neg_Pearson
from evaluation.metrics import calculate_metrics

logger = logging.getLogger(__name__)

# Disable tqdm when stdout is not a terminal (e.g. SLURM batch jobs)
_tqdm_disabled = not sys.stdout.isatty()


class PhysnetTrainer:

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

        drop_rate = float(getattr(config.MODEL, 'DROP_RATE', 0.0))
        self.model = PhysNet_padding_Encoder_Decoder_MAX(
            frames=config.MODEL.PHYSNET.FRAME_NUM,
            drop_rate=drop_rate).to(self.device)

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
        elif mode == "test":
            pass
        else:
            raise ValueError(f"PhysNet trainer: unsupported mode '{mode}'")

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
                rPPG = self.model(
                    batch[0].to(torch.float32).to(self.device))
                BVP_label = batch[1].to(torch.float32).to(self.device)
                rPPG = (rPPG - torch.mean(rPPG)) / (torch.std(rPPG) + 1e-8)
                BVP_label = (BVP_label - torch.mean(BVP_label)) / (torch.std(BVP_label) + 1e-8)
                loss = self.loss_model(rPPG, BVP_label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                train_loss.append(loss.item())
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
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
        """Validation routine. Returns (mean_loss, example_preds, example_labels)."""
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
                BVP_label = valid_batch[1].to(torch.float32).to(self.device)
                rPPG = self.model(
                    valid_batch[0].to(torch.float32).to(self.device))
                rPPG = (rPPG - torch.mean(rPPG)) / (torch.std(rPPG) + 1e-8)
                BVP_label = (BVP_label - torch.mean(BVP_label)) / (torch.std(BVP_label) + 1e-8)
                loss_ecg = self.loss_model(rPPG, BVP_label)
                valid_loss.append(loss_ecg.item())
                vbar.set_postfix(loss=loss_ecg.item())
                if valid_idx == 0:
                    example_preds = rPPG.cpu().numpy()
                    example_labels = BVP_label.cpu().numpy()
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss), example_preds, example_labels

    def test(self, data_loader, save_dir=None, train_dataset=None):
        """Testing routine.

        Args:
            data_loader: dict with 'test' DataLoader
            save_dir: optional directory to save results and figures
            train_dataset: name of dataset the model was trained on
        """
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
                last_model_path = os.path.join(
                    self.model_dir,
                    self.model_file_name + '_last.pth')
                logger.info("Testing uses last epoch model!")
                logger.info(last_model_path)
                self.model.load_state_dict(
                    torch.load(last_model_path, map_location=self.device,
                               weights_only=True))
            else:
                best_model_path = os.path.join(
                    self.model_dir,
                    self.model_file_name + '_best.pth')
                logger.info("Testing uses best epoch model!")
                logger.info(best_model_path)
                self.model.load_state_dict(
                    torch.load(best_model_path, map_location=self.device,
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
                pred_ppg_test = self.model(data)

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
        """Save model checkpoint with given tag (e.g. 'last', 'best')."""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir,
            self.model_file_name + '_' + tag + '.pth')
        torch.save(self.model.state_dict(), model_path)
        logger.info(f'Saved Model Path: {model_path}')
