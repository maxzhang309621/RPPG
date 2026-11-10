"""RADIANT 训练日志。实际写入和出图走 reproduce/common/train_viz.py。"""

import os
import sys

COMMON = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "common")
if COMMON not in sys.path:
    sys.path.insert(0, COMMON)

from train_viz import TrainRecorder


class TrainLog:
    def __init__(self, path, model="RADIANT", dataset=""):
        self.recorder = TrainRecorder(path, model=model, dataset=dataset)

    def add(self, phase, epoch, train_mse, test_mae="", test_rmse="", test_std="",
            valid_loss="", lr="", mape="", pearson="", snr="", macc=""):
        self.recorder.add(
            epoch,
            phase=phase,
            lr=lr,
            train_loss=train_mse,
            valid_loss=valid_loss or None,
            metrics={
                "mae": test_mae,
                "rmse": test_rmse,
                "std": test_std,
                "mape": mape,
                "pearson": pearson,
                "snr": snr,
                "macc": macc,
            },
        )

    def plot(self, image_path):
        self.recorder.plot(image_path)
