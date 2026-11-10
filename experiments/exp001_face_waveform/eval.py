"""Stage 0: POS heart rate, then a short PhysNet fit, on one synthetic face clip."""
import importlib.machinery
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "baselines" / "rPPG-Toolbox"))


def _stub(name):
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    sys.modules[name] = module
    return module


def _install_import_stubs():
    """POS only needs detrend. The toolbox package import also pulls cv2 and skimage."""
    if "skimage.util" not in sys.modules:
        _stub("skimage")
        util = _stub("skimage.util")
        util.img_as_float = lambda image: np.asarray(image, dtype=np.float64)
    if "cv2" not in sys.modules:
        _stub("cv2")
    if "sklearn.metrics" not in sys.modules:
        _stub("sklearn")
        metrics = _stub("sklearn.metrics")
        metrics.mean_squared_error = lambda *args, **kwargs: 0.0


_install_import_stubs()

from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from unsupervised_methods.methods.POS_WANG import POS_WANG

from hr import heart_rate_bpm
from synthetic import skin_video


def load_config():
    cfg = {}
    with open(Path(__file__).with_name("config.yaml"), encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            value = value.strip()
            if value.startswith("["):
                cfg[key.strip()] = [float(part) for part in value.strip("[]").split(",")]
            else:
                try:
                    cfg[key.strip()] = int(value)
                except ValueError:
                    cfg[key.strip()] = value
    return cfg


def negative_pearson(pred, target):
    loss = pred.new_zeros(())
    for i in range(pred.shape[0]):
        x = pred[i] - pred[i].mean()
        y = target[i] - target[i].mean()
        loss = loss + 1 - (x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-8)
    return loss / pred.shape[0]


def run_pos(cfg):
    if not hasattr(np, "mat"):
        np.mat = np.asmatrix
    frames, _ = skin_video(
        cfg["pos_frames"], cfg["fs"], cfg["heart_rate_bpm"], cfg["spatial"], seed=cfg["seed"]
    )
    wave = POS_WANG(frames, cfg["fs"])
    estimate = heart_rate_bpm(wave, cfg["fs"], cfg["band_hz"])
    return estimate, len(wave)


def run_physnet(cfg):
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames, beat = skin_video(
        cfg["physnet_frames"],
        cfg["fs"],
        cfg["heart_rate_bpm"],
        cfg["spatial"],
        seed=cfg["seed"],
    )
    video = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).to(device)
    target = torch.from_numpy(beat).unsqueeze(0).to(device)
    model = PhysNet_padding_Encoder_Decoder_MAX(frames=cfg["physnet_frames"]).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    losses = []
    for _ in range(cfg["physnet_steps"]):
        optim.zero_grad()
        pred, _, _, _ = model(video)
        loss = negative_pearson(pred, target)
        loss.backward()
        optim.step()
        losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        pred, _, _, _ = model(video)
    estimate = heart_rate_bpm(pred.squeeze(0).detach().cpu().numpy(), cfg["fs"], cfg["band_hz"])
    return losses[0], losses[-1], estimate, int(pred.shape[-1])


def main():
    cfg = load_config()
    truth = cfg["heart_rate_bpm"]
    pos_hr, pos_len = run_pos(cfg)
    first, last, net_hr, net_len = run_physnet(cfg)
    pos_err = abs(pos_hr - truth)
    net_err = abs(net_hr - truth)
    print(f"truth_bpm {truth}")
    print(f"pos_bpm {pos_hr:.2f} abs_err {pos_err:.2f} length {pos_len}")
    print(f"physnet_loss {first:.4f} -> {last:.4f} bpm {net_hr:.2f} abs_err {net_err:.2f} length {net_len}")
    if pos_err > 2:
        raise SystemExit("POS heart rate is more than 2 bpm from the planted pulse")
    if not np.isfinite(last) or last > first:
        raise SystemExit("PhysNet loss did not decrease")
    if net_len != cfg["physnet_frames"]:
        raise SystemExit("PhysNet waveform length does not match the clip")


if __name__ == "__main__":
    main()
