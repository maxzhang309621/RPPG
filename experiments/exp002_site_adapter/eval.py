"""Stage 1: face and palm adapters. They do not see each other."""
import importlib.machinery
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "baselines" / "rPPG-Toolbox"))
sys.path.insert(0, str(ROOT / "experiments" / "exp001_face_waveform"))


def _stub(name):
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    sys.modules[name] = module
    return module


def _install_import_stubs():
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

from unsupervised_methods.methods.POS_WANG import POS_WANG

from hr import heart_rate_bpm
from synthetic import SKIN, WEIGHTS, pulse
from model import SiteObservers


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


def face_video(n_frames, fs, heart_rate_bpm_value, spatial):
    beat = pulse(n_frames, fs, heart_rate_bpm_value)
    frames = np.empty((n_frames, spatial, spatial, 3), dtype=np.float32)
    for i, value in enumerate(beat):
        frames[i] = SKIN * (1.0 + 0.03 * WEIGHTS * value)
    return np.clip(frames, 0, 1), beat


def palm_video(n_frames, fs, heart_rate_bpm_value, distractor_bpm, spatial):
    """True pulse sits in the center. A same-RGB flicker covers the whole crop."""
    beat = pulse(n_frames, fs, heart_rate_bpm_value)
    flicker = pulse(n_frames, fs, distractor_bpm)
    yy, xx = np.mgrid[0:spatial, 0:spatial]
    center = ((yy - spatial / 2) ** 2 + (xx - spatial / 2) ** 2) <= (spatial * 0.28) ** 2
    palm_skin = np.array([0.72, 0.48, 0.40], dtype=np.float32)
    frames = np.empty((n_frames, spatial, spatial, 3), dtype=np.float32)
    for i, (value, lamp) in enumerate(zip(beat, flicker)):
        image = palm_skin * (1.0 + 0.08 * lamp)
        image = np.broadcast_to(image, (spatial, spatial, 3)).copy()
        image[center] = palm_skin * (1.0 + 0.05 * WEIGHTS * value)
        frames[i] = image
    return np.clip(frames, 0, 1), beat


def as_tensor(frames, device):
    return torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).to(device)


def negative_pearson(pred, target):
    loss = pred.new_zeros(())
    for i in range(pred.shape[0]):
        x = pred[i] - pred[i].mean()
        y = target[i] - target[i].mean()
        loss = loss + 1 - (x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-8)
    return loss / pred.shape[0]


def reliability(frames, wave, fs, band_hz):
    motion = float(np.mean(np.abs(np.diff(frames.mean(axis=(1, 2)), axis=0))))
    spectrum = np.abs(np.fft.rfft(wave - np.mean(wave))) ** 2
    freqs = np.fft.rfftfreq(len(wave), d=1.0 / fs)
    total = float(spectrum.sum()) + 1e-8
    keep = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    return float(spectrum[keep].sum() / total), motion


def main():
    cfg = load_config()
    if not hasattr(np, "mat"):
        np.mat = np.asmatrix
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiteObservers(cfg["frames"]).to(device)

    face_frames, face_beat = face_video(cfg["frames"], cfg["fs"], cfg["face_bpm"], cfg["spatial"])
    palm_frames, palm_beat = palm_video(
        cfg["frames"], cfg["fs"], cfg["palm_bpm"], cfg["distractor_bpm"], cfg["spatial"]
    )
    face_video_t = as_tensor(face_frames, device)
    palm_video_t = as_tensor(palm_frames, device)
    face_target = torch.from_numpy(face_beat).unsqueeze(0).to(device)
    palm_target = torch.from_numpy(palm_beat).unsqueeze(0).to(device)

    palm_before = next(model.palm.parameters()).detach().clone()
    face_parameters = (
        list(model.block1.parameters())
        + list(model.block2.parameters())
        + list(model.block3.parameters())
        + list(model.block4.parameters())
        + list(model.block5.parameters())
        + list(model.block6.parameters())
        + list(model.block7.parameters())
        + list(model.face.parameters())
    )
    optim = torch.optim.AdamW(face_parameters, lr=1e-3)
    model.train()
    face_losses = []
    for _ in range(cfg["face_steps"]):
        optim.zero_grad()
        loss = negative_pearson(model(face_video_t, "face"), face_target)
        loss.backward()
        optim.step()
        face_losses.append(float(loss.detach()))
    palm_unchanged = torch.allclose(palm_before, next(model.palm.parameters()).detach())

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.palm.parameters():
        parameter.requires_grad_(True)
    optim = torch.optim.AdamW(model.palm.parameters(), lr=1e-3)
    model.train()
    palm_losses = []
    for _ in range(cfg["palm_steps"]):
        optim.zero_grad()
        loss = negative_pearson(model(palm_video_t, "palm"), palm_target)
        loss.backward()
        optim.step()
        palm_losses.append(float(loss.detach()))

    model.eval()
    with torch.no_grad():
        face_on_palm = model(palm_video_t, "face").squeeze(0).cpu().numpy()
        palm_on_palm = model(palm_video_t, "palm").squeeze(0).cpu().numpy()
    face_hr = heart_rate_bpm(face_on_palm, cfg["fs"], cfg["band_hz"])
    palm_hr = heart_rate_bpm(palm_on_palm, cfg["fs"], cfg["band_hz"])
    pos_hr = heart_rate_bpm(POS_WANG(palm_frames, cfg["fs"]), cfg["fs"], cfg["band_hz"])
    band_ratio, motion = reliability(palm_frames, palm_on_palm, cfg["fs"], cfg["band_hz"])

    print(f"palm_truth_bpm {cfg['palm_bpm']} distractor_bpm {cfg['distractor_bpm']}")
    print(f"face_adapter_on_palm_bpm {face_hr:.2f}")
    print(f"palm_adapter_bpm {palm_hr:.2f}")
    print(f"pos_on_palm_bpm {pos_hr:.2f}")
    print(f"face_loss {face_losses[0]:.4f} -> {face_losses[-1]:.4f}")
    print(f"palm_loss {palm_losses[0]:.4f} -> {palm_losses[-1]:.4f}")
    print(f"palm_adapter_unchanged_during_face_fit {palm_unchanged}")
    print(f"palm_reliability_band_ratio {band_ratio:.3f} motion {motion:.5f}")

    if not palm_unchanged:
        raise SystemExit("palm adapter changed while the face path was training")
    if palm_losses[-1] > palm_losses[0]:
        raise SystemExit("palm adapter loss did not decrease")
    if abs(palm_hr - cfg["palm_bpm"]) > 6:
        raise SystemExit("palm adapter did not recover the palm pulse")
    if abs(pos_hr - cfg["palm_bpm"]) > 6:
        raise SystemExit("POS did not recover the palm pulse")


if __name__ == "__main__":
    main()
