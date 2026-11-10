"""Official one-sample inference for PhysFormer VIPL fold1.

Uses Physformer_VIPL_fold1.pkl and data/VIPL_frames/p10/v1/source1.
Heart-rate numbers follow inference_OneSample_VIPL_PhysFormer.py
(per-clip power spectrum) and Inference_HRevaluation.m (three-segment Welch).
"""
import numpy as np
import torch
from scipy.signal import firwin, filtfilt, welch
from torchvision import transforms
from torch.utils.data import DataLoader

from model import ViT_ST_ST_Compact3_TDC_gra_sharp
from Loadtemporal_data_test import VIPL, Normaliztion, ToTensor
from TorchLossComputer import TorchLossComputer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ViT_ST_ST_Compact3_TDC_gra_sharp(
        image_size=(160, 128, 128),
        patches=(4, 4, 4),
        dim=96,
        ff_dim=144,
        num_heads=4,
        num_layers=12,
        dropout_rate=0.1,
        theta=0.7,
    ).to(device)
    state = torch.load("Physformer_VIPL_fold1.pkl", map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    dataset = VIPL(
        "VIPL_onesample.txt",
        "data/VIPL_frames/",
        transform=transforms.Compose([Normaliztion(), ToTensor()]),
    )
    sample = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
    inputs = sample["video_x"].to(device)
    frame_rate = float(sample["framerate"])
    ground_truth = float(sample["clip_average_HR_peaks"])

    pieces = []
    clip_hr = []
    with torch.no_grad():
        for clip in range(inputs.shape[1]):
            rppg, _, _, _ = model(inputs[:, clip], 2.0)
            rppg = rppg[0, 30:30 + 160]
            pieces.append(rppg.detach().cpu())
            hr = TorchLossComputer.cross_entropy_power_spectrum_forward_pred(
                rppg, torch.tensor(frame_rate, device=device)
            )
            clip_hr.append(float(hr) + 40)

    signal = torch.cat(pieces).numpy()
    taps = max(3, int(round(len(signal) / 10)))
    if taps % 2 == 0:
        taps += 1
    coeff = firwin(taps, [0.8, 3.0], pass_zero=False, fs=frame_rate, window="hamming")
    filtered = filtfilt(coeff, [1.0], signal)
    filtered = (filtered - filtered.mean()) / (filtered.std() + 1e-8)
    length = len(filtered)
    bounds = [(0, length // 3), (length // 3, 2 * length // 3), (2 * length // 3, length)]
    segment_hr = []
    for start, end in bounds:
        freqs, power = welch(filtered[start:end], fs=frame_rate, nfft=2**13)
        band = (freqs > 0.7) & (freqs < 4)
        segment_hr.append(float(freqs[band][np.argmax(power[band])] * 60))

    print(f"ground_truth_bpm {ground_truth:.3f}")
    print("clip_hr_bpm " + " ".join(f"{value:.1f}" for value in clip_hr))
    print(f"clip_hr_mean_bpm {float(np.mean(clip_hr)):.3f}")
    print("psd_hr_bpm " + " ".join(f"{value:.3f}" for value in segment_hr))
    print(f"psd_hr_mean_bpm {float(np.mean(segment_hr)):.3f}")


if __name__ == "__main__":
    main()
