"""Skin-colored clip with a known pulse. No real face video is required."""
import numpy as np


SKIN = np.array([0.62, 0.42, 0.36], dtype=np.float32)
# Green carries the strongest pulsatile weight.
WEIGHTS = np.array([0.15, 0.70, 0.15], dtype=np.float32)


def pulse(n_frames, fs, heart_rate_bpm):
    t = np.arange(n_frames, dtype=np.float32) / fs
    return np.sin(2 * np.pi * (heart_rate_bpm / 60.0) * t).astype(np.float32)


def skin_video(n_frames, fs, heart_rate_bpm, spatial, amplitude=0.03, seed=0):
    rng = np.random.default_rng(seed)
    beat = pulse(n_frames, fs, heart_rate_bpm)
    frames = np.empty((n_frames, spatial, spatial, 3), dtype=np.float32)
    noise = rng.normal(0, 0.002, size=frames.shape).astype(np.float32)
    for i, b in enumerate(beat):
        frames[i] = SKIN * (1.0 + amplitude * WEIGHTS * b) + noise[i]
    return np.clip(frames, 0, 1), beat
