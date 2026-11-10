"""合成时间信号，对应论文 3.4.2 与作者 generate_synthetic_temporal_signals.py。

作者原稿的循环缩进和未定义的 offset 无法运行。这里按同一生成式写出可执行版本：
心率正弦、二次谐波、呼吸正弦、高斯噪声和阶跃，频率覆盖约 0.7–4.2 Hz。
"""

import numpy as np


def make_synthetic_clip(rng, freq_hz, n_roi=40, fs=20, seconds=4):
    """返回 chrominance 风格的多 ROI 信号 (n_roi, T) 和脉搏 (T,)。"""
    n = int(fs * seconds)
    t = np.arange(n, dtype=np.float64) / fs
    resp = rng.uniform(0.084, 0.34)
    phi = rng.uniform(0, 2 * np.pi)
    theta = rng.uniform(0, 2 * np.pi)
    pulse = np.sin(2 * np.pi * freq_hz * t + phi)
    pulse = pulse + 0.5 * np.sin(2 * np.pi * (2 * freq_hz) * t + phi)
    breath = np.sin(2 * np.pi * resp * t + theta)
    signals = np.empty((n_roi, n), dtype=np.float32)
    for roi in range(n_roi):
        m1 = rng.uniform(0.3, 1.0)
        m2 = rng.uniform(0.0, 1.0)
        noise = rng.normal(0, 0.1, size=n)
        step_a = int(rng.integers(0, n))
        step_b = int(rng.integers(0, n))
        steps = np.zeros(n, dtype=np.float64)
        steps[step_a:] += 0.6
        steps[step_b:] += 0.4
        signals[roi] = (m1 * pulse + m2 * breath + noise + steps).astype(np.float32)
    target = (pulse + breath).astype(np.float32)
    return signals, target


def generate_dataset(n_samples=5000, n_roi=40, fs=20, seconds=4, seed=42):
    rng = np.random.default_rng(seed)
    lows = np.linspace(0.7, 4.0, 40, endpoint=False)
    span = (4.2 - 0.7) / 40
    signals = np.empty((n_samples, n_roi, fs * seconds), dtype=np.float32)
    pulses = np.empty((n_samples, fs * seconds), dtype=np.float32)
    heart_rates = np.empty(n_samples, dtype=np.float32)
    for i in range(n_samples):
        band = i % len(lows)
        freq = rng.uniform(lows[band], lows[band] + span)
        sig, pulse = make_synthetic_clip(rng, freq, n_roi=n_roi, fs=fs, seconds=seconds)
        signals[i] = sig
        pulses[i] = pulse
        heart_rates[i] = freq * 60.0
    return signals, pulses, heart_rates
