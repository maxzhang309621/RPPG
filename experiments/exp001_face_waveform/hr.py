"""Heart rate from a waveform: bandpass, then the spectral peak."""
import numpy as np
from scipy import signal


def heart_rate_bpm(wave, fs, band_hz):
    wave = np.asarray(wave, dtype=np.float64)
    wave = wave - np.mean(wave)
    low, high = band_hz
    b, a = signal.butter(2, [low / (fs / 2), high / (fs / 2)], btype="bandpass")
    filtered = signal.filtfilt(b, a, wave)
    spec = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(filtered), d=1.0 / fs)
    keep = (freqs >= low) & (freqs <= high)
    if not np.any(keep):
        raise ValueError("band does not cover any FFT bin")
    peak = freqs[keep][np.argmax(spec[keep])]
    return float(peak * 60.0)
