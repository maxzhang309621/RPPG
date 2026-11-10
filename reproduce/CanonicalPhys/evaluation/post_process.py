"""Post processing for calculating heart rate using FFT or peak detection.

Includes helper functions: detrend, power2db, FFT HR, peak HR, SNR, MACC.
"""

import numpy as np
import scipy
import scipy.io
import scipy.sparse.linalg
from scipy.signal import butter
from scipy.sparse import spdiags
from copy import deepcopy


def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _detrend(input_signal, lambda_value):
    """Detrend PPG signal using Whittaker smoother.

    Uses sparse matrix solver instead of dense inverse to handle long signals
    efficiently. The matrix (I + lambda^2 * D^T D) is banded, so spsolve
    runs in O(n) time vs O(n^3) for dense inverse.
    """
    signal_length = input_signal.shape[0]
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length)
    I = scipy.sparse.eye(signal_length)
    A = I + (lambda_value ** 2) * D.T.dot(D)
    smoothed = scipy.sparse.linalg.spsolve(A.tocsc(), input_signal)
    return input_signal - smoothed


def power2db(mag):
    """Convert power to dB."""
    return 10 * np.log10(mag)


def _calculate_fft_hr(ppg_signal, fs=60, low_pass=0.6, high_pass=3.3):
    """Calculate heart rate based on PPG using FFT."""
    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = _next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return fft_hr


def _calculate_peak_hr(ppg_signal, fs):
    """Calculate heart rate based on PPG using peak detection."""
    ppg_peaks, _ = scipy.signal.find_peaks(ppg_signal)
    hr_peak = 60 / (np.mean(np.diff(ppg_peaks)) / fs)
    return hr_peak


def _compute_macc(pred_signal, gt_signal):
    """Calculate maximum amplitude of cross correlation (MACC).

    Uses FFT-based circular cross-correlation: O(n log n) instead of O(n^2)
    from iterating over all lags with np.corrcoef.
    """
    pred = np.squeeze(deepcopy(pred_signal))
    gt = np.squeeze(deepcopy(gt_signal))
    min_len = min(len(pred), len(gt))
    pred = pred[:min_len]
    gt = gt[:min_len]
    # Normalize to zero mean, unit std (same as Pearson correlation)
    pred = (pred - np.mean(pred)) / (np.std(pred) + 1e-8)
    gt = (gt - np.mean(gt)) / (np.std(gt) + 1e-8)
    # Circular cross-correlation via FFT
    corr = np.real(np.fft.ifft(np.fft.fft(pred) * np.conj(np.fft.fft(gt)))) / min_len
    macc = np.max(np.abs(corr))
    return macc


def _calculate_SNR(pred_ppg_signal, hr_label, fs=30, low_pass=0.6, high_pass=3.3):
    """Calculate SNR as the ratio of signal power at harmonics to remainder."""
    first_harmonic_freq = hr_label / 60
    second_harmonic_freq = 2 * first_harmonic_freq
    deviation = 6 / 60  # 6 beats/min converted to Hz

    pred_ppg_signal = np.expand_dims(pred_ppg_signal, 0)
    N = _next_power_of_2(pred_ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(pred_ppg_signal, fs=fs, nfft=N, detrend=False)

    idx_harmonic1 = np.argwhere(
        (f_ppg >= (first_harmonic_freq - deviation)) &
        (f_ppg <= (first_harmonic_freq + deviation)))
    idx_harmonic2 = np.argwhere(
        (f_ppg >= (second_harmonic_freq - deviation)) &
        (f_ppg <= (second_harmonic_freq + deviation)))
    idx_remainder = np.argwhere(
        (f_ppg >= low_pass) & (f_ppg <= high_pass) &
        ~((f_ppg >= (first_harmonic_freq - deviation)) &
          (f_ppg <= (first_harmonic_freq + deviation))) &
        ~((f_ppg >= (second_harmonic_freq - deviation)) &
          (f_ppg <= (second_harmonic_freq + deviation))))

    pxx_ppg = np.squeeze(pxx_ppg)
    pxx_harmonic1 = pxx_ppg[idx_harmonic1]
    pxx_harmonic2 = pxx_ppg[idx_harmonic2]
    pxx_remainder = pxx_ppg[idx_remainder]

    signal_power_hm1 = np.sum(pxx_harmonic1)
    signal_power_hm2 = np.sum(pxx_harmonic2)
    signal_power_rem = np.sum(pxx_remainder)

    if not signal_power_rem == 0:
        SNR = power2db((signal_power_hm1 + signal_power_hm2) / signal_power_rem)
    else:
        SNR = 0
    return SNR


def calculate_metric_per_video(predictions, labels, fs=30, diff_flag=True,
                               use_bandpass=True, hr_method='FFT'):
    """Calculate video-level HR and SNR."""
    if diff_flag:
        predictions = _detrend(np.cumsum(predictions), 100)
        labels = _detrend(np.cumsum(labels), 100)
    else:
        predictions = _detrend(predictions, 100)
        labels = _detrend(labels, 100)
    if use_bandpass:
        [b, a] = butter(1, [0.6 / fs * 2, 3.3 / fs * 2], btype='bandpass')
        predictions = scipy.signal.filtfilt(b, a, np.double(predictions))
        labels = scipy.signal.filtfilt(b, a, np.double(labels))

    macc = _compute_macc(predictions, labels)

    if hr_method == 'FFT':
        hr_pred = _calculate_fft_hr(predictions, fs=fs)
        hr_label = _calculate_fft_hr(labels, fs=fs)
    elif hr_method == 'Peak':
        hr_pred = _calculate_peak_hr(predictions, fs=fs)
        hr_label = _calculate_peak_hr(labels, fs=fs)
    else:
        raise ValueError('Please use FFT or Peak to calculate your HR.')
    SNR = _calculate_SNR(predictions, hr_label, fs=fs)
    return hr_label, hr_pred, SNR, macc
