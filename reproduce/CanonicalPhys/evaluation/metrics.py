"""Metric calculation for rPPG evaluation: MAE, RMSE, MAPE, Pearson, SNR, MACC."""

import logging
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from evaluation.post_process import calculate_metric_per_video

logger = logging.getLogger(__name__)


def _reform_data_from_dict(data, flatten=True):
    """Reformat predictions/labels from dicts to a single numpy array."""
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    sort_data = torch.cat(sort_data, dim=0)
    if flatten:
        sort_data = np.reshape(sort_data.cpu(), (-1))
    else:
        sort_data = np.array(sort_data.cpu())
    return sort_data


def calculate_metrics(predictions, labels, config, save_dir=None,
                      train_dataset=None):
    """Calculate rPPG Metrics (MAE, RMSE, MAPE, Pearson, SNR, MACC).

    Args:
        predictions: dict of {subject: {chunk_idx: tensor}}
        labels: dict of {subject: {chunk_idx: tensor}}
        config: ConfigDict with TEST/INFERENCE settings
        save_dir: optional directory to save results.txt and example figures
        train_dataset: name of dataset the model was trained on
    """
    predict_hr_fft_all = list()
    gt_hr_fft_all = list()
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()
    SNR_all = list()
    MACC_all = list()
    # Collect per-subject waveforms for example plots
    example_subjects = []
    logger.info("Calculating metrics!")
    for index in tqdm(predictions.keys(), ncols=80,
                      file=sys.stdout, mininterval=30):
        prediction = _reform_data_from_dict(predictions[index])
        label = _reform_data_from_dict(labels[index])

        # Collect first few subjects for example visualization
        if len(example_subjects) < 4:
            example_subjects.append((index, prediction, label))

        video_frame_size = prediction.shape[0]
        if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
            window_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * config.TEST.DATA.FS
            if window_frame_size > video_frame_size:
                window_frame_size = video_frame_size
        else:
            window_frame_size = video_frame_size

        for i in range(0, len(prediction), window_frame_size):
            pred_window = prediction[i:i + window_frame_size]
            label_window = label[i:i + window_frame_size]

            if len(pred_window) < 9:
                logger.warning(f"Window frame size of {len(pred_window)} is smaller "
                               f"than minimum pad length of 9. Window ignored!")
                continue

            if config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Standardized" or \
                    config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Raw":
                diff_flag_test = False
            elif config.TEST.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
                diff_flag_test = True
            else:
                raise ValueError("Unsupported label type in testing!")

            if config.INFERENCE.EVALUATION_METHOD == "peak detection":
                gt_hr_peak, pred_hr_peak, SNR, macc = calculate_metric_per_video(
                    pred_window, label_window, diff_flag=diff_flag_test,
                    fs=config.TEST.DATA.FS, hr_method='Peak')
                gt_hr_peak_all.append(gt_hr_peak)
                predict_hr_peak_all.append(pred_hr_peak)
                SNR_all.append(SNR)
                MACC_all.append(macc)
            elif config.INFERENCE.EVALUATION_METHOD == "FFT":
                gt_hr_fft, pred_hr_fft, SNR, macc = calculate_metric_per_video(
                    pred_window, label_window, diff_flag=diff_flag_test,
                    fs=config.TEST.DATA.FS, hr_method='FFT')
                gt_hr_fft_all.append(gt_hr_fft)
                predict_hr_fft_all.append(pred_hr_fft)
                SNR_all.append(SNR)
                MACC_all.append(macc)
            else:
                raise ValueError("Inference evaluation method name wrong!")

    # Collect result lines for both logging and file saving
    result_lines = []
    method = config.INFERENCE.EVALUATION_METHOD

    if method == "FFT":
        gt_hr_fft_all = np.array(gt_hr_fft_all)
        predict_hr_fft_all = np.array(predict_hr_fft_all)
        SNR_all = np.array(SNR_all)
        MACC_all = np.array(MACC_all)
        num_test_samples = len(predict_hr_fft_all)
        pred_hr_all = predict_hr_fft_all
        gt_hr_all = gt_hr_fft_all
        for metric in config.TEST.METRICS:
            if metric == "MAE":
                MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
                se = np.std(np.abs(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)
                line = f"FFT MAE (FFT Label): {MAE_FFT:.4f} +/- {se:.4f}"
                logger.info(line)
                result_lines.append(line)
            elif metric == "RMSE":
                squared_errors = np.square(predict_hr_fft_all - gt_hr_fft_all)
                RMSE_FFT = np.sqrt(np.mean(squared_errors))
                se = np.sqrt(np.std(squared_errors) / np.sqrt(num_test_samples))
                line = f"FFT RMSE (FFT Label): {RMSE_FFT:.4f} +/- {se:.4f}"
                logger.info(line)
                result_lines.append(line)
            elif metric == "MAPE":
                MAPE_FFT = np.mean(
                    np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
                se = np.std(
                    np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) / np.sqrt(
                    num_test_samples) * 100
                line = f"FFT MAPE (FFT Label): {MAPE_FFT:.4f} +/- {se:.4f}"
                logger.info(line)
                result_lines.append(line)
            elif metric == "Pearson":
                Pearson_FFT = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
                cc = Pearson_FFT[0][1]
                se = np.sqrt((1 - cc ** 2) / (num_test_samples - 2))
                line = f"FFT Pearson (FFT Label): {cc:.4f} +/- {se:.4f}"
                logger.info(line)
                result_lines.append(line)
            elif metric == "SNR":
                SNR_FFT = np.mean(SNR_all)
                se = np.std(SNR_all) / np.sqrt(num_test_samples)
                line = f"FFT SNR (FFT Label): {SNR_FFT:.4f} +/- {se:.4f} (dB)"
                logger.info(line)
                result_lines.append(line)
            elif metric == "MACC":
                MACC_avg = np.mean(MACC_all)
                se = np.std(MACC_all) / np.sqrt(num_test_samples)
                line = f"FFT MACC (FFT Label): {MACC_avg:.4f} +/- {se:.4f}"
                logger.info(line)
                result_lines.append(line)
            elif "BA" in metric:
                pass  # Bland-Altman plots not implemented in this baseline
            else:
                raise ValueError("Wrong Test Metric Type")

    elif method == "peak detection":
        gt_hr_peak_all = np.array(gt_hr_peak_all)
        predict_hr_peak_all = np.array(predict_hr_peak_all)
        SNR_all = np.array(SNR_all)
        MACC_all = np.array(MACC_all)
        num_test_samples = len(predict_hr_peak_all)
        pred_hr_all = predict_hr_peak_all
        gt_hr_all = gt_hr_peak_all
        for metric in config.TEST.METRICS:
            if metric == "MAE":
                MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_peak_all))
                se = np.std(np.abs(predict_hr_peak_all - gt_hr_peak_all)) / np.sqrt(num_test_samples)
                line = f"Peak MAE (Peak Label): {MAE_PEAK:.4f} +/- {se:.4f}"
                logger.info(line)
                result_lines.append(line)
            elif metric == "RMSE":
                squared_errors = np.square(predict_hr_peak_all - gt_hr_peak_all)
                RMSE_PEAK = np.sqrt(np.mean(squared_errors))
                se = np.sqrt(np.std(squared_errors) / np.sqrt(num_test_samples))
                line = f"PEAK RMSE (Peak Label): {RMSE_PEAK:.4f} +/- {se:.4f}"
                logger.info(line)
                result_lines.append(line)
            elif metric == "MAPE":
                MAPE_PEAK = np.mean(
                    np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
                se = np.std(
                    np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) / np.sqrt(
                    num_test_samples) * 100
                line = f"PEAK MAPE (Peak Label): {MAPE_PEAK:.4f} +/- {se:.4f}"
                logger.info(line)
                result_lines.append(line)
            elif metric == "Pearson":
                Pearson_PEAK = np.corrcoef(predict_hr_peak_all, gt_hr_peak_all)
                cc = Pearson_PEAK[0][1]
                se = np.sqrt((1 - cc ** 2) / (num_test_samples - 2))
                line = f"PEAK Pearson (Peak Label): {cc:.4f} +/- {se:.4f}"
                logger.info(line)
                result_lines.append(line)
            elif metric == "SNR":
                SNR_PEAK = np.mean(SNR_all)
                se = np.std(SNR_all) / np.sqrt(num_test_samples)
                line = f"PEAK SNR (PEAK Label): {SNR_PEAK:.4f} +/- {se:.4f} (dB)"
                logger.info(line)
                result_lines.append(line)
            elif metric == "MACC":
                MACC_avg = np.mean(MACC_all)
                se = np.std(MACC_all) / np.sqrt(num_test_samples)
                line = f"PEAK MACC (PEAK Label): {MACC_avg:.4f} +/- {se:.4f}"
                logger.info(line)
                result_lines.append(line)
            elif "BA" in metric:
                pass  # Bland-Altman plots not implemented in this baseline
            else:
                raise ValueError("Wrong Test Metric Type")
    else:
        raise ValueError("Inference evaluation method name wrong!")

    # Save results and figures to save_dir if provided
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        # Save results to text file
        results_path = os.path.join(save_dir, "results.txt")
        with open(results_path, 'w') as f:
            f.write(f"Evaluation Method: {method}\n")
            f.write(f"Model Path: {config.INFERENCE.MODEL_PATH}\n")
            f.write(f"Training Dataset: {train_dataset or 'Unknown'}\n")
            f.write(f"Test Dataset: {config.TEST.DATA.DATASET}\n")
            f.write(f"Num Test Samples: {num_test_samples}\n")
            f.write("=" * 50 + "\n")
            for line in result_lines:
                f.write(line + "\n")
        logger.info(f"Results saved to {results_path}")

        # Dump per-video HR (gt, pred) for paired statistical tests
        np.savez(os.path.join(save_dir, 'per_video_hr.npz'),
                 gt=np.asarray(gt_hr_all), pred=np.asarray(pred_hr_all))

        # Save scatter plot: predicted HR vs GT HR
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(gt_hr_all, pred_hr_all, alpha=0.5, s=10)
        hr_min = min(gt_hr_all.min(), pred_hr_all.min()) - 5
        hr_max = max(gt_hr_all.max(), pred_hr_all.max()) + 5
        ax.plot([hr_min, hr_max], [hr_min, hr_max], 'r--', label='y=x')
        ax.set_xlabel('Ground Truth HR (bpm)')
        ax.set_ylabel('Predicted HR (bpm)')
        ax.set_title(f'HR Estimation ({method})')
        ax.legend()
        ax.grid(True)
        ax.set_xlim(hr_min, hr_max)
        ax.set_ylim(hr_min, hr_max)
        ax.set_aspect('equal')
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'hr_scatter.png'), dpi=150)
        plt.close(fig)

        # Save example waveform plots (up to 4 subjects)
        if example_subjects:
            n_examples = len(example_subjects)
            fig, axes = plt.subplots(n_examples, 1,
                                     figsize=(10, 3 * n_examples))
            if n_examples == 1:
                axes = [axes]
            for i, (subj_id, pred_wave, label_wave) in enumerate(example_subjects):
                ax = axes[i]
                # Show first 300 frames for clarity
                show_len = min(300, len(pred_wave))
                ax.plot(pred_wave[:show_len], 'b-', label='Pred', alpha=0.8)
                ax.plot(label_wave[:show_len], 'r-', label='GT', alpha=0.8)
                ax.set_title(f'Subject: {subj_id}')
                ax.set_xlabel('Frame')
                ax.set_ylabel('Signal')
                ax.legend(fontsize=8)
                ax.grid(True)
            fig.suptitle('Example Waveforms (Pred vs GT)', fontsize=12)
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, 'example_waveforms.png'), dpi=150)
            plt.close(fig)
            logger.info(f"Figures saved to {save_dir}")
