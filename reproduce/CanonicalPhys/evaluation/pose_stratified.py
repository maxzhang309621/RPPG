"""Pose-stratified metric calculation for rPPG evaluation.

Groups predictions by head pose bin and computes per-bin metrics.
"""

import json
import logging
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluation.post_process import calculate_metric_per_video

logger = logging.getLogger(__name__)

POSE_BIN_ORDER = ["frontal", "mild", "moderate", "large", "unknown"]


def _reform_data_from_dict(data):
    """Reformat predictions/labels from dict to numpy array."""
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    sort_data = torch.cat(sort_data, dim=0)
    return np.reshape(sort_data.cpu(), (-1))


def load_pose_splits(pose_splits_path):
    """Load pose splits JSON file.

    Returns:
        dict mapping (subject_id, chunk_id_str) -> bin_name
    """
    with open(pose_splits_path, 'r') as f:
        data = json.load(f)

    chunk_to_bin = {}
    for sid, chunks in data["subjects"].items():
        for chunk_id, info in chunks.items():
            chunk_to_bin[(sid, chunk_id)] = info["bin"]

    return chunk_to_bin, data


def calculate_pose_stratified_metrics(predictions, labels, config,
                                      pose_splits_path, save_dir=None,
                                      train_dataset=None):
    """Calculate metrics per pose bin.

    Args:
        predictions: dict of {subject: {chunk_idx: tensor}}
        labels: dict of {subject: {chunk_idx: tensor}}
        config: ConfigDict with TEST settings
        pose_splits_path: path to pose_splits_clXXX.json
        save_dir: optional directory to save results
        train_dataset: name of training dataset
    """
    chunk_to_bin, splits_data = load_pose_splits(pose_splits_path)

    # Group subject predictions by pose bin
    # Each subject's chunks may span multiple bins, so we need chunk-level grouping
    bin_predictions = {b: {} for b in POSE_BIN_ORDER}
    bin_labels = {b: {} for b in POSE_BIN_ORDER}
    unmatched = 0

    for subj_id in predictions:
        for chunk_idx in predictions[subj_id]:
            key = (str(subj_id), str(chunk_idx))
            pose_bin = chunk_to_bin.get(key)
            if pose_bin is None:
                unmatched += 1
                continue

            # Use (subj_id, chunk_idx) as unique key within each bin
            unique_key = f"{subj_id}_{chunk_idx}"
            if unique_key not in bin_predictions[pose_bin]:
                bin_predictions[pose_bin][unique_key] = {}
                bin_labels[pose_bin][unique_key] = {}
            bin_predictions[pose_bin][unique_key][0] = predictions[subj_id][chunk_idx]
            bin_labels[pose_bin][unique_key][0] = labels[subj_id][chunk_idx]

    if unmatched > 0:
        logger.warning(f"{unmatched} chunks not found in pose splits")

    diff_flag = config.TEST.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized"
    fs = config.TEST.DATA.FS
    method = config.INFERENCE.EVALUATION_METHOD

    all_results = {}
    result_lines = []
    result_lines.append("=" * 60)
    result_lines.append("POSE-STRATIFIED EVALUATION RESULTS")
    result_lines.append("=" * 60)

    for pose_bin in POSE_BIN_ORDER:
        preds = bin_predictions[pose_bin]
        labs = bin_labels[pose_bin]
        n_subjects = len(preds)

        if n_subjects == 0:
            result_lines.append(f"\n--- {pose_bin.upper()} (0 chunks) ---")
            result_lines.append("  No data in this bin")
            all_results[pose_bin] = None
            continue

        gt_hr_all = []
        pred_hr_all = []
        snr_all = []

        for subj_key in preds:
            pred_wave = _reform_data_from_dict(preds[subj_key])
            label_wave = _reform_data_from_dict(labs[subj_key])

            if len(pred_wave) < 9:
                continue

            if method == "FFT":
                gt_hr, pred_hr, snr, macc = calculate_metric_per_video(
                    pred_wave, label_wave, diff_flag=diff_flag,
                    fs=fs, hr_method='FFT')
            else:
                gt_hr, pred_hr, snr, macc = calculate_metric_per_video(
                    pred_wave, label_wave, diff_flag=diff_flag,
                    fs=fs, hr_method='Peak')

            gt_hr_all.append(gt_hr)
            pred_hr_all.append(pred_hr)
            snr_all.append(snr)

        if not gt_hr_all:
            result_lines.append(f"\n--- {pose_bin.upper()} (0 valid chunks) ---")
            all_results[pose_bin] = None
            continue

        gt_hr_all = np.array(gt_hr_all)
        pred_hr_all = np.array(pred_hr_all)
        snr_all = np.array(snr_all)
        n = len(gt_hr_all)

        mae = np.mean(np.abs(pred_hr_all - gt_hr_all))
        rmse = np.sqrt(np.mean((pred_hr_all - gt_hr_all) ** 2))
        mape = np.mean(np.abs((pred_hr_all - gt_hr_all) / gt_hr_all)) * 100
        corr = np.corrcoef(pred_hr_all, gt_hr_all)[0, 1] if n > 1 else 0.0
        snr_mean = np.mean(snr_all)

        all_results[pose_bin] = {
            "n_chunks": n,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "Pearson": corr,
            "SNR": snr_mean,
        }

        result_lines.append(f"\n--- {pose_bin.upper()} ({n} chunks) ---")
        result_lines.append(f"  MAE:     {mae:.4f}")
        result_lines.append(f"  RMSE:    {rmse:.4f}")
        result_lines.append(f"  MAPE:    {mape:.4f}")
        result_lines.append(f"  Pearson: {corr:.4f}")
        result_lines.append(f"  SNR:     {snr_mean:.4f} dB")

    # Print summary table
    result_lines.append("\n" + "=" * 60)
    result_lines.append("SUMMARY TABLE")
    result_lines.append(f"{'Bin':<12} {'N':>6} {'MAE':>8} {'RMSE':>8} "
                        f"{'MAPE':>8} {'Pearson':>8} {'SNR':>8}")
    result_lines.append("-" * 60)
    for pose_bin in POSE_BIN_ORDER:
        r = all_results.get(pose_bin)
        if r is None:
            result_lines.append(f"{pose_bin:<12} {'0':>6} {'N/A':>8} "
                                f"{'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
        else:
            result_lines.append(
                f"{pose_bin:<12} {r['n_chunks']:>6} {r['MAE']:>8.4f} "
                f"{r['RMSE']:>8.4f} {r['MAPE']:>8.4f} {r['Pearson']:>8.4f} "
                f"{r['SNR']:>8.4f}")

    for line in result_lines:
        logger.info(line)

    # Save to file
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, "pose_stratified_results.txt")
        with open(out_path, 'w') as f:
            f.write(f"Model Path: {config.INFERENCE.MODEL_PATH}\n")
            f.write(f"Training Dataset: {train_dataset or 'Unknown'}\n")
            f.write(f"Test Dataset: {config.TEST.DATA.DATASET}\n")
            f.write(f"Pose Splits: {pose_splits_path}\n\n")
            for line in result_lines:
                f.write(line + "\n")
        logger.info(f"Pose-stratified results saved to {out_path}")

        # Generate bar chart
        _plot_pose_bar_chart(all_results, save_dir, train_dataset,
                             config.TEST.DATA.DATASET)

    return all_results


def _plot_pose_bar_chart(all_results, save_dir, train_dataset, test_dataset):
    """Generate bar chart of MAE by pose bin."""
    bins = []
    maes = []
    counts = []
    for b in POSE_BIN_ORDER:
        r = all_results.get(b)
        if r is not None:
            bins.append(b)
            maes.append(r["MAE"])
            counts.append(r["n_chunks"])

    if not bins:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(bins, maes, edgecolor='black', alpha=0.7,
                  color=['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c'][:len(bins)])
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f'n={count}', ha='center', va='bottom', fontsize=9)
    ax.set_xlabel('Pose Bin')
    ax.set_ylabel('MAE (BPM)')
    ax.set_title(f'MAE by Pose Bin (Train: {train_dataset}, '
                 f'Test: {test_dataset})')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'pose_stratified_mae.png'), dpi=150)
    plt.close(fig)
