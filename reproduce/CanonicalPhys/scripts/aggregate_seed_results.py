"""Aggregate seed-variance results across cross-dataset eval directories.

Usage:
    python scripts/aggregate_seed_results.py <eval_dir_1> <eval_dir_2> ...

Each eval_dir is the run directory of one cross-dataset eval (e.g.,
runs/eval/exp_20260411_*). The script reads results.txt from each, parses
the train/test dataset and key metrics, and prints a per-seed table plus
mean / std across seeds for each (train_ds, test_ds) cell.
"""

import argparse
import collections
import re
import os
import sys


METRIC_RE = {
    'mae': re.compile(r'FFT MAE \(FFT Label\):\s*([\-\d\.]+)'),
    'rmse': re.compile(r'FFT RMSE \(FFT Label\):\s*([\-\d\.]+)'),
    'pearson': re.compile(r'FFT Pearson \(FFT Label\):\s*([\-\d\.]+)'),
}
TRAIN_RE = re.compile(r'Training Dataset:\s*(\S+)')
TEST_RE = re.compile(r'Test Dataset:\s*(\S+)')
MODEL_RE = re.compile(r'Model Path:\s*(\S+)')


def parse_results_file(path):
    """Parse one results.txt; return dict with train, test, mae, rmse, pearson, model_path."""
    with open(path) as f:
        text = f.read()
    out = {}
    m = TRAIN_RE.search(text)
    if m:
        out['train'] = m.group(1)
    m = TEST_RE.search(text)
    if m:
        out['test'] = m.group(1)
    m = MODEL_RE.search(text)
    if m:
        out['model_path'] = m.group(1)
    for key, regex in METRIC_RE.items():
        m = regex.search(text)
        if m:
            out[key] = float(m.group(1))
    return out


def find_results(roots):
    """Walk eval root dirs and yield parsed results.txt entries."""
    for root in roots:
        if os.path.isfile(root) and root.endswith('results.txt'):
            yield parse_results_file(root)
            continue
        if not os.path.isdir(root):
            continue
        # Allow either a single eval dir or a parent containing many.
        if os.path.exists(os.path.join(root, 'results.txt')):
            yield parse_results_file(os.path.join(root, 'results.txt'))
        else:
            for entry in sorted(os.listdir(root)):
                p = os.path.join(root, entry, 'results.txt')
                if os.path.exists(p):
                    yield parse_results_file(p)


def mean_std(xs):
    if not xs:
        return float('nan'), float('nan')
    n = len(xs)
    mu = sum(xs) / n
    if n == 1:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return mu, var ** 0.5


def main():
    p = argparse.ArgumentParser()
    p.add_argument('roots', nargs='+', help='eval dirs or parent dirs')
    p.add_argument('--metric', default='mae', choices=list(METRIC_RE.keys()))
    args = p.parse_args()

    # group: (train, test) -> list of (model_path, value)
    groups = collections.defaultdict(list)
    for r in find_results(args.roots):
        if 'train' not in r or 'test' not in r or args.metric not in r:
            continue
        key = (r['train'], r['test'])
        groups[key].append((r.get('model_path', '?'), r[args.metric]))

    if not groups:
        print('No results found.', file=sys.stderr)
        sys.exit(1)

    # Print per-cell summary
    print(f"\n=== Per-cell {args.metric.upper()} (mean +/- std across seeds) ===")
    print(f"{'Train':<12} {'Test':<12} {'N':>3}  {'mean':>8}  {'std':>8}  values")
    train_set = sorted({k[0] for k in groups})
    test_set = sorted({k[1] for k in groups})
    for tr in train_set:
        for te in test_set:
            entries = groups.get((tr, te), [])
            if not entries:
                continue
            vals = [v for _, v in entries]
            mu, sd = mean_std(vals)
            vals_str = ' '.join(f'{v:.3f}' for v in vals)
            print(f"{tr:<12} {te:<12} {len(vals):>3}  {mu:>8.3f}  {sd:>8.3f}  [{vals_str}]")

    # Print compact mean table
    print(f"\n=== Mean {args.metric.upper()} matrix ===")
    header = f"{'Train\\Test':<12}" + ''.join(f"{te:>14}" for te in test_set)
    print(header)
    for tr in train_set:
        row = f"{tr:<12}"
        for te in test_set:
            entries = groups.get((tr, te), [])
            if entries:
                vals = [v for _, v in entries]
                mu, sd = mean_std(vals)
                row += f"  {mu:>6.3f}+/-{sd:.2f}"
            else:
                row += f"{'-':>14}"
        print(row)
    print()


if __name__ == '__main__':
    main()
