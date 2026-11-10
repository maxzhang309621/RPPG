"""Paired test: CP vs FP per-video MMPD HR errors (UBFC-trained)."""
import glob, numpy as np
from scipy import stats

def load(jid):
    f = glob.glob(f"runs/eval/exp_*_{jid}/per_video_hr.npz")
    d = np.load(f[0]); return d['gt'], d['pred']

fp_gt, fp_pred = load("19499200")   # FactorizePhys
cp_gt, cp_pred = load("19499201")   # CanonicalPhys
# pair by video index (same MMPD test order, shuffle=False)
n = min(len(fp_gt), len(cp_gt))
assert np.allclose(fp_gt[:n], cp_gt[:n], atol=1e-3), "GT HR mismatch -> not the same videos/order!"
fp_err = np.abs(fp_pred[:n] - fp_gt[:n])
cp_err = np.abs(cp_pred[:n] - cp_gt[:n])
print(f"n videos = {n}")
print(f"MAE  FP = {fp_err.mean():.3f} | CP = {cp_err.mean():.3f}")
print(f"median FP = {np.median(fp_err):.3f} | CP = {np.median(cp_err):.3f}")
w, pw = stats.wilcoxon(cp_err, fp_err)              # paired, two-sided
t, pt = stats.ttest_rel(cp_err, fp_err)
print(f"Wilcoxon signed-rank: W={w:.1f}, p={pw:.2e}")
print(f"Paired t-test:        t={t:.2f}, p={pt:.2e}")
