"""Physiological generator: one ejection, path length, forward plus reflection."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

INK = "#1C1C1C"
MUTED = "#5C6570"
GREEN = "#1E6A45"
GBG = "#E7F3EC"

fig = plt.figure(figsize=(7.2, 3.9), dpi=200)
ax = fig.add_axes([0.02, 0.07, 0.96, 0.90])
ax.set_xlim(0, 120)
ax.set_ylim(0, 100)
ax.axis("off")


def box(x, y, w, h, fc, ec=INK, lw=0.7):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1,rounding_size=0.35",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        mutation_aspect=0.45, clip_on=False,
    ))


def arrow(x1, y1, x2, y2, c=INK):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=6.5,
        lw=0.65, color=c, shrinkA=0, shrinkB=0, clip_on=False,
    ))


def wave(x0, y0, w, h, kind, color):
    t = np.linspace(0, 1, 240)
    base = np.exp(-((t - 0.28) / 0.08) ** 2) + 0.35 * np.exp(-((t - 0.55) / 0.12) ** 2)
    if kind == "forward":
        y = base
    elif kind == "neck":
        y = np.exp(-((t - 0.32) / 0.09) ** 2) + 0.45 * np.exp(-((t - 0.62) / 0.1) ** 2)
    else:
        y = np.exp(-((t - 0.40) / 0.07) ** 2) + 0.7 * np.exp(-((t - 0.58) / 0.08) ** 2)
    y = y / y.max()
    ax.plot(x0 + t * w, y0 + y * h, color=color, lw=1.05, solid_capstyle="round")


ax.text(1, 96, "One ejection, observed through real arterial paths", fontsize=9)

box(1.5, 74, 28, 16, GBG, GREEN, 0.8)
ax.text(15.5, 85.5, "Forward wave", ha="center", color=GREEN, fontsize=8)
ax.text(15.5, 80.5, r"$f(t)$ at the aortic root", ha="center", fontsize=7)
wave(4, 75.2, 22, 4.2, "forward", GREEN)

box(36, 74, 28, 16, "#F7F7F7")
ax.text(50, 85.5, "Path, not a free delay", ha="center", fontsize=8)
ax.text(50, 80.2, r"$\tau_s = L_s / v$", ha="center", fontsize=8)
ax.text(50, 76.2, r"$L_s$ from height", ha="center", fontsize=7, color=MUTED)

box(70, 74, 48, 16, "#F7F7F7")
ax.text(94, 86.2, "Speed stays in the measured range", ha="center", fontsize=8)
ax.text(94, 81.5, "order follows the arteries", ha="center", fontsize=7.5)
ax.text(94, 77.2, "neck before face, both before the palm", ha="center", fontsize=7, color=MUTED)
arrow(29.6, 82, 35.8, 82, GREEN)
arrow(64.2, 82, 69.8, 82, INK)

box(1.5, 40, 76, 28, "white", GREEN, 0.8)
ax.text(39.5, 64.2, "What a site is allowed to show", ha="center", color=GREEN, fontsize=8)
ax.text(39.5, 59.2, r"$\hat{y}_s(t)= a_s f(t-\tau_s)+ b_s f(t-\tau_s-\delta_s)$", ha="center", fontsize=8)
ax.text(39.5, 54.0, "second term is the reflection; it grows toward the periphery", ha="center",
        fontsize=7, color=MUTED)
wave(6, 43.5, 28, 7, "neck", "#24557F")
wave(42, 43.5, 28, 7, "palm", "#B85C2A")
ax.text(20, 42.2, "neck: earlier, rounder", ha="center", fontsize=6.5, color="#24557F")
ax.text(56, 42.2, "palm: later, taller peak", ha="center", fontsize=6.5, color="#B85C2A")

box(82, 40, 36, 28, GBG, GREEN, 0.7)
ax.text(100, 62.5, "Timing uses the foot", ha="center", color=GREEN, fontsize=8)
ax.text(100, 57.2, "not the systolic peak", ha="center", fontsize=7.5)
ax.text(100, 51.5, "the peak is moved", ha="center", fontsize=7, color=MUTED)
ax.text(100, 47.5, "by the reflection", ha="center", fontsize=7, color=MUTED)
ax.text(100, 42.8, "face regions are too close", ha="center", fontsize=6.5, color=MUTED)

box(1.5, 18, 116.5, 16, "#F7F7F7")
ax.text(59.7, 29.5, "All observed sites fit this same ejection", ha="center", fontsize=8)
ax.text(59.7, 24.2, "A missing site removes one equation. The ejection, the lengths, and the speed stay.",
        ha="center", fontsize=7.5, color=MUTED)
ax.text(59.7, 19.8, "Confidence falls when fewer sites agree, or when speed would have to leave the physiological range.",
        ha="center", fontsize=7, color=GREEN)

fig.text(0.02, 0.015,
         "Delays from the central aorta, not the QRS: carotid about 30 ms, brachial about 60 ms, radial about 80 ms, femoral about 75 ms.",
         fontsize=6.5, color=MUTED)

pdf = r"d:\maxzhang\python files\RePSS\docs\figures\multisite_pulse_arch.pdf"
png = r"d:\maxzhang\python files\RePSS\docs\figures\multisite_pulse_arch.png"
fig.savefig(pdf)
fig.savefig(png, dpi=300)
print("ok")
