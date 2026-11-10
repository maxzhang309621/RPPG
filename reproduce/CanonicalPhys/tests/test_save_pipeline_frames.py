"""Smoke test: module imports cleanly and render helpers produce PNGs."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)))


def test_module_imports():
    import scripts.save_pipeline_frames  # noqa: F401


def test_save_rgb_with_landmarks_writes_png(tmp_path):
    from scripts.save_pipeline_frames import _save_rgb_with_landmarks
    frame = np.zeros((72, 72, 3), dtype=np.uint8)
    lm = np.random.RandomState(0).rand(68, 2).astype(np.float32)
    out = tmp_path / "stage01.png"
    _save_rgb_with_landmarks(frame, lm, str(out))
    assert out.exists() and out.stat().st_size > 0


def test_save_canonical_rgb_writes_png(tmp_path):
    from scripts.save_pipeline_frames import _save_canonical_rgb
    frame = np.zeros((72, 72, 3), dtype=np.uint8)
    out = tmp_path / "stage02.png"
    _save_canonical_rgb(frame, str(out))
    assert out.exists() and out.stat().st_size > 0


def test_save_heatmap_writes_png(tmp_path):
    from scripts.save_pipeline_frames import _save_heatmap
    w = np.random.RandomState(0).rand(72, 72).astype(np.float32)
    out = tmp_path / "stage03.png"
    _save_heatmap(w, str(out))
    assert out.exists() and out.stat().st_size > 0


def test_save_signed_writes_png(tmp_path):
    from scripts.save_pipeline_frames import _save_signed
    f = np.random.RandomState(0).randn(72, 72, 3).astype(np.float32)
    out = tmp_path / "stage05.png"
    _save_signed(f, str(out))
    assert out.exists() and out.stat().st_size > 0
