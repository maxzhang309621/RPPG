"""Smoke test: Nature style + op_box/arrow helpers produce artifacts."""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)))


def test_imports_and_style():
    import scripts.draw_architecture_figure as m
    m.setup_nature_style()


def test_helpers_produce_artifacts(tmp_path):
    from scripts.draw_architecture_figure import (
        setup_nature_style, _op_box, _arrow, _loss_box,
    )
    setup_nature_style()
    fig = plt.figure(figsize=(4, 1))
    _op_box(fig, 0.5, 0.5, "op")
    _arrow(fig, 0.1, 0.5, 0.4, 0.5)
    _arrow(fig, 0.6, 0.5, 0.9, 0.5, dashed=True)
    _loss_box(fig, 0.5, 0.2, "loss")
    out = tmp_path / "helpers.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    assert out.exists() and out.stat().st_size > 0
