"""Tests del kit de figuras/tablas de publicación (backend Agg, sin ventana)."""

import numpy as np
import matplotlib.pyplot as plt

from lateralcauchy import plotting as pl


def test_save_fig_pdf_and_png(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out = pl.save_fig(fig, str(tmp_path / "fig"))
    assert [p.endswith(".pdf") for p in out] == [True, False]
    assert all((tmp_path / f"fig.{e}").exists() for e in ("pdf", "png"))


def test_save_fig_respects_extension(tmp_path):
    fig, ax = plt.subplots()
    out = pl.save_fig(fig, str(tmp_path / "solo.png"))
    assert len(out) == 1 and out[0].endswith("solo.png")
    assert not (tmp_path / "solo.pdf").exists()


def test_profiles_and_history(tmp_path):
    h = {k: [1.0, 0.5, 0.1] for k in ("total", "pde", "g", "f", "lat")}
    pl.plot_history(h, path=str(tmp_path / "h"))
    z = np.linspace(0, 1, 5)
    pl.plot_error_vs_z(z, np.exp(-z), path=str(tmp_path / "ez"))
    pl.plot_error_vs_t(z, np.ones_like(z), path=str(tmp_path / "et"))
    pl.plot_gradient_profile(z, np.sin(z), np.sin(z) + 0.01,
                             path=str(tmp_path / "gp"))
    for stem in ("h", "ez", "et", "gp"):
        assert (tmp_path / f"{stem}.pdf").exists()


def test_eval_slice_and_plot_slice(tmp_path):
    fn = lambda X: (X[:, 0:1] ** 2 + X[:, 2:3])          # campo escalar suave
    F, extent = pl.eval_slice(fn, R=1.0, L=1.0, t0=0.5, nx=11, nz=13)
    assert F.shape == (13, 11) and extent == (-1.0, 1.0, 0.0, 1.0)
    pl.plot_slice(F, F * 1.01, extent, path=str(tmp_path / "sl"))
    assert (tmp_path / "sl.pdf").exists()


def test_sweep_and_heatmap(tmp_path):
    pl.plot_frequency_sweep([1.7, 3.0, 4.1], [0.1, 0.5, 0.8], L=1.0,
                            path=str(tmp_path / "sw"))
    pl.plot_heatmap(np.array([[0.1, 0.2], [0.7, 0.9]]),
                    ["0%", "5%"], ["(1,1)", "(2,1)"],
                    "ruido", "modo", path=str(tmp_path / "hm"))
    assert (tmp_path / "sw.pdf").exists() and (tmp_path / "hm.pdf").exists()


def test_latex_table(tmp_path):
    s = pl.latex_table([["modo (1,1)", 0.1337], ["modo (2,1)", 0.698]],
                       header=["caso", "e(z=0)"],
                       path=str(tmp_path / "t.tex"))
    assert "\\toprule" in s and "\\bottomrule" in s
    assert "0.134" in s                                   # formato {:.3g}
    assert (tmp_path / "t.tex").read_text() == s


def test_diagnostics_reexports():
    from lateralcauchy import diagnostics as d
    assert d.plot_history is pl.plot_history
    assert abs(d.ld_ratio(1.0, 2.0, 1.0) - 1.0) < 1e-12
