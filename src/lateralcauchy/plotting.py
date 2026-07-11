"""Figuras y tablas listas para artículo.

Estilo de publicación (serif + mathtext, tamaños de columna de revista, ejes
recesivos, paleta Okabe–Ito segura para daltonismo) y exportación PDF vectorial
(para LaTeX) + PNG 300 dpi. Todas las figuras se guardan con `save_fig` o
pasando `path` (sin extensión o con ella).

    plot_history(op.history, path="figs/history")
    plot_error_vs_z(zc, ez, path="figs/error_z")
    plot_gradient_profile(z, dzT_pinn, dzT_ref, path="figs/perfil")
    plot_slice(F_pinn, F_ref, extent, path="figs/corte")
    plot_frequency_sweep(gammas, errs, L, path="figs/sweep")
    plot_heatmap(grid, cols, filas, ..., path="figs/mapa")
    latex_table(rows, header, path="tabla.tex")     # booktabs, \input-able

Sin títulos dentro de la figura (van en el caption de LaTeX); pásalo en `title`
si se quiere una versión de trabajo.
"""

import os
from contextlib import contextmanager

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler

# Paleta Okabe–Ito (validada CVD-safe); el gris oscuro es para agregados/texto.
BLUE, VERMILLION, GREEN, ORANGE, PURPLE = (
    "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7")
INK = "#333333"
CAT = [BLUE, VERMILLION, GREEN, ORANGE, PURPLE]
CMAP_FIELD, CMAP_ERR = "viridis", "magma"     # secuenciales perceptuales

SINGLE_COL = 3.4      # ancho en pulgadas: columna simple de revista
DOUBLE_COL = 7.0      # ancho a dos columnas

_PAPER_RC = {
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "lines.linewidth": 1.6, "lines.markersize": 5,
    "legend.frameon": False, "figure.dpi": 110,
    "axes.prop_cycle": cycler(color=CAT),
}


@contextmanager
def paper_style():
    """Contexto rcParams de publicación; envuelve la construcción de la figura."""
    with matplotlib.rc_context(_PAPER_RC):
        yield


def save_fig(fig, path, formats=("pdf", "png"), dpi=300):
    """Guarda `path`.pdf (vectorial, para LaTeX) y `path`.png (300 dpi).
    Si `path` trae extensión, se respeta solo ese formato. Devuelve las rutas."""
    root, ext = os.path.splitext(path)
    fmts = (ext.lstrip("."),) if ext else formats
    os.makedirs(os.path.dirname(root) or ".", exist_ok=True)
    out = []
    for f in fmts:
        p = f"{root}.{f}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        out.append(p)
    return out


def _finish(fig, ax, title, path):
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if path:
        save_fig(fig, path)
    return fig


def plot_history(history, path=None, title=None):
    """Curvas de pérdida por término; 'total' en tinta, componentes en paleta."""
    with paper_style():
        fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.55))
        ax.semilogy(history["total"], color=INK, label="total", zorder=5)
        for key, c in zip(("pde", "g", "f", "lat"), CAT):
            ax.semilogy(history[key], color=c, label=key, alpha=0.9)
        ax.set_xlabel("iteración")
        ax.set_ylabel(r"$\mathcal{L}(\theta)$")
        ax.legend(ncol=2)
        return _finish(fig, ax, title, path)


def _plot_error_profile(coord, err, xlabel, path, title):
    with paper_style():
        fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.55))
        ax.semilogy(coord, err, "o-", color=BLUE)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\|\nabla T_\theta-\nabla T\|/\|\nabla T\|$")
        return _finish(fig, ax, title, path)


def plot_error_vs_z(zc, err, path=None, title=None):
    """Perfil e(z) (crece hacia la base: firma del mal condicionamiento)."""
    return _plot_error_profile(zc, err, r"$z$  (tapa $z=L$ a la derecha)", path, title)


def plot_error_vs_t(tc, err, path=None, title=None):
    """Perfil e(t) (control: debería ser ≈ plano)."""
    return _plot_error_profile(tc, err, r"$t$", path, title)


def plot_gradient_profile(z, dzT_pred, dzT_true, path=None, title=None,
                          labels=("PINN", "referencia")):
    """∂_z T a lo largo de z: predicción vs verdad, superpuestas."""
    with paper_style():
        fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.55))
        ax.plot(z, dzT_true, "-", color=BLUE, label=labels[1])
        ax.plot(z, dzT_pred, "o--", color=VERMILLION, label=labels[0],
                markerfacecolor="white")
        ax.set_xlabel(r"$z$")
        ax.set_ylabel(r"$\partial_z T$")
        ax.legend()
        return _finish(fig, ax, title, path)


def eval_slice(fn, R, L, t0, nx=121, nz=121):
    """Evalúa un campo escalar fn(X)→(N,1) en el plano y=0, t=t0.
    Devuelve (F (nz,nx), extent) para plot_slice / imshow."""
    x = np.linspace(-R, R, nx)
    z = np.linspace(0.0, L, nz)
    XX, ZZ = np.meshgrid(x, z)
    X = np.stack([XX.ravel(), np.zeros(XX.size), ZZ.ravel(),
                  np.full(XX.size, t0)], axis=1)
    F = np.asarray(fn(X)).reshape(nz, nx)
    return F, (-R, R, 0.0, L)


def plot_slice(F_pred, F_true, extent, path=None, title=None,
               clabel=r"$T$", labels=("PINN", "referencia")):
    """Tres paneles: predicción | referencia | |diferencia|, escala compartida."""
    with paper_style():
        fig, axs = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.3), sharey=True)
        vmin = min(F_pred.min(), F_true.min())
        vmax = max(F_pred.max(), F_true.max())
        for ax, F, lab in zip(axs[:2], (F_pred, F_true), labels):
            im = ax.imshow(F, origin="lower", extent=extent, aspect="auto",
                           cmap=CMAP_FIELD, vmin=vmin, vmax=vmax)
            ax.set_xlabel(r"$x$"); ax.set_title(lab, fontsize=9)
            ax.grid(False)
        axs[0].set_ylabel(r"$z$")
        fig.colorbar(im, ax=axs[:2], label=clabel, shrink=0.9)
        imd = axs[2].imshow(np.abs(F_pred - F_true), origin="lower",
                            extent=extent, aspect="auto", cmap=CMAP_ERR)
        axs[2].set_xlabel(r"$x$"); axs[2].set_title("|diferencia|", fontsize=9)
        axs[2].grid(False)
        fig.colorbar(imd, ax=axs[2], shrink=0.9)
        if title:
            fig.suptitle(title)
        if path:
            save_fig(fig, path)
        return fig


def plot_frequency_sweep(gammas, errs, L, path=None, title=None):
    """Error en z=0 vs Lγ, con la recta de referencia ∝ e^{Lγ} (teoría)."""
    g = np.asarray(gammas, float)
    e = np.asarray(errs, float)
    with paper_style():
        fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.55))
        x = L * g
        ax.semilogy(x, e, "o-", color=BLUE, label="medido")
        ref = e[0] * np.exp(x - x[0])              # pendiente teórica e^{Lγ}
        ax.semilogy(x, ref, "--", color=INK, alpha=0.6,
                    label=r"$\propto e^{L\gamma}$")
        ax.set_xlabel(r"$L\gamma$")
        ax.set_ylabel(r"error de $\nabla T$ en $z=0$")
        ax.legend()
        return _finish(fig, ax, title, path)


def plot_heatmap(grid, xlabels, ylabels, xlabel, ylabel, path=None, title=None,
                 cbar_label="error relativo"):
    """Mapa (p. ej. ruido × frecuencia) con celdas anotadas."""
    G = np.asarray(grid, float)
    with paper_style():
        fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.7))
        im = ax.imshow(G, origin="lower", aspect="auto", cmap=CMAP_FIELD)
        ax.set_xticks(range(len(xlabels)), xlabels)
        ax.set_yticks(range(len(ylabels)), ylabels)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.grid(False)
        lo, hi = G.min(), G.max()
        for i in range(G.shape[0]):
            for j in range(G.shape[1]):
                frac = 0.5 if hi == lo else (G[i, j] - lo) / (hi - lo)
                ax.text(j, i, f"{G[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if frac > 0.6 else "white")
        fig.colorbar(im, ax=ax, label=cbar_label)
        return _finish(fig, ax, title, path)


def latex_table(rows, header, path=None, fmt="{:.3g}"):
    """Tabla booktabs lista para \\input{} en LaTeX.

    rows: lista de filas (str o números; los números se formatean con `fmt`).
    header: lista de encabezados. Devuelve el string; lo escribe si hay path."""
    def cell(v):
        return fmt.format(v) if isinstance(v, (int, float, np.floating)) else str(v)

    ncol = len(header)
    lines = ["\\begin{tabular}{l" + "c" * (ncol - 1) + "}", "\\toprule",
             " & ".join(header) + r" \\", "\\midrule"]
    lines += [" & ".join(cell(v) for v in row) + r" \\" for row in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    out = "\n".join(lines) + "\n"
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf8") as fh:
            fh.write(out)
    return out
