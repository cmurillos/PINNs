"""Figura de la §3.3 del articulo: figs/aten_curvas.pdf (§1.4 del spec).

Curvas a(omega) por diagnostics.atenuacion para (a) medio constante rc=k=1 y
(b) el k(z)=1+z de ex3; 40 valores de omega log-espaciados tales que a recorre
~[1e-3, 1]. Ejes: log a vs sqrt(omega), con la recta teorica del caso
constante (semiespacio, Apendice C): a = exp(-L*sqrt(omega/2)).

Ejecutar desde la raiz del repo:  python scripts/aten_curvas.py
"""

import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np

from lateralcauchy import plotting as pl
from lateralcauchy.diagnostics import atenuacion
from lateralcauchy.runconfig import FIGS_DIR

import matplotlib.pyplot as plt

L = 1.0
MEDIOS = {
    r"$\rho c = k = 1$": (lambda z: np.ones_like(z), lambda z: np.ones_like(z)),
    r"$k(z) = 1 + z$":   (lambda z: np.ones_like(z), lambda z: 1.0 + z),
}


def main():
    # a = exp(-sqrt(w/2)) recorre [1e-3, 1] para sqrt(w/2) in [0, 6.9]
    omegas = np.logspace(-2, 2.0, 40)
    with pl.paper_style():
        fig, ax = plt.subplots(figsize=(pl.SINGLE_COL, 2.55))
        for (nombre, medio), color in zip(MEDIOS.items(), pl.CAT):
            a = np.array([atenuacion(w, medio, L) for w in omegas])
            ax.semilogy(np.sqrt(omegas), a, "-", color=color, label=nombre)
        ax.semilogy(np.sqrt(omegas), np.exp(-L * np.sqrt(omegas / 2.0)), "--",
                    color=pl.INK, alpha=0.6,
                    label=r"$e^{-L\sqrt{\omega/2}}$ (semiespacio)")
        ax.set_xlabel(r"$\sqrt{\omega}$")
        ax.set_ylabel(r"$a(\omega)$")
        ax.legend()
        fig.tight_layout()
        pl.save_fig(fig, os.path.join(FIGS_DIR, "aten_curvas"))
    print(f"[aten_curvas] figura: {FIGS_DIR}/aten_curvas.pdf")


if __name__ == "__main__":
    main()
