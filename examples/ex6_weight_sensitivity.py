"""ex6 - Sensibilidad al balance de pesos (spec §2-ex6). Base: ex2.

Malla logaritmica (lambda_g, lambda_f) in {0.1, 1, 10}^2 con
lambda_PDE = lambda_lat = 1 fijos; 3 semillas por celda (27 corridas).
Solo se MIDE la sensibilidad (nada de pesos adaptativos, §0.5): esperado
sensibilidad moderada con optimo plano.

Figura: figs/ex6_heatmap.pdf — mapa de calor 3x3 de la media de E0, con la
std anotada en cada celda.

Ejecutar desde la raiz del repo:  python examples/ex6_weight_sensitivity.py [--smoke]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from lateralcauchy import LateralCauchyCylinder
from lateralcauchy import plotting as pl
from lateralcauchy import runconfig as rc
from lateralcauchy.diagnostics import atenuacion, regimen
from lateralcauchy.metrics import malla_e0, e0, sample_cylinder, rel_l2, torchify
from ex2_bessel import (
    A_OBJETIVO, MEDIO_NP, construir_referencia, medio_torch_const,
    omega_para_atenuacion, senal_tapa,
)

EX = "ex6"
R = L = TMAX = 1.0
LAMBDAS = (0.1, 1.0, 10.0)


def figura(media, std):
    """figs/ex6_heatmap.pdf: media de E0 (3x3) con std anotada por celda."""
    import matplotlib.pyplot as plt
    with pl.paper_style():
        fig, ax = plt.subplots(figsize=(pl.SINGLE_COL, 2.8))
        im = ax.imshow(media, origin="lower", cmap=pl.CMAP_FIELD)
        etiquetas = [f"{v:g}" for v in LAMBDAS]
        ax.set_xticks(range(3), etiquetas)
        ax.set_yticks(range(3), etiquetas)
        ax.set_xlabel(r"$\lambda_f$")
        ax.set_ylabel(r"$\lambda_g$")
        ax.grid(False)
        lo, hi = media.min(), media.max()
        for i in range(3):
            for j in range(3):
                frac = 0.5 if hi == lo else (media[i, j] - lo) / (hi - lo)
                ax.text(j, i, f"{media[i, j]:.2f}\n$\\pm${std[i, j]:.2f}",
                        ha="center", va="center", fontsize=6.5,
                        color="black" if frac > 0.6 else "white")
        fig.colorbar(im, ax=ax, label=r"$E_0$ (media de 3 semillas)")
        fig.tight_layout()
        pl.save_fig(fig, os.path.join(rc.FIGS_DIR, "ex6_heatmap"))


def main(smoke=False):
    omega = omega_para_atenuacion(A_OBJETIVO, MEDIO_NP, L)
    a = atenuacion(omega, MEDIO_NP, L)
    print(f"[{EX}] base ex2: omega={omega:.4f}  a={a:.4f}")
    ref, _, _ = construir_referencia(omega, MEDIO_NP)
    print(f"[{EX}] regimen (§4.4):", regimen(senal_tapa(ref), MEDIO_NP, L, eta=0.0))

    rho, c, k = medio_torch_const()
    seeds = rc.SEEDS_RED[:1] if smoke else rc.SEEDS_RED[0:3]
    opts = rc.fit_opts(smoke)
    malla = malla_e0(R, TMAX)
    Xg = sample_cylinder(R, L, TMAX, 3000, seed=6)

    media = np.zeros((3, 3))
    std = np.zeros((3, 3))
    for i, lg in enumerate(LAMBDAS):
        for j, lf in enumerate(LAMBDAS):
            vals = []
            for s in seeds:
                op = LateralCauchyCylinder(R, L, TMAX, rho, c, k,
                                           net_config=rc.net_config(s))
                o = dict(opts, weights=(1.0, lg, lf, 1.0))
                t0 = time.time()
                op.fit(torchify(ref.g, op.device), torchify(ref.f, op.device), **o)
                segs = time.time() - t0
                E0 = e0(op, ref, malla)
                eg = rel_l2(op.grad_T(torch.as_tensor(Xg, device=op.device)),
                            ref.grad_T(Xg))
                rc.escribir_raw(EX, caso=f"lg{lg:g}_lf{lf:g}", semilla_red=s,
                                a=a, omega=omega, L=L, E0=E0, err_global=eg,
                                tiempo_s=round(segs, 2))
                vals.append(E0)
                print(f"[{EX}] lg={lg:g} lf={lf:g} seed {s}: E0={E0:.3e} "
                      f"({segs:.0f}s)")
            media[i, j] = np.mean(vals)
            std[i, j] = np.std(vals)

    figura(media, std)
    print(f"[{EX}] figura: figs/ex6_heatmap.pdf")
    return media


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="presupuesto reducido (CI): 1 semilla por celda, pocas iteraciones")
    main(smoke=ap.parse_args().smoke)
