"""ex1 - Control de maquinaria: PINN vs solucion manufacturada (spec §2-ex1).

T*(z,t) = exp(-lam t) cos(beta z + psi) con lam=1, beta=1, psi=pi/4 (traza
basal no trivial). No varia en (x,y), asi que NO estresa el mal
condicionamiento: si E0 > ~1e-2 es bug de maquinaria, no del metodo.

Protocolo (docs/SIMULACIONES.md): 5 corridas (SEEDS_RED), E0 contra grad T*
analitico + error global + perfiles vs z y vs t; figuras figs/ex1_corte.pdf y
figs/ex1_traza_basal.pdf; raw.csv y modelos segun el contrato §0.4.

Ejecutar desde la raiz del repo:  python examples/ex1_manufactured.py [--smoke]
"""

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import torch

from lateralcauchy import LateralCauchyCylinder, ManufacturedZ
from lateralcauchy import plotting as pl
from lateralcauchy import runconfig as rc
from lateralcauchy.diagnostics import regimen
from lateralcauchy.metrics import (
    malla_e0, e0, perfil_error_z, perfil_error_t, sample_cylinder, rel_l2,
    torchify,
)

import matplotlib.pyplot as plt

EX = "ex1"
R = L = TMAX = 1.0
LAM, PSI = 1.0, math.pi / 4          # beta = sqrt(rhoc*lam/k) = 1

MEDIO_NP = (lambda z: np.ones_like(z), lambda z: np.ones_like(z))


def _senal_tapa(g, n=1001):
    """Muestrea g en (0,0,L,t) para el diagnostico regimen (senal temporal)."""
    t = np.linspace(0.0, TMAX, n)
    X = np.stack([np.zeros_like(t), np.zeros_like(t), np.full_like(t, L), t], 1)
    return np.asarray(g(X)).ravel(), t[1] - t[0]


def figuras(op, exact):
    # figs/ex1_corte.pdf: T_theta vs T* sobre el eje x=y=0, t en {0.25,0.5,0.9}
    z = np.linspace(0.0, L, 201)
    with pl.paper_style():
        fig, ax = plt.subplots(figsize=(pl.SINGLE_COL, 2.55))
        for t0, color in zip((0.25, 0.5, 0.9), pl.CAT):
            X = np.stack([np.zeros_like(z), np.zeros_like(z), z,
                          np.full_like(z, t0)], 1)
            ax.plot(z, exact.T(X).ravel(), "-", color=color, alpha=0.55)
            Tp = op.T(torch.as_tensor(X, device=op.device)).cpu().numpy().ravel()
            ax.plot(z, Tp, "--", color=color, label=fr"$t={t0}$")
        ax.set_xlabel(r"$z$")
        ax.set_ylabel(r"$T$ en $x{=}y{=}0$")
        ax.legend(title=r"PINN (- -) vs $T^*$ (—)")
        fig.tight_layout()
        pl.save_fig(fig, os.path.join(rc.FIGS_DIR, "ex1_corte"))

    # figs/ex1_traza_basal.pdf: dz T_theta|_{z=0} vs exacto, en t
    t = np.linspace(TMAX / 200, TMAX, 200)
    X = np.stack([np.zeros_like(t), np.zeros_like(t), np.zeros_like(t), t], 1)
    dz_pinn = op.grad_T(torch.as_tensor(X, device=op.device))[:, 2].cpu().numpy()
    dz_true = exact.grad_T(X)[:, 2]
    with pl.paper_style():
        fig, ax = plt.subplots(figsize=(pl.SINGLE_COL, 2.55))
        ax.plot(t, dz_true, "-", color=pl.BLUE, label="exacta")
        ax.plot(t, dz_pinn, "o--", color=pl.VERMILLION, markerfacecolor="white",
                markevery=10, label="PINN")
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\partial_z T|_{z=0}$")
        ax.legend()
        fig.tight_layout()
        pl.save_fig(fig, os.path.join(rc.FIGS_DIR, "ex1_traza_basal"))


def main(smoke=False):
    exact = ManufacturedZ(lam=LAM, psi=PSI, rhoc=1.0, k=1.0, L=L)
    print(f"[{EX}] regimen (§4.4):",
          regimen(_senal_tapa(exact.g), MEDIO_NP, L, eta=0.0))

    one = lambda X: torch.ones_like(X[:, :1])
    seeds = rc.SEEDS_RED[:1] if smoke else rc.SEEDS_RED
    opts = rc.fit_opts(smoke)
    malla = malla_e0(R, TMAX)
    Xg = sample_cylinder(R, L, TMAX, 3000, seed=1)

    e0s, primero = [], None
    for s in seeds:
        op = LateralCauchyCylinder(R, L, TMAX, one, one, one,
                                   net_config=rc.net_config(s))
        t0 = time.time()
        op.fit(torchify(exact.g, op.device), torchify(exact.f, op.device), **opts)
        segs = time.time() - t0
        E0 = e0(op, exact, malla)
        eg = rel_l2(op.grad_T(torch.as_tensor(Xg, device=op.device)),
                    exact.grad_T(Xg))
        rc.escribir_raw(EX, caso="manufacturada", semilla_red=s, L=L,
                        E0=E0, err_global=eg, tiempo_s=round(segs, 2))
        rc.guardar_modelo(op, EX, f"seed{s}")
        e0s.append(E0)
        if primero is None:
            primero = op
        print(f"[{EX}] seed {s}: E0={E0:.3e}  err_global={eg:.3e}  ({segs:.0f}s)")

    media = float(np.mean(e0s))
    ok = "OK" if media <= 1e-2 else "FALLA (bug de maquinaria, no del metodo)"
    print(f"[{EX}] E0 media = {media:.3e}  [criterio E0 <~ 1e-2: {ok}]")

    # perfiles de control §0.3 (primer modelo)
    zc, ez = perfil_error_z(primero, exact, R, L, TMAX)
    print(f"[{EX}] perfil e(z): max={ez.max():.3e} en z={zc[np.argmax(ez)]:.2f}")
    tt, et = perfil_error_t(primero, exact, malla)
    print(f"[{EX}] perfil e(t): max={et.max():.3e} en t={tt[np.argmax(et)]:.2f}")

    figuras(primero, exact)
    print(f"[{EX}] figuras: figs/ex1_corte.pdf, figs/ex1_traza_basal.pdf")
    return media


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="presupuesto reducido (CI): 1 semilla, pocas iteraciones")
    main(smoke=ap.parse_args().smoke)
