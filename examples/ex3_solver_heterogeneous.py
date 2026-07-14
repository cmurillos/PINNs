"""ex3 - Medio heterogeneo k(z) = 1 + z (spec §2-ex3).

Variacion 2x entre base y tapa, C-infinito. NO hay solucion analitica: la
referencia es el solver modal con k(z) (unico disponible). La frecuencia se
elige por biseccion sobre diagnostics.atenuacion CON el medio heterogeneo,
misma atenuacion objetivo a ~ e^{-1} que ex2 (sin difusividad representativa:
la curva medida es del medio real).

Esperado: E0 comparable al de ex2 a igual a — la heterogeneidad determinista
no degrada mas alla de lo previsto por la curva. 5 corridas (SEEDS_RED).
Figuras: figs/ex3_k_perfil.pdf, figs/ex3_traza_basal.pdf, figs/ex3_error_vs_z.pdf.
Los modelos guardados en results/ex3/models/ alimentan la visualizacion 5.7.

Ejecutar desde la raiz del repo:  python examples/ex3_solver_heterogeneous.py [--smoke]
"""

import argparse
import math
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
from lateralcauchy.metrics import (
    malla_e0, e0, perfil_error_z, sample_cylinder, rel_l2, torchify,
)
from ex2_bessel import (
    A_OBJETIVO, M, N, construir_referencia, omega_para_atenuacion,
    senal_tapa, verificacion_cruzada,
)

import matplotlib.pyplot as plt

EX = "ex3"
R = L = TMAX = 1.0

# medio del spec: rho c = 1, k(z) = 1 + z
MEDIO_NP = (lambda z: np.ones_like(z), lambda z: 1.0 + z)


def medio_torch():
    one = lambda X: torch.ones_like(X[:, :1])
    k = lambda X: 1.0 + X[:, 2:3]
    return one, one, k


def figuras(op, ref, omega):
    # figs/ex3_k_perfil.pdf: el k(z) usado
    z = np.linspace(0.0, L, 201)
    with pl.paper_style():
        fig, ax = plt.subplots(figsize=(pl.SINGLE_COL, 2.55))
        ax.plot(z, MEDIO_NP[1](z), "-", color=pl.BLUE)
        ax.set_xlabel(r"$z$")
        ax.set_ylabel(r"$k(z) = 1 + z$")
        fig.tight_layout()
        pl.save_fig(fig, os.path.join(rc.FIGS_DIR, "ex3_k_perfil"))

    # figs/ex3_traza_basal.pdf: dz T_theta|_{z=0} vs referencia modal, en t
    # (en el eje r=0, donde J0 es maximo)
    t = np.linspace(TMAX / 200, TMAX, 200)
    X = np.stack([np.zeros_like(t), np.zeros_like(t), np.zeros_like(t), t], 1)
    dz_pinn = op.grad_T(torch.as_tensor(X, device=op.device))[:, 2].cpu().numpy()
    dz_ref = ref.grad_T(X)[:, 2]
    with pl.paper_style():
        fig, ax = plt.subplots(figsize=(pl.SINGLE_COL, 2.55))
        ax.plot(t, dz_ref, "-", color=pl.BLUE, label="referencia modal")
        ax.plot(t, dz_pinn, "o--", color=pl.VERMILLION, markerfacecolor="white",
                markevery=10, label="PINN")
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\partial_z T|_{z=0}$ en $r=0$")
        ax.legend()
        fig.tight_layout()
        pl.save_fig(fig, os.path.join(rc.FIGS_DIR, "ex3_traza_basal"))

    # figs/ex3_error_vs_z.pdf: perfil e(z), 20 franjas (§0.3)
    zc, ez = perfil_error_z(op, ref, R, L, TMAX)
    pl.plot_error_vs_z(zc, ez, path=os.path.join(rc.FIGS_DIR, "ex3_error_vs_z"))


def main(smoke=False):
    omega = omega_para_atenuacion(A_OBJETIVO, MEDIO_NP, L)
    a = atenuacion(omega, MEDIO_NP, L)
    print(f"[{EX}] biseccion con k(z)=1+z: omega={omega:.4f}  ->  a={a:.4f} "
          f"(objetivo e^-1)")

    ref, _, _ = construir_referencia(omega, MEDIO_NP)
    err_solver = verificacion_cruzada(ref, omega, MEDIO_NP, analitico=False)
    print(f"[{EX}] verificacion cruzada solver<->BVP complejo: "
          f"rel err = {err_solver:.2e}")

    print(f"[{EX}] regimen (§4.4):", regimen(senal_tapa(ref), MEDIO_NP, L, eta=0.0))

    rho, c, k = medio_torch()
    seeds = rc.SEEDS_RED[:1] if smoke else rc.SEEDS_RED
    opts = rc.fit_opts(smoke)
    malla = malla_e0(R, TMAX)
    Xg = sample_cylinder(R, L, TMAX, 3000, seed=3)

    e0s, primero = [], None
    for s in seeds:
        op = LateralCauchyCylinder(R, L, TMAX, rho, c, k,
                                   net_config=rc.net_config(s))
        t0 = time.time()
        op.fit(torchify(ref.g, op.device), torchify(ref.f, op.device), **opts)
        segs = time.time() - t0
        E0 = e0(op, ref, malla)
        eg = rel_l2(op.grad_T(torch.as_tensor(Xg, device=op.device)),
                    ref.grad_T(Xg))
        rc.escribir_raw(EX, caso="k_lineal", semilla_red=s, a=a, omega=omega,
                        L=L, E0=E0, err_global=eg, tiempo_s=round(segs, 2))
        rc.guardar_modelo(op, EX, f"seed{s}")
        e0s.append(E0)
        if primero is None:
            primero = op
        print(f"[{EX}] seed {s}: E0={E0:.3e}  err_global={eg:.3e}  ({segs:.0f}s)")

    print(f"[{EX}] E0 media = {np.mean(e0s):.3e}  "
          f"(esperado comparable a ex2 a igual a)")

    figuras(primero, ref, omega)
    print(f"[{EX}] figuras: figs/ex3_k_perfil.pdf, figs/ex3_traza_basal.pdf, "
          f"figs/ex3_error_vs_z.pdf")
    return float(np.mean(e0s))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="presupuesto reducido (CI): 1 semilla, pocas iteraciones")
    main(smoke=ap.parse_args().smoke)
