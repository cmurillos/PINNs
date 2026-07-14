"""ex4 - Robustez al ruido (spec §2-ex4). Base: configuracion de ex2.

Ruido gaussiano aditivo iid por punto de muestreo, INDEPENDIENTE sobre g y f,
amplitud relativa eta in {0, 1e-3, 1e-2, 5e-2} respecto de ||g||_inf y
||f||_inf respectivamente. Protocolo: por nivel, 5 semillas de ruido
(SEEDS_RUIDO) x 3 semillas de red (SEEDS_RED[0:3]) = 15 corridas; el nivel
eta=0 usa solo las 3 de red.

Esperado: amplificacion conforme al criterio (3.8), factor 1/a(omega); la
suavidad de la red fija el piso a eta pequeno. Figura: figs/ex4_ruido.pdf
(E0 vs eta, log-log, media +/- std; eta=0 como piso horizontal).

Ejecutar desde la raiz del repo:  python examples/ex4_noise.py [--smoke]
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
from lateralcauchy.metrics import (
    malla_e0, e0, sample_cylinder, sample_disk_slice, rel_l2, torchify,
)
from ex2_bessel import (
    A_OBJETIVO, MEDIO_NP, construir_referencia, medio_torch_const,
    omega_para_atenuacion, senal_tapa,
)

EX = "ex4"
R = L = TMAX = 1.0
ETAS = (0.0, 1e-3, 1e-2, 5e-2)


def con_ruido(fn, eta, escala, rng):
    """fn + eta*escala*N(0,1) iid por punto de muestreo (gaussiano aditivo)."""
    def ruidosa(X):
        v = np.asarray(fn(X))
        return v + eta * escala * rng.standard_normal(v.shape)
    return ruidosa


def figura(filas):
    """figs/ex4_ruido.pdf: E0 vs eta log-log, media +/- std; eta=0 como piso."""
    import matplotlib.pyplot as plt
    por_eta = {}
    for f in filas:
        por_eta.setdefault(float(f["eta"]), []).append(float(f["E0"]))
    etas = sorted(e for e in por_eta if e > 0)
    med = np.array([np.mean(por_eta[e]) for e in etas])
    std = np.array([np.std(por_eta[e]) for e in etas])
    with pl.paper_style():
        fig, ax = plt.subplots(figsize=(pl.SINGLE_COL, 2.55))
        ax.errorbar(etas, med, yerr=std, fmt="o-", color=pl.BLUE, capsize=2.5,
                    label=r"$E_0(\eta)$")
        if 0.0 in por_eta:
            piso = float(np.mean(por_eta[0.0]))
            ax.axhline(piso, ls="--", color=pl.INK, alpha=0.6,
                       label=r"piso $\eta=0$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\eta$ (amplitud relativa del ruido)")
        ax.set_ylabel(r"$E_0$")
        ax.legend()
        fig.tight_layout()
        pl.save_fig(fig, os.path.join(rc.FIGS_DIR, "ex4_ruido"))


def main(smoke=False):
    omega = omega_para_atenuacion(A_OBJETIVO, MEDIO_NP, L)
    a = atenuacion(omega, MEDIO_NP, L)
    print(f"[{EX}] base ex2: omega={omega:.4f}  a={a:.4f}  (1/a={1/a:.2f} de "
          f"amplificacion prevista)")
    ref, _, _ = construir_referencia(omega, MEDIO_NP)

    # escalas ||g||_inf y ||f||_inf sobre una muestra fija de la tapa
    Xtop = sample_disk_slice(R, TMAX, L, 4000, seed=9)
    esc_g = float(np.max(np.abs(ref.g(Xtop))))
    esc_f = float(np.max(np.abs(ref.f(Xtop))))

    rho, c, k = medio_torch_const()
    seeds_red = rc.SEEDS_RED[:1] if smoke else rc.SEEDS_RED[0:3]
    seeds_ruido = rc.SEEDS_RUIDO[:1] if smoke else rc.SEEDS_RUIDO
    opts = rc.fit_opts(smoke)
    malla = malla_e0(R, TMAX)
    Xg = sample_cylinder(R, L, TMAX, 3000, seed=4)

    for eta in ETAS:
        print(f"[{EX}] regimen (§4.4, eta={eta:g}):",
              regimen(senal_tapa(ref), MEDIO_NP, L, eta=eta))
        combos = ([(s, None) for s in seeds_red] if eta == 0.0 else
                  [(s, sr) for sr in seeds_ruido for s in seeds_red])
        for s, sr in combos:
            if sr is None:
                g_dato, f_dato = ref.g, ref.f
            else:
                # generadores separados por rol e independientes entre g y f
                g_dato = con_ruido(ref.g, eta, esc_g, np.random.default_rng([sr, 0]))
                f_dato = con_ruido(ref.f, eta, esc_f, np.random.default_rng([sr, 1]))
            op = LateralCauchyCylinder(R, L, TMAX, rho, c, k,
                                       net_config=rc.net_config(s))
            t0 = time.time()
            op.fit(torchify(g_dato, op.device), torchify(f_dato, op.device), **opts)
            segs = time.time() - t0
            E0 = e0(op, ref, malla)
            eg = rel_l2(op.grad_T(torch.as_tensor(Xg, device=op.device)),
                        ref.grad_T(Xg))
            rc.escribir_raw(EX, caso=f"eta{eta:g}", semilla_red=s,
                            semilla_ruido=sr, eta=eta, a=a, omega=omega, L=L,
                            E0=E0, err_global=eg, tiempo_s=round(segs, 2))
            rc.guardar_modelo(op, EX, f"eta{eta:g}_seed{s}" +
                              ("" if sr is None else f"_ruido{sr}"))
            print(f"[{EX}] eta={eta:g} red={s} ruido={sr}: E0={E0:.3e} ({segs:.0f}s)")

    figura(rc.leer_raw(EX))
    print(f"[{EX}] figura: figs/ex4_ruido.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="presupuesto reducido (CI): 1x1 semillas, pocas iteraciones")
    main(smoke=ap.parse_args().smoke)
