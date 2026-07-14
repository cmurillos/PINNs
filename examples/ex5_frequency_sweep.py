"""ex5 - Mapa de degradacion (RESULTADO CENTRAL, spec §2-ex5).

Dos barridos sobre la base de ex2 (modo m=0,n=1), MISMO presupuesto de
entrenamiento por caso (presupuesto fijo, no convergencia; §0.1):

  (i)  frecuencia: L=1 fijo; 7 omegas por biseccion tales que a(omega)
       recorre {0.7, 0.5, 0.37, 0.2, 0.1, 0.03, 0.01}; 5 semillas por caso.
  (ii) longitud: dato fijo (la omega de a=0.37 en L=1); L in {0.5, 0.75, 1,
       1.5, 2}; a de cada caso por diagnostics.atenuacion; 3 semillas.

Segunda curva: barrido (i) con ruido eta=1e-2 (3 semillas de ruido x 1 de red).

Figura central figs/ex5_colapso.pdf — log E0 vs log(1/a): las dos familias
(colapso esperado en una sola curva: la dificultad es a, no L ni omega por
separado), la recta de la cota inferior (3.8) E0 = eta/(2a) con la eta efectiva
(piso de optimizacion del caso mas facil sin ruido), la vertical a = eta y la
curva con ruido superpuesta. La prediccion de regimen() se imprime ANTES de
entrenar cada caso.

Ejecutar desde la raiz del repo:  python examples/ex5_frequency_sweep.py [--smoke]
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
    MEDIO_NP, construir_referencia, medio_torch_const, omega_para_atenuacion,
    senal_tapa,
)
from ex4_noise import con_ruido

EX = "ex5"
R = TMAX = 1.0
A_FREC = (0.7, 0.5, 0.37, 0.2, 0.1, 0.03, 0.01)   # barrido (i), L=1
L_LARGO = (0.5, 0.75, 1.0, 1.5, 2.0)              # barrido (ii), dato fijo
ETA_RUIDO = 1e-2


def _corrida(ref, L, s, opts, malla, Xg, g_dato=None, f_dato=None):
    """Entrena una semilla contra `ref` (geometria de largo L) y mide E0."""
    rho, c, k = medio_torch_const()
    op = LateralCauchyCylinder(R, L, TMAX, rho, c, k, net_config=rc.net_config(s))
    t0 = time.time()
    op.fit(torchify(g_dato or ref.g, op.device),
           torchify(f_dato or ref.f, op.device), **opts)
    segs = time.time() - t0
    E0 = e0(op, ref, malla)
    eg = rel_l2(op.grad_T(torch.as_tensor(Xg, device=op.device)), ref.grad_T(Xg))
    return op, E0, eg, segs


def figura():
    """figs/ex5_colapso.pdf: log E0 vs log(1/a), familias + cota + vertical."""
    import matplotlib.pyplot as plt
    filas = rc.leer_raw(EX)
    fam = {"freq": {}, "long": {}, "ruido": {}}
    for f in filas:
        pref = f["caso"].split("_")[0]
        if pref in fam:
            fam[pref].setdefault(float(f["a"]), []).append(float(f["E0"]))

    def serie(d):
        a = np.array(sorted(d, reverse=True))
        med = np.array([np.mean(d[x]) for x in a])
        std = np.array([np.std(d[x]) for x in a])
        return a, med, std

    with pl.paper_style():
        fig, ax = plt.subplots(figsize=(pl.SINGLE_COL, 2.8))
        estilos = {"freq": ("o", pl.BLUE, r"barrido $\omega$ ($L=1$)"),
                   "long": ("s", pl.GREEN, r"barrido $L$ ($\omega$ fija)"),
                   "ruido": ("^", pl.VERMILLION,
                             rf"barrido $\omega$, $\eta={ETA_RUIDO:g}$")}
        for nombre, (mk, color, lab) in estilos.items():
            if not fam[nombre]:
                continue
            a, med, std = serie(fam[nombre])
            ax.errorbar(1.0 / a, med, yerr=std, fmt=mk, ls="-", ms=4.5,
                        color=color, capsize=2, label=lab,
                        markerfacecolor="white" if nombre == "long" else None)

        # cota inferior (3.8): E0 = eta_ef/(2a), con eta_ef del piso de
        # optimizacion (caso mas facil sin ruido, a maxima)
        if fam["freq"]:
            a, med, _ = serie(fam["freq"])
            eta_ef = 2.0 * a[0] * med[0]
            xx = np.array([1.0 / a[0], 1.0 / a[-1]])
            ax.plot(xx, eta_ef * xx / 2.0, "--", color=pl.INK, alpha=0.6,
                    label=rf"cota $E_0=\eta_{{\rm ef}}/(2a)$")
        ax.axvline(1.0 / ETA_RUIDO, color=pl.PURPLE, ls=":", alpha=0.8,
                   label=rf"$a=\eta={ETA_RUIDO:g}$")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$1/a$ (amplificacion)")
        ax.set_ylabel(r"$E_0$")
        ax.legend(fontsize=6.5)
        fig.tight_layout()
        pl.save_fig(fig, os.path.join(rc.FIGS_DIR, "ex5_colapso"))


def main(smoke=False):
    opts = rc.fit_opts(smoke)
    malla = malla_e0(R, TMAX)
    Xg1 = sample_cylinder(R, 1.0, TMAX, 3000, seed=5)
    seeds_i = rc.SEEDS_RED[:1] if smoke else rc.SEEDS_RED
    seeds_ii = rc.SEEDS_RED[:1] if smoke else rc.SEEDS_RED[0:3]
    seeds_ruido = rc.SEEDS_RUIDO[:1] if smoke else rc.SEEDS_RUIDO[0:3]

    # presupuesto: verificar tiempo de una corrida base antes de lanzar (§2-ex5)
    n_total = (len(A_FREC) * len(seeds_i) + len(L_LARGO) * len(seeds_ii)
               + len(A_FREC) * len(seeds_ruido))
    omega037 = omega_para_atenuacion(0.37, MEDIO_NP, 1.0)
    ref037, _, _ = construir_referencia(omega037, MEDIO_NP)
    t0 = time.time()
    _corrida(ref037, 1.0, rc.SEEDS_RED[0], opts, malla, Xg1)
    por_corrida = time.time() - t0
    print(f"[{EX}] presupuesto: {n_total} corridas x ~{por_corrida:.0f}s "
          f"~ {n_total * por_corrida / 60:.0f} min")

    # ------------------------------------------------ barrido (i): frecuencia
    for a_obj in A_FREC:
        omega = omega_para_atenuacion(a_obj, MEDIO_NP, 1.0)
        a = atenuacion(omega, MEDIO_NP, 1.0)
        ref, _, _ = construir_referencia(omega, MEDIO_NP)
        print(f"[{EX}] (i) a_obj={a_obj:g}: omega={omega:.3f}  prediccion:",
              regimen(senal_tapa(ref), MEDIO_NP, 1.0, eta=0.0))
        for s in seeds_i:
            _, E0, eg, segs = _corrida(ref, 1.0, s, opts, malla, Xg1)
            rc.escribir_raw(EX, caso=f"freq_a{a_obj:g}", semilla_red=s, a=a,
                            omega=omega, L=1.0, E0=E0, err_global=eg,
                            tiempo_s=round(segs, 2))
            print(f"[{EX}] (i) a={a:.3f} seed {s}: E0={E0:.3e} ({segs:.0f}s)")

    # ------------------------------------------------- barrido (ii): longitud
    for Lc in L_LARGO:
        a = atenuacion(omega037, MEDIO_NP, Lc)
        ref, _, _ = construir_referencia(omega037, MEDIO_NP, L=Lc)
        Xg = sample_cylinder(R, Lc, TMAX, 3000, seed=5)
        print(f"[{EX}] (ii) L={Lc:g}: prediccion:",
              regimen(senal_tapa(ref), MEDIO_NP, Lc, eta=0.0))
        for s in seeds_ii:
            _, E0, eg, segs = _corrida(ref, Lc, s, opts, malla, Xg)
            rc.escribir_raw(EX, caso=f"long_L{Lc:g}", semilla_red=s, a=a,
                            omega=omega037, L=Lc, E0=E0, err_global=eg,
                            tiempo_s=round(segs, 2))
            print(f"[{EX}] (ii) L={Lc:g} a={a:.3f} seed {s}: E0={E0:.3e} ({segs:.0f}s)")

    # ------------------------------- segunda curva: barrido (i) con eta=1e-2
    from lateralcauchy.metrics import sample_disk_slice
    for a_obj in A_FREC:
        omega = omega_para_atenuacion(a_obj, MEDIO_NP, 1.0)
        a = atenuacion(omega, MEDIO_NP, 1.0)
        ref, _, _ = construir_referencia(omega, MEDIO_NP)
        Xtop = sample_disk_slice(R, TMAX, 1.0, 4000, seed=9)
        esc_g = float(np.max(np.abs(ref.g(Xtop))))
        esc_f = float(np.max(np.abs(ref.f(Xtop))))
        print(f"[{EX}] (ruido) a_obj={a_obj:g}: prediccion:",
              regimen(senal_tapa(ref), MEDIO_NP, 1.0, eta=ETA_RUIDO))
        for sr in seeds_ruido:
            g_dato = con_ruido(ref.g, ETA_RUIDO, esc_g, np.random.default_rng([sr, 0]))
            f_dato = con_ruido(ref.f, ETA_RUIDO, esc_f, np.random.default_rng([sr, 1]))
            _, E0, eg, segs = _corrida(ref, 1.0, rc.SEEDS_RED[0], opts, malla,
                                       Xg1, g_dato=g_dato, f_dato=f_dato)
            rc.escribir_raw(EX, caso=f"ruido_a{a_obj:g}",
                            semilla_red=rc.SEEDS_RED[0], semilla_ruido=sr,
                            eta=ETA_RUIDO, a=a, omega=omega, L=1.0, E0=E0,
                            err_global=eg, tiempo_s=round(segs, 2))
            print(f"[{EX}] (ruido) a={a:.3f} ruido={sr}: E0={E0:.3e} ({segs:.0f}s)")

    figura()
    print(f"[{EX}] figura central: figs/ex5_colapso.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="presupuesto reducido (CI): 1 semilla por caso, pocas iteraciones")
    main(smoke=ap.parse_args().smoke)
