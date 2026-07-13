"""Visualizacion espacio-temporal (subseccion 5.7 del articulo; spec §2-5.7).

CARGA el modelo entrenado de ex3 desde results/ex3/models/ (NO reentrena) y
reconstruye su referencia modal (biseccion determinista) para los datos g, f.

Salidas:
  figs/vis_espaciotemporal.pdf — reticula 3 filas x 4 columnas, instantes
    t in {0.2, 0.4, 0.6, 0.8}·Tmax: fila 1 campo T_theta en el corte
    longitudinal y=0 (semiplano r-z, malla 100x100); fila 2 dato g sobre el
    disco de la tapa (malla polar); fila 3 dato f sobre el disco. Barra de
    color COMUN por fila (mismos vmin/vmax en las 4 columnas).
  figs/vis_animacion.gif — los mismos tres paneles sincronizados, 40 cuadros
    en (0, Tmax], ~10 s.

Requiere haber corrido antes ex3 (guarda seed0 en results/ex3/models/).
Ejecutar desde la raiz del repo:  python scripts/vis_espaciotemporal.py
"""

import os
import sys

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_RAIZ, "src"))
sys.path.insert(0, os.path.join(_RAIZ, "examples"))

import numpy as np
import torch

from lateralcauchy import plotting as pl
from lateralcauchy import runconfig as rc
from ex2_bessel import A_OBJETIVO, construir_referencia, omega_para_atenuacion
from ex3_solver_heterogeneous import MEDIO_NP, medio_torch

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

R = L = TMAX = 1.0
INSTANTES = (0.2, 0.4, 0.6, 0.8)     # x Tmax (figura estatica)
N_CUADROS = 40                       # gif


def cargar():
    rho, c, k = medio_torch()
    op = rc.cargar_modelo("ex3", "seed0", rho, c, k, map_location="cpu")
    omega = omega_para_atenuacion(A_OBJETIVO, MEDIO_NP, L)
    ref, _, _ = construir_referencia(omega, MEDIO_NP)
    return op, ref


def corte_T(op, t0, n=100):
    """T_theta en el semiplano r-z (y=0, x=r>=0), malla n x n."""
    r = np.linspace(0.0, R, n)
    z = np.linspace(0.0, L, n)
    RRg, ZZ = np.meshgrid(r, z)
    X = np.stack([RRg.ravel(), np.zeros(RRg.size), ZZ.ravel(),
                  np.full(RRg.size, t0)], axis=1)
    T = op.T(torch.as_tensor(X, device=op.device)).cpu().numpy().reshape(n, n)
    return T


def malla_disco(n_r=50, n_th=121):
    r = np.linspace(0.0, R, n_r)
    th = np.linspace(0.0, 2.0 * np.pi, n_th)
    RRg, TH = np.meshgrid(r, th, indexing="ij")
    return RRg * np.cos(TH), RRg * np.sin(TH)


def dato_disco(fn, XX, YY, t0):
    X = np.stack([XX.ravel(), YY.ravel(), np.full(XX.size, L),
                  np.full(XX.size, t0)], axis=1)
    return np.asarray(fn(X)).reshape(XX.shape)


def _campos(op, ref, tiempos):
    """Precalcula (T, g, f) por instante y los (vmin, vmax) comunes por fila."""
    XX, YY = malla_disco()
    campos = [(corte_T(op, t), dato_disco(ref.g, XX, YY, t),
               dato_disco(ref.f, XX, YY, t)) for t in tiempos]
    lims = [(min(c[i].min() for c in campos), max(c[i].max() for c in campos))
            for i in range(3)]
    return campos, lims, (XX, YY)


def figura_estatica(op, ref):
    tiempos = [ti * TMAX for ti in INSTANTES]
    campos, lims, (XX, YY) = _campos(op, ref, tiempos)
    etiquetas = (r"$T_\theta$  (corte $y=0$)", r"$g$  (tapa $z=L$)",
                 r"$f$  (tapa $z=L$)")
    with pl.paper_style():
        fig, axs = plt.subplots(3, 4, figsize=(pl.DOUBLE_COL, 5.4))
        for j, (t0, (T, G, F)) in enumerate(zip(tiempos, campos)):
            ims = []
            im = axs[0, j].imshow(T, origin="lower", extent=(0, R, 0, L),
                                  aspect="auto", cmap=pl.CMAP_FIELD,
                                  vmin=lims[0][0], vmax=lims[0][1])
            ims.append(im)
            axs[0, j].set_title(fr"$t={t0:g}$", fontsize=8)
            axs[0, j].set_xlabel(r"$r$")
            for i, C in enumerate((G, F), start=1):
                im = axs[i, j].pcolormesh(XX, YY, C, cmap=pl.CMAP_FIELD,
                                          vmin=lims[i][0], vmax=lims[i][1],
                                          shading="gouraud")
                ims.append(im)
                axs[i, j].set_aspect("equal")
            for i in range(3):
                axs[i, j].grid(False)
                if j > 0:
                    axs[i, j].set_yticklabels([])
        axs[0, 0].set_ylabel(r"$z$")
        for i in range(3):
            # barra de color comun por fila (la atenuacion queda legible)
            fig.colorbar(axs[i, -1].collections[0] if i else axs[i, -1].images[0],
                         ax=axs[i, :], shrink=0.85, label=etiquetas[i])
        pl.save_fig(fig, os.path.join(rc.FIGS_DIR, "vis_espaciotemporal"))
    print("[vis] figs/vis_espaciotemporal.pdf")


def animacion(op, ref):
    tiempos = np.linspace(TMAX / N_CUADROS, TMAX, N_CUADROS)
    campos, lims, (XX, YY) = _campos(op, ref, tiempos)
    with pl.paper_style():
        fig, axs = plt.subplots(1, 3, figsize=(pl.DOUBLE_COL, 2.4))
        T0, G0, F0 = campos[0]
        imT = axs[0].imshow(T0, origin="lower", extent=(0, R, 0, L),
                            aspect="auto", cmap=pl.CMAP_FIELD,
                            vmin=lims[0][0], vmax=lims[0][1])
        axs[0].set_xlabel(r"$r$")
        axs[0].set_ylabel(r"$z$")
        imG = axs[1].pcolormesh(XX, YY, G0, cmap=pl.CMAP_FIELD,
                                vmin=lims[1][0], vmax=lims[1][1],
                                shading="gouraud")
        imF = axs[2].pcolormesh(XX, YY, F0, cmap=pl.CMAP_FIELD,
                                vmin=lims[2][0], vmax=lims[2][1],
                                shading="gouraud")
        for ax, ti in zip(axs, (r"$T_\theta$ ($y=0$)", r"$g$", r"$f$")):
            ax.set_title(ti, fontsize=8)
            ax.grid(False)
        for ax in axs[1:]:
            ax.set_aspect("equal")
        titulo = fig.suptitle(f"t = {tiempos[0]:.3f}", fontsize=9)
        fig.tight_layout()

        def cuadro(i):
            T, G, F = campos[i]
            imT.set_data(T)
            imG.set_array(G.ravel())
            imF.set_array(F.ravel())
            titulo.set_text(f"t = {tiempos[i]:.3f}")
            return imT, imG, imF, titulo

        anim = FuncAnimation(fig, cuadro, frames=N_CUADROS, blit=False)
        os.makedirs(rc.FIGS_DIR, exist_ok=True)
        ruta = os.path.join(rc.FIGS_DIR, "vis_animacion.gif")
        anim.save(ruta, writer=PillowWriter(fps=4), dpi=110)   # ~10 s
        plt.close(fig)
    print("[vis] figs/vis_animacion.gif")


def main():
    op, ref = cargar()
    figura_estatica(op, ref)
    animacion(op, ref)


if __name__ == "__main__":
    main()
