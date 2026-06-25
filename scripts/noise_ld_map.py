"""Mapa cruzado: error de grad_T en z=0 vs (nivel de ruido) x (frecuencia espacial).

Cruza los dos ejes que gobiernan la inversion mal puesta:
  - frecuencia espacial del modo de Bessel -> tasa de amplificacion exp(L*gamma)
    (mayor gamma = mayor distancia de continuacion efectiva),
  - nivel de ruido aditivo en los datos de Cauchy (g, f).
Para cada celda entrena la PINN y mide el error en la base z=0 (recuperacion
genuina). Guarda un heatmap. Es el resultado de fondo: muestra DONDE el metodo
deja de recuperar. Ejecutar:  python scripts/noise_ld_map.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lateralcauchy import LateralCauchyCylinder, ManufacturedBessel
from lateralcauchy.metrics import sample_disk_slice, rel_l2, torchify

R = L = Tmax = 1.0
RHOC = K = 1.0
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")


def _perturbation(seed=0):
    """Perturbacion suave y determinista delta(x,y,t) en [-1,1] (fija por punto)."""
    rng = np.random.default_rng(seed)
    a, b, c, ph = rng.uniform(-3, 3, 4)
    return lambda X: np.sin(a * X[:, 0:1] + b * X[:, 1:2] + c * X[:, 3:4] + ph)


def main(modes=((1, 1), (2, 1), (3, 1)), noises=(0.0, 0.02, 0.05),
         adam_iters=2500, lbfgs_iters=600, savefig=True, **opts):
    one = lambda X: torch.ones_like(X[:, :1])
    Xbase = sample_disk_slice(R, Tmax, 0.0, 3000, seed=200)
    delta = _perturbation()
    grid = np.zeros((len(modes), len(noises)))
    gammas = []

    print("[map] error de grad_T en z=0  (modo/gamma x nivel de ruido):")
    for i, (m, n) in enumerate(modes):
        exact = ManufacturedBessel(m=m, n=n, R=R, lam=0.5, rhoc=RHOC, k=K, A=1.0, B=0.3)
        gammas.append(exact.gamma)
        Xtop = sample_disk_slice(R, Tmax, L, 4000, seed=9)
        rms = float(np.sqrt(np.mean(exact.g(Xtop) ** 2)))
        for j, eps in enumerate(noises):
            gN = lambda X, e=eps: exact.g(X) + e * rms * delta(X)
            fN = lambda X, e=eps: exact.f(X) + e * rms * delta(X)
            op = LateralCauchyCylinder(R, L, Tmax, one, one, one)
            op.fit(torchify(gN, op.device), torchify(fN, op.device),
                   adam_iters=adam_iters, lbfgs_iters=lbfgs_iters, **opts)
            err = rel_l2(op.grad_T(torch.as_tensor(Xbase, device=op.device)),
                         exact.grad_T(Xbase))
            grid[i, j] = err
            print(f"  modo({m},{n}) gamma={exact.gamma:4.2f}  ruido={eps:4.0%}  "
                  f"err(z=0)={err:.3e}")

    if savefig:
        os.makedirs(FIGDIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(noises)))
        ax.set_xticklabels([f"{e:.0%}" for e in noises])
        ax.set_yticks(range(len(modes)))
        ax.set_yticklabels([f"({m},{n}) γ={g:.1f}" for (m, n), g in zip(modes, gammas)])
        ax.set_xlabel("nivel de ruido en (g, f)")
        ax.set_ylabel("modo (frecuencia espacial)")
        ax.set_title("Error de grad_T en z=0  (ruido × distancia de continuación)")
        fig.colorbar(im, ax=ax, label="error relativo")
        for i in range(len(modes)):
            for j in range(len(noises)):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                        color="w", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGDIR, "noise_ld_map.png"), dpi=120)
        print(f"[map] figura en {FIGDIR}/noise_ld_map.png")
    return grid


if __name__ == "__main__":
    main()
