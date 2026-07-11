"""Benchmark reproducible con salida lista para artículo.

Entrena varias semillas contra un modo de Bessel (verdad analítica), valida con
op.validate y emite en scripts/figs/:

    history.{pdf,png}      pérdidas de la última corrida
    error_vs_z.{pdf,png}   perfil e(z)  (firma del mal condicionamiento)
    error_vs_t.{pdf,png}   perfil e(t)  (control, ≈ plano)
    perfil_dzT.{pdf,png}   ∂_z T a lo largo de z: PINN vs exacta
    corte_T.{pdf,png}      corte y=0, t=0.5: PINN | exacta | |diferencia|
    tabla_z0.tex           tabla booktabs: e(z=0) por semilla y media ± std

El error en la BASE z=0 es la métrica principal.  python scripts/benchmark.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import torch

from lateralcauchy import LateralCauchyCylinder, ManufacturedBessel
from lateralcauchy import plotting as pl
from lateralcauchy.metrics import torchify

R = L = Tmax = 1.0
RHOC = K = 1.0
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")


def main(seeds=(0, 1, 2), savefigs=True,
         adam_iters=4000, lbfgs_iters=1000, **opts):
    exact = ManufacturedBessel(m=2, n=1, R=R, lam=0.5, rhoc=RHOC, k=K, A=1.0, B=0.3)
    one = lambda X: torch.ones_like(X[:, :1])

    errs, last = [], None
    for s in seeds:
        op = LateralCauchyCylinder(R, L, Tmax, one, one, one, net_config={"seed": s})
        op.fit(torchify(exact.g, op.device), torchify(exact.f, op.device),
               adam_iters=adam_iters, lbfgs_iters=lbfgs_iters, **opts)
        rep = op.validate(exact, n=3000, seed=100)
        errs.append(rep["err_base"])
        last = (op, rep)
        print(f"  seed {s}:  e(z=0) = {rep['err_base']:.3e}   "
              f"(global {rep['err_global']:.3e})")

    errs = np.array(errs)
    print(f"\n[benchmark] e(z=0) modo (2,1), 1 modo: "
          f"{errs.mean():.3e} +/- {errs.std():.1e}  (n={len(seeds)} semillas)")

    if savefigs:
        op, rep = last
        os.makedirs(FIGDIR, exist_ok=True)
        p = lambda name: os.path.join(FIGDIR, name)

        op.plot_history(p("history"))
        pl.plot_error_vs_z(*rep["z"], path=p("error_vs_z"))
        pl.plot_error_vs_t(*rep["t"], path=p("error_vs_t"))

        # perfil dzT(z) en (x0,y0)=(0.5,0), t=0.5 (fuera del eje: phi_{2,1}(0)=0)
        z = np.linspace(0.0, L, 41)
        Xz = np.stack([np.full_like(z, 0.5), np.zeros_like(z), z,
                       np.full_like(z, 0.5)], axis=1)
        dz_pinn = op.grad_T(torch.as_tensor(Xz, device=op.device))[:, 2].cpu().numpy()
        dz_true = exact.grad_T(Xz)[:, 2]
        pl.plot_gradient_profile(z, dz_pinn, dz_true, path=p("perfil_dzT"),
                                 labels=("PINN", "exacta"))

        # corte 2D del campo T en el plano y=0, t=0.5
        T_pinn = lambda X: op.T(torch.as_tensor(X, device=op.device)).cpu().numpy()
        F, ext = pl.eval_slice(T_pinn, R, L, t0=0.5)
        G, _ = pl.eval_slice(exact.T, R, L, t0=0.5)
        pl.plot_slice(F, G, ext, path=p("corte_T"), labels=("PINN", "exacta"))

        rows = [[f"semilla {s}", e] for s, e in zip(seeds, errs)]
        rows.append([r"media $\pm$ std", f"{errs.mean():.3g} $\\pm$ {errs.std():.1g}"])
        pl.latex_table(rows, header=["caso", r"$e(z{=}0)$"], path=p("tabla_z0.tex"))
        print(f"[benchmark] figuras (pdf+png) y tabla_z0.tex en {FIGDIR}/")
    return errs


if __name__ == "__main__":
    main()
