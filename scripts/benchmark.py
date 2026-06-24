"""Benchmark reproducible: error de grad_T en la base z=0 sobre varias semillas.

Entrena la PINN para un modo de Bessel (verdad analitica) con distintas semillas
y reporta el error en z=0 como media +/- std (cifra defendible, no una sola
corrida). Guarda ademas las figuras del paper: historia de perdidas, error-vs-z
(firma del mal condicionamiento) y error-vs-t (control, ~plano).

    python scripts/benchmark.py                      # regimen moderado por defecto
    python scripts/benchmark.py  (editar opts)       # mas iteraciones

Las figuras se guardan en scripts/figs/ (no versionadas).
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import torch

from lateralcauchy import LateralCauchyCylinder, ManufacturedBessel, diagnostics
from lateralcauchy.metrics import (
    sample_cylinder, sample_disk_slice, rel_l2, error_vs_z, error_vs_t, torchify,
)

R = L = Tmax = 1.0
RHOC = K = 1.0
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")


def _medium():
    one = lambda X: torch.ones_like(X[:, :1])
    return one, one, one


def main(seeds=(0, 1, 2), savefigs=True,
         adam_iters=4000, lbfgs_iters=1000, **opts):
    exact = ManufacturedBessel(m=2, n=1, R=R, lam=0.5, rhoc=RHOC, k=K, A=1.0, B=0.3)
    rho, c, k = _medium()
    Xbase = sample_disk_slice(R, Tmax, 0.0, 3000, seed=100)

    errs, last = [], None
    for s in seeds:
        op = LateralCauchyCylinder(R, L, Tmax, rho, c, k, net_config={"seed": s})
        hist = op.fit(torchify(exact.g, op.device), torchify(exact.f, op.device),
                      adam_iters=adam_iters, lbfgs_iters=lbfgs_iters, **opts)
        err = rel_l2(op.grad_T(torch.as_tensor(Xbase, device=op.device)),
                     exact.grad_T(Xbase))
        errs.append(err)
        last = (op, hist)
        print(f"  seed {s}:  err(z=0) = {err:.3e}")

    errs = np.array(errs)
    print(f"\n[benchmark] grad_T en z=0 (modo 2,1, 1 modo): "
          f"{errs.mean():.3e} +/- {errs.std():.1e}  (n={len(seeds)} semillas)")

    if savefigs:
        op, hist = last
        os.makedirs(FIGDIR, exist_ok=True)
        X = sample_cylinder(R, L, Tmax, 3000, seed=7)
        pred = op.grad_T(torch.as_tensor(X, device=op.device))
        true = exact.grad_T(X)
        diagnostics.plot_history(hist, os.path.join(FIGDIR, "history.png"))
        zc, ez = error_vs_z(pred, true, X, nbins=8)
        diagnostics.plot_error_vs_z(zc, ez, os.path.join(FIGDIR, "error_vs_z.png"))
        tc, et = error_vs_t(pred, true, X, nbins=8)
        diagnostics.plot_error_vs_t(tc, et, os.path.join(FIGDIR, "error_vs_t.png"))
        print(f"[benchmark] figuras guardadas en {FIGDIR}/")
    return errs


if __name__ == "__main__":
    main()
