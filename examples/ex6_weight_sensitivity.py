"""Ejemplo 6 - Sensibilidad al balance de pesos (lambda_g, lambda_f).

En un problema mal puesto el resultado depende del balance de la perdida. Los
pesos adaptativos estan aplazados (CLAUDE.md §8), pero conviene MEDIR la
sensibilidad: aqui se barre (lambda_g, lambda_f) y se reporta el error de grad_T
en la BASE z=0 (la recuperacion genuina) para cada combinacion. Es material de
paper: muestra cuanto mueve el resultado el balance del dato de Cauchy.
Ejecutar:  python -m examples.ex6_weight_sensitivity
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch

from lateralcauchy import LateralCauchyCylinder, ManufacturedBessel
from lateralcauchy.metrics import sample_disk_slice, rel_l2, torchify

R = L = Tmax = 1.0
RHOC = K = 1.0


def main(grid=(2.0, 10.0, 50.0), **opts):
    exact = ManufacturedBessel(m=2, n=1, R=R, lam=0.5, rhoc=RHOC, k=K, A=1.0, B=0.3)
    rho = lambda X: torch.ones_like(X[:, :1])
    c = lambda X: RHOC * torch.ones_like(X[:, :1])
    k = lambda X: K * torch.ones_like(X[:, :1])
    Xbase = sample_disk_slice(R, Tmax, 0.0, 3000, seed=6)

    print("[ex6] sensibilidad al balance de pesos (error de grad_T en z=0):")
    print("        lambda_g  lambda_f   err(z=0)")
    out = {}
    for lg in grid:
        for lf in grid:
            op = LateralCauchyCylinder(R, L, Tmax, rho, c, k)
            op.fit(torchify(exact.g, op.device), torchify(exact.f, op.device),
                   weights=(1.0, lg, lf, 1.0), **opts)
            pred = op.grad_T(torch.as_tensor(Xbase, device=op.device))
            err = rel_l2(pred, exact.grad_T(Xbase))
            out[(lg, lf)] = err
            print(f"        {lg:8.1f}  {lf:8.1f}   {err:.3e}")
    best = min(out, key=out.get)
    print(f"[ex6] mejor balance: lambda_g={best[0]}, lambda_f={best[1]}  ->  {out[best]:.3e}")
    return out


if __name__ == "__main__":
    main()
