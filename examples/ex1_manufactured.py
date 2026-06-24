"""Ejemplo 1 - Sanity check: PINN vs solucion exacta sin frecuencia espacial.

T* = exp(-lam t) cos(beta z + psi). No varia en (x,y), asi que NO estresa el
mal condicionamiento (CLAUDE.md §6): solo confirma que la maquinaria recupera
grad_T correctamente. Ejecutar:  python -m examples.ex1_manufactured
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch

from lateralcauchy import LateralCauchyCylinder, ManufacturedZ
from lateralcauchy.metrics import sample_cylinder, rel_l2, torchify

R = L = Tmax = 1.0
RHOC, K = 1.0, 1.0


def main(**opts):
    rho = lambda X: torch.ones_like(X[:, :1])
    c = lambda X: RHOC * torch.ones_like(X[:, :1])
    k = lambda X: K * torch.ones_like(X[:, :1])

    exact = ManufacturedZ(lam=2.0, psi=0.7, rhoc=RHOC, k=K, L=L)
    op = LateralCauchyCylinder(R, L, Tmax, rho, c, k)
    op.fit(torchify(exact.g, op.device), torchify(exact.f, op.device), **opts)

    X = sample_cylinder(R, L, Tmax, 2000, seed=1)
    pred = op.grad_T(torch.as_tensor(X, device=op.device))
    err = rel_l2(pred, exact.grad_T(X))
    print(f"[ex1] PINN vs exacta (sin frec. espacial): grad_T rel err = {err:.3e}")
    return err


if __name__ == "__main__":
    main()
