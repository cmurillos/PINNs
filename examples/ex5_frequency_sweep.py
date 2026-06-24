"""Ejemplo 5 - Mapa de degradacion vs frecuencia espacial (mal condicionamiento).

Barre modos de Bessel de frecuencia creciente. Cada modo crece en z como
exp(gamma z) con gamma = sqrt(mu - rho c lam / k); el factor de amplificacion de
la continuacion de la tapa a la base es ~ exp(L gamma). A mayor frecuencia,
mayor gamma, mayor amplificacion y peor recuperacion: el error de grad_T deberia
crecer con exp(L gamma). Ejecutar:  python -m examples.ex5_frequency_sweep
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from lateral_cauchy_cylinder import LateralCauchyCylinder
from numerics import ManufacturedBessel
from metrics import sample_cylinder, rel_l2, torchify

R = L = Tmax = 1.0
RHOC = K = 1.0


def main(modes=((1, 1), (2, 1), (3, 1), (4, 1)), **opts):
    rho = lambda X: torch.ones_like(X[:, :1])
    c = lambda X: RHOC * torch.ones_like(X[:, :1])
    k = lambda X: K * torch.ones_like(X[:, :1])
    Xc = sample_cylinder(R, L, Tmax, 3000, seed=4)

    print("[ex5] degradacion vs frecuencia espacial:")
    print("        modo    mu      gamma   exp(L*gamma)   grad_T rel err")
    out = {}
    for (m, n) in modes:
        exact = ManufacturedBessel(m=m, n=n, R=R, lam=0.5, rhoc=RHOC, k=K, A=1.0, B=0.3)
        op = LateralCauchyCylinder(R, L, Tmax, rho, c, k)
        op.fit(torchify(exact.g, op.device), torchify(exact.f, op.device), **opts)
        pred = op.grad_T(torch.as_tensor(Xc, device=op.device))
        err = rel_l2(pred, exact.grad_T(Xc))
        amp = np.exp(L * exact.gamma)
        out[(m, n)] = err
        print(f"        ({m},{n})  {exact.mode.mu:6.2f}  {exact.gamma:5.2f}   "
              f"{amp:10.1f}     {err:.3e}")
    return out


if __name__ == "__main__":
    main()
