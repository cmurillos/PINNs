"""Ejemplo 2 - Test exigente: PINN vs modo de Bessel (frecuencia espacial real).

T* = amp * phi_{m,n}(x,y) * (A cosh(gamma z) + B sinh(gamma z)) * exp(-lam t).
Ahora SI varia en (x,y), asi que estresa la inversion mal puesta (CLAUDE.md §6,
advertencia). Se reporta el error global de grad_T y su perfil en z: deberia
crecer al alejarse de la tapa z=L (la base z=0 es lo mas dificil de recuperar).

Tambien comprueba que el solver numerico reproduce la solucion analitica.
Ejecutar:  python -m examples.ex2_bessel
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from lateral_cauchy_cylinder import LateralCauchyCylinder
from numerics import ManufacturedBessel, ReferenceSolution
from metrics import sample_cylinder, rel_l2, error_vs_z, torchify

R = L = Tmax = 1.0
RHOC, K = 1.0, 1.0


def main(m=2, n=1, **opts):
    exact = ManufacturedBessel(m=m, n=n, R=R, lam=0.5, rhoc=RHOC, k=K,
                               A=1.0, B=0.3, amp=1.0)

    # (a) el solver numerico reproduce la analitica (control de calidad del solver)
    Z = lambda z: exact.A * np.cosh(exact.gamma * z) + exact.B * np.sinh(exact.gamma * z)
    modes = [dict(m=m, n=n, kind="cos", amp=1.0, u0=Z,
                  bc0=lambda t: Z(0.0) * np.exp(-exact.lam * t),
                  bcL=lambda t: Z(L) * np.exp(-exact.lam * t))]
    ref = ReferenceSolution(R, L, Tmax, lambda z: RHOC * np.ones_like(z),
                            lambda z: K * np.ones_like(z), modes)
    Xc = sample_cylinder(R, L, Tmax, 3000, seed=2)
    print(f"[ex2] solver numerico vs analitico: grad_T rel err = {rel_l2(ref.grad_T(Xc), exact.grad_T(Xc)):.2e}")

    # (b) PINN sobre los datos de Cauchy exactos
    rho = lambda X: torch.ones_like(X[:, :1])
    c = lambda X: RHOC * torch.ones_like(X[:, :1])
    k = lambda X: K * torch.ones_like(X[:, :1])
    op = LateralCauchyCylinder(R, L, Tmax, rho, c, k)
    op.fit(torchify(exact.g, op.device), torchify(exact.f, op.device), **opts)

    pred = op.grad_T(torch.as_tensor(Xc, device=op.device))
    err = rel_l2(pred, exact.grad_T(Xc))
    print(f"[ex2] PINN vs exacta (modo {m},{n}): grad_T rel err = {err:.3e}")
    zc, ez = error_vs_z(pred, exact.grad_T(Xc), Xc, nbins=6)
    print("[ex2] error grad_T por profundidad z (tapa z=L a la derecha):")
    for zz, ee in zip(zc, ez):
        print(f"        z={zz:.2f}  err={ee:.3e}")
    return err


if __name__ == "__main__":
    main()
