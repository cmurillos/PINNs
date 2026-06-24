"""Ejemplo 3 - Medio heterogeneo k(z): PINN vs solver numerico (sin formula exacta).

Aqui k depende de z (perfil suave por capas), de modo que NO hay solucion
analitica: el solver numerico de referencia es la unica verdad disponible. Se
genera una solucion directa, se extraen sus datos de Cauchy (g, f) en z=L, se
entrena la PINN con ellos y se compara grad_T contra el solver en todo el
cilindro. Ejecutar:  python -m examples.ex3_solver_heterogeneous
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import torch

from lateralcauchy import LateralCauchyCylinder, ReferenceSolution
from lateralcauchy.metrics import (
    sample_cylinder, sample_disk_slice, rel_l2, error_vs_z, error_vs_t, torchify,
)

R = L = Tmax = 1.0

# medio: rho c = 1, k(z) suave (C1, polinomio) que varia ~2x con la profundidad
rhoc_np = lambda z: np.ones_like(z)
k_np = lambda z: 1.0 + 0.8 * z ** 2           # k(0)=1, k(L)=1.8


def main(**opts):
    # referencia directa: superposicion de un par de modos del disco
    Z0 = lambda z: np.cos(1.5 * z) + 0.2
    Z1 = lambda z: 0.5 * (1.0 - z)
    modes = [
        dict(m=1, n=1, kind="cos", amp=1.0, u0=Z0,
             bc0=lambda t: (Z0(0.0) * np.exp(-0.4 * t)),
             bcL=lambda t: (Z0(L) * np.exp(-0.4 * t))),
        dict(m=2, n=1, kind="sin", amp=0.6, u0=Z1,
             bc0=lambda t: (Z1(0.0) * np.exp(-0.3 * t)),
             bcL=lambda t: (Z1(L) * np.exp(-0.3 * t))),
    ]
    ref = ReferenceSolution(R, L, Tmax, rhoc_np, k_np, modes, nz=301, nt=600)

    # PINN con el MISMO medio (k(z) como callable torch sobre X)
    rho = lambda X: torch.ones_like(X[:, :1])
    c = lambda X: torch.ones_like(X[:, :1])
    k = lambda X: 1.0 + 0.8 * X[:, 2:3] ** 2
    op = LateralCauchyCylinder(R, L, Tmax, rho, c, k)
    op.fit(torchify(ref.g, op.device), torchify(ref.f, op.device), **opts)

    # METRICA PRINCIPAL: error en la base z=0 (lo que la PINN nunca ve)
    Xbase = sample_disk_slice(R, Tmax, 0.0, 3000, seed=30)
    base_err = rel_l2(op.grad_T(torch.as_tensor(Xbase, device=op.device)),
                      ref.grad_T(Xbase))
    print(f"[ex3] *** grad_T en la BASE z=0 (k(z) heterogeneo, {len(modes)} modos): "
          f"rel err = {base_err:.3e} ***")

    X = sample_cylinder(R, L, Tmax, 3000, seed=3)
    pred = op.grad_T(torch.as_tensor(X, device=op.device))
    print(f"[ex3] (referencia) error global en Omega = {rel_l2(pred, ref.grad_T(X)):.3e}")
    zc, ez = error_vs_z(pred, ref.grad_T(X), X, nbins=6)
    print("[ex3] error grad_T por profundidad z:")
    for zz, ee in zip(zc, ez):
        print(f"        z={zz:.2f}  err={ee:.3e}")
    tc, et = error_vs_t(pred, ref.grad_T(X), X, nbins=6)
    print("[ex3] error grad_T por tiempo t (control, deberia ser ~plano):")
    for tt, ee in zip(tc, et):
        print(f"        t={tt:.2f}  err={ee:.3e}")
    return base_err


if __name__ == "__main__":
    main()
