"""Ejemplo 2 - Test exigente: PINN vs modo de Bessel (frecuencia espacial real).

T* = amp * phi_{m,n}(x,y) * (A cosh(gamma z) + B sinh(gamma z)) * exp(-lam t).
Ahora SI varia en (x,y), asi que estresa la inversion mal puesta (CLAUDE.md §6,
advertencia).

METRICA PRINCIPAL: el error de grad_T en la BASE z=0, que la PINN nunca ve y que
debe recuperar por continuacion. El error promediado en todo Omega esta dominado
por la zona facil (cerca de la tapa) y vende de mas; el de z=0 es la recuperacion
genuina. Se reportan ademas el perfil en z (crece hacia la base) y el perfil en t
(control: deberia ser ~plano, la continuacion es espacial, no temporal).

NOTA (sesgo optimista): el dato (g,f) es UN solo modo de Bessel -> banda limitada;
recuperar eso es mas facil que una g real de espectro ancho. La validacion
sintetica es optimista respecto a datos reales. Ejecutar: python -m examples.ex2_bessel
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import torch

from lateralcauchy import LateralCauchyCylinder, ManufacturedBessel, ReferenceSolution
from lateralcauchy.metrics import (
    sample_cylinder, sample_disk_slice, rel_l2, error_vs_z, error_vs_t, torchify,
)

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

    # METRICA PRINCIPAL: error en la base z=0 (recuperacion genuina)
    Xbase = sample_disk_slice(R, Tmax, 0.0, 3000, seed=20)
    base_err = rel_l2(op.grad_T(torch.as_tensor(Xbase, device=op.device)),
                      exact.grad_T(Xbase))
    print(f"[ex2] *** grad_T en la BASE z=0 (modo {m},{n}, 1 modo): "
          f"rel err = {base_err:.3e} ***")

    pred = op.grad_T(torch.as_tensor(Xc, device=op.device))
    print(f"[ex2] (referencia) error global en Omega = {rel_l2(pred, exact.grad_T(Xc)):.3e}")
    zc, ez = error_vs_z(pred, exact.grad_T(Xc), Xc, nbins=6)
    print("[ex2] error grad_T por z (distancia a la tapa; tapa z=L a la derecha):")
    for zz, ee in zip(zc, ez):
        print(f"        z={zz:.2f}  err={ee:.3e}")
    tc, et = error_vs_t(pred, exact.grad_T(Xc), Xc, nbins=6)
    print("[ex2] error grad_T por tiempo t (control, deberia ser ~plano):")
    for tt, ee in zip(tc, et):
        print(f"        t={tt:.2f}  err={ee:.3e}")
    return base_err


if __name__ == "__main__":
    main()
