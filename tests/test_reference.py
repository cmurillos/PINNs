"""El solver de referencia debe reproducir la solucion analitica de Bessel."""

import numpy as np

from numerics import ReferenceSolution, ManufacturedBessel
from metrics import sample_cylinder, rel_l2


def _build():
    R = L = Tmax = 1.0
    ex = ManufacturedBessel(m=2, n=1, R=R, lam=0.5, rhoc=1.0, k=1.0, A=1.0, B=0.3)
    Z = lambda z: ex.A * np.cosh(ex.gamma * z) + ex.B * np.sinh(ex.gamma * z)
    modes = [dict(m=2, n=1, kind="cos", amp=1.0, u0=Z,
                  bc0=lambda t: Z(0.0) * np.exp(-ex.lam * t),
                  bcL=lambda t: Z(L) * np.exp(-ex.lam * t))]
    ref = ReferenceSolution(R, L, Tmax, lambda z: np.ones_like(z),
                            lambda z: np.ones_like(z), modes)
    return ex, ref, R, L, Tmax


def test_reference_matches_analytic():
    ex, ref, R, L, Tmax = _build()
    X = sample_cylinder(R, L, Tmax, 3000, seed=1)
    assert rel_l2(ref.T(X), ex.T(X)) < 1e-3
    assert rel_l2(ref.grad_T(X), ex.grad_T(X)) < 1e-3
    XL = X.copy(); XL[:, 2] = L
    assert rel_l2(ref.g(XL), ex.g(XL)) < 1e-4
    assert rel_l2(ref.f(XL), ex.f(XL)) < 1e-3
