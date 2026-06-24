"""Las soluciones exactas deben cumplir la PDE y la Neumann lateral (coef. const)."""

import numpy as np
import pytest

from numerics import ManufacturedZ, ManufacturedBessel

RHOC = K = 1.0


def _residual(sol, X, h=1e-4):
    """rho c T_t - k Lap(T) por diferencias finitas (k constante)."""
    def T(x, y, z, t):
        return sol.T(np.stack([x, y, z, t], axis=1))[:, 0]
    x, y, z, t = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    Tt = (T(x, y, z, t + h) - T(x, y, z, t - h)) / (2 * h)
    lap = (T(x + h, y, z, t) + T(x - h, y, z, t)
           + T(x, y + h, z, t) + T(x, y - h, z, t)
           + T(x, y, z + h, t) + T(x, y, z - h, t) - 6 * T(x, y, z, t)) / h ** 2
    return RHOC * Tt - K * lap


def _cloud(seed=0, n=300, R=0.6, L=1.0, Tmax=1.0):
    rng = np.random.default_rng(seed)
    r = R * np.sqrt(rng.random(n))
    th = 2 * np.pi * rng.random(n)
    z = 0.2 + 0.6 * rng.random(n)            # lejos de bordes (FD limpio)
    t = 0.2 + 0.6 * rng.random(n)
    return np.stack([r * np.cos(th), r * np.sin(th), z, t], axis=1)


@pytest.mark.parametrize("sol", [
    ManufacturedZ(lam=2.0, psi=0.7, rhoc=RHOC, k=K, L=1.0),
    ManufacturedBessel(m=2, n=1, R=1.0, lam=0.5, rhoc=RHOC, k=K, A=1.0, B=0.3),
])
def test_satisfies_pde(sol):
    X = _cloud()
    res = _residual(sol, X)
    scale = np.max(np.abs(sol.T(X)))
    assert np.max(np.abs(res)) / scale < 1e-2


def test_bessel_lateral_neumann():
    sol = ManufacturedBessel(m=2, n=1, R=1.0, lam=0.5, rhoc=RHOC, k=K)
    th = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    X = np.stack([np.cos(th), np.sin(th), 0.5 * np.ones_like(th),
                  0.3 * np.ones_like(th)], axis=1)
    g = sol.grad_T(X)                         # d_n T = (x/R)T_x + (y/R)T_y en r=R
    dn = np.cos(th) * g[:, 0] + np.sin(th) * g[:, 1]
    assert np.max(np.abs(dn)) < 1e-9
