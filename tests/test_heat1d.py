"""Test del solver 1D (z,t) contra solucion analitica de coeficientes constantes."""

import numpy as np

from lateralcauchy.numerics.heat1d import Heat1D


def test_vs_analytic():
    L = Tmax = 1.0
    mu, lam = 5.0, 2.0
    gamma = np.sqrt(mu - lam)                       # gamma^2 = mu - rhoc lam / k
    one = lambda z: np.ones_like(z)
    sol = Heat1D(L, Tmax, one, one, mu, nz=201, nt=400)
    Z = lambda z: np.cosh(gamma * z)
    U = sol.solve(Z(sol.z), lambda t: Z(0.0) * np.exp(-lam * t),
                  lambda t: Z(L) * np.exp(-lam * t))

    TT, ZZ = np.meshgrid(sol.t, sol.z, indexing="ij")
    U_exact = np.cosh(gamma * ZZ) * np.exp(-lam * TT)
    assert np.linalg.norm(U - U_exact) / np.linalg.norm(U_exact) < 1e-4

    # derivada en el borde z=L (esquema unilateral de 3er orden)
    tq = np.linspace(0, Tmax, 20)
    uzL = sol.uz_at(np.full_like(tq, L), tq)
    uzL_exact = gamma * np.sinh(gamma * L) * np.exp(-lam * tq)
    assert np.linalg.norm(uzL - uzL_exact) / np.linalg.norm(uzL_exact) < 1e-3
