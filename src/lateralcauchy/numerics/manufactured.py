"""Soluciones exactas (coeficientes constantes) para validar solver y PINN.

Dos familias, ambas con callables numpy T, grad_T, g, f sobre X (N,4):

- ManufacturedZ:
      T*(x,y,z,t) = e^{−λt} · cos(βz + ψ),        β = √(ρc·λ/k).
  No depende de (x,y) ⇒ NO estresa el mal condicionamiento (sanity check,
  CLAUDE.md §6).

- ManufacturedBessel:
      T*(x,y,z,t) = a · φ_{m,n}(x,y) · Z(z) · e^{−λt},
      Z(z) = A cosh(γz) + B sinh(γz),      γ² = μ − ρc·λ/k.
  SÍ tiene frecuencia espacial en (x,y): estresa la inversión mal puesta con
  amplificación ~ e^{Lγ} de la tapa a la base.

Ambas cumplen la PDE y la Neumann lateral de forma exacta.
"""

import numpy as np

from .disk_modes import DiskMode


class ManufacturedZ:
    def __init__(self, lam=2.0, psi=0.7, rhoc=1.0, k=1.0, L=1.0):
        self.lam, self.psi, self.k, self.L = lam, psi, k, L
        self.beta = np.sqrt(rhoc * lam / k)

    def _cols(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, 2], X[:, 3]                  # solo z, t

    def T(self, X):
        z, t = self._cols(X)
        return (np.exp(-self.lam * t) * np.cos(self.beta * z + self.psi)).reshape(-1, 1)

    def grad_T(self, X):
        z, t = self._cols(X)
        gz = -self.beta * np.sin(self.beta * z + self.psi) * np.exp(-self.lam * t)
        return np.stack([np.zeros_like(gz), np.zeros_like(gz), gz], axis=1)

    def g(self, X):
        _, t = self._cols(X)
        return (np.exp(-self.lam * t) * np.cos(self.beta * self.L + self.psi)).reshape(-1, 1)

    def f(self, X):
        _, t = self._cols(X)
        s = np.sin(self.beta * self.L + self.psi)
        return (self.k * self.beta * s * np.exp(-self.lam * t)).reshape(-1, 1)


class ManufacturedBessel:
    def __init__(self, m=2, n=1, R=1.0, lam=0.5, rhoc=1.0, k=1.0,
                 A=1.0, B=0.3, amp=1.0, kind="cos"):
        self.mode = DiskMode(m, n, R, kind)
        self.lam, self.k, self.amp = lam, k, amp
        self.A, self.B = A, B
        g2 = self.mode.mu - rhoc * lam / k
        if g2 <= 0:
            raise ValueError("gamma^2 <= 0: reduce lam o sube el modo (mu).")
        self.gamma = np.sqrt(g2)

    def _Z(self, z):
        return self.A * np.cosh(self.gamma * z) + self.B * np.sinh(self.gamma * z)

    def _dZ(self, z):
        return self.gamma * (self.A * np.sinh(self.gamma * z) + self.B * np.cosh(self.gamma * z))

    def _cols(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, 0], X[:, 1], X[:, 2], X[:, 3]

    def T(self, X):
        x, y, z, t = self._cols(X)
        phi = self.mode.value(x, y)
        return (self.amp * phi * self._Z(z) * np.exp(-self.lam * t)).reshape(-1, 1)

    def grad_T(self, X):
        x, y, z, t = self._cols(X)
        e = np.exp(-self.lam * t)
        phi = self.mode.value(x, y)
        phi_x, phi_y = self.mode.grad(x, y)
        Z, dZ = self._Z(z), self._dZ(z)
        return self.amp * np.stack([phi_x * Z * e, phi_y * Z * e, phi * dZ * e], axis=1)

    def g(self, X):
        x, y, z, t = self._cols(X)
        phi = self.mode.value(x, y)
        return (self.amp * phi * self._Z(z) * np.exp(-self.lam * t)).reshape(-1, 1)

    def f(self, X):
        x, y, z, t = self._cols(X)
        phi = self.mode.value(x, y)
        return (-self.k * self.amp * phi * self._dZ(z) * np.exp(-self.lam * t)).reshape(-1, 1)
