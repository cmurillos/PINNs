"""Solver de referencia 3D+t por superposición de modos del disco.

Resuelve el problema DIRECTO (bien puesto) y extrae de él la traza de Cauchy que
alimenta al operador Λ de la PINN:

    T(x,y,z,t) = Σⱼ aⱼ · φⱼ(x,y) · uⱼ(z,t),
    (g, f)  =  ( T|_{z=L} ,  −k ∂_z T|_{z=L} ),

donde φⱼ es un modo de Neumann del disco (−Δφ = μφ, ∂_r φ|_{r=R} = 0;
disk_modes) y uⱼ resuelve el problema 1D reducido en (z,t) (heat1d). Por
linealidad, toda superposición es solución. Válido para ρc, k dependientes solo
de z (medios por capas).

Expone callables numpy con el contrato de la PINN: X (N,4) = [x,y,z,t] ↦ (N,1)
o (N,3).
"""

import numpy as np

from .disk_modes import DiskMode
from .heat1d import Heat1D


class ReferenceSolution:
    def __init__(self, R, L, Tmax, rhoc, k, modes, nz=301, nt=600):
        # rhoc, k : callables z(array) -> array.  modes : lista de dicts con
        # claves m, n, kind, amp, u0(z), bc0(t), bcL(t).
        self.R, self.L, self.Tmax = R, L, Tmax
        self.kL = float(k(np.array([L]))[0])       # k en la tapa (para f)
        self.parts = []
        for spec in modes:
            md = DiskMode(spec["m"], spec["n"], R, spec.get("kind", "cos"))
            h = Heat1D(L, Tmax, rhoc, k, md.mu, nz=nz, nt=nt)
            h.solve(spec["u0"](h.z), spec["bc0"], spec["bcL"])
            self.parts.append((md, h, spec.get("amp", 1.0)))

    @staticmethod
    def _cols(X):
        X = np.asarray(X, dtype=float)
        return X[:, 0], X[:, 1], X[:, 2], X[:, 3]

    def T(self, X):
        """X ↦ T(X) = Σⱼ aⱼ φⱼ(x,y) uⱼ(z,t)."""
        x, y, z, t = self._cols(X)
        out = sum(a * md.value(x, y) * h.u_at(z, t) for md, h, a in self.parts)
        return out.reshape(-1, 1)

    def grad_T(self, X):
        """X ↦ ∇T(X) = Σⱼ aⱼ (∂ₓφⱼ·uⱼ, ∂ᵧφⱼ·uⱼ, φⱼ·∂_z uⱼ)."""
        x, y, z, t = self._cols(X)
        gx = gy = gz = 0.0
        for md, h, a in self.parts:
            phi = md.value(x, y)
            phi_x, phi_y = md.grad(x, y)
            u, uz = h.u_at(z, t), h.uz_at(z, t)
            gx = gx + a * phi_x * u
            gy = gy + a * phi_y * u
            gz = gz + a * phi * uz
        return np.stack([gx, gy, gz], axis=1)

    def g(self, X):
        """Dato de Cauchy 1:  g = T|_{z=L}."""
        x, y, _, t = self._cols(X)
        zL = np.full_like(t, self.L)
        out = sum(a * md.value(x, y) * h.u_at(zL, t) for md, h, a in self.parts)
        return out.reshape(-1, 1)

    def f(self, X):
        """Dato de Cauchy 2:  f = −k ∂_z T|_{z=L}."""
        x, y, _, t = self._cols(X)
        zL = np.full_like(t, self.L)
        out = sum(a * md.value(x, y) * h.uz_at(zL, t) for md, h, a in self.parts)
        return (-self.kL * out).reshape(-1, 1)
