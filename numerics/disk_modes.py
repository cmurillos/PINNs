"""Modos propios del disco para la condicion lateral de Neumann.

La pared lateral pide  d(phi)/dr = 0  en r = R. Las funciones propias del
Laplaciano en el disco que la satisfacen son

    phi_{m,n}(r, theta) = J_m(kappa r) * trig(m theta),   kappa = j'_{m,n} / R

con  J_m'(kappa R) = 0  (de ahi los ceros de la derivada de Bessel). Cumplen
   -Laplaciano(phi) = mu phi,   mu = kappa^2.

Cada modo es producto separable: usar phi(x,y) por un perfil u(z,t) reduce la
PDE 3D a un problema 1D en (z,t) (ver heat1d.py). trig = cos para 'cos', sin
para 'sin'; m = 0 solo admite 'cos' (modo radial).
"""

import numpy as np
from scipy.special import jv, jvp, jnp_zeros


class DiskMode:
    """Funcion propia de Neumann del disco; evaluable en (x, y) con su gradiente."""

    def __init__(self, m, n, R, kind="cos"):
        if m == 0 and kind == "sin":
            raise ValueError("m=0 solo admite kind='cos' (modo radial).")
        self.m, self.R, self.kind = m, R, kind
        self.kappa = jnp_zeros(m, n)[n - 1] / R   # j'_{m,n} / R  -> Neumann en R
        self.mu = self.kappa ** 2                 # -Lap(phi) = mu phi

    def _trig(self, th):
        return np.cos(self.m * th) if self.kind == "cos" else np.sin(self.m * th)

    def _dtrig(self, th):  # d(trig)/d(theta)
        m = self.m
        return -m * np.sin(m * th) if self.kind == "cos" else m * np.cos(m * th)

    def value(self, x, y):
        r = np.hypot(x, y)
        th = np.arctan2(y, x)
        return jv(self.m, self.kappa * r) * self._trig(th)

    def grad(self, x, y):
        """(phi_x, phi_y) por regla de la cadena polar -> cartesiana."""
        r = np.hypot(x, y)
        rs = np.where(r == 0.0, 1.0, r)           # evita 0/0; terminos se anulan
        th = np.arctan2(y, x)
        k = self.kappa
        J = jv(self.m, k * r)
        dphi_dr = k * jvp(self.m, k * r) * self._trig(th)
        dphi_dth = J * self._dtrig(th)
        cos_t, sin_t = x / rs, y / rs
        phi_x = cos_t * dphi_dr - sin_t / rs * dphi_dth
        phi_y = sin_t * dphi_dr + cos_t / rs * dphi_dth
        return phi_x, phi_y
