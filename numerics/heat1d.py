"""Solver 1D en (z, t) para un modo del disco (Crank-Nicolson, forma conservativa).

Al separar T(x,y,z,t) = phi(x,y) u(z,t) con -Lap(phi)=mu phi, la PDE
   rho c dT/dt = div(k grad T)   (con rho c, k dependientes solo de z)
se reduce a

   rho c(z) u_t = d/dz( k(z) u_z ) - mu k(z) u,   z in [0, L].

Se discretiza con diferencias finitas conservativas en z (k en las caras) y
Crank-Nicolson en t, con Dirichlet en ambos extremos. La forma conservativa
soporta k(z) heterogeneo (perfiles por capas) sin cambios.
"""

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.interpolate import RegularGridInterpolator


class Heat1D:
    def __init__(self, L, Tmax, rhoc, k, mu, nz=201, nt=400):
        self.z = np.linspace(0.0, L, nz)
        self.t = np.linspace(0.0, Tmax, nt + 1)
        dz = self.z[1] - self.z[0]
        zc = 0.5 * (self.z[:-1] + self.z[1:])     # caras (midpoints)
        kf = k(zc)                                 # k en las caras
        rc = rhoc(self.z)
        kn = k(self.z)

        # operador L u = d/dz(k u_z) - mu k u  (matriz tridiagonal, nodos interiores)
        A = np.zeros((nz, nz))
        for i in range(1, nz - 1):
            A[i, i - 1] = kf[i - 1] / dz ** 2
            A[i, i + 1] = kf[i] / dz ** 2
            A[i, i] = -(kf[i - 1] + kf[i]) / dz ** 2 - mu * kn[i]
        M = np.diag(rc)                            # masa rho c
        self.dt = self.t[1] - self.t[0]

        # Crank-Nicolson:  (M - dt/2 A) u^{n+1} = (M + dt/2 A) u^n
        Lhs = M - 0.5 * self.dt * A
        self.Rhs = M + 0.5 * self.dt * A
        Lhs[0, :] = 0.0; Lhs[0, 0] = 1.0          # filas Dirichlet -> identidad
        Lhs[-1, :] = 0.0; Lhs[-1, -1] = 1.0
        self._lu = lu_factor(Lhs)

    def solve(self, u0, bc0, bcL):
        """u0: array(nz) inicial; bc0, bcL: callables t->valor en z=0 y z=L."""
        nz, nt = len(self.z), len(self.t)
        U = np.zeros((nt, nz))
        U[0] = u0
        for n in range(nt - 1):
            rhs = self.Rhs @ U[n]
            rhs[0] = bc0(self.t[n + 1])
            rhs[-1] = bcL(self.t[n + 1])
            U[n + 1] = lu_solve(self._lu, rhs)
        self.U = U
        Uz = np.gradient(U, self.z, axis=1)        # u_z (2do orden en el interior)
        dz = self.z[1] - self.z[0]                 # borde: unilateral 3er orden
        Uz[:, 0] = (-11 * U[:, 0] + 18 * U[:, 1] - 9 * U[:, 2] + 2 * U[:, 3]) / (6 * dz)
        Uz[:, -1] = (11 * U[:, -1] - 18 * U[:, -2] + 9 * U[:, -3] - 2 * U[:, -4]) / (6 * dz)
        self._iU = RegularGridInterpolator((self.t, self.z), U)
        self._iUz = RegularGridInterpolator((self.t, self.z), Uz)
        return U

    def u_at(self, z, t):
        return self._iU(np.stack([t, z], axis=-1))

    def uz_at(self, z, t):
        return self._iUz(np.stack([t, z], axis=-1))
