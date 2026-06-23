"""Validacion con solucion manufacturada de coeficientes constantes (CLAUDE.md §6).

T*(x,y,z,t) = exp(-lam t) cos(beta z + psi),  beta = sqrt(rho c lam / k)

Satisface la PDE, la Neumann lateral exacta (no depende de x,y) y da datos de
Cauchy analiticos en z=L. Sirve de sanity check de la maquinaria (no estresa el
mal condicionamiento, ver advertencia en §6).
"""

import math

import torch

from lateral_cauchy_cylinder import LateralCauchyCylinder

RHOC, K = 1.0, 1.0
LAM, PSI = 2.0, 0.7
R = L = TMAX = 1.0
BETA = math.sqrt(RHOC * LAM / K)

rho = lambda X: torch.ones_like(X[:, :1])
c = lambda X: RHOC * torch.ones_like(X[:, :1])
k = lambda X: K * torch.ones_like(X[:, :1])

# datos de Cauchy en z = L
g = lambda X: torch.exp(-LAM * X[:, 3:4]) * math.cos(BETA * L + PSI)
f = lambda X: K * BETA * math.sin(BETA * L + PSI) * torch.exp(-LAM * X[:, 3:4])


def grad_true(X):
    """Gradiente espacial exacto: (0, 0, -beta sin(beta z + psi) exp(-lam t))."""
    z, t = X[:, 2:3], X[:, 3:4]
    gz = -BETA * torch.sin(BETA * z + PSI) * torch.exp(-LAM * t)
    return torch.cat([torch.zeros_like(gz), torch.zeros_like(gz), gz], 1)


def main(**opts):
    op = LateralCauchyCylinder(R, L, TMAX, rho, c, k)
    op.fit(g, f, **opts)

    # nube de puntos dentro del cilindro para medir error relativo de grad_T
    r = R * torch.sqrt(torch.rand(2000, 1))
    th = 2 * math.pi * torch.rand(2000, 1)
    X = torch.cat([r * torch.cos(th), r * torch.sin(th),
                   L * torch.rand(2000, 1), TMAX * torch.rand(2000, 1)], 1)

    pred, true = op.grad_T(X), grad_true(X)
    rel = (torch.linalg.norm(pred - true) / torch.linalg.norm(true)).item()
    print(f"error relativo ||grad_pred - grad_true|| / ||grad_true|| = {rel:.3e}")
    return rel


if __name__ == "__main__":
    main()
