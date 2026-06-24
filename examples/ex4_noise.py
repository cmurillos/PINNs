"""Ejemplo 4 - Robustez al ruido: cuanto se degrada grad_T si (g,f) tienen ruido.

En la practica (g,f) vienen de una regresion, no son exactos. Como el problema
es mal puesto, el ruido se amplifica al continuar hacia la base. Aqui se anade
una perturbacion suave y fija (reproducible) de amplitud eps * rms(senal) a los
datos de Cauchy de un modo de Bessel y se mide el error de grad_T para varios
eps. Ejecutar:  python -m examples.ex4_noise
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from lateral_cauchy_cylinder import LateralCauchyCylinder
from numerics import ManufacturedBessel
from metrics import sample_cylinder, rel_l2, torchify

R = L = Tmax = 1.0
RHOC = K = 1.0


def _perturbation(seed=0):
    """Perturbacion suave determinista delta(x,y,t) en [-1,1] (fija por punto)."""
    rng = np.random.default_rng(seed)
    a, b, c, ph = rng.uniform(-3, 3, 4)
    return lambda X: np.sin(a * X[:, 0:1] + b * X[:, 1:2] + c * X[:, 3:4] + ph)


def main(levels=(0.0, 0.02, 0.05), **opts):
    exact = ManufacturedBessel(m=2, n=1, R=R, lam=0.5, rhoc=RHOC, k=K, A=1.0, B=0.3)
    Xtop = sample_cylinder(R, L, Tmax, 4000, seed=9); Xtop[:, 2] = L
    rms = np.sqrt(np.mean(exact.g(Xtop) ** 2))     # escala de la senal
    delta = _perturbation()

    rho = lambda X: torch.ones_like(X[:, :1])
    c = lambda X: RHOC * torch.ones_like(X[:, :1])
    k = lambda X: K * torch.ones_like(X[:, :1])
    Xc = sample_cylinder(R, L, Tmax, 3000, seed=3)

    print("[ex4] robustez al ruido (modo 2,1):")
    out = {}
    for eps in levels:
        g_noisy = lambda X, e=eps: exact.g(X) + e * rms * delta(X)
        f_noisy = lambda X, e=eps: exact.f(X) + e * rms * delta(X)
        op = LateralCauchyCylinder(R, L, Tmax, rho, c, k)
        op.fit(torchify(g_noisy, op.device), torchify(f_noisy, op.device), **opts)
        pred = op.grad_T(torch.as_tensor(Xc, device=op.device))
        err = rel_l2(pred, exact.grad_T(Xc))
        out[eps] = err
        print(f"        ruido eps={eps:.0%}  ->  grad_T rel err = {err:.3e}")
    return out


if __name__ == "__main__":
    main()
