"""ex2 - Frecuencia espacial: modo de Bessel axisimetrico m=0, n=1 (spec §2-ex2).

phi(r) = J0(kappa r), kappa = j'_{0,1}/R ~ 3.8317 (Neumann lateral exacta).
Excitacion: armonico temporal en la base del problema 1D reducido, con omega
elegida por BISECCION sobre diagnostics.atenuacion tal que a(omega) ~ e^{-1}.
Los datos (g, f) salen del solver modal (Crank-Nicolson 400x400), verificado
antes contra la solucion separable exacta del modo (k constante).

METRICA: E0 en la base z=0 (la PINN nunca la ve). Se espera E0 ~ 1e-1, perfil
e(z) CRECIENTE hacia la base (firma del mal condicionamiento) y e(t) ~ plano
(la continuacion es espacial). 5 corridas (SEEDS_RED).

Este runner define ademas la BASE COMUN de ex3-ex6 y la visualizacion 5.7:
omega_para_atenuacion (biseccion), construir_referencia (solucion periodica
modal via el BVP complejo + Crank-Nicolson) y senal_tapa (para regimen).

Ejecutar desde la raiz del repo:  python examples/ex2_bessel.py [--smoke]
"""

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import torch

from lateralcauchy import LateralCauchyCylinder, ReferenceSolution
from lateralcauchy.numerics import DiskMode
from lateralcauchy import plotting as pl
from lateralcauchy import runconfig as rc
from lateralcauchy.diagnostics import atenuacion, perfil_armonico, regimen
from lateralcauchy.metrics import (
    malla_e0, e0, perfil_error_z, perfil_error_t, sample_cylinder, rel_l2,
    torchify,
)

EX = "ex2"
R = L = TMAX = 1.0
M, N = 0, 1                          # modo axisimetrico J0(kappa r)
A_OBJETIVO = math.exp(-1.0)          # atenuacion objetivo ~ 0.37

MEDIO_NP = (lambda z: np.ones_like(z), lambda z: np.ones_like(z))


# ---------------------------------------------------- base comun (ex2-ex6, 5.7)
def omega_para_atenuacion(a_objetivo, medio, L, lo=1e-3, hi=1e4, tol=1e-4):
    """Biseccion (en log omega) sobre atenuacion(omega) hasta a(omega)=a_objetivo.

    a(omega) es decreciente; el bracket [lo, hi] debe encerrar el objetivo."""
    if not atenuacion(lo, medio, L) > a_objetivo > atenuacion(hi, medio, L):
        raise ValueError("a_objetivo fuera del bracket [lo, hi] de la biseccion.")
    while hi / lo > 1.0 + tol:
        mid = math.sqrt(lo * hi)
        if atenuacion(mid, medio, L) > a_objetivo:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def construir_referencia(omega, medio, R=R, L=L, Tmax=TMAX, m=M, n=N,
                         nz=400, nt=400):
    """Referencia modal periodica para la excitacion armonica en la base.

    El perfil complejo U(z) del BVP (con el termino -mu k U del modo) da la
    solucion periodica exacta u(z,t) = Re[U(z) e^{i omega t}] del problema 1D
    reducido; se usa para fijar u0 y las Dirichlet del solver Crank-Nicolson
    (malla nz x nt), de modo que el solver reproduce el regimen periodico desde
    t=0. Devuelve (ref, U, modo): ReferenceSolution con g, f, grad_T."""
    modo = DiskMode(m, n, R)
    zb, U = perfil_armonico(omega, medio, L, n_z=2000, mu=modo.mu)
    UL = U[-1]
    modes = [dict(
        m=m, n=n, kind="cos", amp=1.0,
        u0=lambda zq: np.interp(zq, zb, U.real),
        bc0=lambda t: np.cos(omega * t),
        bcL=lambda t: (UL * np.exp(1j * omega * t)).real,
    )]
    ref = ReferenceSolution(R, L, Tmax, medio[0], medio[1], modes, nz=nz, nt=nt)
    return ref, (zb, U), modo


def verificacion_cruzada(ref, omega, medio, analitico=True):
    """Error de discretizacion del solver modal sobre su malla (z,t).

    Con k constante compara contra la solucion separable exacta del modo
    u(z,t) = Re[u_hat(z) e^{i omega t}], u_hat = cosh(nu(L-z))/cosh(nu L),
    nu^2 = mu + i omega. Con medio heterogeneo compara contra el perfil del
    BVP complejo (discretizacion independiente). Devuelve el error rel L2."""
    modo, h = ref.parts[0][0], ref.parts[0][1]
    TT, ZZ = np.meshgrid(h.t, h.z, indexing="ij")
    if analitico:
        nu = np.sqrt(modo.mu + 1j * omega)
        u_hat = np.cosh(nu * (h.z[-1] - h.z)) / np.cosh(nu * h.z[-1])
    else:
        zb, U = perfil_armonico(omega, medio, h.z[-1], n_z=2000, mu=modo.mu)
        u_hat = np.interp(h.z, zb, U.real) + 1j * np.interp(h.z, zb, U.imag)
    u_exacta = (u_hat[None, :] * np.exp(1j * omega * TT)).real
    return float(np.linalg.norm(h.U - u_exacta) / np.linalg.norm(u_exacta))


def senal_tapa(ref, n=1001):
    """g en el punto (0,0,L,t): senal temporal para diagnostics.regimen."""
    t = np.linspace(0.0, ref.Tmax, n)
    X = np.stack([np.zeros_like(t), np.zeros_like(t),
                  np.full_like(t, ref.L), t], 1)
    return np.asarray(ref.g(X)).ravel(), t[1] - t[0]


def medio_torch_const():
    one = lambda X: torch.ones_like(X[:, :1])
    return one, one, one


# ------------------------------------------------------------------- runner
def main(smoke=False):
    omega = omega_para_atenuacion(A_OBJETIVO, MEDIO_NP, L)
    a = atenuacion(omega, MEDIO_NP, L)
    print(f"[{EX}] biseccion: omega={omega:.4f}  ->  a={a:.4f} (objetivo e^-1)")

    ref, _, _ = construir_referencia(omega, MEDIO_NP)
    err_solver = verificacion_cruzada(ref, omega, MEDIO_NP, analitico=True)
    print(f"[{EX}] verificacion cruzada solver<->analitica (k cte): "
          f"rel err = {err_solver:.2e}  (< 1e-4 esperado)")

    print(f"[{EX}] regimen (§4.4):", regimen(senal_tapa(ref), MEDIO_NP, L, eta=0.0))

    rho, c, k = medio_torch_const()
    seeds = rc.SEEDS_RED[:1] if smoke else rc.SEEDS_RED
    opts = rc.fit_opts(smoke)
    malla = malla_e0(R, TMAX)
    Xg = sample_cylinder(R, L, TMAX, 3000, seed=2)

    e0s, primero = [], None
    for s in seeds:
        op = LateralCauchyCylinder(R, L, TMAX, rho, c, k,
                                   net_config=rc.net_config(s))
        t0 = time.time()
        op.fit(torchify(ref.g, op.device), torchify(ref.f, op.device), **opts)
        segs = time.time() - t0
        E0 = e0(op, ref, malla)
        eg = rel_l2(op.grad_T(torch.as_tensor(Xg, device=op.device)),
                    ref.grad_T(Xg))
        rc.escribir_raw(EX, caso=f"bessel_m{M}n{N}", semilla_red=s, a=a,
                        omega=omega, L=L, E0=E0, err_global=eg,
                        tiempo_s=round(segs, 2))
        rc.guardar_modelo(op, EX, f"seed{s}")
        e0s.append(E0)
        if primero is None:
            primero = op
        print(f"[{EX}] seed {s}: E0={E0:.3e}  err_global={eg:.3e}  ({segs:.0f}s)")

    print(f"[{EX}] E0 media = {np.mean(e0s):.3e}  (esperado orden 1e-1)")

    # figuras: perfil e(z) (log y, debe CRECER hacia la base) y e(t) (~plano)
    zc, ez = perfil_error_z(primero, ref, R, L, TMAX)
    pl.plot_error_vs_z(zc, ez, path=os.path.join(rc.FIGS_DIR, "ex2_error_vs_z"))
    tt, et = perfil_error_t(primero, ref, malla)
    pl.plot_error_vs_t(tt, et, path=os.path.join(rc.FIGS_DIR, "ex2_error_vs_t"))
    print(f"[{EX}] figuras: figs/ex2_error_vs_z.pdf, figs/ex2_error_vs_t.pdf")
    return float(np.mean(e0s))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="presupuesto reducido (CI): 1 semilla, pocas iteraciones")
    main(smoke=ap.parse_args().smoke)
