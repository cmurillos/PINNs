"""Experimentos del artículo (ex1–ex6, visualización 5.7) como funciones.

Cada función realiza el protocolo de su sección en docs/SIMULACIONES.md — los
DEFAULTS son exactamente el spec — e imprime el diagnóstico `regimen()` al
inicio. La lógica experimental vive aquí y NO en la clase PINN: la clase es el
operador Λ; el experimento es protocolo. Los `examples/exN_*.py` y `scripts/`
son envoltorios CLI de ~5 líneas sobre este módulo.

Convenciones comunes:
  - `smoke=True`: presupuesto reducido para CI (1 semilla por caso, pocas
    iteraciones); misma estructura y mismas salidas.
  - `opts=None`: overrides del presupuesto de `runconfig.fit_opts` (dict), solo
    para tests/presupuestos custom; None = presupuesto del spec.
  - `outdir=None`: escribe en `results/` y `figs/` del contrato §0.4; con otro
    valor, TODO (csv, modelos, figuras) va bajo `{outdir}/results` y
    `{outdir}/figs` (p. ej. un directorio montado en Drive desde Colab).
  - Retorno: dict con rutas escritas (`raw`, `figs`, `models`) y `resumen`
    (E0 media ± std por caso), para consumo directo desde un notebook.

Nota sobre `a_objetivo`: el spec fija la atenuación objetivo `a ≈ e^{-1} ≈
0.37`; el default es `math.exp(-1)` (el 0.37 es su redondeo).
"""

import csv
import math
import os
import time

import numpy as np
import torch

from .pinn import LateralCauchyCylinder
from .numerics import DiskMode, ReferenceSolution
from . import plotting as pl
from .diagnostics import atenuacion, perfil_armonico, regimen
from .metrics import (
    malla_e0, e0, perfil_error_z, perfil_error_t, sample_cylinder,
    sample_disk_slice, rel_l2, torchify,
)
from .runconfig import (
    COLUMNAS_RAW, SEEDS_RED, SEEDS_RUIDO, fit_opts, net_config,
)

import matplotlib.pyplot as plt

R = TMAX = 1.0
L = 1.0
A_OBJETIVO = math.exp(-1.0)          # spec: a ~ e^-1 ~ 0.37

MEDIO_CONST = (lambda z: np.ones_like(z), lambda z: np.ones_like(z))
K_PERFIL_EX3 = lambda z: 1.0 + z     # spec ex3 (valido para numpy y torch)


# ================================================================ salidas
def _rutas(outdir):
    """(results_dir, figs_dir) del contrato §0.4, opcionalmente bajo outdir."""
    if outdir:
        return os.path.join(outdir, "results"), os.path.join(outdir, "figs")
    return "results", "figs"


def _escribir_raw(rdir, experimento, **campos):
    """Fila en {rdir}/{experimento}/raw.csv, esquema fijo COLUMNAS_RAW (§0.4).
    (Mismo formato que runconfig.escribir_raw, con raíz parametrizable.)"""
    desconocidos = set(campos) - set(COLUMNAS_RAW)
    if desconocidos:
        raise ValueError(f"columnas fuera del contrato §0.4: {sorted(desconocidos)}")
    d = os.path.join(rdir, experimento)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "raw.csv")
    nuevo = not os.path.exists(path)
    fila = {"experimento": experimento, **campos}
    with open(path, "a", newline="", encoding="utf8") as fh:
        wr = csv.DictWriter(fh, fieldnames=COLUMNAS_RAW)
        if nuevo:
            wr.writeheader()
        wr.writerow({c: ("" if fila.get(c) is None else fila.get(c, ""))
                     for c in COLUMNAS_RAW})
    return path


def _leer_raw(rdir, experimento):
    with open(os.path.join(rdir, experimento, "raw.csv"),
              newline="", encoding="utf8") as fh:
        return list(csv.DictReader(fh))


def _guardar_modelo(rdir, op, experimento, nombre):
    d = os.path.join(rdir, experimento, "models")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{nombre}.pt")
    op.save(path)
    return path


def _resumen(por_caso):
    """{caso: [E0...]} -> {caso: {E0_media, E0_std, n}}."""
    return {caso: {"E0_media": float(np.mean(v)), "E0_std": float(np.std(v)),
                   "n": len(v)}
            for caso, v in por_caso.items()}


def _opts(smoke, opts):
    o = fit_opts(smoke)
    if opts:
        o.update(opts)
    return o


# ==================================================== base común (ex2–ex6, 5.7)
def omega_para_atenuacion(a_objetivo, medio, L, lo=1e-3, hi=1e4, tol=1e-4):
    """Bisección (en log omega) sobre atenuacion(omega) hasta a(omega)=a_objetivo.

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


def construir_referencia(omega, medio, R=R, L=L, Tmax=TMAX, m=0, n=1,
                         nz=400, nt=400):
    """Referencia modal periódica para la excitación armónica en la base.

    El perfil complejo U(z) del BVP (con el término -mu k U del modo) da la
    solución periódica exacta u(z,t) = Re[U(z) e^{i omega t}] del problema 1D
    reducido; se usa para fijar u0 y las Dirichlet del solver Crank–Nicolson
    (malla nz x nt), de modo que el solver reproduce el régimen periódico desde
    t=0. Devuelve (ref, (z, U), modo): ReferenceSolution con g, f, grad_T."""
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
    """Error de discretización del solver modal sobre su malla (z,t).

    Con k constante compara contra la solución separable exacta del modo
    u(z,t) = Re[u_hat(z) e^{i omega t}], u_hat = cosh(nu(L-z))/cosh(nu L),
    nu^2 = mu + i omega. Con medio heterogéneo compara contra el perfil del
    BVP complejo (discretización independiente). Devuelve el error rel L2."""
    modo, h = ref.parts[0][0], ref.parts[0][1]
    TT = np.meshgrid(h.t, h.z, indexing="ij")[0]
    if analitico:
        nu = np.sqrt(modo.mu + 1j * omega)
        u_hat = np.cosh(nu * (h.z[-1] - h.z)) / np.cosh(nu * h.z[-1])
    else:
        zb, U = perfil_armonico(omega, medio, h.z[-1], n_z=2000, mu=modo.mu)
        u_hat = np.interp(h.z, zb, U.real) + 1j * np.interp(h.z, zb, U.imag)
    u_exacta = (u_hat[None, :] * np.exp(1j * omega * TT)).real
    return float(np.linalg.norm(h.U - u_exacta) / np.linalg.norm(u_exacta))


def senal_tapa(ref, n=1001):
    """g en el punto (0,0,L,t): señal temporal para diagnostics.regimen."""
    t = np.linspace(0.0, ref.Tmax, n)
    X = np.stack([np.zeros_like(t), np.zeros_like(t),
                  np.full_like(t, ref.L), t], 1)
    return np.asarray(ref.g(X)).ravel(), t[1] - t[0]


def con_ruido(fn, eta, escala, rng):
    """fn + eta*escala*N(0,1) iid por punto de muestreo (gaussiano aditivo)."""
    def ruidosa(X):
        v = np.asarray(fn(X))
        return v + eta * escala * rng.standard_normal(v.shape)
    return ruidosa


def _medio_torch_const():
    one = lambda X: torch.ones_like(X[:, :1])
    return one, one, one


def _medio_torch_k(k_perfil):
    one = lambda X: torch.ones_like(X[:, :1])
    return one, one, lambda X: k_perfil(X[:, 2:3])


# ========================================================================= ex1
def ex1(seeds=SEEDS_RED, smoke=False, opts=None, outdir=None):
    """ex1 — Control de maquinaria: solución manufacturada (spec §2-ex1).

    T*(z,t) = exp(-t)·cos(z + pi/4); sin frecuencia espacial. Criterio de
    éxito: E0 ≲ 1e-2 en media (si falla es bug de maquinaria, no del método)."""
    from .numerics import ManufacturedZ
    EXN = "ex1"
    rdir, fdir = _rutas(outdir)
    exact = ManufacturedZ(lam=1.0, psi=math.pi / 4, rhoc=1.0, k=1.0, L=L)

    t = np.linspace(0.0, TMAX, 1001)
    Xt = np.stack([np.zeros_like(t), np.zeros_like(t), np.full_like(t, L), t], 1)
    print(f"[{EXN}] regimen (§4.4):",
          regimen((np.asarray(exact.g(Xt)).ravel(), t[1] - t[0]),
                  MEDIO_CONST, L, eta=0.0))

    one = lambda X: torch.ones_like(X[:, :1])
    seeds = list(seeds)[:1] if smoke else list(seeds)
    o = _opts(smoke, opts)
    malla = malla_e0(R, TMAX)
    Xg = sample_cylinder(R, L, TMAX, 3000, seed=1)

    e0s, modelos, primero = [], [], None
    raw = None
    for s in seeds:
        op = LateralCauchyCylinder(R, L, TMAX, one, one, one,
                                   net_config=net_config(s))
        t0 = time.time()
        op.fit(torchify(exact.g, op.device), torchify(exact.f, op.device), **o)
        segs = time.time() - t0
        E0 = e0(op, exact, malla)
        eg = rel_l2(op.grad_T(torch.as_tensor(Xg, device=op.device)),
                    exact.grad_T(Xg))
        raw = _escribir_raw(rdir, EXN, caso="manufacturada", semilla_red=s, L=L,
                            E0=E0, err_global=eg, tiempo_s=round(segs, 2))
        modelos.append(_guardar_modelo(rdir, op, EXN, f"seed{s}"))
        e0s.append(E0)
        if primero is None:
            primero = op
        print(f"[{EXN}] seed {s}: E0={E0:.3e}  err_global={eg:.3e}  ({segs:.0f}s)")

    media = float(np.mean(e0s))
    ok = "OK" if media <= 1e-2 else "FALLA (bug de maquinaria, no del metodo)"
    print(f"[{EXN}] E0 media = {media:.3e}  [criterio E0 <~ 1e-2: {ok}]")

    zc, ez = perfil_error_z(primero, exact, R, L, TMAX)
    print(f"[{EXN}] perfil e(z): max={ez.max():.3e} en z={zc[np.argmax(ez)]:.2f}")
    tt, et = perfil_error_t(primero, exact, malla)
    print(f"[{EXN}] perfil e(t): max={et.max():.3e} en t={tt[np.argmax(et)]:.2f}")

    figs = _figs_ex1(fdir, primero, exact)
    print(f"[{EXN}] figuras: {', '.join(figs)}")
    return {"raw": raw, "figs": figs, "models": modelos,
            "resumen": _resumen({"manufacturada": e0s})}


def _figs_ex1(fdir, op, exact):
    # figs/ex1_corte.pdf: T_theta vs T* sobre el eje x=y=0, t en {0.25,0.5,0.9}
    z = np.linspace(0.0, L, 201)
    with pl.paper_style():
        fig, ax = plt.subplots(figsize=(pl.SINGLE_COL, 2.55))
        for t0, color in zip((0.25, 0.5, 0.9), pl.CAT):
            X = np.stack([np.zeros_like(z), np.zeros_like(z), z,
                          np.full_like(z, t0)], 1)
            ax.plot(z, exact.T(X).ravel(), "-", color=color, alpha=0.55)
            Tp = op.T(torch.as_tensor(X, device=op.device)).cpu().numpy().ravel()
            ax.plot(z, Tp, "--", color=color, label=fr"$t={t0}$")
        ax.set_xlabel(r"$z$")
        ax.set_ylabel(r"$T$ en $x{=}y{=}0$")
        ax.legend(title=r"PINN (- -) vs $T^*$ (—)")
        fig.tight_layout()
        pl.save_fig(fig, os.path.join(fdir, "ex1_corte"))

    # figs/ex1_traza_basal.pdf: dz T_theta|_{z=0} vs exacto, en t
    t = np.linspace(TMAX / 200, TMAX, 200)
    X = np.stack([np.zeros_like(t), np.zeros_like(t), np.zeros_like(t), t], 1)
    dz_pinn = op.grad_T(torch.as_tensor(X, device=op.device))[:, 2].cpu().numpy()
    dz_true = exact.grad_T(X)[:, 2]
    with pl.paper_style():
        fig, ax = plt.subplots(figsize=(pl.SINGLE_COL, 2.55))
        ax.plot(t, dz_true, "-", color=pl.BLUE, label="exacta")
        ax.plot(t, dz_pinn, "o--", color=pl.VERMILLION, markerfacecolor="white",
                markevery=10, label="PINN")
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\partial_z T|_{z=0}$")
        ax.legend()
        fig.tight_layout()
        pl.save_fig(fig, os.path.join(fdir, "ex1_traza_basal"))
    return [os.path.join(fdir, "ex1_corte.pdf"),
            os.path.join(fdir, "ex1_traza_basal.pdf")]


# ========================================================================= ex2
def ex2(a_objetivo=A_OBJETIVO, modo=(0, 1), seeds=SEEDS_RED, smoke=False,
        opts=None, outdir=None):
    """ex2 — Frecuencia espacial: modo de Bessel m=0, n=1 (spec §2-ex2).

    omega por bisección hasta a(omega)=a_objetivo (~e^-1); datos del solver
    modal 400x400 con verificación cruzada contra la separable exacta."""
    EXN = "ex2"
    m, n = modo
    rdir, fdir = _rutas(outdir)

    omega = omega_para_atenuacion(a_objetivo, MEDIO_CONST, L)
    a = atenuacion(omega, MEDIO_CONST, L)
    print(f"[{EXN}] biseccion: omega={omega:.4f}  ->  a={a:.4f} "
          f"(objetivo {a_objetivo:g})")

    ref, _, _ = construir_referencia(omega, MEDIO_CONST, m=m, n=n)
    err_solver = verificacion_cruzada(ref, omega, MEDIO_CONST, analitico=True)
    print(f"[{EXN}] verificacion cruzada solver<->analitica (k cte): "
          f"rel err = {err_solver:.2e}  (< 1e-4 esperado)")
    print(f"[{EXN}] regimen (§4.4):",
          regimen(senal_tapa(ref), MEDIO_CONST, L, eta=0.0))

    rho, c, k = _medio_torch_const()
    seeds = list(seeds)[:1] if smoke else list(seeds)
    o = _opts(smoke, opts)
    malla = malla_e0(R, TMAX)
    Xg = sample_cylinder(R, L, TMAX, 3000, seed=2)

    caso = f"bessel_m{m}n{n}"
    e0s, modelos, primero, raw = [], [], None, None
    for s in seeds:
        op = LateralCauchyCylinder(R, L, TMAX, rho, c, k,
                                   net_config=net_config(s))
        t0 = time.time()
        op.fit(torchify(ref.g, op.device), torchify(ref.f, op.device), **o)
        segs = time.time() - t0
        E0 = e0(op, ref, malla)
        eg = rel_l2(op.grad_T(torch.as_tensor(Xg, device=op.device)),
                    ref.grad_T(Xg))
        raw = _escribir_raw(rdir, EXN, caso=caso, semilla_red=s, a=a,
                            omega=omega, L=L, E0=E0, err_global=eg,
                            tiempo_s=round(segs, 2))
        modelos.append(_guardar_modelo(rdir, op, EXN, f"seed{s}"))
        e0s.append(E0)
        if primero is None:
            primero = op
        print(f"[{EXN}] seed {s}: E0={E0:.3e}  err_global={eg:.3e}  ({segs:.0f}s)")

    print(f"[{EXN}] E0 media = {np.mean(e0s):.3e}  (esperado orden 1e-1)")

    zc, ez = perfil_error_z(primero, ref, R, L, TMAX)
    pl.plot_error_vs_z(zc, ez, path=os.path.join(fdir, "ex2_error_vs_z"))
    tt, et = perfil_error_t(primero, ref, malla)
    pl.plot_error_vs_t(tt, et, path=os.path.join(fdir, "ex2_error_vs_t"))
    figs = [os.path.join(fdir, "ex2_error_vs_z.pdf"),
            os.path.join(fdir, "ex2_error_vs_t.pdf")]
    print(f"[{EXN}] figuras: {', '.join(figs)}")
    return {"raw": raw, "figs": figs, "models": modelos,
            "resumen": _resumen({caso: e0s}),
            "omega": omega, "a": a, "err_solver": err_solver}


# ========================================================================= ex3
def ex3(k_perfil=None, a_objetivo=A_OBJETIVO, seeds=SEEDS_RED, smoke=False,
        opts=None, outdir=None):
    """ex3 — Medio heterogéneo k(z) (spec §2-ex3). k_perfil=None => k(z)=1+z.

    k_perfil debe ser un callable polinómico z ↦ k(z) válido para numpy y
    torch (se usa para el solver de referencia y para la PINN)."""
    EXN = "ex3"
    k_perfil = k_perfil or K_PERFIL_EX3
    rdir, fdir = _rutas(outdir)
    medio = (lambda z: np.ones_like(z), k_perfil)

    omega = omega_para_atenuacion(a_objetivo, medio, L)
    a = atenuacion(omega, medio, L)
    print(f"[{EXN}] biseccion con k(z) heterogeneo: omega={omega:.4f}  ->  "
          f"a={a:.4f} (objetivo {a_objetivo:g})")

    ref, _, _ = construir_referencia(omega, medio)
    err_solver = verificacion_cruzada(ref, omega, medio, analitico=False)
    print(f"[{EXN}] verificacion cruzada solver<->BVP complejo: "
          f"rel err = {err_solver:.2e}")
    print(f"[{EXN}] regimen (§4.4):",
          regimen(senal_tapa(ref), medio, L, eta=0.0))

    rho, c, k = _medio_torch_k(k_perfil)
    seeds = list(seeds)[:1] if smoke else list(seeds)
    o = _opts(smoke, opts)
    malla = malla_e0(R, TMAX)
    Xg = sample_cylinder(R, L, TMAX, 3000, seed=3)

    e0s, modelos, primero, raw = [], [], None, None
    for s in seeds:
        op = LateralCauchyCylinder(R, L, TMAX, rho, c, k,
                                   net_config=net_config(s))
        t0 = time.time()
        op.fit(torchify(ref.g, op.device), torchify(ref.f, op.device), **o)
        segs = time.time() - t0
        E0 = e0(op, ref, malla)
        eg = rel_l2(op.grad_T(torch.as_tensor(Xg, device=op.device)),
                    ref.grad_T(Xg))
        raw = _escribir_raw(rdir, EXN, caso="k_lineal", semilla_red=s, a=a,
                            omega=omega, L=L, E0=E0, err_global=eg,
                            tiempo_s=round(segs, 2))
        modelos.append(_guardar_modelo(rdir, op, EXN, f"seed{s}"))
        e0s.append(E0)
        if primero is None:
            primero = op
        print(f"[{EXN}] seed {s}: E0={E0:.3e}  err_global={eg:.3e}  ({segs:.0f}s)")

    print(f"[{EXN}] E0 media = {np.mean(e0s):.3e}  "
          f"(esperado comparable a ex2 a igual a)")

    figs = _figs_ex3(fdir, primero, ref, k_perfil)
    print(f"[{EXN}] figuras: {', '.join(figs)}")
    return {"raw": raw, "figs": figs, "models": modelos,
            "resumen": _resumen({"k_lineal": e0s}),
            "omega": omega, "a": a, "err_solver": err_solver}


def _figs_ex3(fdir, op, ref, k_perfil):
    z = np.linspace(0.0, L, 201)
    with pl.paper_style():
        fig, ax = plt.subplots(figsize=(pl.SINGLE_COL, 2.55))
        ax.plot(z, k_perfil(z), "-", color=pl.BLUE)
        ax.set_xlabel(r"$z$")
        ax.set_ylabel(r"$k(z)$")
        fig.tight_layout()
        pl.save_fig(fig, os.path.join(fdir, "ex3_k_perfil"))

    t = np.linspace(TMAX / 200, TMAX, 200)
    X = np.stack([np.zeros_like(t), np.zeros_like(t), np.zeros_like(t), t], 1)
    dz_pinn = op.grad_T(torch.as_tensor(X, device=op.device))[:, 2].cpu().numpy()
    dz_ref = ref.grad_T(X)[:, 2]
    with pl.paper_style():
        fig, ax = plt.subplots(figsize=(pl.SINGLE_COL, 2.55))
        ax.plot(t, dz_ref, "-", color=pl.BLUE, label="referencia modal")
        ax.plot(t, dz_pinn, "o--", color=pl.VERMILLION, markerfacecolor="white",
                markevery=10, label="PINN")
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\partial_z T|_{z=0}$ en $r=0$")
        ax.legend()
        fig.tight_layout()
        pl.save_fig(fig, os.path.join(fdir, "ex3_traza_basal"))

    zc, ez = perfil_error_z(op, ref, R, L, TMAX)
    pl.plot_error_vs_z(zc, ez, path=os.path.join(fdir, "ex3_error_vs_z"))
    return [os.path.join(fdir, f)
            for f in ("ex3_k_perfil.pdf", "ex3_traza_basal.pdf",
                      "ex3_error_vs_z.pdf")]


# ========================================================================= ex4
def ex4(etas=(0.0, 1e-3, 1e-2, 5e-2), seeds_ruido=SEEDS_RUIDO[:5],
        seeds_red=SEEDS_RED[:3], smoke=False, opts=None, outdir=None):
    """ex4 — Robustez al ruido (spec §2-ex4). Base: configuración de ex2.

    Ruido gaussiano aditivo iid por punto, independiente sobre g y f, relativo
    a ||g||_inf y ||f||_inf; eta=0 usa solo las semillas de red."""
    EXN = "ex4"
    rdir, fdir = _rutas(outdir)

    omega = omega_para_atenuacion(A_OBJETIVO, MEDIO_CONST, L)
    a = atenuacion(omega, MEDIO_CONST, L)
    print(f"[{EXN}] base ex2: omega={omega:.4f}  a={a:.4f}  (1/a={1 / a:.2f} de "
          f"amplificacion prevista)")
    ref, _, _ = construir_referencia(omega, MEDIO_CONST)

    Xtop = sample_disk_slice(R, TMAX, L, 4000, seed=9)
    esc_g = float(np.max(np.abs(ref.g(Xtop))))
    esc_f = float(np.max(np.abs(ref.f(Xtop))))

    rho, c, k = _medio_torch_const()
    seeds_red = list(seeds_red)[:1] if smoke else list(seeds_red)
    seeds_ruido = list(seeds_ruido)[:1] if smoke else list(seeds_ruido)
    o = _opts(smoke, opts)
    malla = malla_e0(R, TMAX)
    Xg = sample_cylinder(R, L, TMAX, 3000, seed=4)

    por_caso, modelos, raw = {}, [], None
    for eta in etas:
        print(f"[{EXN}] regimen (§4.4, eta={eta:g}):",
              regimen(senal_tapa(ref), MEDIO_CONST, L, eta=eta))
        combos = ([(s, None) for s in seeds_red] if eta == 0.0 else
                  [(s, sr) for sr in seeds_ruido for s in seeds_red])
        for s, sr in combos:
            if sr is None:
                g_dato, f_dato = ref.g, ref.f
            else:
                g_dato = con_ruido(ref.g, eta, esc_g,
                                   np.random.default_rng([sr, 0]))
                f_dato = con_ruido(ref.f, eta, esc_f,
                                   np.random.default_rng([sr, 1]))
            op = LateralCauchyCylinder(R, L, TMAX, rho, c, k,
                                       net_config=net_config(s))
            t0 = time.time()
            op.fit(torchify(g_dato, op.device), torchify(f_dato, op.device), **o)
            segs = time.time() - t0
            E0 = e0(op, ref, malla)
            eg = rel_l2(op.grad_T(torch.as_tensor(Xg, device=op.device)),
                        ref.grad_T(Xg))
            raw = _escribir_raw(rdir, EXN, caso=f"eta{eta:g}", semilla_red=s,
                                semilla_ruido=sr, eta=eta, a=a, omega=omega,
                                L=L, E0=E0, err_global=eg,
                                tiempo_s=round(segs, 2))
            modelos.append(_guardar_modelo(
                rdir, op, EXN, f"eta{eta:g}_seed{s}" +
                ("" if sr is None else f"_ruido{sr}")))
            por_caso.setdefault(f"eta{eta:g}", []).append(E0)
            print(f"[{EXN}] eta={eta:g} red={s} ruido={sr}: E0={E0:.3e} "
                  f"({segs:.0f}s)")

    figs = [_fig_ex4(fdir, _leer_raw(rdir, EXN))]
    print(f"[{EXN}] figura: {figs[0]}")
    return {"raw": raw, "figs": figs, "models": modelos,
            "resumen": _resumen(por_caso), "omega": omega, "a": a}


def _fig_ex4(fdir, filas):
    """figs/ex4_ruido.pdf: E0 vs eta log-log, media +/- std; eta=0 como piso."""
    por_eta = {}
    for f in filas:
        por_eta.setdefault(float(f["eta"]), []).append(float(f["E0"]))
    etas = sorted(e for e in por_eta if e > 0)
    med = np.array([np.mean(por_eta[e]) for e in etas])
    std = np.array([np.std(por_eta[e]) for e in etas])
    with pl.paper_style():
        fig, ax = plt.subplots(figsize=(pl.SINGLE_COL, 2.55))
        ax.errorbar(etas, med, yerr=std, fmt="o-", color=pl.BLUE, capsize=2.5,
                    label=r"$E_0(\eta)$")
        if 0.0 in por_eta:
            piso = float(np.mean(por_eta[0.0]))
            ax.axhline(piso, ls="--", color=pl.INK, alpha=0.6,
                       label=r"piso $\eta=0$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\eta$ (amplitud relativa del ruido)")
        ax.set_ylabel(r"$E_0$")
        ax.legend()
        fig.tight_layout()
        pl.save_fig(fig, os.path.join(fdir, "ex4_ruido"))
    return os.path.join(fdir, "ex4_ruido.pdf")


# ========================================================================= ex5
def ex5(a_objetivos=(0.7, 0.5, 0.37, 0.2, 0.1, 0.03, 0.01),
        Ls=(0.5, 0.75, 1.0, 1.5, 2.0), eta_ruido=1e-2,
        seeds_frec=SEEDS_RED, seeds_L=SEEDS_RED[:3], smoke=False, opts=None,
        outdir=None):
    """ex5 — Mapa de degradación (RESULTADO CENTRAL, spec §2-ex5).

    Barrido (i) en frecuencia (L=1), barrido (ii) en longitud (dato fijo, la
    omega de a=0.37 en L=1), y el barrido (i) repetido con ruido eta_ruido.
    Imprime la predicción regimen() de cada caso ANTES de entrenar y verifica
    el presupuesto con una corrida base antes de lanzar."""
    EXN = "ex5"
    rdir, fdir = _rutas(outdir)
    o = _opts(smoke, opts)
    malla = malla_e0(R, TMAX)
    Xg1 = sample_cylinder(R, 1.0, TMAX, 3000, seed=5)
    seeds_frec = list(seeds_frec)[:1] if smoke else list(seeds_frec)
    seeds_L = list(seeds_L)[:1] if smoke else list(seeds_L)
    seeds_ruido = SEEDS_RUIDO[:1] if smoke else SEEDS_RUIDO[:3]

    def corrida(ref, Lc, s, Xg, g_dato=None, f_dato=None):
        rho, c, k = _medio_torch_const()
        op = LateralCauchyCylinder(R, Lc, TMAX, rho, c, k,
                                   net_config=net_config(s))
        t0 = time.time()
        op.fit(torchify(g_dato or ref.g, op.device),
               torchify(f_dato or ref.f, op.device), **o)
        segs = time.time() - t0
        E0 = e0(op, ref, malla)
        eg = rel_l2(op.grad_T(torch.as_tensor(Xg, device=op.device)),
                    ref.grad_T(Xg))
        return op, E0, eg, segs

    n_total = (len(a_objetivos) * len(seeds_frec) + len(Ls) * len(seeds_L)
               + len(a_objetivos) * len(seeds_ruido))
    omega037 = omega_para_atenuacion(0.37, MEDIO_CONST, 1.0)
    ref037, _, _ = construir_referencia(omega037, MEDIO_CONST)
    t0 = time.time()
    corrida(ref037, 1.0, SEEDS_RED[0], Xg1)
    por_corrida = time.time() - t0
    print(f"[{EXN}] presupuesto: {n_total} corridas x ~{por_corrida:.0f}s "
          f"~ {n_total * por_corrida / 60:.0f} min")

    por_caso, raw = {}, None

    # ------------------------------------------------ barrido (i): frecuencia
    for a_obj in a_objetivos:
        omega = omega_para_atenuacion(a_obj, MEDIO_CONST, 1.0)
        a = atenuacion(omega, MEDIO_CONST, 1.0)
        ref, _, _ = construir_referencia(omega, MEDIO_CONST)
        print(f"[{EXN}] (i) a_obj={a_obj:g}: omega={omega:.3f}  prediccion:",
              regimen(senal_tapa(ref), MEDIO_CONST, 1.0, eta=0.0))
        for s in seeds_frec:
            _, E0, eg, segs = corrida(ref, 1.0, s, Xg1)
            caso = f"freq_a{a_obj:g}"
            raw = _escribir_raw(rdir, EXN, caso=caso, semilla_red=s, a=a,
                                omega=omega, L=1.0, E0=E0, err_global=eg,
                                tiempo_s=round(segs, 2))
            por_caso.setdefault(caso, []).append(E0)
            print(f"[{EXN}] (i) a={a:.3f} seed {s}: E0={E0:.3e} ({segs:.0f}s)")

    # ------------------------------------------------- barrido (ii): longitud
    for Lc in Ls:
        a = atenuacion(omega037, MEDIO_CONST, Lc)
        ref, _, _ = construir_referencia(omega037, MEDIO_CONST, L=Lc)
        Xg = sample_cylinder(R, Lc, TMAX, 3000, seed=5)
        print(f"[{EXN}] (ii) L={Lc:g}: prediccion:",
              regimen(senal_tapa(ref), MEDIO_CONST, Lc, eta=0.0))
        for s in seeds_L:
            _, E0, eg, segs = corrida(ref, Lc, s, Xg)
            caso = f"long_L{Lc:g}"
            raw = _escribir_raw(rdir, EXN, caso=caso, semilla_red=s, a=a,
                                omega=omega037, L=Lc, E0=E0, err_global=eg,
                                tiempo_s=round(segs, 2))
            por_caso.setdefault(caso, []).append(E0)
            print(f"[{EXN}] (ii) L={Lc:g} a={a:.3f} seed {s}: E0={E0:.3e} "
                  f"({segs:.0f}s)")

    # ------------------------------- segunda curva: barrido (i) con eta_ruido
    for a_obj in a_objetivos:
        omega = omega_para_atenuacion(a_obj, MEDIO_CONST, 1.0)
        a = atenuacion(omega, MEDIO_CONST, 1.0)
        ref, _, _ = construir_referencia(omega, MEDIO_CONST)
        Xtop = sample_disk_slice(R, TMAX, 1.0, 4000, seed=9)
        esc_g = float(np.max(np.abs(ref.g(Xtop))))
        esc_f = float(np.max(np.abs(ref.f(Xtop))))
        print(f"[{EXN}] (ruido) a_obj={a_obj:g}: prediccion:",
              regimen(senal_tapa(ref), MEDIO_CONST, 1.0, eta=eta_ruido))
        for sr in seeds_ruido:
            g_dato = con_ruido(ref.g, eta_ruido, esc_g,
                               np.random.default_rng([sr, 0]))
            f_dato = con_ruido(ref.f, eta_ruido, esc_f,
                               np.random.default_rng([sr, 1]))
            _, E0, eg, segs = corrida(ref, 1.0, SEEDS_RED[0], Xg1,
                                      g_dato=g_dato, f_dato=f_dato)
            caso = f"ruido_a{a_obj:g}"
            raw = _escribir_raw(rdir, EXN, caso=caso, semilla_red=SEEDS_RED[0],
                                semilla_ruido=sr, eta=eta_ruido, a=a,
                                omega=omega, L=1.0, E0=E0, err_global=eg,
                                tiempo_s=round(segs, 2))
            por_caso.setdefault(caso, []).append(E0)
            print(f"[{EXN}] (ruido) a={a:.3f} ruido={sr}: E0={E0:.3e} "
                  f"({segs:.0f}s)")

    figs = [_fig_ex5(fdir, _leer_raw(rdir, EXN), eta_ruido)]
    print(f"[{EXN}] figura central: {figs[0]}")
    return {"raw": raw, "figs": figs, "models": [],
            "resumen": _resumen(por_caso)}


def _fig_ex5(fdir, filas, eta_ruido):
    """figs/ex5_colapso.pdf: log E0 vs log(1/a), familias + cota + vertical."""
    fam = {"freq": {}, "long": {}, "ruido": {}}
    for f in filas:
        pref = f["caso"].split("_")[0]
        if pref in fam:
            fam[pref].setdefault(float(f["a"]), []).append(float(f["E0"]))

    def serie(d):
        a = np.array(sorted(d, reverse=True))
        med = np.array([np.mean(d[x]) for x in a])
        std = np.array([np.std(d[x]) for x in a])
        return a, med, std

    with pl.paper_style():
        fig, ax = plt.subplots(figsize=(pl.SINGLE_COL, 2.8))
        estilos = {"freq": ("o", pl.BLUE, r"barrido $\omega$ ($L=1$)"),
                   "long": ("s", pl.GREEN, r"barrido $L$ ($\omega$ fija)"),
                   "ruido": ("^", pl.VERMILLION,
                             rf"barrido $\omega$, $\eta={eta_ruido:g}$")}
        for nombre, (mk, color, lab) in estilos.items():
            if not fam[nombre]:
                continue
            a, med, std = serie(fam[nombre])
            ax.errorbar(1.0 / a, med, yerr=std, fmt=mk, ls="-", ms=4.5,
                        color=color, capsize=2, label=lab,
                        markerfacecolor="white" if nombre == "long" else None)
        if fam["freq"]:
            a, med, _ = serie(fam["freq"])
            eta_ef = 2.0 * a[0] * med[0]
            xx = np.array([1.0 / a[0], 1.0 / a[-1]])
            ax.plot(xx, eta_ef * xx / 2.0, "--", color=pl.INK, alpha=0.6,
                    label=r"cota $E_0=\eta_{\rm ef}/(2a)$")
        ax.axvline(1.0 / eta_ruido, color=pl.PURPLE, ls=":", alpha=0.8,
                   label=rf"$a=\eta={eta_ruido:g}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$1/a$ (amplificacion)")
        ax.set_ylabel(r"$E_0$")
        ax.legend(fontsize=6.5)
        fig.tight_layout()
        pl.save_fig(fig, os.path.join(fdir, "ex5_colapso"))
    return os.path.join(fdir, "ex5_colapso.pdf")


# ========================================================================= ex6
def ex6(lambdas=(0.1, 1.0, 10.0), seeds=SEEDS_RED[:3], smoke=False, opts=None,
        outdir=None):
    """ex6 — Sensibilidad al balance de pesos (spec §2-ex6). Base: ex2.

    Malla (lambda_g, lambda_f) in lambdas^2 con lambda_PDE = lambda_lat = 1;
    solo MIDE la sensibilidad (sin pesos adaptativos, §0.5)."""
    EXN = "ex6"
    rdir, fdir = _rutas(outdir)

    omega = omega_para_atenuacion(A_OBJETIVO, MEDIO_CONST, L)
    a = atenuacion(omega, MEDIO_CONST, L)
    print(f"[{EXN}] base ex2: omega={omega:.4f}  a={a:.4f}")
    ref, _, _ = construir_referencia(omega, MEDIO_CONST)
    print(f"[{EXN}] regimen (§4.4):",
          regimen(senal_tapa(ref), MEDIO_CONST, L, eta=0.0))

    rho, c, k = _medio_torch_const()
    seeds = list(seeds)[:1] if smoke else list(seeds)
    o = _opts(smoke, opts)
    malla = malla_e0(R, TMAX)
    Xg = sample_cylinder(R, L, TMAX, 3000, seed=6)

    nl = len(lambdas)
    media = np.zeros((nl, nl))
    std = np.zeros((nl, nl))
    por_caso, raw = {}, None
    for i, lg in enumerate(lambdas):
        for j, lf in enumerate(lambdas):
            vals = []
            for s in seeds:
                op = LateralCauchyCylinder(R, L, TMAX, rho, c, k,
                                           net_config=net_config(s))
                oo = dict(o, weights=(1.0, lg, lf, 1.0))
                t0 = time.time()
                op.fit(torchify(ref.g, op.device), torchify(ref.f, op.device),
                       **oo)
                segs = time.time() - t0
                E0 = e0(op, ref, malla)
                eg = rel_l2(op.grad_T(torch.as_tensor(Xg, device=op.device)),
                            ref.grad_T(Xg))
                caso = f"lg{lg:g}_lf{lf:g}"
                raw = _escribir_raw(rdir, EXN, caso=caso, semilla_red=s, a=a,
                                    omega=omega, L=L, E0=E0, err_global=eg,
                                    tiempo_s=round(segs, 2))
                vals.append(E0)
                por_caso.setdefault(caso, []).append(E0)
                print(f"[{EXN}] lg={lg:g} lf={lf:g} seed {s}: E0={E0:.3e} "
                      f"({segs:.0f}s)")
            media[i, j] = np.mean(vals)
            std[i, j] = np.std(vals)

    figs = [_fig_ex6(fdir, media, std, lambdas)]
    print(f"[{EXN}] figura: {figs[0]}")
    return {"raw": raw, "figs": figs, "models": [],
            "resumen": _resumen(por_caso)}


def _fig_ex6(fdir, media, std, lambdas):
    """figs/ex6_heatmap.pdf: media de E0 con std anotada por celda."""
    nl = len(lambdas)
    with pl.paper_style():
        fig, ax = plt.subplots(figsize=(pl.SINGLE_COL, 2.8))
        im = ax.imshow(media, origin="lower", cmap=pl.CMAP_FIELD)
        etiquetas = [f"{v:g}" for v in lambdas]
        ax.set_xticks(range(nl), etiquetas)
        ax.set_yticks(range(nl), etiquetas)
        ax.set_xlabel(r"$\lambda_f$")
        ax.set_ylabel(r"$\lambda_g$")
        ax.grid(False)
        lo, hi = media.min(), media.max()
        for i in range(nl):
            for j in range(nl):
                frac = 0.5 if hi == lo else (media[i, j] - lo) / (hi - lo)
                ax.text(j, i, f"{media[i, j]:.2f}\n$\\pm${std[i, j]:.2f}",
                        ha="center", va="center", fontsize=6.5,
                        color="black" if frac > 0.6 else "white")
        fig.colorbar(im, ax=ax, label=r"$E_0$ (media por celda)")
        fig.tight_layout()
        pl.save_fig(fig, os.path.join(fdir, "ex6_heatmap"))
    return os.path.join(fdir, "ex6_heatmap.pdf")


# ============================================================ visualización 5.7
def visualizacion(experimento="ex3", modelo="seed0",
                  tiempos=(0.2, 0.4, 0.6, 0.8), outdir=None):
    """Visualización 5.7: CARGA un modelo guardado de ex3 (no reentrena) y
    produce figs/vis_espaciotemporal.pdf (3 filas x len(tiempos) columnas,
    barra de color común por fila) y figs/vis_animacion.gif (40 cuadros).

    El modelo se lee de {results}/{experimento}/models/{modelo}.pt con el
    medio del spec de ex3 (k(z)=1+z); la referencia modal se reconstruye de
    forma determinista (misma bisección que ex3)."""
    rdir, fdir = _rutas(outdir)
    rho, c, k = _medio_torch_k(K_PERFIL_EX3)
    ruta_pt = os.path.join(rdir, experimento, "models", f"{modelo}.pt")
    op = LateralCauchyCylinder.load(ruta_pt, rho, c, k, map_location="cpu")

    medio = (lambda z: np.ones_like(z), K_PERFIL_EX3)
    omega = omega_para_atenuacion(A_OBJETIVO, medio, L)
    ref, _, _ = construir_referencia(omega, medio)

    figs = [_vis_estatica(fdir, op, ref, [ti * TMAX for ti in tiempos]),
            _vis_animacion(fdir, op, ref)]
    return {"raw": None, "figs": figs, "models": [ruta_pt], "resumen": {}}


def _corte_T(op, t0, n=100):
    """T_theta en el semiplano r-z (y=0, x=r>=0), malla n x n."""
    r = np.linspace(0.0, R, n)
    z = np.linspace(0.0, L, n)
    RRg, ZZ = np.meshgrid(r, z)
    X = np.stack([RRg.ravel(), np.zeros(RRg.size), ZZ.ravel(),
                  np.full(RRg.size, t0)], axis=1)
    return op.T(torch.as_tensor(X, device=op.device)).cpu().numpy().reshape(n, n)


def _malla_disco(n_r=50, n_th=121):
    r = np.linspace(0.0, R, n_r)
    th = np.linspace(0.0, 2.0 * np.pi, n_th)
    RRg, TH = np.meshgrid(r, th, indexing="ij")
    return RRg * np.cos(TH), RRg * np.sin(TH)


def _dato_disco(fn, XX, YY, t0):
    X = np.stack([XX.ravel(), YY.ravel(), np.full(XX.size, L),
                  np.full(XX.size, t0)], axis=1)
    return np.asarray(fn(X)).reshape(XX.shape)


def _campos_vis(op, ref, tiempos):
    """Precalcula (T, g, f) por instante y los (vmin, vmax) comunes por fila."""
    XX, YY = _malla_disco()
    campos = [(_corte_T(op, t), _dato_disco(ref.g, XX, YY, t),
               _dato_disco(ref.f, XX, YY, t)) for t in tiempos]
    lims = [(min(c[i].min() for c in campos), max(c[i].max() for c in campos))
            for i in range(3)]
    return campos, lims, (XX, YY)


def _vis_estatica(fdir, op, ref, tiempos):
    campos, lims, (XX, YY) = _campos_vis(op, ref, tiempos)
    etiquetas = (r"$T_\theta$  (corte $y=0$)", r"$g$  (tapa $z=L$)",
                 r"$f$  (tapa $z=L$)")
    nc = len(tiempos)
    with pl.paper_style():
        fig, axs = plt.subplots(3, nc, figsize=(pl.DOUBLE_COL, 5.4),
                                squeeze=False)
        for j, (t0, (T, G, F)) in enumerate(zip(tiempos, campos)):
            axs[0, j].imshow(T, origin="lower", extent=(0, R, 0, L),
                             aspect="auto", cmap=pl.CMAP_FIELD,
                             vmin=lims[0][0], vmax=lims[0][1])
            axs[0, j].set_title(fr"$t={t0:g}$", fontsize=8)
            axs[0, j].set_xlabel(r"$r$")
            for i, C in enumerate((G, F), start=1):
                axs[i, j].pcolormesh(XX, YY, C, cmap=pl.CMAP_FIELD,
                                     vmin=lims[i][0], vmax=lims[i][1],
                                     shading="gouraud")
                axs[i, j].set_aspect("equal")
            for i in range(3):
                axs[i, j].grid(False)
                if j > 0:
                    axs[i, j].set_yticklabels([])
        axs[0, 0].set_ylabel(r"$z$")
        for i in range(3):
            fig.colorbar(axs[i, -1].collections[0] if i else axs[i, -1].images[0],
                         ax=axs[i, :], shrink=0.85, label=etiquetas[i])
        pl.save_fig(fig, os.path.join(fdir, "vis_espaciotemporal"))
    print(f"[vis] {os.path.join(fdir, 'vis_espaciotemporal.pdf')}")
    return os.path.join(fdir, "vis_espaciotemporal.pdf")


def _vis_animacion(fdir, op, ref, n_cuadros=40):
    from matplotlib.animation import FuncAnimation, PillowWriter
    tiempos = np.linspace(TMAX / n_cuadros, TMAX, n_cuadros)
    campos, lims, (XX, YY) = _campos_vis(op, ref, tiempos)
    with pl.paper_style():
        fig, axs = plt.subplots(1, 3, figsize=(pl.DOUBLE_COL, 2.4))
        T0, G0, F0 = campos[0]
        imT = axs[0].imshow(T0, origin="lower", extent=(0, R, 0, L),
                            aspect="auto", cmap=pl.CMAP_FIELD,
                            vmin=lims[0][0], vmax=lims[0][1])
        axs[0].set_xlabel(r"$r$")
        axs[0].set_ylabel(r"$z$")
        imG = axs[1].pcolormesh(XX, YY, G0, cmap=pl.CMAP_FIELD,
                                vmin=lims[1][0], vmax=lims[1][1],
                                shading="gouraud")
        imF = axs[2].pcolormesh(XX, YY, F0, cmap=pl.CMAP_FIELD,
                                vmin=lims[2][0], vmax=lims[2][1],
                                shading="gouraud")
        for ax, ti in zip(axs, (r"$T_\theta$ ($y=0$)", r"$g$", r"$f$")):
            ax.set_title(ti, fontsize=8)
            ax.grid(False)
        for ax in axs[1:]:
            ax.set_aspect("equal")
        titulo = fig.suptitle(f"t = {tiempos[0]:.3f}", fontsize=9)
        fig.tight_layout()

        def cuadro(i):
            T, G, F = campos[i]
            imT.set_data(T)
            imG.set_array(G.ravel())
            imF.set_array(F.ravel())
            titulo.set_text(f"t = {tiempos[i]:.3f}")
            return imT, imG, imF, titulo

        anim = FuncAnimation(fig, cuadro, frames=n_cuadros, blit=False)
        os.makedirs(fdir, exist_ok=True)
        ruta = os.path.join(fdir, "vis_animacion.gif")
        anim.save(ruta, writer=PillowWriter(fps=4), dpi=110)   # ~10 s
        plt.close(fig)
    print(f"[vis] {ruta}")
    return ruta


# ============================================================== figura §3.3
def aten_curvas(outdir=None):
    """Figura de la §3.3 (spec §1.4): figs/aten_curvas.pdf — curvas a(omega)
    para el medio constante y el k(z)=1+z de ex3, ejes log a vs sqrt(omega),
    con la recta teórica del semiespacio."""
    _, fdir = _rutas(outdir)
    medios = {r"$\rho c = k = 1$": MEDIO_CONST,
              r"$k(z) = 1 + z$": (lambda z: np.ones_like(z), K_PERFIL_EX3)}
    omegas = np.logspace(-2, 2.0, 40)
    with pl.paper_style():
        fig, ax = plt.subplots(figsize=(pl.SINGLE_COL, 2.55))
        for (nombre, medio), color in zip(medios.items(), pl.CAT):
            a = np.array([atenuacion(w, medio, L) for w in omegas])
            ax.semilogy(np.sqrt(omegas), a, "-", color=color, label=nombre)
        ax.semilogy(np.sqrt(omegas), np.exp(-L * np.sqrt(omegas / 2.0)), "--",
                    color=pl.INK, alpha=0.6,
                    label=r"$e^{-L\sqrt{\omega/2}}$ (semiespacio)")
        ax.set_xlabel(r"$\sqrt{\omega}$")
        ax.set_ylabel(r"$a(\omega)$")
        ax.legend()
        fig.tight_layout()
        pl.save_fig(fig, os.path.join(fdir, "aten_curvas"))
    ruta = os.path.join(fdir, "aten_curvas.pdf")
    print(f"[aten_curvas] figura: {ruta}")
    return {"raw": None, "figs": [ruta], "models": [], "resumen": {}}
