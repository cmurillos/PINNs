"""Metricas de comparacion y puente numpy <-> torch (capa compartida).

La PINN trabaja en torch; el solver de referencia en numpy. Aqui viven las
utilidades que cruzan ambos mundos: error relativo, muestreo del cilindro,
error en funcion de la coordenada z / distancia a la tapa (clave en un problema
mal puesto: el error crece al alejarse de la tapa) y `torchify` para usar callables numpy como datos
de Cauchy de la PINN.
"""

import numpy as np
import torch


def to_numpy(a):
    return a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)


def rel_l2(pred, true):
    """e = ‖pred − true‖ / ‖true‖   (norma de Frobenius sobre la nube)."""
    p, t = to_numpy(pred), to_numpy(true)
    return float(np.linalg.norm(p - t) / np.linalg.norm(t))


def sample_cylinder(R, L, Tmax, n, seed=0):
    """Nube uniforme de n puntos (n,4) [x,y,z,t] dentro del cilindro."""
    rng = np.random.default_rng(seed)
    r = R * np.sqrt(rng.random(n))
    th = 2 * np.pi * rng.random(n)
    return np.stack([r * np.cos(th), r * np.sin(th),
                     L * rng.random(n), Tmax * rng.random(n)], axis=1)


def sample_disk_slice(R, Tmax, z0, n, seed=0):
    """Nube (n,4) en un corte z = z0 (disco x tiempo). Para la metrica basal z=0."""
    rng = np.random.default_rng(seed)
    r = R * np.sqrt(rng.random(n))
    th = 2 * np.pi * rng.random(n)
    return np.stack([r * np.cos(th), r * np.sin(th),
                     np.full(n, z0), Tmax * rng.random(n)], axis=1)


def _error_vs(grad_pred, grad_true, X, col, nbins):
    """Perfil e(c) = ‖∇T_pred − ∇T‖/‖∇T‖ por franjas de la columna c (2=z, 3=t)."""
    p, t, c = to_numpy(grad_pred), to_numpy(grad_true), to_numpy(X)[:, col]
    edges = np.linspace(c.min(), c.max(), nbins + 1)
    cc, err = [], []
    for i in range(nbins):
        m = (c >= edges[i]) & (c <= edges[i + 1])
        if m.sum() == 0:
            continue
        cc.append(0.5 * (edges[i] + edges[i + 1]))
        err.append(np.linalg.norm(p[m] - t[m]) / np.linalg.norm(t[m]))
    return np.array(cc), np.array(err)


def error_vs_z(grad_pred, grad_true, X, nbins=8):
    """z ↦ e(z): crece al alejarse de la tapa z=L (firma del mal condicionamiento)."""
    return _error_vs(grad_pred, grad_true, X, 2, nbins)


def error_vs_t(grad_pred, grad_true, X, nbins=8):
    """Error de grad_T por tiempo. CONTROL: deberia salir ~plano, porque la
    continuacion es espacial (en z), no temporal (CLAUDE.md §1.6.2). Un pico en
    t=0 indicaria efecto de borde de colocacion, no la inversion mal puesta."""
    return _error_vs(grad_pred, grad_true, X, 3, nbins)


def malla_e0(R, Tmax, n_r=24, n_theta=48, n_t=100):
    """Malla de cuadratura de la métrica E0 (§0.3 de docs/SIMULACIONES.md).

    Base Γ0 (z=0): producto polar n_r radios × n_theta ángulos (excluye r=0
    duplicado; pesos de área r·Δr·Δθ) × n_t instantes equiespaciados en
    (0, Tmax] con peso trapezoidal en t. Devuelve dict con
        X : (n_r·n_theta·n_t, 4) puntos [x, y, 0, t]  (t varía más rápido),
        w : (N,) pesos de cuadratura,  t : (n_t,),  n : (n_r, n_theta, n_t).
    """
    dr = R / n_r
    r = dr * np.arange(1, n_r + 1)
    dth = 2.0 * np.pi / n_theta
    th = dth * np.arange(n_theta)
    dt = Tmax / n_t
    t = dt * np.arange(1, n_t + 1)
    wt = np.full(n_t, dt)
    wt[0] *= 0.5
    wt[-1] *= 0.5                                  # trapecio en t
    wa = r * dr * dth                              # área polar por celda

    RR, TH, TT = np.meshgrid(r, th, t, indexing="ij")
    X = np.stack([(RR * np.cos(TH)).ravel(), (RR * np.sin(TH)).ravel(),
                  np.zeros(RR.size), TT.ravel()], axis=1)
    w = (wa[:, None, None] * np.ones((1, n_theta, 1)) * wt[None, None, :]).ravel()
    return {"X": X, "w": w, "t": t, "n": (n_r, n_theta, n_t)}


def _e_ponderado(P, G, w):
    """‖P − G‖_w / ‖G‖_w con pesos de cuadratura w (N,)."""
    w = w[:, None]
    return float(np.sqrt(np.sum(w * (P - G) ** 2) / np.sum(w * G ** 2)))


def e0(model, referencia, malla):
    """E0 = ‖∇T_θ − ∇T_ref‖₂ / ‖∇T_ref‖₂ sobre la malla de `malla_e0` (§0.3).

    model      : objeto con grad_T(X (N,4)) → (N,3)  (la PINN entrenada),
    referencia : objeto con grad_T numpy (exacta o ReferenceSolution),
    malla      : dict de malla_e0. ÚNICA discretización de E0: prohibido
                 re-discretizar por script (§0.3).
    """
    X, w = malla["X"], malla["w"]
    P = to_numpy(model.grad_T(X))
    G = np.asarray(referencia.grad_T(X), dtype=float)
    return _e_ponderado(P, G, w)


def perfil_error_z(model, referencia, R, L, Tmax, n_franjas=20, **malla_kw):
    """Métrica de control §0.3: error relativo por franja de z (20 franjas),
    misma discretización polar×t que E0 evaluada en el centro de cada franja.
    Devuelve (z_centros, err)."""
    base = malla_e0(R, Tmax, **malla_kw)
    zc = (np.arange(n_franjas) + 0.5) * (L / n_franjas)
    errs = []
    for z in zc:
        X = base["X"].copy()
        X[:, 2] = z
        P = to_numpy(model.grad_T(X))
        G = np.asarray(referencia.grad_T(X), dtype=float)
        errs.append(_e_ponderado(P, G, base["w"]))
    return zc, np.array(errs)


def perfil_error_t(model, referencia, malla):
    """Métrica de control §0.3: error relativo por instante sobre la malla E0
    (base z=0, pesos de área polar). Devuelve (t, err)."""
    n_r, n_th, n_t = malla["n"]
    X = malla["X"]
    P = to_numpy(model.grad_T(X)).reshape(n_r, n_th, n_t, 3)
    G = np.asarray(referencia.grad_T(X), dtype=float).reshape(n_r, n_th, n_t, 3)
    w = malla["w"].reshape(n_r, n_th, n_t)[:, :, 0]        # área (sin peso en t)
    errs = [_e_ponderado(P[:, :, i].reshape(-1, 3),
                         G[:, :, i].reshape(-1, 3), w.ravel())
            for i in range(n_t)]
    return malla["t"], np.array(errs)


def torchify(fn, device="cpu", dtype=None):
    """Envuelve un callable numpy (X(N,4)->(N,k)) para que la PINN lo use:
    acepta un tensor torch y devuelve torch en el device/dtype indicados."""
    dtype = dtype or torch.get_default_dtype()

    def wrapped(X):
        out = fn(to_numpy(X))
        return torch.as_tensor(out, dtype=dtype, device=device)

    return wrapped
