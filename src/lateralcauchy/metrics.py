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


def torchify(fn, device="cpu", dtype=None):
    """Envuelve un callable numpy (X(N,4)->(N,k)) para que la PINN lo use:
    acepta un tensor torch y devuelve torch en el device/dtype indicados."""
    dtype = dtype or torch.get_default_dtype()

    def wrapped(X):
        out = fn(to_numpy(X))
        return torch.as_tensor(out, dtype=dtype, device=device)

    return wrapped
