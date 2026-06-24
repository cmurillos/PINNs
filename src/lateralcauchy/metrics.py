"""Metricas de comparacion y puente numpy <-> torch (capa compartida).

La PINN trabaja en torch; el solver de referencia en numpy. Aqui viven las
utilidades que cruzan ambos mundos: error relativo, muestreo del cilindro,
error en funcion de la profundidad z (clave en un problema mal puesto: el error
crece al alejarse de la tapa) y `torchify` para usar callables numpy como datos
de Cauchy de la PINN.
"""

import numpy as np
import torch


def to_numpy(a):
    return a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)


def rel_l2(pred, true):
    """Error relativo ||pred - true|| / ||true|| (norma de Frobenius)."""
    p, t = to_numpy(pred), to_numpy(true)
    return float(np.linalg.norm(p - t) / np.linalg.norm(t))


def sample_cylinder(R, L, Tmax, n, seed=0):
    """Nube uniforme de n puntos (n,4) [x,y,z,t] dentro del cilindro."""
    rng = np.random.default_rng(seed)
    r = R * np.sqrt(rng.random(n))
    th = 2 * np.pi * rng.random(n)
    return np.stack([r * np.cos(th), r * np.sin(th),
                     L * rng.random(n), Tmax * rng.random(n)], axis=1)


def error_vs_z(grad_pred, grad_true, X, nbins=8):
    """Error relativo por franjas de z. Devuelve (centros_z, errores)."""
    p, t, z = to_numpy(grad_pred), to_numpy(grad_true), to_numpy(X)[:, 2]
    edges = np.linspace(z.min(), z.max(), nbins + 1)
    zc, err = [], []
    for i in range(nbins):
        m = (z >= edges[i]) & (z <= edges[i + 1])
        if m.sum() == 0:
            continue
        zc.append(0.5 * (edges[i] + edges[i + 1]))
        err.append(np.linalg.norm(p[m] - t[m]) / np.linalg.norm(t[m]))
    return np.array(zc), np.array(err)


def torchify(fn, device="cpu", dtype=None):
    """Envuelve un callable numpy (X(N,4)->(N,k)) para que la PINN lo use:
    acepta un tensor torch y devuelve torch en el device/dtype indicados."""
    dtype = dtype or torch.get_default_dtype()

    def wrapped(X):
        out = fn(to_numpy(X))
        return torch.as_tensor(out, dtype=dtype, device=device)

    return wrapped
