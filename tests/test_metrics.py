"""Tests de las utilidades de metricas y del puente numpy<->torch."""

import numpy as np
import torch

from lateralcauchy.metrics import (
    rel_l2, sample_cylinder, sample_disk_slice, error_vs_z, error_vs_t,
    torchify, to_numpy, malla_e0, e0, perfil_error_t,
)


class _CampoSintetico:
    """grad_T = factor * (x, y, 1+t): campo sintético para probar e0."""

    def __init__(self, factor):
        self.factor = factor

    def grad_T(self, X):
        X = np.asarray(X, dtype=float)
        return self.factor * np.stack(
            [X[:, 0], X[:, 1], 1.0 + X[:, 3]], axis=1)


def test_malla_e0_discretizacion_spec():
    # §0.3: 24 radios x 48 angulos (sin r=0) x 100 instantes en (0, Tmax].
    m = malla_e0(R=1.0, Tmax=1.0)
    n_r, n_th, n_t = m["n"]
    assert (n_r, n_th, n_t) == (24, 48, 100)
    X, w = m["X"], m["w"]
    assert X.shape == (24 * 48 * 100, 4) and w.shape == (X.shape[0],)
    r = np.hypot(X[:, 0], X[:, 1])
    assert r.min() > 0.0 and r.max() <= 1.0 + 1e-12      # excluye r=0
    assert np.allclose(X[:, 2], 0.0)                     # base z=0
    assert X[:, 3].min() > 0.0 and abs(X[:, 3].max() - 1.0) < 1e-12
    assert (w > 0).all()


def test_e0_error_conocido():
    # pred = (1+eps)*ref  ->  E0 = eps exactamente (independiente de los pesos).
    eps = 0.02
    m = malla_e0(R=1.0, Tmax=1.0)
    val = e0(_CampoSintetico(1.0 + eps), _CampoSintetico(1.0), m)
    assert abs(val - eps) < 1e-12
    assert e0(_CampoSintetico(1.0), _CampoSintetico(1.0), m) == 0.0


def test_perfil_error_t_error_conocido():
    m = malla_e0(R=1.0, Tmax=1.0, n_r=6, n_theta=8, n_t=10)
    t, err = perfil_error_t(_CampoSintetico(1.05), _CampoSintetico(1.0), m)
    assert len(t) == len(err) == 10
    assert np.allclose(err, 0.05)


def test_rel_l2_zero_and_value():
    a = np.array([[3.0, 4.0]])
    assert rel_l2(a, a) == 0.0
    assert abs(rel_l2(np.zeros_like(a), a) - 1.0) < 1e-12   # error == senal


def test_rel_l2_accepts_torch():
    a = torch.ones(5, 3)
    assert rel_l2(a, a) == 0.0


def test_sample_cylinder_in_domain():
    R, L, Tmax = 2.0, 3.0, 1.5
    X = sample_cylinder(R, L, Tmax, 1000, seed=1)
    r = np.hypot(X[:, 0], X[:, 1])
    assert X.shape == (1000, 4)
    assert r.max() <= R + 1e-9
    assert 0 <= X[:, 2].min() and X[:, 2].max() <= L
    assert 0 <= X[:, 3].min() and X[:, 3].max() <= Tmax


def test_torchify_returns_torch():
    fn = lambda X: X[:, :1] * 2.0                 # numpy -> numpy
    g = torchify(fn, device="cpu")
    out = g(torch.ones(4, 4))
    assert torch.is_tensor(out) and out.shape == (4, 1)
    assert torch.allclose(out, 2 * torch.ones(4, 1))


def test_error_vs_z_and_t_shapes():
    X = sample_cylinder(1, 1, 1, 500, seed=2)
    pred, true = np.zeros((500, 3)), np.ones((500, 3))
    for fn in (error_vs_z, error_vs_t):
        c, err = fn(pred, true, X, nbins=5)
        assert len(c) == len(err) <= 5
        assert np.allclose(err, 1.0)              # pred=0 -> error relativo 1


def test_sample_disk_slice_fixed_z():
    X = sample_disk_slice(2.0, 1.5, 0.0, 500, seed=1)
    assert np.allclose(X[:, 2], 0.0)
    assert np.hypot(X[:, 0], X[:, 1]).max() <= 2.0 + 1e-9


def test_default_dtype_is_float64():
    assert torch.get_default_dtype() == torch.float64


def test_bridge_preserves_float64():
    g = torchify(lambda X: X[:, :1], device="cpu")
    out = g(torch.ones(3, 4, dtype=torch.float64))
    assert out.dtype == torch.float64                       # sin cast a float32
    assert to_numpy(torch.ones(2, dtype=torch.float64)).dtype == np.float64
