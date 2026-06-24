"""Tests de las utilidades de metricas y del puente numpy<->torch."""

import numpy as np
import torch

from metrics import rel_l2, sample_cylinder, error_vs_z, torchify, to_numpy


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


def test_error_vs_z_shapes():
    X = sample_cylinder(1, 1, 1, 500, seed=2)
    pred = np.zeros((500, 3))
    true = np.ones((500, 3))
    zc, err = error_vs_z(pred, true, X, nbins=5)
    assert len(zc) == len(err) <= 5
    assert np.allclose(err, 1.0)                  # pred=0 -> error relativo 1
