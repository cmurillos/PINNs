"""Tests rapidos de la PINN (entrenamiento minimo: solo verifican la mecanica)."""

import warnings

import pytest
import torch

from lateralcauchy import LateralCauchyCylinder

ONE = lambda X: torch.ones_like(X[:, :1])
ZERO = lambda X: torch.zeros_like(X[:, :1])
TINY = dict(adam_iters=2, lbfgs_iters=1, n_int=80, n_top=40, n_lat=40)


def _op():
    return LateralCauchyCylinder(1.0, 1.0, 1.0, ONE, ONE, ONE)


def test_error_before_fit():
    op = _op()
    with pytest.raises(RuntimeError):
        op.grad_T(torch.rand(3, 4))


def test_history_length_matches_iterations():
    # un registro por iteracion (no por evaluacion de closure de L-BFGS)
    op = _op()
    h = op.fit(ZERO, ZERO, **TINY)
    assert len(h["total"]) == TINY["adam_iters"] + TINY["lbfgs_iters"]
    assert all(len(h[k]) == len(h["total"]) for k in h)


def test_resample_runs():
    op = _op()
    op.fit(ZERO, ZERO, adam_iters=4, lbfgs_iters=1,
           n_int=60, n_top=30, n_lat=30, resample_every=2)
    assert callable(op.grad_T)


def test_outputs_after_fit():
    op = _op()
    h = op.fit(ZERO, ZERO, **TINY)
    assert set(h) == {"total", "pde", "g", "f", "lat"}
    X = torch.rand(5, 4) * 0.5
    assert op.T(X).shape == (5, 1)
    assert op.grad_T(X).shape == (5, 3)
    assert op.flux(X).shape == (5, 3)
    assert op.alpha(X).shape == (5, 1)


def test_flux_equals_minus_k_grad():
    op = _op()
    op.fit(ZERO, ZERO, **TINY)
    X = torch.rand(10, 4) * 0.5
    assert torch.allclose(op.flux(X), -op.grad_T(X), atol=1e-10)


def test_create_graph_allows_second_derivative():
    op = _op()
    op.fit(ZERO, ZERO, **TINY)
    X = (torch.rand(5, 4) * 0.5).requires_grad_(True)
    g = op.grad_T(X, create_graph=True)
    d2 = torch.autograd.grad(g[:, 2].sum(), X)[0]
    assert d2.shape == (5, 4)


def test_detached_by_default():
    op = _op()
    op.fit(ZERO, ZERO, **TINY)
    assert not op.grad_T(torch.rand(4, 4) * 0.5).requires_grad


def test_save_load_roundtrip(tmp_path):
    op = _op()
    op.fit(ZERO, ZERO, **TINY)
    X = torch.rand(6, 4) * 0.5
    before = op.grad_T(X)
    path = str(tmp_path / "modelo.pt")
    op.save(path)
    op2 = LateralCauchyCylinder.load(path, ONE, ONE, ONE)
    assert torch.allclose(before, op2.grad_T(X), atol=1e-12)   # mismos pesos


def test_domain_warning_outside():
    op = _op()
    op.fit(ZERO, ZERO, **TINY)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        op.T(torch.tensor([[5.0, 5.0, 0.5, 0.5]]))     # r=7 > R=1
    assert len(w) == 1
