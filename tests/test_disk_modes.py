"""Tests de los modos de Neumann del disco."""

import numpy as np
import pytest

from numerics.disk_modes import DiskMode

MODES = [(0, 1, "cos"), (1, 1, "cos"), (2, 2, "sin"), (3, 1, "cos")]


def _interior_points(seed=0, n=300, R=1.0):
    rng = np.random.default_rng(seed)
    r = 0.7 * R * np.sqrt(rng.random(n))
    th = 2 * np.pi * rng.random(n)
    return r * np.cos(th), r * np.sin(th)


@pytest.mark.parametrize("m,n,kind", MODES)
def test_eigenfunction(m, n, kind):
    md = DiskMode(m, n, 1.0, kind)
    x, y = _interior_points()
    h = 1e-5
    lap = (md.value(x + h, y) + md.value(x - h, y)
           + md.value(x, y + h) + md.value(x, y - h) - 4 * md.value(x, y)) / h ** 2
    rel = np.max(np.abs(lap + md.mu * md.value(x, y))) / np.max(np.abs(md.value(x, y)))
    assert rel < 1e-3                      # -Laplaciano(phi) = mu phi


@pytest.mark.parametrize("m,n,kind", MODES)
def test_gradient(m, n, kind):
    md = DiskMode(m, n, 1.0, kind)
    x, y = _interior_points()
    h = 1e-6
    gx, gy = md.grad(x, y)
    gx_fd = (md.value(x + h, y) - md.value(x - h, y)) / (2 * h)
    gy_fd = (md.value(x, y + h) - md.value(x, y - h)) / (2 * h)
    assert np.allclose(gx, gx_fd, atol=1e-5) and np.allclose(gy, gy_fd, atol=1e-5)


@pytest.mark.parametrize("m,n,kind", MODES)
def test_lateral_neumann(m, n, kind):
    md = DiskMode(m, n, 1.0, kind)
    th = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    gx, gy = md.grad(np.cos(th), np.sin(th))      # r = R = 1
    dphi_dr = gx * np.cos(th) + gy * np.sin(th)
    assert np.max(np.abs(dphi_dr)) < 1e-10        # Neumann en r=R
