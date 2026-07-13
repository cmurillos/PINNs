"""Tests de la infraestructura de diagnóstico (§1 de docs/SIMULACIONES.md)."""

import warnings

import numpy as np

from lateralcauchy.diagnostics import atenuacion, omega_efectiva, regimen

MEDIO_CONST = (lambda z: np.ones_like(z), lambda z: np.ones_like(z))


def _a_exacta_intervalo(omega, L=1.0):
    """Exacta del BVP en el intervalo con U'(L)=0: a = |1/cosh(μL)|, μ=√(iω/α)."""
    mu = np.sqrt(1j * omega)
    return abs(1.0 / np.cosh(mu * L))


def test_atenuacion_vs_exacta_intervalo():
    # medio constante (rc=k=1, alpha=1), L=1; tolerancia relativa 2 % (§1.1).
    for omega in (1.0, 5.0, 20.0, 80.0):
        a = atenuacion(omega, MEDIO_CONST, 1.0)
        exacta = _a_exacta_intervalo(omega)
        assert abs(a - exacta) / exacta < 0.02
        # la exponencial del semiespacio queda como COTA inferior, no referencia
        assert a >= 0.98 * np.exp(-np.sqrt(omega / 2.0))


def test_atenuacion_decrece_con_omega_y_L():
    a1 = atenuacion(5.0, MEDIO_CONST, 1.0)
    assert atenuacion(20.0, MEDIO_CONST, 1.0) < a1 < atenuacion(1.0, MEDIO_CONST, 1.0)
    assert atenuacion(5.0, MEDIO_CONST, 2.0) < a1


def test_omega_efectiva_coseno():
    # g = cos(ωt) sobre ventana múltiplo del período -> ω_ef = ω (< 0.5 %).
    omega = 5.0
    periodo = 2.0 * np.pi / omega
    t = np.linspace(0.0, 4 * periodo, 2001)
    w = omega_efectiva(np.cos(omega * t), t[1] - t[0])
    assert abs(w - omega) / omega < 0.005


def test_regimen_junta_ambos():
    omega, eta = 5.0, 1e-3
    periodo = 2.0 * np.pi / omega
    t = np.linspace(0.0, 4 * periodo, 2001)
    d = regimen((np.cos(omega * t), t[1] - t[0]), MEDIO_CONST, 1.0, eta)
    assert set(d) == {"omega_ef", "a", "margen", "recuperable"}
    assert abs(d["omega_ef"] - omega) / omega < 0.005
    assert abs(d["a"] - atenuacion(d["omega_ef"], MEDIO_CONST, 1.0)) < 1e-12
    assert abs(d["margen"] - d["a"] / eta) < 1e-12
    assert d["recuperable"] == (d["a"] > eta)
    assert regimen((np.cos(omega * t), t[1] - t[0]),
                   MEDIO_CONST, 1.0, eta=0.0)["margen"] == float("inf")


def test_skin_depth_ld_ratio_deprecadas():
    from lateralcauchy.diagnostics import skin_depth, ld_ratio
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert abs(skin_depth(1.0, 2.0) - 1.0) < 1e-12
        assert abs(ld_ratio(1.0, 2.0, 1.0) - 1.0) < 1e-12
    assert sum(issubclass(w.category, DeprecationWarning) for w in rec) == 2
