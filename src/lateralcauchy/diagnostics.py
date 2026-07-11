"""Diagnóstico L/δ y acceso a las gráficas (compatibilidad).

- skin_depth, ld_ratio: el número L/δ (δ = √(2α/ω)) predice, ANTES de entrenar,
  si el régimen es recuperable. Régimen sano L/δ ≲ 1; valores grandes ⇒ la
  continuación hacia la base amplifica demasiado (CLAUDE.md §3).
- Las gráficas viven en `plotting` (estilo de artículo, PDF+PNG); aquí se
  re-exportan plot_history / plot_error_vs_z / plot_error_vs_t por
  compatibilidad con código existente.
"""

import math

from .plotting import (                                      # noqa: F401
    plot_history, plot_error_vs_z, plot_error_vs_t,
)


def skin_depth(alpha, omega):
    """δ(α, ω) = √(2α/ω)   (longitud de penetración del armónico temporal)."""
    return math.sqrt(2.0 * alpha / omega)


def ld_ratio(alpha, omega, L):
    """L/δ: distancia de continuación en unidades de δ. ≲1 recuperable; ≫1 mal condicionado."""
    return L / skin_depth(alpha, omega)
