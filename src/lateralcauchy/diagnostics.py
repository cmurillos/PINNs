"""Diagnostico y graficas para la PINN del cilindro.

- ld_ratio: el numero L/delta (delta = sqrt(2 alpha / omega)) que predice, ANTES
  de entrenar, si el regimen es recuperable. Regimen sano L/delta <~ 1; valores
  grandes => la continuacion hacia la base amplifica demasiado (CLAUDE.md §3).
- plot_history: curvas de perdida por termino (lo que devuelve fit).
- plot_error_vs_z: error relativo de grad_T por coordenada z / distancia a la tapa
  (firma del mal condicionamiento: crece al alejarse de la tapa z=L).
- plot_error_vs_t: el mismo error por tiempo (control: deberia salir ~plano).

Usa el backend 'Agg' (sin ventana): guarda a archivo o devuelve la figura.
"""

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def skin_depth(alpha, omega):
    """δ(α, ω) = √(2α/ω)   (longitud de penetración del armónico temporal)."""
    return math.sqrt(2.0 * alpha / omega)


def ld_ratio(alpha, omega, L):
    """L/δ: distancia de continuación en unidades de δ. ≲1 recuperable; ≫1 mal condicionado."""
    return L / skin_depth(alpha, omega)


def plot_history(history, path=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    for key in ("total", "pde", "g", "f", "lat"):
        ax.semilogy(history[key], label=key)
    ax.set_xlabel("iteracion"); ax.set_ylabel("perdida")
    ax.set_title("Historia de entrenamiento"); ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig


def _plot_profile(coord, err, xlabel, title, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(coord, err, "o-")
    ax.set_xlabel(xlabel); ax.set_ylabel("error relativo de grad_T")
    ax.set_title(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig


def plot_error_vs_z(zc, err, path=None):
    return _plot_profile(zc, err, "z (tapa z=L a la derecha)",
                         "Error por distancia a la tapa (mal condicionamiento)", path)


def plot_error_vs_t(tc, err, path=None):
    return _plot_profile(tc, err, "t",
                         "Error por tiempo (control: deberia ser ~plano)", path)
