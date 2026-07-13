"""Diagnóstico de atenuación (§1 de docs/SIMULACIONES.md) y compatibilidad.

Infraestructura del artículo:
  - atenuacion(omega, medio, L)   : BVP complejo (3.4) por diferencias finitas
    conservativas; a = |U(L)| es la atenuación del armónico temporal a lo largo
    del cilindro. Su inversa 1/a es el factor de amplificación de la
    continuación tapa→base (gobierna ex2–ex5).
  - omega_efectiva(g, dt)         : ec. (3.9), ω_ef = ‖∂ₜg‖_L2 / ‖g‖_L2.
  - regimen(g, medio, L, eta)     : diagnóstico de la §4.4; junta las dos
    anteriores y predice si el dato es recuperable (a > η).

`medio` es el par de callables `(rhoc, k)` sobre z (mismo contrato que el
solver de referencia: z array → array), o un dict {"rhoc": ..., "k": ...}.

skin_depth / ld_ratio (el diagnóstico L/δ anterior) quedan DEPRECADAS: la curva
medida `atenuacion` reemplaza a la estimación de semiespacio. Las gráficas
siguen re-exportadas desde `plotting` por compatibilidad.
"""

import math
import warnings

import numpy as np
from scipy.linalg import solve_banded

from .plotting import (                                      # noqa: F401
    plot_history, plot_error_vs_z, plot_error_vs_t,
)


def _medio(medio):
    """Normaliza `medio` a (rhoc, k): tupla/lista o dict {"rhoc","k"}."""
    if isinstance(medio, dict):
        return medio["rhoc"], medio["k"]
    rhoc, k = medio
    return rhoc, k


def perfil_armonico(omega, medio, L, n_z=2000, mu=0.0):
    """Perfil complejo U(z) del armónico temporal e^{iωt} a lo largo de z.

    Resuelve el BVP complejo (ec. (3.4) del artículo; con μ>0 sirve además para
    construir la solución periódica exacta del problema 1D reducido de un modo
    del disco, usada por ex2/ex3):

        i·ω·ρc(z)·U = (k(z)·U')' − μ·k(z)·U   en (0,L),   U(0)=1,   U'(L)=0,

    por diferencias finitas centradas sobre malla uniforme de n_z nodos, con
    (k U')' en forma conservativa (k en los puntos medios) y U'(L)=0 por nodo
    fantasma (cierre simétrico, segundo orden). Sistema tridiagonal complejo
    resuelto con scipy.linalg.solve_banded. Devuelve (z, U) con U complex128.
    """
    rhoc, k = _medio(medio)
    z = np.linspace(0.0, float(L), int(n_z))
    h = z[1] - z[0]
    kf = np.asarray(k(0.5 * (z[:-1] + z[1:])), dtype=float)   # k en las caras
    kn = np.asarray(k(z), dtype=float)
    rc = np.asarray(rhoc(z), dtype=float)

    n = int(n_z)
    diag = np.zeros(n, dtype=complex)
    sup = np.zeros(n, dtype=complex)     # coef. de U_{j+1} en la fila j
    sub = np.zeros(n, dtype=complex)     # coef. de U_{j-1} en la fila j
    b = np.zeros(n, dtype=complex)

    diag[0] = 1.0                        # U(0) = 1
    b[0] = 1.0
    j = np.arange(1, n - 1)
    sub[j] = kf[j - 1] / h ** 2
    sup[j] = kf[j] / h ** 2
    diag[j] = -(kf[j - 1] + kf[j]) / h ** 2 - mu * kn[j] - 1j * omega * rc[j]
    # z = L: nodo fantasma U_{n} = U_{n-2}  (U'(L)=0, reflexión simétrica)
    sub[-1] = 2.0 * kf[-1] / h ** 2
    diag[-1] = -2.0 * kf[-1] / h ** 2 - mu * kn[-1] - 1j * omega * rc[-1]

    ab = np.zeros((3, n), dtype=complex)
    ab[0, 1:] = sup[:-1]
    ab[1, :] = diag
    ab[2, :-1] = sub[1:]
    U = solve_banded((1, 1), ab, b)
    return z, U


def atenuacion(omega, medio, L, n_z=2000):
    """a(ω) = |U(L)| del BVP complejo (3.4): atenuación base→tapa del armónico.

    Para medio constante (α=1) la exacta del intervalo es a = |1/cosh(μL)| con
    μ = √(iω/α); la exponencial del semiespacio exp(−L√(ω/2)) es cota inferior.
    """
    _, U = perfil_armonico(omega, medio, L, n_z=n_z)
    return float(abs(U[-1]))


def omega_efectiva(g_muestreado, dt):
    """ω_ef = ‖∂ₜg‖_L2 / ‖g‖_L2  (ec. (3.9)) sobre una señal muestreada.

    ∂ₜg por diferencias centradas (unilaterales de 2º orden en los bordes) y
    normas discretas con peso trapezoidal.
    """
    g = np.asarray(g_muestreado, dtype=float).ravel()
    dg = np.gradient(g, dt)
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(math.sqrt(trapz(dg ** 2, dx=dt) / trapz(g ** 2, dx=dt)))


def regimen(g, medio, L, eta, dt=None):
    """Diagnóstico de la §4.4: ¿es recuperable el dato ANTES de entrenar?

    g : señal temporal muestreada — tupla (muestras, dt) o array (con dt aparte).
    Devuelve {omega_ef, a, margen: a/eta, recuperable: a > eta}.
    """
    if isinstance(g, (tuple, list)) and len(g) == 2 and np.isscalar(g[1]):
        g, dt = g
    if dt is None:
        raise ValueError("regimen: falta dt (pasar g=(muestras, dt) o dt=...).")
    w = omega_efectiva(g, dt)
    a = atenuacion(w, medio, L)
    margen = a / eta if eta > 0 else float("inf")
    return {"omega_ef": w, "a": a, "margen": float(margen),
            "recuperable": bool(a > eta)}


# ------------------------------------------------------------- DEPRECADAS
def skin_depth(alpha, omega):
    """DEPRECADA: usar `atenuacion` (curva medida del BVP (3.4)).

    δ(α, ω) = √(2α/ω), la longitud de penetración del semiespacio."""
    warnings.warn("skin_depth está deprecada; usar diagnostics.atenuacion "
                  "(curva medida del BVP complejo).", DeprecationWarning,
                  stacklevel=2)
    return math.sqrt(2.0 * alpha / omega)


def ld_ratio(alpha, omega, L):
    """DEPRECADA: usar `atenuacion` (a = |U(L)| reemplaza la heurística L/δ)."""
    warnings.warn("ld_ratio está deprecada; usar diagnostics.atenuacion "
                  "(curva medida del BVP complejo).", DeprecationWarning,
                  stacklevel=2)
    return L / math.sqrt(2.0 * alpha / omega)
