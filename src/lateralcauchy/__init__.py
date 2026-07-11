"""Realización PINN del operador de continuación lateral Λ : (g, f) ↦ T para la
ecuación de calor ρc ∂ₜT = ∇·(k∇T) en un cilindro heterogéneo, con solver
numérico de referencia para validación (docs/planteamiento_pde.pdf).

Submódulos:
  - pinn         : la clase LateralCauchyCylinder (el operador Λ, PyTorch).
  - numerics     : solver de referencia independiente (numpy/scipy) + exactas.
  - metrics      : métricas ‖·‖ de comparación y puente numpy↔torch.
  - diagnostics  : gráficas y diagnóstico L/δ (import explícito; matplotlib).
"""

from .pinn import LateralCauchyCylinder
from .numerics import (
    DiskMode, Heat1D, ReferenceSolution, ManufacturedZ, ManufacturedBessel,
)
from . import metrics

__all__ = [
    "LateralCauchyCylinder",
    "DiskMode", "Heat1D", "ReferenceSolution",
    "ManufacturedZ", "ManufacturedBessel",
    "metrics",
]

__version__ = "0.1.0"
