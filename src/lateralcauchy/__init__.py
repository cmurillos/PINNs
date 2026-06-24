"""LateralCauchyCylinder: PINN para el problema de Cauchy lateral del calor en un
cilindro, con un solver numerico de referencia para validacion.

Submodulos:
  - pinn         : la clase LateralCauchyCylinder (PINN en PyTorch).
  - numerics     : solver numerico independiente (numpy/scipy) + soluciones exactas.
  - metrics      : metricas de comparacion y puente numpy<->torch.
  - diagnostics  : graficas y diagnostico L/delta (import explicito; usa matplotlib).
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
