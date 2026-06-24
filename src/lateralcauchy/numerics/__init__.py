"""Metodos numericos de referencia para la PDE del calor en el cilindro.

Solver independiente (numpy/scipy) para comparar contra la PINN:
  - disk_modes : funciones propias de Neumann del disco (Bessel).
  - heat1d     : solver 1D en (z,t) por modo (Crank-Nicolson).
  - reference  : solver 3D+t por superposicion de modos; extrae datos de Cauchy.
  - manufactured : soluciones exactas para validar solver y PINN.
"""

from .disk_modes import DiskMode
from .heat1d import Heat1D
from .reference import ReferenceSolution
from .manufactured import ManufacturedZ, ManufacturedBessel

__all__ = ["DiskMode", "Heat1D", "ReferenceSolution",
           "ManufacturedZ", "ManufacturedBessel"]
