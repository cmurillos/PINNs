"""Vuelca la configuración base congelada (§0.1 de docs/SIMULACIONES.md) a
results/tabla1.csv — el CSV que llena la Tabla 1 del artículo.

Ejecutar desde la raíz del repositorio:  python scripts/tabla1.py
"""

import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from lateralcauchy.runconfig import volcar_tabla1

if __name__ == "__main__":
    print(f"[tabla1] escrita en {volcar_tabla1()}")
