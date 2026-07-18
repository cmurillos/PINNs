"""Visualizacion 5.7 (envoltorio CLI de experiments.visualizacion; carga
el modelo guardado de ex3, no reentrena)."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from lateralcauchy import experiments

if __name__ == "__main__":
    experiments.visualizacion()
