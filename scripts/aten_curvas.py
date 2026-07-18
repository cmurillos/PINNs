"""Figura de la §3.3 (envoltorio CLI de experiments.aten_curvas)."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from lateralcauchy import experiments

if __name__ == "__main__":
    experiments.aten_curvas()
