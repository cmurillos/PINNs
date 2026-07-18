"""ex2 - Frecuencia espacial, modo de Bessel (envoltorio CLI de experiments.ex2)."""

import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from lateralcauchy import experiments

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="presupuesto reducido (CI)")
    experiments.ex2(smoke=ap.parse_args().smoke)
