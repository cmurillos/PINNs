"""ex6 - Sensibilidad al balance de pesos (envoltorio CLI de experiments.ex6)."""

import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from lateralcauchy import experiments

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="presupuesto reducido (CI)")
    experiments.ex6(smoke=ap.parse_args().smoke)
