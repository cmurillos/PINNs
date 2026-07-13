"""Configuración base congelada y contrato de salidas (§0 de docs/SIMULACIONES.md).

Un solo lugar para:
  - BASE_CONFIG (§0.1): la configuración de red/entrenamiento compartida por
    TODOS los experimentos (presupuesto fijo, no convergencia). Se vuelca a
    results/tabla1.csv (Tabla 1 del artículo) con `volcar_tabla1`.
  - SEEDS_RED / SEEDS_RUIDO (§0.2): listas fijas e independientes; generadores
    separados por rol (numpy.random.default_rng(seed)), nunca el global.
  - raw.csv (§0.4): una fila por corrida, columnas fijas COLUMNAS_RAW; los
    campos no aplicables van VACÍOS (no cero) y nunca se sobreescribe con
    agregados (media ± std se calcula al graficar).
  - modelos entrenados en results/exN/models/ para regenerar figuras sin
    reentrenar (incluida la visualización 5.7).

Las rutas son relativas al directorio de trabajo (ejecutar desde la raíz del
repositorio): results/exN/... y figs/....
"""

import csv
import os

from .pinn import LateralCauchyCylinder

# ---------------------------------------------------- §0.1 configuración base
# Leída de los defaults de la clase PINN (pinn.py); congelada aquí para la
# Tabla 1. Cada ex solo desvía donde su sección del spec lo indique.
BASE_CONFIG = {
    "capas": 4,              # capas ocultas de la MLP [4, 96, 96, 96, 96, 1]
    "ancho": 96,
    "N_int": 8000,
    "N_sup": 3000,
    "N_lat": 2000,
    "lambda_pde": 1.0,
    "lambda_g": 10.0,
    "lambda_f": 10.0,
    "lambda_lat": 1.0,
    "iter_adam": 15000,
    "iter_lbfgs": 3000,
    "lr_adam": 1e-3,
}

# ------------------------------------------------------------- §0.2 semillas
SEEDS_RED = [0, 1, 2, 3, 4]
SEEDS_RUIDO = [100, 101, 102, 103, 104]

# ------------------------------------------------- §0.4 contrato de salidas
COLUMNAS_RAW = ["experimento", "caso", "semilla_red", "semilla_ruido",
                "eta", "a", "omega", "L", "E0", "err_global", "tiempo_s"]

RESULTS_DIR = "results"
FIGS_DIR = "figs"


def fit_opts(smoke=False):
    """opts de op.fit según BASE_CONFIG; `smoke` reduce el presupuesto para
    correr de punta a punta en CI (misma estructura, pocas iteraciones)."""
    o = {
        "weights": (BASE_CONFIG["lambda_pde"], BASE_CONFIG["lambda_g"],
                    BASE_CONFIG["lambda_f"], BASE_CONFIG["lambda_lat"]),
        "adam_iters": BASE_CONFIG["iter_adam"],
        "lbfgs_iters": BASE_CONFIG["iter_lbfgs"],
        "lr": BASE_CONFIG["lr_adam"],
        "n_int": BASE_CONFIG["N_int"],
        "n_top": BASE_CONFIG["N_sup"],
        "n_lat": BASE_CONFIG["N_lat"],
    }
    if smoke:
        o.update(adam_iters=150, lbfgs_iters=30,
                 n_int=1200, n_top=500, n_lat=300)
    return o


def net_config(seed):
    """net_config de la clase PINN para una semilla de red (§0.2)."""
    capas = [4] + [BASE_CONFIG["ancho"]] * BASE_CONFIG["capas"] + [1]
    return {"layers": capas, "seed": int(seed)}


# ------------------------------------------------------------------ raw.csv
def dir_resultados(experimento):
    d = os.path.join(RESULTS_DIR, experimento)
    os.makedirs(d, exist_ok=True)
    return d


def escribir_raw(experimento, **campos):
    """Añade UNA fila a results/<experimento>/raw.csv (crea el archivo con la
    cabecera fija si no existe). Campos no pasados o None quedan vacíos."""
    desconocidos = set(campos) - set(COLUMNAS_RAW)
    if desconocidos:
        raise ValueError(f"columnas fuera del contrato §0.4: {sorted(desconocidos)}")
    path = os.path.join(dir_resultados(experimento), "raw.csv")
    nuevo = not os.path.exists(path)
    fila = {"experimento": experimento, **campos}
    with open(path, "a", newline="", encoding="utf8") as fh:
        wr = csv.DictWriter(fh, fieldnames=COLUMNAS_RAW)
        if nuevo:
            wr.writeheader()
        wr.writerow({c: ("" if fila.get(c) is None else fila.get(c, ""))
                     for c in COLUMNAS_RAW})
    return path


def leer_raw(experimento):
    """Lee results/<experimento>/raw.csv como lista de dicts (strings crudos;
    campos vacíos = no aplicables)."""
    path = os.path.join(RESULTS_DIR, experimento, "raw.csv")
    with open(path, newline="", encoding="utf8") as fh:
        return list(csv.DictReader(fh))


# ------------------------------------------------------------------- modelos
def ruta_modelo(experimento, nombre):
    d = os.path.join(dir_resultados(experimento), "models")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{nombre}.pt")


def guardar_modelo(op, experimento, nombre):
    """op.save en results/<experimento>/models/<nombre>.pt (§0.4)."""
    path = ruta_modelo(experimento, nombre)
    op.save(path)
    return path


def cargar_modelo(experimento, nombre, rho, c, k, map_location=None):
    """LateralCauchyCylinder.load desde results/<experimento>/models/.
    Requiere el MISMO medio (rho, c, k) con que se entrenó."""
    return LateralCauchyCylinder.load(ruta_modelo(experimento, nombre),
                                      rho, c, k, map_location=map_location)


# -------------------------------------------------------------- Tabla 1
def volcar_tabla1(path=None):
    """Escribe la configuración base a results/tabla1.csv (§0.1)."""
    path = path or os.path.join(RESULTS_DIR, "tabla1.csv")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(BASE_CONFIG))
        wr.writeheader()
        wr.writerow(BASE_CONFIG)
    return path
