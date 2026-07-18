"""Tests mínimos del módulo de experimentos (REFACTOR_EXPERIMENTOS PR-A)."""

import csv
import inspect
import math

from lateralcauchy import experiments as ex
from lateralcauchy.runconfig import COLUMNAS_RAW, SEEDS_RED, SEEDS_RUIDO


def _defaults(fn):
    return {k: v.default for k, v in inspect.signature(fn).parameters.items()
            if v.default is not inspect.Parameter.empty}


def test_las_siete_funciones_existen():
    for nombre in ("ex1", "ex2", "ex3", "ex4", "ex5", "ex6", "visualizacion"):
        assert callable(getattr(ex, nombre))


def test_defaults_del_spec():
    assert _defaults(ex.ex1)["seeds"] == SEEDS_RED

    d2 = _defaults(ex.ex2)
    # spec: a ~ e^-1 ~ 0.37 (el default exacto es exp(-1))
    assert abs(d2["a_objetivo"] - 0.37) < 0.01
    assert d2["a_objetivo"] == math.exp(-1.0)
    assert d2["modo"] == (0, 1) and d2["seeds"] == SEEDS_RED

    d3 = _defaults(ex.ex3)
    assert d3["k_perfil"] is None            # None => k(z)=1+z del spec
    assert abs(d3["a_objetivo"] - 0.37) < 0.01

    d4 = _defaults(ex.ex4)
    assert d4["etas"] == (0.0, 1e-3, 1e-2, 5e-2)
    assert d4["seeds_ruido"] == SEEDS_RUIDO[:5]
    assert d4["seeds_red"] == SEEDS_RED[:3]

    d5 = _defaults(ex.ex5)
    assert d5["a_objetivos"] == (0.7, 0.5, 0.37, 0.2, 0.1, 0.03, 0.01)
    assert d5["Ls"] == (0.5, 0.75, 1.0, 1.5, 2.0)
    assert d5["eta_ruido"] == 1e-2
    assert d5["seeds_frec"] == SEEDS_RED and d5["seeds_L"] == SEEDS_RED[:3]

    d6 = _defaults(ex.ex6)
    assert d6["lambdas"] == (0.1, 1.0, 10.0) and d6["seeds"] == SEEDS_RED[:3]

    dv = _defaults(ex.visualizacion)
    assert dv["experimento"] == "ex3" and dv["modelo"] == "seed0"
    assert dv["tiempos"] == (0.2, 0.4, 0.6, 0.8)

    for fn in (ex.ex1, ex.ex2, ex.ex3, ex.ex4, ex.ex5, ex.ex6,
               ex.visualizacion):
        assert _defaults(fn)["outdir"] is None


def test_ex1_presupuesto_infimo_punta_a_punta(tmp_path):
    out = ex.ex1(seeds=[0], smoke=True, outdir=str(tmp_path),
                 opts={"adam_iters": 20, "lbfgs_iters": 0,
                       "n_int": 200, "n_top": 100, "n_lat": 50})
    with open(out["raw"], newline="") as fh:
        filas = list(csv.reader(fh))
    assert filas[0] == COLUMNAS_RAW                  # esquema §0.4 intacto
    assert len(filas) == 2                           # 1 corrida = 1 fila
    fila = dict(zip(filas[0], filas[1]))
    assert fila["experimento"] == "ex1" and fila["semilla_red"] == "0"
    assert float(fila["E0"]) > 0 and fila["eta"] == ""   # no aplicable: vacio
    assert str(tmp_path) in out["raw"]               # respeta outdir
    assert out["resumen"]["manufacturada"]["n"] == 1
    for f in out["figs"]:
        assert str(tmp_path) in f
        import os
        assert os.path.exists(f)
