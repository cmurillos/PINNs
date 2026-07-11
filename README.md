# LateralCauchyCylinder

[![CI](https://github.com/cmurillos/PINNs/actions/workflows/ci.yml/badge.svg)](https://github.com/cmurillos/PINNs/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Realización numérica (PINN, PyTorch) del **operador de continuación lateral**
para la ecuación de calor en un cilindro heterogéneo:

```
Λ : (g, f) ↦ T,        rho(x) c(x) ∂t T = ∇·( k(x) ∇T )   en  Ω × (0, Tmax]
```

donde `Ω = D × (0,L)` (disco `D` de radio `R`), con los **dos datos de Cauchy**
sobre la tapa: `T = g` y `−k ∂z T = f` en `Γ_sup = D × {L}` (ambos como pérdida
blanda), flujo nulo `∂n T = 0` en la pared lateral, **base `z=0` libre y sin
condición inicial**. Tras entrenar, la clase expone `T = Λ(g,f)` y sus vistas
derivadas `grad_T` (el objetivo práctico) y `flux = −k∇T`, invocables sobre todo
el cilindro espacio-temporal.

El planteamiento riguroso está en
[`docs/planteamiento_pde.pdf`](docs/planteamiento_pde.pdf):

- **Unicidad (Proposición 1).** En la clase `H^{2,1}_loc((Ω ∪ Γ_sup) × (0,Tmax])`
  hay a lo más una solución, luego `Λ` está bien definido. La prueba extiende por
  cero a través de la tapa — **no característica**, pues el símbolo principal es
  `k|ξ|²` — y aplica el principio de **continuación única espacial** para
  operadores parabólicos con coeficiente principal Lipschitz (estimaciones de
  Carleman: Saut–Scheurer 1987; Escauriaza–Fernández 2003; Escauriaza–Vessella
  2003). Ni la condición lateral ni condición inicial alguna intervienen: la CI
  queda determinada por el dato (Puzyrev–Shlapunov 2012).
- **Existencia — condicional.** El conjunto de datos `(g,f)` admisibles es denso
  pero no cerrado; no se aborda.
- **Estabilidad — NO.** La continuación de `z=L` a `z=0` amplifica cada modo con
  factor `~exp(L·Re√(|ξ'|² + iωρc/k))`: **mal puesto exponencialmente**. Esto
  gobierna las decisiones de diseño (`tanh`, doble precisión, sin Fourier
  features, normalización interna, `g` y `f` blandos).

> **Alcance.** Estudio **determinista** de métodos y verificación: el operador de
> continuación lateral parabólico realizado como software y verificado contra un
> solver de referencia independiente. Sin datos de campo, sin componente
> estocástica. El número `L/δ` se reporta como *distancia de continuación*
> recuperable, no como profundidad física.

## Instalación

```bash
pip install -e ".[dev]"     # paquete + dependencias de desarrollo (pytest)
```

## Estructura del repositorio

```
src/lateralcauchy/          paquete instalable
  pinn.py                   clase LateralCauchyCylinder (la PINN, PyTorch)
  numerics/                 solver numérico de referencia (numpy/scipy)
    disk_modes.py           funciones propias de Neumann del disco (Bessel)
    heat1d.py               solver 1D en (z,t) por modo (Crank-Nicolson)
    reference.py            solver 3D+t por superposición; extrae datos de Cauchy
    manufactured.py         soluciones exactas (validan solver y PINN)
  metrics.py                error relativo, muestreo, error-vs-z, puente numpy<->torch
  diagnostics.py            gráficas y diagnóstico L/delta (matplotlib)
examples/                   comparaciones PINN <-> referencia (ex1..ex6)
docs/planteamiento_pde.pdf  planteamiento riguroso (operador, clase, unicidad)
tests/                      suite pytest
pyproject.toml              metadatos y dependencias del paquete
CLAUDE.md                   especificación completa del proyecto
```

## Uso

```python
import math, torch
from lateralcauchy import LateralCauchyCylinder

RHOC, K = 1.0, 1.0
rho = lambda X: torch.ones_like(X[:, :1])
c   = lambda X: RHOC * torch.ones_like(X[:, :1])
k   = lambda X: K * torch.ones_like(X[:, :1])

LAM, PSI, L = 2.0, 0.7, 1.0
BETA = math.sqrt(RHOC * LAM / K)
g = lambda X: torch.exp(-LAM * X[:, 3:4]) * math.cos(BETA * L + PSI)
f = lambda X: K * BETA * math.sin(BETA * L + PSI) * torch.exp(-LAM * X[:, 3:4])

op = LateralCauchyCylinder(R=1.0, L=L, Tmax=1.0, rho=rho, c=c, k=k)
history = op.fit(g, f)

X = torch.rand(100, 4)   # puntos [x, y, z, t]
G = op.grad_T(X)         # (100, 3) -> (dT/dx, dT/dy, dT/dz)
```

## Sistema de validación numérica

La PINN resuelve un problema **inverso mal puesto**. Para comprobar que su `∇T`
es correcto se incluye un **solver de referencia independiente**
(`lateralcauchy.numerics`) que resuelve el problema *directo* (bien puesto), del
que se extraen los datos de Cauchy `(g, f)` que alimentan a la PINN.

| Ejemplo                                | Compara                                  | Qué verifica                                   |
|----------------------------------------|------------------------------------------|------------------------------------------------|
| `examples/ex1_manufactured.py`         | PINN ↔ solución exacta (sin frec. espac.)| sanity check de la maquinaria (§6)             |
| `examples/ex2_bessel.py`               | PINN ↔ modo de Bessel; solver ↔ exacta   | estrés con frecuencia espacial en `(x,y)`      |
| `examples/ex3_solver_heterogeneous.py` | PINN ↔ solver numérico, `k(z)` variable  | medio heterogéneo, **sin** solución analítica  |
| `examples/ex4_noise.py`                | PINN con datos `(g,f)` ruidosos          | robustez al ruido (la inversión lo amplifica)  |
| `examples/ex5_frequency_sweep.py`      | PINN sobre modos de frecuencia creciente | mapa de degradación vs `exp(L·γ)`              |
| `examples/ex6_weight_sensitivity.py`   | barrido de `(λ_g, λ_f)`                   | sensibilidad del resultado al balance de pesos |

```bash
python -m examples.ex1_manufactured            # sanity check
python -m examples.ex2_bessel                  # frecuencia espacial + error-vs-z
python -m examples.ex3_solver_heterogeneous    # medio k(z) heterogéneo
```

Cada ejemplo acepta un régimen reducido para correr rápido:

```python
from examples.ex2_bessel import main
main(adam_iters=600, lbfgs_iters=200)          # defaults completos: 15000 / 3000
```

El solver de referencia es válido para medios `ρc, k` dependientes solo de `z`
(perfiles por capas), exactamente la heterogeneidad del medio descrita en
`CLAUDE.md` §1.2.

### Notas metodológicas (importantes para interpretar los resultados)

- **Métrica principal = error en la base `z=0`.** Es la recuperación *genuina*: la
  PINN nunca ve `z=0` (lo continúa desde la tapa). El error promediado sobre todo
  `Ω` está dominado por la zona fácil cerca de la tapa y **vende de más**; los
  ejemplos reportan el error en `z=0` como número principal.
- **`error_vs_t` es un control, no un headline.** La continuación es espacial (en
  `z`), no temporal (`CLAUDE.md` §1.6.2), así que el perfil en `t` debe salir
  **≈plano**. Un pico en `t=0` sería efecto de borde de colocación, no la
  inversión mal puesta.
- **Sesgo optimista por banda limitada.** Los datos `(g,f)` sintéticos son una
  superposición de **pocos** modos de Bessel (banda limitada); recuperar eso es
  más fácil que una `g` real de espectro ancho. La validación sintética es
  **optimista** respecto a datos reales — los ejemplos indican cuántos modos usan.
- **Entrenamiento.** Adam con paso explícito (un registro de pérdida por
  iteración) y resampleo de colocación cada `resample_every` iters (opción de
  `fit`, 0 = off); L-BFGS con colocación **fija** y un registro por iteración
  aceptada (nunca se resamplea en L-BFGS: rompería su aproximación del Hessiano).

## Tests y diagnóstico

```bash
pytest -q          # suite de tests (modos, solver, PINN, métricas)
```

`lateralcauchy.diagnostics` ofrece el diagnóstico **`L/δ`** (`ld_ratio`, con
`δ=√(2α/ω)`) que predice *antes* de entrenar si el régimen es recuperable
(`L/δ ≲ 1`). La integración continua (`.github/workflows/ci.yml`) corre la
suite en cada push y pull request.

### Validación integrada y figuras listas para artículo

Tras `fit`, la validación completa contra cualquier verdad (exacta o solver de
referencia) es un método de la clase:

```python
rep = op.validate(exacta)        # exacta o ReferenceSolution (algo con .grad_T)
rep["err_base"]                  # e(z=0)  ← métrica principal
rep["err_global"], rep["z"], rep["t"]   # global y perfiles e(z), e(t)
op.history                       # pérdidas de la última corrida
op.plot_history("figs/history")  # figura de entrenamiento
```

`lateralcauchy.plotting` genera todas las figuras en **modo publicación**
(serif + mathtext, tamaños de columna de revista `SINGLE_COL`/`DOUBLE_COL`,
paleta Okabe–Ito segura para daltonismo, sin títulos embebidos) y cada `path`
exporta **PDF vectorial** (para `\includegraphics`) **+ PNG 300 dpi**:

```python
from lateralcauchy import plotting as pl

pl.plot_error_vs_z(*rep["z"], path="figs/error_z")       # firma e(z)
pl.plot_error_vs_t(*rep["t"], path="figs/error_t")       # control (≈ plano)
pl.plot_gradient_profile(z, dzT_pinn, dzT_ref, path="figs/perfil")
F, ext = pl.eval_slice(lambda X: op.T(torch.as_tensor(X)).numpy(), R, L, t0=0.5)
G, _   = pl.eval_slice(ref.T, R, L, t0=0.5)
pl.plot_slice(F, G, ext, path="figs/corte")              # PINN | ref | |dif|
pl.plot_frequency_sweep(gammas, errs, L, path="figs/sweep")  # vs e^{Lγ} teórico
pl.plot_heatmap(grid, ruidos, modos, "ruido", "modo", path="figs/mapa")
pl.latex_table(filas, header, path="figs/tabla.tex")     # booktabs, \input-able
```

### Guardar y cargar un modelo entrenado

```python
op.fit(g, f)
op.save("modelo.pt")

# mas tarde: el medio (rho, c, k) hay que volver a pasarlo (define el operador)
op2 = LateralCauchyCylinder.load("modelo.pt", rho, c, k)
G = op2.grad_T(X)
```

### Benchmark reproducible

`scripts/benchmark.py` entrena varias semillas y reporta el error de `grad_T` en
la base `z=0` como `media ± std` (cifras defendibles, no una sola corrida):

```bash
python scripts/benchmark.py        # guarda figuras en scripts/figs/
```

## Licencia

MIT — ver [`LICENSE`](LICENSE).
