# PINNs

Physics-Informed Neural Networks.

## `LateralCauchyCylinder`

PINN (PyTorch) que resuelve el **problema de Cauchy lateral** para la ecuación de
calor en un cilindro con medio heterogéneo:

```
rho(x) c(x) dT/dt = div( k(x) grad T )      en  Q = D x (0,L) x (0,Tmax]
```

con dato de Cauchy `(g, f)` en la tapa `z=L` (ambos como pérdida blanda), Neumann
homogéneo en la pared lateral, base `z=0` libre y sin condición inicial. Tras
entrenar expone el campo gradiente espacial `grad_T` como objeto invocable sobre
todo el cilindro espacio-temporal.

El problema es **mal puesto exponencialmente**; eso gobierna las decisiones de
diseño (`tanh`, doble precisión, sin Fourier features, normalización interna).
Ver `CLAUDE.md` para el enunciado matemático y de software completo.

### Uso

```python
import math, torch
from lateral_cauchy_cylinder import LateralCauchyCylinder

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

### Estructura del repositorio

| Ruta                          | Contenido                                                       |
|-------------------------------|-----------------------------------------------------------------|
| `lateral_cauchy_cylinder.py`  | clase `LateralCauchyCylinder` (la **PINN**)                     |
| `numerics/`                   | **solver numérico de referencia** (independiente, numpy/scipy)  |
| `numerics/disk_modes.py`      | funciones propias de Neumann del disco (Bessel)                 |
| `numerics/heat1d.py`          | solver 1D en `(z,t)` por modo (Crank–Nicolson, forma conserv.)  |
| `numerics/reference.py`       | solver 3D+t por superposición de modos; extrae datos de Cauchy  |
| `numerics/manufactured.py`    | soluciones exactas (para validar solver **y** PINN)             |
| `metrics.py`                  | error relativo, muestreo, error-vs-`z`, puente numpy↔torch       |
| `examples/`                   | comparaciones PINN ↔ referencia                                 |
| `CLAUDE.md`                   | especificación completa del proyecto                            |

### Sistema de validación numérica

La PINN resuelve un problema **inverso mal puesto**. Para comprobar que su `∇T`
es correcto se incluye un **solver de referencia independiente** (`numerics/`)
que resuelve el problema *directo* (bien puesto), del que se extraen los datos
de Cauchy `(g, f)` que alimentan a la PINN. Hay tres validaciones encadenadas:

| Ejemplo                                | Compara                                  | Qué verifica                                   |
|----------------------------------------|------------------------------------------|------------------------------------------------|
| `examples/ex1_manufactured.py`         | PINN ↔ solución exacta (sin frec. espac.)| sanity check de la maquinaria (§6)             |
| `examples/ex2_bessel.py`               | PINN ↔ modo de Bessel; solver ↔ exacta   | estrés con frecuencia espacial en `(x,y)`      |
| `examples/ex3_solver_heterogeneous.py` | PINN ↔ solver numérico, `k(z)` variable  | medio heterogéneo, **sin** solución analítica  |
| `examples/ex4_noise.py`                | PINN con datos `(g,f)` ruidosos          | robustez al ruido (la inversión lo amplifica)  |
| `examples/ex5_frequency_sweep.py`      | PINN sobre modos de frecuencia creciente | mapa de degradación vs `exp(L·γ)`              |

El solver de referencia es válido para medios `ρc, k` dependientes solo de `z`
(perfiles por capas), exactamente la heterogeneidad prevista en `CLAUDE.md` §8.

### Tests y diagnóstico

```bash
pip install -r requirements-dev.txt
pytest tests/ -q          # suite de tests (modos, solver, PINN, métricas)
```

`diagnostics.py` ofrece gráficas (`plot_history`, `plot_error_vs_z`) y el
diagnóstico **`L/δ`** (`ld_ratio`, con `δ=√(2α/ω)`) que predice *antes* de
entrenar si el régimen es recuperable (`L/δ ≲ 1`). La integración continua
(`.github/workflows/ci.yml`) corre la suite en cada push y pull request.

### Instalación y ejecución

```bash
pip install -r requirements.txt
python -m examples.ex1_manufactured     # sanity check
python -m examples.ex2_bessel           # test con frecuencia espacial
python -m examples.ex3_solver_heterogeneous   # medio k(z) heterogéneo
```

Cada ejemplo acepta un régimen reducido para correr rápido, p. ej.:

```python
from examples.ex2_bessel import main
main(adam_iters=600, lbfgs_iters=200)   # defaults completos: 15000 / 3000
```
