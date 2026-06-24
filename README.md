# LateralCauchyCylinder

Physics-Informed Neural Networks (PINNs) para el **problema de Cauchy lateral**
de la ecuación de calor en un cilindro con medio heterogéneo:

```
rho(x) c(x) dT/dt = div( k(x) grad T )      en  Q = D x (0,L) x (0,Tmax]
```

con dato de Cauchy `(g, f)` en la tapa `z=L` (ambos como pérdida blanda), Neumann
homogéneo en la pared lateral, base `z=0` libre y sin condición inicial. Tras
entrenar, la PINN expone el campo gradiente espacial `grad_T` como objeto
invocable sobre todo el cilindro espacio-temporal.

El problema es **mal puesto exponencialmente**; eso gobierna las decisiones de
diseño (`tanh`, doble precisión, sin Fourier features, normalización interna).
Ver `CLAUDE.md` para el enunciado matemático y de software completo.

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
examples/                   comparaciones PINN <-> referencia (ex1..ex5)
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
(perfiles por capas), exactamente la heterogeneidad prevista en `CLAUDE.md` §8.

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

`lateralcauchy.diagnostics` ofrece gráficas (`plot_history`, `plot_error_vs_z`) y
el diagnóstico **`L/δ`** (`ld_ratio`, con `δ=√(2α/ω)`) que predice *antes* de
entrenar si el régimen es recuperable (`L/δ ≲ 1`). La integración continua
(`.github/workflows/ci.yml`) corre la suite en cada push y pull request.
