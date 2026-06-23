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

### Archivos

| Archivo                        | Contenido                                              |
|--------------------------------|--------------------------------------------------------|
| `lateral_cauchy_cylinder.py`   | clase `LateralCauchyCylinder` (la PINN)                |
| `validate.py`                  | validación con solución manufacturada (§6 de CLAUDE.md)|
| `CLAUDE.md`                    | especificación completa del proyecto                   |

### Instalación y validación

```bash
pip install -r requirements.txt
python validate.py        # imprime el error relativo de grad_T
```

Con los defaults (`adam_iters=15000`, `lbfgs_iters=3000`) el error relativo de
`grad_T` en el caso manufacturado es pequeño; con una corrida reducida
(`python -c "import validate; validate.main(adam_iters=400, lbfgs_iters=200)"`)
ya baja del orden de `1e-2`.
