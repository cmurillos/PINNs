# CLAUDE.md — Proyecto `LateralCauchyCylinder`

> Documento de contexto autocontenido. Quien lo lea **no tiene historial previo**:
> aquí está todo lo necesario para implementar el proyecto desde cero.

---

## 0. Resumen en una frase

Implementar una clase `LateralCauchyCylinder` que realiza, mediante una PINN
(PyTorch), el **operador de continuación lateral** `Λ : (g,f) ↦ T` para la
ecuación de calor en un cilindro con medio heterogéneo, y que tras entrenar
expone `T` y el campo **gradiente espacial** `∇T` (objetivo práctico) como
objetos invocables sobre el cilindro espacio-temporal.

**Planteamiento riguroso:** `docs/planteamiento_pde.pdf` — dominio, clase de
soluciones `H^{2,1}_loc`, demostración de unicidad (Proposición 1, vía
continuación única espacial con estimaciones de Carleman) y referencias. Este
documento es la fuente de verdad matemática; §1 de aquí es su resumen operativo.

**Alcance:** proyecto determinista (coeficientes dados, no aleatorios) de métodos
y verificación. No es un trabajo de aplicación geofísica ni de cuantificación de
incertidumbre.

---

## 1. Problema matemático (enunciado completo)

### 1.1 Dominio

- `D ⊂ ℝ²` = disco de radio `R` (la sección transversal del cilindro).
- Cilindro espacial: `Ω = D × (0, L) ⊂ ℝ³`, con coordenadas `x = (x, y, z)`, `z ∈ (0, L)`.
- Ventana temporal: `(0, Tmax]`.
- Cilindro espacio-temporal de trabajo: `Q = Ω × (0, Tmax]`.

Frontera de `Ω`, en tres piezas disjuntas:

| Pieza            | Definición          | Nombre        |
|------------------|---------------------|---------------|
| Tapa (superior)  | `z = L`             | `Γ_sup`       |
| Base (inferior)  | `z = 0`             | `Γ_0`         |
| Pared lateral    | `r = R`, `z ∈ (0,L)`| `Γ_lat`       |

### 1.2 Coeficientes (el medio)

Funciones dadas `ρ, c, k : Ω → ℝ`, con:

- `ρ·c ∈ L∞(Ω)`, `ρ·c ≥ γ₀ > 0`.
- `k` **Lipschitz** (`C^{0,1}`) es el umbral teórico para la continuación única;
  en la práctica se usan **polinomios** (`C^∞`), que además dan el doble autodiff
  limpio de §4.3. `k ≥ k₀ > 0`.

La suavidad de `k` **no es opcional**: ver §4.3 (forma divergencia).

### 1.3 Ecuación (calor en medio heterogéneo)

```
ρ(x) c(x) ∂ₜT = ∇·( k(x) ∇T )          en Q
```

`∇` y `∇·` son **espaciales** (en `x, y, z`).

### 1.4 Condiciones de frontera y temporal

| Frontera     | Condición                          | Rol                                  |
|--------------|------------------------------------|--------------------------------------|
| `Γ_sup` (z=L)| `T = g`  **y**  `-k ∂_z T = f`     | **dato de Cauchy** (los dos, blandos)|
| `Γ_lat` (r=R)| `∂_n T = 0`                        | Neumann homogéneo (flujo lateral nulo)|
| `Γ_0`  (z=0) | — ninguna —                        | **libre**                            |
| `t = 0`      | — ninguna —                        | **sin condición inicial**            |

- En la tapa se prescriben **DOS** condiciones (sobredeterminación de Cauchy):
  la traza `g` y el flujo normal `f`. Ambas entran como **pérdida blanda**
  (NUNCA `g` como Dirichlet duro — ver §4.2).
- En `Γ_lat`, `∂_n` es la derivada en la normal exterior, que para el disco es
  `n = (x/R, y/R, 0)`, de modo que `∂_n T = (x/R)∂_x T + (y/R)∂_y T`.
- La base es libre y NO hay condición inicial: ver §1.6.

### 1.5 El operador de continuación lateral

El problema es: dado el par admisible `(g, f)`, hallar `T` que satisfaga la PDE
con las condiciones de §1.4. Ello define el **operador de continuación lateral**

```
Λ : (g, f) ↦ T
```

(ec. (8) del planteamiento, `docs/planteamiento_pde.pdf`). El **objetivo práctico**
de la clase es el campo gradiente espacial derivado de él:

```
∇T = ∇(Λ(g,f)) : Ω × (0, Tmax] → ℝ³,   (x,y,z,t) ↦ (∂_x T, ∂_y T, ∂_z T)
```

en TODO el cilindro espacio-temporal (no solo en la base `z=0`).

- `∇T` es el gradiente **espacial**. NO incluye `∂_t T`.
- Todo depende de `t`: la entrada siempre es `(x, y, z, t)`.
- En el planteamiento el tiempo final se denota `T`; en el código es `Tmax`
  (para no colisionar con la temperatura `T`).

### 1.6 Estatus matemático del problema (3 hechos)

El enunciado riguroso completo, con la demostración de unicidad, está en
`docs/planteamiento_pde.pdf`. Resumen:

1. **Existencia — condicional.** No todo par `(g, f)` es admisible; solo los que
   provienen de la traza de Cauchy de una solución real de la PDE. El conjunto de
   datos admisibles es denso pero **no cerrado**; la existencia no se aborda.

2. **Unicidad — SÍ (Proposición 1 del planteamiento).** Se trabaja en la clase
   `T ∈ H^{2,1}_loc((Ω ∪ Γ_sup) × (0,Tmax])`, de modo que las trazas `T|_{z=L}` y
   `∂_z T|_{z=L}` están bien definidas. El sistema admite **a lo más una** solución
   en esa clase, luego `Λ` está bien definido sobre los datos admisibles.
   Esquema de la prueba (todo demostrado salvo el lema citado):
   - *Tapa no característica:* el símbolo principal de `P u = ρc ∂ₜu − ∇·(k∇u)` es
     `p = k|ξ|²`; sobre la conormal de `Γ_sup` vale `k ≥ k₀ > 0`. Las superficies
     características son exactamente los niveles `t = cte` — por eso el dato de
     Cauchy es admisible en la tapa y NO lo sería en `{t = 0}`.
   - *Extensión por cero:* la diferencia `w = T₁ − T₂` (dato de Cauchy nulo) se
     extiende por cero a un collar `D × (L, L+ε)`; el pegado en `H¹` y la fórmula
     de Green muestran que `w̃` es solución débil a través de la tapa (los
     coeficientes se extienden con McShane + truncación, preservando Lipschitz).
   - *Continuación única espacial (Lema 1, citado):* estimaciones de Carleman
     para operadores parabólicos con coeficiente principal Lipschitz
     [Saut–Scheurer; Escauriaza–Fernández; Escauriaza–Vessella] propagan
     `w̃ ≡ 0` del collar a todo el dominio conexo: `T₁ = T₂`.
   - **Ni la condición lateral (7) ni condición inicial alguna intervienen en la
     demostración**: la continuación es espacial en `z`, desde la tapa, y la CI
     queda determinada por el dato [Puzyrev–Shlapunov]. Imponer una CI
     sobredeterminaría el problema.
   Esta es la razón por la que la regularidad `k ∈ C^{0,1}` de §1.2 es el umbral
   teórico: es la hipótesis del lema de continuación única.

3. **Estabilidad — NO.** `Λ` es lineal, **cerrado, densamente definido y no
   acotado**. Sobre un modo de Fourier `(ξ', ω)` en `(x, y, t)`, la continuación
   de `z=L` a `z=0` amplifica con factor
   `~ exp( L · Re√(|ξ'|² + i ω ρc/k) )`,
   que crece sin cota en alta frecuencia. **Mal puesto exponencialmente.**
   Esto gobierna TODAS las decisiones de diseño (ver §3).

**Referencias del planteamiento**

1. J.-C. Saut, B. Scheurer, *Unique continuation for some evolution equations*,
   J. Differential Equations 66 (1987), 118–139.
2. L. Escauriaza, F. J. Fernández, *Unique continuation for parabolic operators*,
   Ark. Mat. 41 (2003), 35–60.
3. L. Escauriaza, S. Vessella, *Optimal three cylinder inequalities for solutions
   to parabolic equations with Lipschitz leading coefficients*, Contemp. Math. 333,
   AMS (2003), 79–87.
4. R. E. Puzyrev, A. A. Shlapunov, *On an ill-posed problem for the heat equation*,
   J. Siberian Federal Univ. Math. Phys. 5 (2012), no. 3, 337–348.

---

## 2. La clase `LateralCauchyCylinder` (arquitectura de software)

### 2.1 Concepto

La **instancia es el operador `Λ` para un medio fijo**. La geometría `(R, L, Tmax)`
y el medio `(ρ, c, k)` van en el constructor porque *definen* el operador. El dato
`(g, f)` va en `fit` porque es el *input* del operador. Esta es la separación
natural del operador: un medio distinto (otros `ρ, c, k`) es otra instancia, sin
tocar la lógica de entrenamiento.

### 2.2 Firma

```python
class LateralCauchyCylinder:

    def __init__(self, R, L, Tmax, rho, c, k, net_config=None):
        # R, L, Tmax : float    -> geometría del cilindro espacio-temporal
        # rho, c, k  : callables -> definen el medio (ver §5 contrato)
        # net_config : dict|None -> DEFINICIÓN del modelo (ver §2.4)
        #
        # Guarda geometría y medio. Construye la red (backend).
        # Define alpha = k/(rho*c) como CALLABLE (difusividad, para diagnósticos).
        # Inicializa los callables de salida en None (aún no entrenado):
        self.grad_T = None    # Ω×(0,Tmax] -> ℝ³   (OBJETIVO)
        self.T      = None    # Ω×(0,Tmax] -> ℝ
        self.flux   = None    # Ω×(0,Tmax] -> ℝ³   (= -k ∇T)
        self.alpha  = None    # callable k/(rho c)

    def fit(self, g, f, **opts) -> "history":
        # g, f : callables sobre la tapa z=L (los DOS datos de Cauchy)
        # opts : RÉGIMEN de entrenamiento (ver §2.4)
        #
        # Entrena el backend con la pérdida de §4.
        # Al terminar INSTALA los tres callables (self.grad_T, self.T, self.flux).
        # Devuelve history (curvas de pérdida por término, para diagnóstico).
        # Re-llamar a fit REINICIA el entrenamiento (sin warm-start en esta versión).
```

### 2.3 Las tres salidas instaladas por `fit`

Tras `fit`, tres atributos **callables hermanos**, evaluables por separado,
que comparten el mismo backend entrenado (son vistas distintas de él):

```python
op.grad_T(X)  # (N,3) -> (∂_x T, ∂_y T, ∂_z T)   <-- objetivo
op.T(X)       # (N,1) -> T
op.flux(X)    # (N,3) -> -k ∇T
```

- Antes de `fit` valen `None`; llamarlos antes de entrenar => **error claro**.
- Por defecto devuelven tensor **detached** (función pura, sin grafo de autograd).
  Flag opcional `create_graph=True` para derivadas de orden mayor.
- Internamente abren grafo de autograd (NO pueden correr bajo `torch.no_grad()`),
  pero encapsulado: de cara al usuario se sienten funciones limpias `X ↦ ...`.
- Evaluación fuera de `Ω`: chequeo de dominio que **avisa** (warning), no falla.

### 2.4 Corte `net_config` (modelo) vs `opts` de fit (corrida)

Separación por naturaleza: lo que *define el modelo* vs lo que *define la corrida*.

`net_config` (en `__init__`, define el modelo):
- `layers`           (default `[4, 96, 96, 96, 96, 1]`)
- `activation`       (default `tanh` — derivadas suaves; NO ReLU)
- `seed`             (default `0`)
- `fourier_features` (default **`False`** — ver §3, amplifican lo mal puesto)

`opts` (en `fit`, define la corrida):
- `weights`     tupla `(λ_pde, λ_g, λ_f, λ_lat)`, default `(1.0, 10.0, 10.0, 1.0)`
- `adam_iters`  (default `15000`)
- `lbfgs_iters` (default `3000`)
- `lr`          (default `1e-3`)
- `n_int, n_top, n_lat`  tamaños de colocación (defaults `8000, 3000, 2000`)

Razón del corte: puedes reentrenar el mismo modelo con otro régimen sin
reconstruirlo, y viceversa.

---

## 3. Por qué el mal condicionamiento gobierna el diseño

El problema es mal puesto exponencialmente (§1.6.3). Implicaciones de diseño
que NO son negociables:

- **`tanh`, no ReLU.** Se necesitan derivadas segundas limpias para el residuo
  `∇·(k∇T)`. La suavidad de la red ES la regularización implícita contra el `e^{L/δ}`.
- **Fourier features apagadas por defecto.** Las frecuencias altas que ayudan a
  ajustar `g` son EXACTAMENTE la dirección que la continuación hacia abajo
  amplifica. Activarlas agresivas => se ajusta ruido de `g` y `∇T` explota.
  Lo contrario de un forward bien puesto.
- **`g, f` se asumen suaves** (vienen de una regresión, no de píxeles crudos).
  La clase NO limpia ruido: filtrar/regularizar el dato es responsabilidad de
  quien llama.
- **Normalización interna** de entradas a `[-1,1]` por `(R, L, Tmax)`: crítica
  para el condicionamiento del entrenamiento (no es cosmética).
- **Distancia de continuación viable**, controlada por `L/δ` con `δ = √(2α/ω)`
  (longitud de penetración del armónico temporal dominante). Régimen sano
  `L/δ ≲ 1`. Por eso se guarda `alpha` para diagnóstico.

---

## 4. Método PINN (detalle de implementación)

### 4.1 Red (backend)

- MLP: entrada `4` (x,y,z,t) → capas ocultas `tanh` → salida `1` (T escalar).
- Default `[4, 96, 96, 96, 96, 1]`. Init Xavier normal, bias 0.
- Normalización de entrada a `[-1,1]`: `H = 2(X - lb)/(ub - lb) - 1`,
  con `lb = [-R, -R, 0, 0]`, `ub = [R, R, L, Tmax]`.
- `torch.set_default_dtype(torch.float64)` (doble precisión: importa en PINNs).
- Device: CUDA si disponible, si no CPU.

### 4.2 Pérdida

```
L(θ) = λ_pde·L_pde + λ_g·L_g + λ_f·L_f + λ_lat·L_lat
```

- `L_pde = mean| ρc ∂ₜT - ∇·(k∇T) |²`   sobre colocación interior.
- `L_g   = mean| T - g |²`               sobre `Γ_sup` (z=L).  **Dirichlet BLANDO.**
- `L_f   = mean| -k ∂_z T - f |²`         sobre `Γ_sup` (z=L).  Segundo dato Cauchy.
- `L_lat = mean| (x/R)∂_x T + (y/R)∂_y T |²`  sobre `Γ_lat` (r=R).
- **Base `z=0`: SIN término.** Es la incógnita; se lee al final.

CRÍTICO: `g` entra como pérdida blanda, **NUNCA** como Dirichlet duro. Si se clava
dura, el problema colapsa al régimen *forward* y se mata la inversión. `g` y `f`
ambos blandos es lo que realiza el dato de Cauchy.

### 4.3 Residuo en FORMA DIVERGENCIA (doble autodiff)

Calcular SIEMPRE como `∇·(k∇T)`, no como `k ΔT + ∇k·∇T`:

```python
T   = net(X)                         # X requiere grad
TX  = grad(T, X)                     # [T_x, T_y, T_z, T_t]
flux = k(X) * TX[:, 0:3]             # k ∇T   (espacial)
div  = Σ_i  grad(flux[:, i:i+1], X)[:, i:i+1]   # ∂_x(k T_x)+∂_y(k T_y)+∂_z(k T_z)
res  = rhoc(X) * TX[:, 3:4] - div    # ρc ∂ₜT - ∇·(k∇T)
```

donde `grad(u,x) = torch.autograd.grad(u, x, ones_like(u), create_graph=True, retain_graph=True)[0]`.

**Por qué importa la suavidad de `k`:** el segundo autodiff de `flux` deriva
TAMBIÉN a `k(X)` por la regla del producto. Si `k` es suave (polinomios), correcto.
Si `k` tuviera saltos duros, el autodiff de `k` en el salto da basura. Por eso el
contrato exige `k ∈ C¹`. (Forma divergencia elegida para NO tener que pasar `∇k`
analítico a mano.)

### 4.4 Muestreo (disco de radio R)

Aislar en un método propio (para que un `D` general sea un cambio futuro localizado).

- Interior: `r = R·√(U)`, `θ = 2πU`  (el `√` da densidad uniforme en el disco),
  `x=r cosθ, y=r sinθ`, `z = L·U`, `t = Tmax·U`.
- Tapa `Γ_sup`: igual pero `z = L` fijo.
- Lateral `Γ_lat`: `r = R` fijo, `θ = 2πU`, normales `(n_x,n_y) = (cosθ, sinθ)`.
- Base `Γ_0` (solo para leer): `z = 0` fijo.

### 4.5 Entrenamiento

- **Adam** (~15000 iters, `lr=1e-3`) para salir del régimen inicial.
- **L-BFGS** después (`max_iter≈3000`, `line_search_fn="strong_wolfe"`,
  `tolerance_grad=1e-9`, `tolerance_change=1e-12`, `history_size=50`) — afina; en
  PINNs el salto de calidad final con L-BFGS es grande.
- El balance de `weights` es lo que más afecta el resultado (problema mal puesto).
  Pesos adaptativos => FUTURO, no en esta versión.

---

## 5. Contrato de los callables (entrada/salida)

- Todas las entradas son tensores torch `X` de forma `(N, 4)`,
  columnas en orden **`[x, y, z, t]`**.
- API pública puede aceptar numpy y convertir internamente (capa de conveniencia);
  el núcleo trabaja en torch.

| Callable | Recibe        | Devuelve | Notas                                            |
|----------|---------------|----------|--------------------------------------------------|
| `rho(X)` | `(N,4)`       | `(N,1)`  | medio; debe IGNORAR `t` (se pasa X completo)     |
| `c(X)`   | `(N,4)`       | `(N,1)`  | medio; ignora `t`                                |
| `k(X)`   | `(N,4)`       | `(N,1)`  | medio; ignora `t`; **C¹ (polinomios)**           |
| `g(X)`   | `(N,4)`, z=L  | `(N,1)`  | dato de Cauchy 1; usa `x,y,t`; suave             |
| `f(X)`   | `(N,4)`, z=L  | `(N,1)`  | dato de Cauchy 2; usa `x,y,t`; suave             |

Salidas instaladas tras `fit`:

| Callable        | Recibe  | Devuelve | Definición                       |
|-----------------|---------|----------|----------------------------------|
| `op.grad_T(X)`  | `(N,4)` | `(N,3)`  | `(∂_x T, ∂_y T, ∂_z T)`          |
| `op.T(X)`       | `(N,4)` | `(N,1)`  | `T`                              |
| `op.flux(X)`    | `(N,4)` | `(N,3)`  | `-k(X) · ∇T`                     |
| `op.alpha(X)`   | `(N,4)` | `(N,1)`  | `k/(ρc)` (difusividad, diagnóst.)|

---

## 6. Validación: solución manufacturada

Caso de referencia con **coeficientes constantes** `ρc, k` para sanity check.
Solución exacta:

```
T*(x,y,z,t) = exp(-λ t) · cos(β z + ψ),     β = √(ρc·λ / k)
```

Propiedades (verificables a mano):
- Satisface la PDE con `ρc, k` constantes.
- Satisface la Neumann lateral EXACTA: no depende de `x,y` ⇒ `∂_n T = 0`.
- Datos de Cauchy en `z = L`:
  - `g(t)  = exp(-λ t) · cos(β L + ψ)`
  - `f(t)  = -k ∂_z T|_{z=L} = k β sin(β L + ψ) · exp(-λ t)`
- Gradiente exacto (para comparar `op.grad_T`):
  - `∂_x T* = 0`,  `∂_y T* = 0`,  `∂_z T* = -β sin(β z + ψ) · exp(-λ t)`
- Traza basal de referencia: `∂_z T|_{z=0} = -β sin(ψ) · exp(-λ t)`

Parámetros sugeridos: `λ=2.0, ψ=0.7, ρc=1, k=1, R=1, L=1, Tmax=1`.
Métrica: error relativo `‖grad_pred - grad_true‖ / ‖grad_true‖` en una nube de
puntos del cilindro.

ADVERTENCIA sobre este test: como `T*` no varía en `x,y`, NO estresa el mal
condicionamiento. Recuperará `∇T` con error bajo, pero eso solo prueba que la
maquinaria está bien armada. El test que de verdad mide el método (FUTURO) usa
`g` con frecuencia espacial en `(x,y)` + ruido aditivo calibrado, para mapear a
partir de qué `L/δ` se degrada la recuperación.

---

## 7. Ejemplo de uso esperado

```python
import torch

# medio constante (ejemplo de validación)
RHOC, K = 1.0, 1.0
rho = lambda X: torch.ones_like(X[:, :1])
c   = lambda X: RHOC * torch.ones_like(X[:, :1])
k   = lambda X: K * torch.ones_like(X[:, :1])

# datos de Cauchy de la solución manufacturada
import math
LAM, PSI, L = 2.0, 0.7, 1.0
BETA = math.sqrt(RHOC * LAM / K)
g = lambda X: torch.exp(-LAM*X[:,3:4]) * math.cos(BETA*L + PSI)
f = lambda X: K*BETA*math.sin(BETA*L + PSI) * torch.exp(-LAM*X[:,3:4])

op = LateralCauchyCylinder(R=1.0, L=L, Tmax=1.0, rho=rho, c=c, k=k)
history = op.fit(g, f)

# ahora op.grad_T es el campo ∇T : Ω×(0,Tmax] -> ℝ³
X = torch.rand(100, 4)          # puntos [x,y,z,t]
G = op.grad_T(X)                # (100, 3)
```

---

## 8. Alcance de esta versión

**Implementado (dentro de alcance):**
- Medio heterogéneo **determinista** `ρ, c, k`, incluido `k(z)` por capas (la
  forma divergencia lo soporta sin cambios de código: solo cambia el callable `k`).
- Solver de referencia modal independiente y soluciones manufacturadas.
- Experimentos de estrés: frecuencia espacial (Bessel) y ruido aditivo.
- `save` / `load` del modelo entrenado.

**Fuera de alcance (NO implementar):**
- Cierre Robin `f = h(g - T_atm)` dentro de `fit` (irá como helper externo, luego).
- Pesos adaptativos (self-adaptive / annealing). `ex6` solo MIDE la sensibilidad
  al balance; no lo auto-ajusta.
- Dominio `D` general (por ahora solo disco; el muestreo queda aislado para
  facilitar el cambio futuro).
- Helper de validación como método de la clase (por ahora externo / `__main__`).

> **Carácter del proyecto.** Es **determinista**: no hay campos aleatorios `G`,
> ensemble Monte Carlo ni cuantificación de incertidumbre. La heterogeneidad es de
> coeficientes dados, no estocástica.

---

## 9. Checklist de implementación

- [ ] Clase `LateralCauchyCylinder` con `__init__(R, L, Tmax, rho, c, k, net_config=None)`.
- [ ] Backend MLP `tanh` con normalización `[-1,1]`, float64, Xavier init.
- [ ] `alpha` callable `k/(ρc)` guardado en init.
- [ ] Salidas inicializadas en `None`; error claro si se invocan antes de `fit`.
- [ ] Muestreo en disco aislado en método propio (interior, tapa, lateral+normales, base).
- [ ] Residuo PDE en forma divergencia (doble autodiff).
- [ ] Pérdida de 4 términos; `g`, `f` BLANDOS; base sin término.
- [ ] Entrenamiento Adam → L-BFGS; `fit` devuelve `history` por término.
- [ ] `fit` instala `grad_T (ℝ³)`, `T (ℝ)`, `flux=-k∇T (ℝ³)`, detached por defecto,
      flag `create_graph`.
- [ ] Chequeo de dominio que avisa (no falla) fuera de `Ω`.
- [ ] Re-`fit` reinicia (sin warm-start).
- [ ] Script de validación con solución manufacturada (error relativo de `∇T`).
