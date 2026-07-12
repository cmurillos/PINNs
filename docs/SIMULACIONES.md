# SIMULACIONES.md — Especificación de experimentos del artículo

> Documento para Codex. Acompaña al borrador `articulo_borrador.pdf` y al repositorio
> `cmurillos/PINNs` (paquete `lateralcauchy`). Define la infraestructura, las
> convenciones globales y la especificación de cada experimento (`ex1`–`ex6` y la
> visualización). Los números de ecuación citados ((3.4), (4.10), (3.8), etc.)
> refieren al PDF del artículo. **No modificar la lógica central de la PINN ni del
> solver**: todo lo de abajo son runners, funciones de diagnóstico, métricas y figuras.

---

## 0. Convenciones globales (aplican a TODOS los experimentos)

### 0.1 Configuración base congelada
- Una única configuración base de red y entrenamiento, compartida por todos los
  experimentos; cada `ex` solo desvía donde su sección lo indique.
- Leerla del código actual (la clase PINN del paquete) y volcarla a
  `results/tabla1.csv` con columnas:
  `capas, ancho, N_int, N_sup, N_lat, lambda_pde, lambda_g, lambda_f, lambda_lat, iter_adam, iter_lbfgs, lr_adam`.
  Ese CSV llena la Tabla 1 del artículo.
- **Presupuesto fijo, no convergencia**: mismo número de iteraciones por caso
  (crítico en `ex5`; si se entrena "hasta converger", los casos difíciles reciben
  más cómputo y el mapa se contamina).
- Doble precisión en todo (red, datos, métricas).

### 0.2 Semillas
- Semillas de red: lista fija `SEEDS_RED = [0, 1, 2, 3, 4]`.
- Semillas de ruido: lista fija e independiente `SEEDS_RUIDO = [100, 101, 102, 103, 104]`.
- Generadores separados (`numpy.random.default_rng(seed)` por rol); nunca el
  generador global.

### 0.3 Métrica principal E0 — discretización obligatoria
Versión discreta de la ec. (4.10) del artículo, idéntica en todos los `ex`:
- Malla en la base `Γ0`: producto polar `n_r = 24` radios × `n_θ = 48` ángulos
  (excluir r=0 duplicado; incluir pesos de área `r·Δr·Δθ`).
- Malla temporal: `n_t = 100` instantes equiespaciados en `(0, 𝒯]`.
- `E0 = || ∇T_θ − ∇T_ref ||_2 / || ∇T_ref ||_2` sobre esa malla con pesos de
  cuadratura (trapecio en t, área polar en el disco).
- Implementar UNA función `metrics.e0(model, referencia, malla)` y usarla en todos
  los experimentos; prohibido re-discretizar por script.
- Métricas de control (misma malla): perfil error-vs-z (20 franjas en z, error
  relativo por franja) y error-vs-t (error relativo por instante).

### 0.4 Contrato de salidas
- Por experimento: `results/exN/raw.csv` — una fila por corrida. Columnas fijas:
  `experimento, caso, semilla_red, semilla_ruido, eta, a, omega, L, E0, err_global, tiempo_s`.
  (Campos no aplicables: vacío, no cero.)
- Nunca sobreescribir `raw.csv` con agregados; media ± std se calcula al graficar.
- Figuras en `figs/` con nombres fijos (ver cada `ex`), formato PDF (y PNG para
  previsualización). Un único módulo de estilo (`plotting_style.py` con rcParams:
  misma fuente, tamaños, grosores) importado por todos los scripts de figuras.
- Guardar cada modelo entrenado (`save`) en `results/exN/models/` para regenerar
  figuras sin reentrenar (incluida la visualización 5.7).

### 0.5 Prohibiciones
- Sin pesos adaptativos (ex6 solo MIDE sensibilidad, no auto-ajusta).
- Sin Fourier features.
- Sin cambiar la forma divergencia del residuo ni el esquema Adam→L-BFGS.

---

## 1. INFRAESTRUCTURA (implementar ANTES que los experimentos)

Estas funciones son de carga: alimentan la figura de la §3.3, el diagnóstico de la
§4.4 y el eje x de `ex5`.

### 1.1 `diagnostics.atenuacion(omega, medio, L, n_z=2000) -> float`
Resuelve el problema de frontera complejo (3.4) del artículo:
```
i·ω·ρc(z)·U = (k(z)·U')'  en (0,L),   U(0)=1,   U'(L)=0,
```
por diferencias finitas centradas sobre malla uniforme de `n_z` nodos:
- Discretizar `(k U')'` en forma conservativa: `(k_{j+1/2}(U_{j+1}-U_j) − k_{j-1/2}(U_j−U_{j-1}))/h²`
  con `k_{j±1/2}` evaluada en puntos medios.
- Condición `U'(L)=0` por nodo fantasma o esquema unilateral de segundo orden.
- Sistema tridiagonal complejo; resolver con `scipy.linalg.solve_banded` (dtype complex128).
- Devuelve `a = abs(U[-1])`.

**Test unitario obligatorio (gratis por el Apéndice C):** para medio constante
(`ρc=k=1`, `α=1`), comparar contra la forma cerrada `a_exacta(ω) = exp(−L·sqrt(ω/2))`
con tolerancia relativa 2 % para `ω ∈ {1, 5, 20, 80}` y `L=1`.
(Nota: la cerrada es la del semiespacio; con la reflexión de Neumann la numérica
puede quedar hasta ~2× encima a frecuencias bajas — si el test falla por eso,
comparar contra `|cosh|`-forma exacta del intervalo: `a = |1/cosh(μL)|` con
`μ = sqrt(iω/α)`. Esa es la exacta del BVP con U'(L)=0; usarla como referencia
del test y dejar la exponencial como cota.)

### 1.2 `diagnostics.omega_efectiva(g_muestreado, dt) -> float`
Ec. (3.9) del artículo: `ω_ef = ||∂t g||_L2 / ||g||_L2`.
- `∂t g` por diferencias centradas sobre la malla temporal.
- Normas discretas con peso trapezoidal.
- Test: para `g = cos(ω t)` sobre ventana múltiplo del período, `ω_ef = ω` con
  error < 0.5 %.

### 1.3 `diagnostics.regimen(g, medio, L, eta) -> dict`
Junta 1.1 y 1.2: devuelve
`{omega_ef, a, margen: a/eta, recuperable: a > eta}`.
Es el diagnóstico de la §4.4; se ejecuta e imprime al inicio de cada runner.

### 1.4 Figura de la §3.3 — `figs/aten_curvas.pdf`
- Curvas `a(ω)` por 1.1 para: (a) medio constante `ρc=k=1`; (b) el `k(z)` de `ex3`.
- Malla: 40 valores de ω log-espaciados tales que `a` recorra ~[1e−3, 1].
- Ejes: `log a` vs `sqrt(ω)`. Superponer la recta teórica del caso constante
  (Apéndice C). Leyenda con ambos medios.

---

## 2. EXPERIMENTOS

Base común salvo indicación: `R = L = 𝒯 = 1`, configuración de §0.1,
`SEEDS_RED` de §0.2, métricas de §0.3, salidas de §0.4.

### ex1 — Control de maquinaria (solución manufacturada)
- **Medio:** `ρc = k = 1`.
- **Solución exacta:** `T*(z,t) = exp(−λt)·cos(βz+ψ)` con `λ=1`, `β=sqrt(ρc·λ/k)=1`,
  `ψ=π/4` (garantiza traza basal no trivial: `∂z T*|_{z=0} = −β e^{−λt} sin ψ ≠ 0`).
- **Datos:** `g(t)=e^{−t}cos(β+ψ)`, `f(t)=kβ e^{−t} sin(β+ψ)` evaluados
  analíticamente sobre los puntos de la tapa. Sin ruido.
- **Protocolo:** 5 corridas (SEEDS_RED).
- **Mediciones:** `E0` (contra ∇T* analítico), error global en Ω, perfiles vs z y vs t.
- **Figuras:** `figs/ex1_corte.pdf` (T_θ vs T* sobre el eje x=y=0, tres instantes
  t ∈ {0.25, 0.5, 0.9}); `figs/ex1_traza_basal.pdf` (∂z T_θ|_{z=0} vs exacto, en t).
- **Criterio de éxito:** `E0 ≲ 1e−2` en media. Si falla, es bug de maquinaria, no
  del método (el dato no tiene frecuencia espacial).

### ex2 — Frecuencia espacial (modo de Bessel)
- **Medio:** `ρc = k = 1`.
- **Modo:** axisimétrico `m=0, n=1`: `φ(r) = J0(κ r)` con `κ = j'_{0,1}/R ≈ 3.8317`
  (primer cero de `J0'`; `scipy.special.jnp_zeros(0,1)`).
- **Excitación:** armónico temporal en la base del problema 1D reducido, con ω
  elegida (por bisección sobre 1.1) tal que `a(ω) ≈ e^{−1} ≈ 0.37`. Registrar la ω
  resultante en `raw.csv`.
- **Datos:** `(g, f)` extraídos del solver modal (reducción 1D Crank–Nicolson,
  malla `n_z × n_t = 400 × 400`). **Verificación cruzada previa:** para k constante
  comparar el solver contra la solución separable exacta del modo; reportar el
  error de discretización en el log del runner (< 1e−4 esperado).
- **Protocolo:** 5 corridas.
- **Figuras:** `figs/ex2_error_vs_z.pdf` (perfil, eje y logarítmico; debe CRECER
  hacia la base — la firma del mal condicionamiento);
  `figs/ex2_error_vs_t.pdf` (debe salir ≈ plano — la continuación es espacial).
- **Esperado:** `E0` moderado (orden 1e−1); ambos controles estructurales se cumplen.

### ex3 — Medio heterogéneo k(z)
- **Medio:** `ρc = 1`, `k(z) = 1 + z` (variación 2× entre base y tapa; C∞).
- **Frecuencia:** por bisección sobre 1.1 **con el medio heterogéneo**, misma
  atenuación objetivo `a ≈ e^{−1}`. (Sin difusividad representativa: la curva
  medida es del medio real.)
- **Datos:** del solver modal con `k(z)` (único disponible; no hay analítica).
- **Protocolo:** 5 corridas.
- **Figuras:** `figs/ex3_k_perfil.pdf` (k(z) usado);
  `figs/ex3_traza_basal.pdf` (∂z T_θ|_{z=0} vs referencia modal, en t);
  `figs/ex3_error_vs_z.pdf`.
- **Esperado:** `E0` comparable al de ex2 a igual `a`: la heterogeneidad
  determinista no degrada más allá de lo previsto por la curva.

### ex4 — Robustez al ruido
- **Base:** configuración de ex2 (mismo modo, misma ω).
- **Ruido:** gaussiano aditivo iid por punto de muestreo, independiente sobre g y f,
  amplitud relativa `η ∈ {0, 1e−3, 1e−2, 5e−2}` respecto de `||g||_∞` y `||f||_∞`
  respectivamente.
- **Protocolo:** por nivel de η: 5 semillas de ruido × 3 semillas de red
  (SEEDS_RED[0:3]) = 15 corridas por nivel (el nivel η=0 usa solo las 3 de red).
- **Figura:** `figs/ex4_ruido.pdf` — `E0` vs `η` en log–log, media ± std;
  marcar el nivel η=0 como piso horizontal.
- **Esperado:** amplificación conforme al criterio (3.8), factor `1/a(ω)`; la
  suavidad de la red fija el piso a η pequeño.

### ex5 — Mapa de degradación (RESULTADO CENTRAL)
Dos barridos sobre la base de ex2, mismo presupuesto de entrenamiento por caso:
- **Barrido (i) — frecuencia:** `L=1` fijo; 7 frecuencias elegidas por bisección
  para que `a(ω)` recorra `{0.7, 0.5, 0.37, 0.2, 0.1, 0.03, 0.01}`.
- **Barrido (ii) — longitud:** dato fijo (la ω de `a=0.37` en L=1);
  `L ∈ {0.5, 0.75, 1, 1.5, 2}`; registrar la `a` de cada caso por 1.1 (depende de L).
- **Protocolo:** 5 semillas por caso en (i); 3 semillas por caso en (ii)
  (presupuesto: 7×5 + 5×3 = 50 entrenamientos; verificar tiempo de una corrida
  base × 50 antes de lanzar).
- **Segunda curva:** repetir el barrido (i) con ruido `η = 1e−2` (3 semillas de
  ruido × 1 de red por caso).
- **Figura central:** `figs/ex5_colapso.pdf` — `log E0` vs `log(1/a)`:
  - familia (i) y familia (ii) con símbolos distintos → **esperado: colapso en una
    sola curva** (la dificultad es `a`, no L ni ω por separado);
  - recta de la cota inferior (3.8): `E0 = η/(2a)`, pendiente 1 en estos ejes
    (con la η efectiva: piso de optimización estimado del caso más fácil sin ruido);
  - línea vertical en `a = η`;
  - curva con ruido `η=1e−2` superpuesta.
- **Esperado:** colapso de las dos familias; curva por encima y ~paralela a la
  cota; degradación brusca al cruzar `a = η`, predicha por el diagnóstico 1.3
  (imprimir la predicción de cada caso ANTES de entrenar en el log).

### ex6 — Sensibilidad al balance de pesos
- **Base:** ex2. Malla logarítmica `(λ_g, λ_f) ∈ {0.1, 1, 10}²` con
  `λ_PDE = λ_lat = 1` fijos.
- **Protocolo:** 3 semillas por celda (27 corridas).
- **Figura:** `figs/ex6_heatmap.pdf` — mapa de calor 3×3 de la media de `E0`,
  anotando std en cada celda.
- **Esperado:** sensibilidad moderada, óptimo plano. Solo se MIDE; nada de
  auto-ajuste.

### Visualización espacio-temporal (subsección 5.7)
- **Caso:** modelo entrenado de ex3 (cargar de `results/ex3/models/`, no reentrenar).
- **Figura estática:** `figs/vis_espaciotemporal.pdf` — retícula 3 filas × 4 columnas,
  instantes `t_i ∈ {0.2, 0.4, 0.6, 0.8}·𝒯`:
  - fila 1: campo `T_θ` en el corte longitudinal `y=0` (semiplano r–z con contorno
    del cilindro), malla 100×100;
  - fila 2: dato `g(·, t_i)` sobre el disco de la tapa (malla polar);
  - fila 3: dato `f(·, t_i)` sobre el disco.
  - Barra de color COMÚN por fila (mismos vmin/vmax en las 4 columnas) para que la
    atenuación entre instantes sea legible.
- **Animación (material suplementario):** `figs/vis_animacion.gif` — los mismos
  tres paneles sincronizados, 40 cuadros en `(0, 𝒯]`, `matplotlib.animation` o
  `plotly`; ~10 s de duración.

---

## 3. Orden de implementación

1. Infraestructura §1 (con sus dos tests unitarios) + `plotting_style.py`.
2. Volcado de la configuración base a `results/tabla1.csv` (§0.1).
3. `ex1` → validar maquinaria.
4. `ex2` (incluida la verificación cruzada del solver) → `ex3` → `ex4`.
5. `ex5` (verificar presupuesto de cómputo antes de lanzar los 50+ entrenamientos).
6. `ex6`.
7. Visualización 5.7 (usa modelos guardados de ex3).

Cada `exN` es un runner independiente (`examples/exN_*.py` ya existen como base:
extenderlos, no reescribirlos) que: imprime el diagnóstico 1.3, entrena o carga,
escribe `raw.csv`, genera sus figuras.
