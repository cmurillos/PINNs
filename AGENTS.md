# AGENTS.md — Traspaso: experimentos del artículo (`ex1`–`ex6` + visualización)

> Documento de arranque para el agente que implemente los experimentos del
> artículo. No asume historial previo: aquí está el mapa de documentos, la regla
> de precedencia entre ellos, el orden de trabajo y el contrato de calidad del
> código. La especificación técnica completa de cada experimento está en
> [`docs/SIMULACIONES.md`](docs/SIMULACIONES.md).

---

## 1. Los documentos y quién manda sobre quién

| Documento                        | Rol                                                        |
|----------------------------------|------------------------------------------------------------|
| `docs/SIMULACIONES.md`           | **Especificación operativa de los experimentos.** Manda.   |
| Borrador del artículo (externo)  | Contexto y ecuaciones citadas ((3.4), (3.8), (3.9), (4.10)), recuadros de experimentos. **No distribuido en este repositorio público** (pedirlo al autor). |
| `CLAUDE.md`                      | Especificación de la clase PINN y del proyecto base (ya implementado). |
| Planteamiento riguroso (externo) | Planteamiento matemático (unicidad, mal condicionamiento). No distribuido en el repo; resumen autocontenido en `CLAUDE.md` §1. |

**Regla de precedencia (explícita, no negociable):** cuando un recuadro o
descripción de experimento del artículo difiera en nivel de detalle de
`docs/SIMULACIONES.md`, **manda SIMULACIONES.md**. El MD es más específico a
propósito: fija discretizaciones, semillas, tolerancias, nombres de archivo y
protocolos que el artículo solo esboza. El artículo se usa para entender el
*porqué* (las ecuaciones citadas) — nunca para contradecir el *cómo* del MD.

> **Nota sobre los PDFs.** El borrador del artículo y el planteamiento riguroso
> **no se versionan en este repositorio público**. Las ecuaciones que
> `SIMULACIONES.md` cita por número (`(3.4)`, `(4.10)`, …) están en el borrador
> del artículo: consíguelo del autor antes de implementar la infraestructura §1
> si necesitas el enunciado exacto de esas ecuaciones.

**No modificar la lógica central de la PINN ni del solver de referencia**
(`src/lateralcauchy/pinn.py`, `src/lateralcauchy/numerics/`). Todo el trabajo
nuevo son runners, funciones de diagnóstico, métricas y figuras alrededor de
lo que ya existe.

## 2. Por dónde empezar (orden obligatorio)

El orden de implementación es el §3 de `docs/SIMULACIONES.md`. El primer paso
no es opcional ni intercambiable:

1. **Infraestructura §1 de SIMULACIONES.md, con sus DOS tests unitarios:**
   - `diagnostics.atenuacion(omega, medio, L, n_z=2000)` — BVP complejo (3.4)
     por diferencias finitas conservativas; **test contra la forma cerrada**
     del medio constante (Apéndice C del PDF; ver la nota del MD sobre la
     referencia `|1/cosh(μL)|` del intervalo con Neumann).
   - `diagnostics.omega_efectiva(g_muestreado, dt)` — ec. (3.9); test con
     `g = cos(ωt)` a error < 0.5 %.

   **Regla de bloqueo:** si `atenuacion()` no pasa su test contra la forma
   cerrada, DETENERSE ahí y arreglarlo. La curva `a(ω)` es el eje x de `ex5`
   (el resultado central), el diagnóstico previo de todos los runners y la
   figura de la §3.3: si está mal, nada de lo que sigue tiene sentido.

2. `plotting_style.py` + volcado de la configuración base a `results/tabla1.csv`.
3. `ex1` (valida maquinaria) → `ex2` (con verificación cruzada del solver) →
   `ex3` → `ex4` → `ex5` (verificar presupuesto de cómputo ANTES de lanzar los
   50+ entrenamientos) → `ex6` → visualización 5.7 (reutiliza modelos guardados
   de `ex3`, no reentrena).

## 3. Contrato de calidad del código (requisito del artículo)

Este repositorio acompaña a un artículo: **todo el código que produce un número
o una figura del paper debe estar versionado aquí, legible y re-ejecutable**.
En concreto:

- **Modular y bien separado.** Lógica reutilizable en el paquete
  (`src/lateralcauchy/`): `diagnostics.py`, `metrics.py`, `plotting_style.py`.
  Los runners `examples/exN_*.py` quedan delgados: leen configuración, imprimen
  el diagnóstico previo (§1.3 del MD), entrenan o cargan, escriben `raw.csv`,
  generan figuras. Nada de duplicar fórmulas entre scripts: una función, un
  lugar (p. ej. `metrics.e0` es LA única discretización de la métrica (4.10);
  prohibido re-discretizar por script).
- **Opciones siempre visibles y fáciles de mover.** Cada parámetro de un
  experimento (mallas, semillas, pesos, iteraciones, niveles de ruido,
  frecuencias objetivo) se declara como constante nombrada al inicio del runner
  o en la sección de configuración compartida — nunca un número mágico enterrado
  en medio de una función. Si un experimento falla, cambiar una opción debe ser
  editar UNA línea obvia, no cazarla.
- **Comentado de acuerdo a los documentos.** Cada función/bloque no trivial cita
  la sección o ecuación que implementa (`SIMULACIONES.md §0.3`, `ec. (3.4)`,
  `Apéndice C`), como ya hacen los módulos existentes con `CLAUDE.md`. El lector
  del artículo debe poder ir del código al documento y de vuelta.
- **Extender, no reescribir.** Los `examples/exN_*.py` existentes son la base:
  se extienden al protocolo del MD (semillas, `raw.csv`, figuras con nombres
  fijos) conservando su estructura. Ídem `diagnostics.py` y `metrics.py`.
- **Salidas reproducibles.** `results/exN/raw.csv` una fila por corrida (columnas
  fijas del §0.4; campos no aplicables vacíos, no cero), modelos entrenados en
  `results/exN/models/`, figuras en `figs/` con los nombres exactos del MD
  (PDF + PNG). Nunca sobreescribir `raw.csv` con agregados.
- **Tests.** Los dos tests del §1 del MD van a `tests/` (pytest, como el resto
  de la suite); la CI ya los recogerá.

## 4. Prohibiciones (§0.5 del MD)

- Sin pesos adaptativos (`ex6` solo MIDE la sensibilidad, no auto-ajusta).
- Sin Fourier features.
- Sin cambiar la forma divergencia del residuo ni el esquema Adam → L-BFGS.
- Presupuesto de iteraciones FIJO por caso, no «hasta converger» (crítico en
  `ex5`: entrenar más los casos difíciles contamina el mapa).

## 5. Comandos útiles

```bash
pip install -e ".[dev]"   # instalar paquete + pytest
pytest -q                 # suite existente (debe seguir verde tras cada cambio)
python -m examples.ex1_manufactured   # runner de ejemplo (acepta iters reducidas)
```
