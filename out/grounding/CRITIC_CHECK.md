# CRITIC_CHECK — P1-lite slot critic (REGLAS, jamás LLM)

- Método: **reglas deterministas** (`scripts/slot_critic.py`: regex + listas
  cerradas, solo stdlib). Cero ML, cero pesos, cero red, cero embeddings,
  cero llamadas a modelos. Etiqueta: **reglas, jamás LLM**.
- Gramática: fases {pick, set-down, handover, place} (fases del relay en
  `scripts/verify_repair.py:PHASES`), brazos {A=left, B=right} (A coloca en
  relay y aparca, B re-agarra — `scripts/run_seeds.py` docstring), objetos
  {mug, plate} (cuerpos libres del relay en `sim/dual_so101_dinner.xml`).
- Set propio: `out/grounding/paraphrases.jsonl` — 30 líneas redactadas por el
  autor (15 aceptables del relay + 15 adversariales ambiguas/fuera de
  gramática). Ninguna generada por modelo.
- Comando: `python3 scripts/slot_critic.py --check out/grounding/paraphrases.jsonl`
- Log: `out/grounding/critic_run.log` (exit 0).

## Números propios (reales, del log)

author-locked: mide conformidad con spec, no generalización; un claim de generalización exigiría paráfrasis independientes

- **aceptadasOK/total: 15/15** — 30/30 author-locked (spec lock), unwired (deployment sigue single-sentence router)
- **rechazadasOK/total: 15/15** — 30/30 author-locked (spec lock), unwired (deployment sigue single-sentence router)
- Total: 30/30 author-locked (spec lock), unwired (deployment sigue single-sentence router), líneas con `expect == got`.

Cobertura aceptadas: 4 fases × ambos brazos × ambos objetos (muestreo),
alias (pick up, grasp, set down, hand over, hands off), flexiones
(picks/sets/places/picked/placing), mayúsculas/puntuación (a12–a14).
Rechazos: objeto ambiguo (r01), brazo ambiguo (r02), fase ambigua (r03),
sin fase/verbo de tarea completa (r04), sin brazo (r05), pronombre sin
antecedente (r06), OOG cup/spoon/peg+socket/open+drawer/pour (r07–r09,
r12–r13), negación (r14), balbuceo (r15), frases de despliegue/swap r10–r11.

## Capas (honesto)

- El **router** (`scripts/eval_policy.py:route`, igualdad exacta con
  `INSTRUCTION`) acepta la frase de despliegue del relay; el **crítico**
  la rechaza (r10) **por diseño**: valida órdenes a nivel de fase con
  slots {fase, brazo, objeto}, no la frase de despliegue. r11 muestra que
  la frase swap colapsa también a nivel de crítico.
- Limitaciones honestas (no se persiguen aquí): adjuntos no listados se
  ignoran (a15 "park" emite WARN por stderr + campo "warn" como precondición
  de cableado futuro safety-relevante, sin cambiar el outcome); "place down"
  mapearía a place; solo inglés; sin
  alcance de negación (toda negación se rechaza); sin correferencia entre
  frases.
