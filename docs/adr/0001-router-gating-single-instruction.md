# ADR 0001 — Router-gating single-instruction

Fecha: 2026-09-16
Estado: aceptada

## Contexto

- Doctrina: la única red que genera acciones en inferencia es ACT 52M (`docs/ARCHITECTURE.md` Sec 1). El checkpoint en servicio es baseline single-instruction (imágenes + estado articular medido → chunk articular, sin slot de texto) (`docs/ARCHITECTURE.md` Sec 1, Estado E3).
- Lenguaje: los comandos pasan por gramática acotada con accept/refuse; el verificador externo solo puntúa y nunca emite deltas (`docs/submission_draft/SUBMISSION_TEXT.md` Sec 2; `docs/ARCHITECTURE.md` Sec 5).
- Evidencia de grounding: control swap-instruction 0/10 por refusal del router con 0 pasos; baseline random 0/10; vehicle 3/10 en seeds 0-9 (`docs/submission_draft/SUBMISSION_TEXT.md` Sec 4; `docs/submission_draft/SUBMISSION_COPY.md` Short Description).
- Variante con condicionamiento por frase declarada stretch DEAD en el reloj de 24 h; el manejo multi-frase sigue siendo trabajo del router, no de la red (`docs/submission_draft/SUBMISSION_TEXT.md` Sec 7; `docs/submission_draft/EVIDENCE_INDEX.md` Sec 6).
- Crítico de slots: reglas deterministas, jamás modelo (`out/grounding/CRITIC_CHECK.md`).

## Decisión

Router-gating single-instruction: se acepta exactamente una frase de relay; cualquier otra entrada se rechaza con 0 pasos. La política nunca adivina; la seguridad es estructural, no solicitada.

## Consecuencias

- El swap colapsa como exige el gate (0/10 por refusal), sin pasos de política.
- El despliegue sigue single-sentence; el crítico de slots queda unwired para despliegue.
- Toda ampliación a multi-frase exige nuevo ADR y nuevo gate swap.

## Alternativas descartadas

- Instruction-embedding en el slot de estado: declarado muerto para este pase (stretch DEAD, unwired). Se descarta por falta de gate y de ventana de cómputo, no por mérito futuro.
- Manejo multi-frase en la red: descartado; es trabajo del router accept/refuse hasta nuevo aviso.
