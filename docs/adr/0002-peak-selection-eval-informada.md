# ADR 0002 — Peak-selection eval-informada 20.5ép vs 41ép

Fecha: 2026-09-16
Estado: aceptada (disclosed)

## Contexto

- Vehicle v3-100k 3/10 en seeds 0-9 (seeds 1, 4, 7 con place) vs v3-200k 0/10 en las mismas seeds (`docs/submission_draft/SUBMISSION_TEXT.md` Sec 4; `docs/submission_draft/SUBMISSION_COPY.md` Business Value).
- Pico intermedio 20.5ép (v3-100k, train loss 0.034) vs degradación por memorización 41ép (v3-200k 0/10) (`docs/submission_draft/SUBMISSION_TEXT.md` Sec 4, veredicto FINAL VEHICLE transcrito).
- Regla congelada: `Regla vigente desde frozen_at_utc; la elección 100k>200k fue eval-informada (3/10 vs 0/10 en 0-9) y queda disclosed como tal, no como selección ciega` (`out/frozen-selection.json` `no_tune_rule`; `docs/submission_draft/FROZEN_PROTOCOL.md` Sec 4; `frozen_at_utc` 2026-09-16T19:43:24Z).
- El submission celebra peak-selection y muestra la curva de colapso, no selección ciega (`docs/submission_draft/SUBMISSION_COPY.md` Business Value; `docs/submission_draft/SUBMISSION_TEXT.md` Sec 4).

## Decisión

Peak-selection eval-informada y disclosed: se sirve v3-100k por su 3/10 frente al 0/10 de v3-200k, declarado como elección informada por evaluación. La prohibición de tunear rige desde `frozen_at_utc` hacia adelante; no reescribe el pasado.

## Consecuencias

- El 3/10 reproducible prima sobre un 10/10 irreproducible; los fallos se conservan.
- Cualquier nuevo checkpoint exige nuevo congelado y nuevo disclosed; no se re-ejecutan baterías para quedarse con el mejor run.
- Comparar 3/10 place contra gates estrictos de otras baterías cruza seeds y gates y no es apples-to-apples.

## Alternativas descartadas

- Selección ciega: descartada por deshonesta frente al submission, que celebra el pico y publica el colapso.
- Last-checkpoint (servir 200k por ser el último): descartado; habría servido 0/10 por degradación documentada.
