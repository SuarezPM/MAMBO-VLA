# Swap-Gate Standard — protocolo para policies futuras

Fecha: 2026-09-16
Alcance: gate estándar bit-idéntico para cualquier policy futura. Solo texto, cero evals nuevas.

## Protocolo

1. Fijar seed y escena (hash-file ya congelado, p. ej. `out/seeds/seed_hashes.json`).
2. Ejecutar dos ramas con la MISMA seed y MISMO stats-root/checkpoint/EXEC_STEPS:
   - Rama A: instrucción de despliegue exacta (debe producir rollout).
   - Rama B: instrucción cambiada fuera de gramática (debe producir refusal con 0 pasos).
3. Comparar por seed con scope dividido: rama B bit a bit (`success`, `steps`, `note`); rama A solo acuerdo `success`/fail + `note`, con `steps` exento por no-determinismo documentado.
4. Criterio PASS del gate:
   - B refusal determinista bit-idéntico en todas las seeds (0 steps, sin física).
   - A acuerdo success/fail+nota (steps exento).
   - Justificación: la física closed-loop es no-determinista (seed 93: 7750 vs 7775 per `out/seeds_ood/logs_ood_80_99/eval_80_99.log` vs `out/seeds_ood/logs_ood_80_99/eval_80_99_rerun.log` + `out/seeds_ood/logs_ood_80_99/results_80_99.json`), solo el refusal sin física admite bit-identidad.
5. Registrar tabla + logs por seed; conservar fallos; no re-rollear.

## Precedente citado (sin números nuevos)

- Swap 0/10 refusal: rama swap 0/10 por refusal del router con 0 pasos en seeds 0-9 (`out/gates/v3_200k_rollout.log` `=== SWAP200K ===` + `swap control: success collapses to 0 (refusal, arms never moved)`; `docs/submission_draft/SUBMISSION_TEXT.md` Sec 4; `docs/submission_draft/EVIDENCE_INDEX.md` Sec 6).
- Null-swap idéntico como ejemplo de rigor solo en B: repetir la instrucción cambiada debe dar refusal idéntico bit a bit; cualquier deriva en B invalida el gate y exige reportarla, no tunearla.

## Qué NO es este gate

- No mide place ni generalización; solo grounding del router.
- No redefine frozen place ni P6-strict (`out/determinism/GOAL_PREDICATES.md`).
- No autoriza re-evaluar seeds congeladas para elegir; rige desde su adopción hacia adelante.
