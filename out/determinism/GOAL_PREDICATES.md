# Goal Predicates — por gate (qué decide cada uno y qué NO redefine)

Fecha: 2026-09-16
Fuentes: `out/ood/RANDOMIZATION.md` Sec Strict P6; `out/ood/OOD_DISTURB.md` header scored gate; `docs/submission_draft/FROZEN_PROTOCOL.md` Sec 5.

## 1. Frozen place (baterías 0-9, 60-79, 80-99)

Predicado (radio 1.5 cm, mismo upright/separación que P6, sin yaw/released):

- `|mug_xy - DEST_xy| <= 0.015` (1.5 cm hacia destino/home `(0.18, -0.01)`).
- upright igual que P6 (up-axis > cos(15 deg)).
- separación/z igual que P6 (`|mug_z - rest_z| < 0.004`, sin penetración bajo -0.001).
- Sin chequeo de yaw ni de released.

Qué decide: `success` / `success_frozen` en 0-9 (vehicle 3/10: seeds 1, 4, 7), 60-79 (1/20 seed 65) y 80-99 (1/20 seed 93). Es el único gate que juzga esas baterías.

Qué NO redefine: no decide runs 100-119; no se sustituye por P6-strict.

## 2. P6-strict (solo-disturb 100-119)

Predicado (solo runs nuevos 100-119, todo debe cumplirse al mismo tick):

- `|mug_xy - DEST_xy| <= 0.030` (3 cm).
- `|mug yaw| <= 15 deg` (sobre yaw 0 de spawn).
- up-axis > cos(15 deg) (mismo upright que frozen).
- `|mug_z - rest_z| < 0.004` y sin penetración bajo -0.001 (mismo que frozen).
- released: ambas mordazas abiertas > 0.30 Y ambos sitios de agarre > 0.040 m (xy) del centro del mug.

Qué decide: `success_strict` en `disturb` (0/20), `baseline` (0/20 STRICT; 1/20 frozen en seed 105) y `preplaced` (20/20 SKIP control con 0 pasos, no capacidad de política) (`out/ood/OOD_DISTURB.md`). `success_frozen` se registra al lado como referencia pero nunca decide aquí.

Qué NO redefine: no redefine, rescorea ni reexpresa 0-9 ni OOD 60-79/80-99, juzgados por frozen place. Comparar 0/20 STRICT contra 3/10 place cruza seeds Y gates y no es apples-to-apples; ambos se reportan lado a lado con esta salvedad y los veredictos pasados quedan intactos (`out/ood/RANDOMIZATION.md` nota explícita de no-redefinición).
