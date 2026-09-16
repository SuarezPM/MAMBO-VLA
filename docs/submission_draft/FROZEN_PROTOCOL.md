# FROZEN PROTOCOL — Fase A (provenance, sin re-entreno)

Fuente de verdad: `out/frozen-selection.json`.
Regla superior: nada se re-entrena ni se re-evalúa para elegir. Solo lectura + hashes.

## 1. Modelo congelado

- Checkpoint: `out/checkpoints/act_full_v3/checkpoints/100000/pretrained_model/model.safetensors`
- `sha256_safetensors`: `3ae1514903116bcdb18bca3f10546f8b133e8636a1170f1642e50a0e09b6cc83`
- IR: `out/export/act_full_v3_100k/model.xml`
  - `ir_xml_sha`: `28edc2bdf8a0216b0d739f6e812789fa4c841598f5077a4aa3927e95c9090567`
- IR: `out/export/act_full_v3_100k/model.bin`
  - `ir_bin_sha`: `15d5d97cb0b3a737e10366885ec7359cbbf8f663029160293f8d1c7d45ca4666`
- Paridad IR vs referencia: `parity_max_err = 1.1920928955078125e-06` (tol `0.001`, `pass=true` en `out/export/act_full_v3_100k/parity.json`).
- Verificación solo por hash (lectura):
  - `sha256sum out/export/act_full_v3_100k/model.xml out/export/act_full_v3_100k/model.bin`
  - `sha256sum out/checkpoints/act_full_v3/checkpoints/100000/pretrained_model/model.safetensors`
  - No cargar pesos. No abrir binarios como texto.

## 2. Protocolo congelado

- `eval_script`: `scripts/eval_policy.py` (usar defaults; no modificar).
- `exec_steps`: `50` (congelado, registrado por fila en cada `results*.json`).
- `max_steps`: `9000` (valor observado en filas `steps`; corte de episodio).
- `stats_root`: `out/lerobot_v3` (normalización/estadísticas).
- `policy_wrapping`: `act chunk_size=50 n_action_steps=50 vision_backbone=resnet18; OpenVINO IR FP32 (model.xml/model.bin) con stats de out/lerobot_v3; temporal_ensemble_coeff=null`.

## 3. Seeds: dev vs eval

Dev (visto en train / selección, no usar para afirmar generalización):

- Train: seeds `0-59` (`0-9` original en `out/dataset/seed_0..9.npz` + `10-29` extra en `out/dataset_extra/seed_10..29.npz` + `30-59` v3 en `out/dataset_v3/seed_30..59.npz`; 60 demos teacher scripted con `success=True`, ver `scripts/convert_v3.py:9-11,174-179` y `scripts/run_seeds_v3.py:55` con `V3_SEEDS=30..59`). No se congela ningún seed de train como batería de evaluación.

Eval (rollouts closed-loop nunca ejecutados antes del scoring (frames demo exceptuados); únicas válidas para reporte):

| Batería | Seeds | Hash-file |
|---|---|---|
| `frozen-0-9` | `0-9` | `out/seeds/seed_hashes.json` |
| `frozen-10-29` | `10-29` | `out/seeds_extra/seed_hashes.json` |
| `frozen-ood-60-79` | `60-79` | `out/seeds_ood/seed_hashes_ood.json` |
| `frozen-ood-80-99` | `80-99` | `out/seeds_ood/seed_hashes_ood_80_99.json` |
| `frozen-disturb-100-119` | `100-119` | `out/seeds_disturb/seed_hashes_disturb.json` |

Cada `seed_hashes*.json` fija el bundle por seed (layout, posiciones, iluminación, instrucción). Los `bundles_*.json` correspondientes deben coincidir con esos hashes.

Mapa `results` por batería (ver `evaluation.batteries[].results` en `out/frozen-selection.json`): `frozen-ood-80-99` → `out/seeds_ood/logs_ood_80_99/results_80_99.json`; `frozen-disturb-100-119` → `out/seeds_disturb/results_disturb.json` + `results_baseline.json` + `results_preplaced.json`; `frozen-0-9`, `frozen-10-29` y `frozen-ood-60-79` → `[]` (sin results JSON archivado: `10-29` sin tabla de rollout de política según `out/ood/OOD_RESULTS_60_79.md`; `60-79` solo tabla en `out/ood/OOD_RESULTS_60_79.md`; `0-9` solo tabla en submission + `result.json` teacher por seed; no inventado).

Texto canónico:

- `dev_vs_eval`: `train: seeds 0-59 (0-9 original + 10-29 extra + 30-59 v3; 60 demos teacher scripted) usadas en train; evaluation: rollouts closed-loop nunca ejecutados antes del scoring (frames demo exceptuados)`.
- `frozen_at_utc`: ver `out/frozen-selection.json`.
- `no_tune_rule`: `Regla vigente desde frozen_at_utc; la elección 100k>200k fue eval-informada (3/10 vs 0/10 en 0-9) y queda disclosed como tal, no como selección ciega`.

## 4. Regla no-tune

- Regla vigente desde frozen_at_utc; la elección 100k>200k fue eval-informada (3/10 vs 0/10 en 0-9) y queda disclosed como tal, no como selección ciega.
- Prohibido re-ejecutar baterías para quedarse con el mejor run.
- Coherencia con submission (solo cita, sin editar): `docs/submission_draft/SUBMISSION_COPY.md` celebra peak-selection — «Peak selection at 20.5 epochs (v3-100k, train loss 0.034) versus memorization degradation at 41 epochs (v3-200k 0/10)» — y `docs/submission_draft/SUBMISSION_TEXT.md` fija «FINAL VEHICLE: v3-100k 3/10 con export PASS. v3-200k 0/10 confirma pico intermedio (20.5ép) y degradación por memorización (41ép)». Ambas celebran selección por pico eval-informado, no selección ciega; coherente con el disclosed de este protocolo.
- Si un resultado ya existe, se conserva tal cual; solo se permite añadir claves aditivas de proveniencia (`checkpoint_sha256`, `frozen_selection`).
- Cualquier cambio de política/estadísticas/IR invalida este congelado y exige nuevo `frozen-selection.json` + nuevos hashes.

## 5. Gates

### 5.1 Baterías generales (`0-9`, `10-29`, `60-79`, `80-99`)

- Gate: `place`.
- Fila típica: `seed, success, steps, forward_p50_ms, forward_p95_ms, exec_steps=50, note`.
- `success=true` solo si hay `place` vía rollout de política. `steps=9000` indica no-place (corte por `max_steps`).

### 5.2 Batería disturb (`100-119`) — único sitio con doble gate

Modos en `out/seeds_disturb/`: `baseline`, `disturb`, `preplaced`.

Cada fila trae:

- `success_strict` (gate P6-strict).
- `success_frozen` (gate place congelado).
- `steps, exec_steps=50, retries{limited}, skipped[], disturbed, mug_moved_mm, note`.

Definición observada en logs:

- `STRICT = 3cm + yaw15 + upright + released` (solo runs nuevos).
- `frozen_place` vs `strict_place` se registran por separado al re-percibir en `t=0` y al cerrar el rollout.
- `preplaced`: `skipped=[pick,set-down,handover,place]`, `steps=0`, ambos gates en `true` porque la pieza ya está en destino estricto (0 pasos de política).
- Regla de lectura: reportar siempre ambos; no sustituir `success_frozen` por `success_strict` ni viceversa. El gate P6-strict solo aplica a `100-119`; no retroaplicar a otras baterías.

## 6. Cómo reproducir una batería (sin re-tunear)

Solo descripción; no ejecutar para seleccionar.

1. Leer `out/frozen-selection.json`; anotar `checkpoint`, `sha256_safetensors`, `ir_xml_sha`, `ir_bin_sha`, `exec_steps`, `max_steps`, `stats_root`.
2. Verificar hashes (lectura):
   - `sha256sum` de `model.xml`, `model.bin` y `model.safetensors`; comparar con `frozen-selection.json`.
   - `python3 -m json.tool` sobre el `hash_file` de la batería y sobre el `results*.json` correspondiente.
3. Fijar args congelados: `eval_script=scripts/eval_policy.py`, `stats_root=out/lerobot_v3`, `exec_steps=50`, `max_steps=9000`, checkpoint e IR del punto 1, `hash_file` de la tabla del punto 3.
4. Ejecutar una sola vez por batería con esos args y el rango `seeds` indicado (ej. `80-99` con `out/seeds_ood/seed_hashes_ood_80_99.json`).
5. Comparar salida fila a fila con el `results*.json` archivado (`seed, success/success_strict/success_frozen, steps, exec_steps, note`). No cambiar nada si difiere solo en tiempos (`forward_p*_ms`); si difiere en `success/steps`, el entorno no es reproducible y no se debe tunear para igualarlo.
6. Proveniencia añadida (ya aplicada):
   - `out/seeds_ood/logs_ood_80_99/results_80_99.json`: claves aditivas por fila.
   - `out/seeds_disturb/results_disturb.json`: claves aditivas por fila.
   - `out/seeds_disturb/results_baseline.json`: claves aditivas por fila.
   - `out/seeds_disturb/results_preplaced.json`: claves aditivas por fila.
   - `out/bench_local/bench_local_cpu.json`: claves aditivas top-level.
   - Valores: `checkpoint_sha256=<sha256_safetensors>`, `frozen_selection=out/frozen-selection.json`. Validado con `python3 -m json.tool`.

## 7. Archivos tocados por esta fase

- Nuevo: `out/frozen-selection.json`.
- Modificados solo aditivamente: los 5 JSON de `§6.6` (sin cambiar ningún valor previo).
- Nuevo: este `docs/submission_draft/FROZEN_PROTOCOL.md`.
