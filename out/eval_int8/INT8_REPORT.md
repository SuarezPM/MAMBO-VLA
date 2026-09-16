# INT8 Report — veredicto NEGATIVO archivado (NNCF ausente, env congelado)

Fecha (UTC): 2026-09-16. Carril: Deepwork F2. Sin imágenes, solo texto/código.
Tablas congeladas 0-9 FP32 intactas (ningún log 0-9 escrito ni re-evaluado aquí).
`out/bench_local/` y `out/export/act_full_v3_100k/` solo lectura. Sin entrenar, sin instalar paquetes.

## P7 (barato, primero) — método FP32: VEREDICTO

`scripts/export_openvino.py` wrapea el predictor de chunks, NO `select_action` con deque:
`ACTChunkWrapper.forward` llama `self.net(batch)` → `actions` (chunk completo `[1,50,12]`,
net-only, sin cola temporal), compilado con `INFERENCE_PRECISION_HINT=f32`
(`scripts/export_openvino.py:70-78,145`; el rollout usa `policy.predict_action_chunk`
en `scripts/eval_policy.py:226`; cero ocurrencias de `select_action`/`deque` en el export).
Paridad FP32 existente (no re-exportada): max_err **1.1920928955078125e-06** (tol 1e-3,
pass=true, 66.658 MB) — `out/export/act_full_v3_100k/parity.json`.

## NNCF: honest-negative, stop en paso 2

```
.venv/bin/python -c "import nncf" → ModuleNotFoundError: No module named 'nncf' (exit 1)
```

NNCF ausente en el env congelado (openvino 2026.3.0, `.venv` Python 3.10.21).
Env congelado: NO se instala nada. Pasos 3-5 (PTQ/TRANSFORMER o fallback
compress_weights INT8_ASYM, paridad INT8-vs-FP32, closed-loop 0-9, bench INT8)
NO ejecutados por falta de herramienta, no por decisión de diseño.

## Paridad INT8-vs-FP32

No medida (sin IR INT8; `out/export/int8/` vacío por este motivo).
Referencia FP32 leída (solo lectura): max_err 1.192093e-06 — `out/export/act_full_v3_100k/parity.json`.

## Closed-loop (veredicto por err rad NO: por closed-loop)

No evaluado en INT8 (sin IR que evaluar). Congelado FP32 citado, no tocado:
vehicle 3/10 (seeds 1/7650, 4/7775, 7/7750), random 0/10, swap 0/10 —
README Sec Result + `docs/submission_draft/SUBMISSION_TEXT.md` Sec 4.
Comparativa X/10 INT8 vs 3/10 FP32: PENDIENTE (bloqueada por NNCF ausente).
Logs por seed en `out/eval_int8/`: ninguno (nada que loguear sin IR).

## Bench

INT8 sin bench (sin modelo). Referencia FP32 local leída (solo lectura, sin re-ejecutar):
AMD Ryzen 5 3600, p50 68.10 ms / p95 71.53 ms, 16.90 FPS agg / 4.22 per-stream
(niter 100+warmup 10) — `out/bench_local/bench_local_cpu.json`. Xeon no mezclado.

## Veredicto: NEGATIVO archivado (no SHIP)

Sin IR INT8 no hay nada que shipear. Qué citarían los lablab texts si SHIP:
nada de INT8 — solo las cifras FP32 ya citadas (paridad 1.192093e-06, Xeon
40.80 ms/24.51 FPS latency y 27.85 FPS agg throughput, 3/10 frozen).
Desbloqueo futuro (fuera de este pase): proveer NNCF en un env separado
(sin tocar el congelado), PTQ 300 obs reales modo TRANSFORMER con fallback
documentado a compress_weights INT8_ASYM, paridad max_err, closed-loop 0-9
mismo protocolo (EXEC_STEPS=50, stats-root out/lerobot_v3), bench AMD
etiquetado, y solo entonces veredicto SHIP/negativo por closed-loop.
