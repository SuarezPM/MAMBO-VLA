# ADR 0003 — FP32-only scored

Fecha: 2026-09-16
Estado: aceptada

## Contexto

- Artefacto servido: IR FP32 `out/export/act_full_v3_100k/` — 66.658 MB, paridad max_err 1.1920928955078125e-06 vs tol 1e-3, pass=true (`out/export/act_full_v3_100k/parity.json`; `out/frozen-selection.json` `model`).
- Bench Xeon congelado (otro host, citado no mezclado): latency-sync median 40.80 ms / 24.51 FPS y throughput-async 27.85 FPS aggregate (`out/bench_intel_incoming/vehicle_20260916T135349Z/bench_fp32_LATENCY_sync.log`; `out/bench_intel_incoming/vehicle_20260916T135349Z/bench_fp32_THROUGHPUT_async.log`; citado en `docs/submission_draft/SUBMISSION_COPY.md` Additional Info).
- FP16 no-op medido: rehearsal fp32 ≈ fp16 en las cuatro celdas (latency-sync 40.83 vs 40.86 ms) (`docs/submission_draft/SUBMISSION_TEXT.md` Sec 5; `docs/submission_draft/EVIDENCE_INDEX.md` Sec Intel rehearsal); sin squeeze reclamado.
- INT8 bloqueado: NNCF ausente en el env congelado, sin IR INT8, veredicto NEGATIVO archivado (`out/eval_int8/INT8_REPORT.md`).

## Decisión

Scored ÚNICAMENTE FP32 CPU LATENCY con la tolerancia 1e-3 citada. FP16 queda como parity probe, no como artefacto servido. INT8 no se sirve en este pase.

## Consecuencias

- El closed-loop servido con el artefacto puntuado es 3/10 (seeds 1, 4, 7) (`docs/submission_draft/SUBMISSION_TEXT.md` Sec 5).
- Ninguna cifra de latencia/throughput amortiza replay entre chunk sizes distintos; solo latencia de inferencia del dispositivo.
- Desbloquear INT8 exige env separado con herramienta, PTQ documentado, paridad y closed-loop con el mismo protocolo; fuera de este pase.

## Alternativas descartadas

- Servir FP16: descartado por no-op medido en este path CPU; no hay squeeze que reclamar.
- Servir INT8: descartado por ausencia de herramienta e IR; no hay nada que shipear hasta nuevo veredicto por closed-loop.
