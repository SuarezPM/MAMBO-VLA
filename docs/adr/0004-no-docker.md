# ADR 0004 — No-Docker

Fecha: 2026-09-16
Estado: aceptada (honest-negative)

## Contexto

- Sin imagen Docker construida: `n/a — no Dockerfile/image built` (`docs/submission_draft/EVIDENCE_INDEX.md` Sec handoff); disclosures CPU-only/host, sin imagen Docker ni VPS (`docs/submission_draft/SUBMISSION_COPY.md` Additional Info).
- Stack pinnado en texto: Python 3.10, mujoco==3.12.0, lerobot==0.4.4, openvino==2026.3.0, torch CUDA train con versión exacta en lock (`requirements.txt` + `requirements.lock`; `docs/submission_draft/SUBMISSION_TEXT.md` Sec 3).
- Reproducibilidad por hashes + scripts con defaults congelados: `out/seeds/seed_hashes.json`, `out/seeds_extra/seed_hashes.json`, `out/seeds_ood/seed_hashes_ood.json`, `out/seeds_ood/seed_hashes_ood_80_99.json`, `out/seeds_disturb/seed_hashes_disturb.json` (`out/frozen-selection.json`); `scripts/bench_matrix_cpu.sh`; tabla FRESH verificable (`scripts/results_table.py --check`).

## Decisión

Sin imagen Docker en este pase. La reproducibilidad se sostiene vía lock + scripts + hashes + logs congelados, declarada como honest-negative.

## Consecuencias

- El handoff Intel queda en prosa + logs (`docs/submission_draft/SUBMISSION_TEXT.md` Sec 7); la parte de imagen queda FINAL honest-negative.
- Cualquier imagen futura exige pin exacto al stack del lock y re-verificación de paridad y bench; fuera de este pase.

## Alternativas descartadas

- Construir imagen Docker ahora: descartado por alcance y por ausencia de Dockerfile; habría tocado env congelado.
- Imagen VPS: descartada por la misma razón; sin imagen que citar.
