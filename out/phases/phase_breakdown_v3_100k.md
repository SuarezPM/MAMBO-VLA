# Phase breakdown — vehicle v3-100k, seeds 0–9 (from existing traces)

Vehicle: `out/checkpoints/act_full_v3/checkpoints/100000` (ACT-52M, router-gated).
Protocol: frozen seeds exactly 0–9, EXEC_STEPS=50, stats v3.
Generator: `scripts/analyze_phases.py` (stdlib only, deterministic).
Lane boundary: Carril A never opens `/tmp/mambo-vla-submit`, `out/ood/`,
`out/export/`, `scripts/eval_policy.py`, `sim/`, `training/`, `out/seeds_ood/`.
No PDFs/PNGs/MP4s read. Zero invented numbers.

## 1. Real schema of `out/seeds/traces.jsonl`

- Rows: 10; seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- Keys (union): `instruction, note, outcome, poses_hash, run, seed, skill`
- Outcomes: {'success': 10}; skills: {'table_relay': 10}
- Has dedicated phase field (phase/pick/setdown/handover/place): False
- Rows mentioning a phase token: 0
- Verdict: this file carries NO per-phase telemetry. It records one
  end-to-end `outcome` per frozen seed for the scripted teacher
  (`note: scripted teacher; physics-step p50, no NN forward`).
  It cannot yield vehicle per-phase rates by itself.

## 2. What `out/seeds/traces.full.jsonl` adds (and does not add)

- Rows: 215; keys (union): `instruction, note, outcome, poses_hash, run, seed, skill`
- Rows whose note mentions a carry outcome: 164
- Rows that are v3-100k policy rollouts: False (none found)
- The carry strings (`A carry failed ...`, `B1 carry failed ...`,
  `B carry failed ...`) belong to teacher exploration attempts, not to
  the v3-100k learned policy. They document how the scripted teacher
  failed while generating data; they are NOT vehicle phase passes.
- Therefore: no numerator/denominator for vehicle pick / set-down /
  handover exists in either traces file.

## 3. Teacher contrast (NOT the vehicle — do not conflate)

- `out/seeds/seed_*/result.json` (seeds 0–9, scripted teacher): 10/10
  end-to-end success, each log holding 3 placed carries
  (`A carry ... placed`, `B1 carry ... placed`, `B carry ... placed`).
- Example `out/seeds/seed_1/result.json`: steps 8144, 3/3 carries placed.
- This is the data-generation ceiling, not the learned policy.
  Vehicle rates below never reuse these 10/10 numbers.

## 4. Vehicle phase table (seeds 0–9, v3-100k)

| phase | definition used here | v3-100k rate 0–9 | status | log cited |
|---|---|---|---|---|
| pick | arm A first grasp + lift of the mug | honest-negative — no per-phase telemetry in traces | NOT CITABLE | `out/seeds/traces.jsonl` keys lack any phase field; `out/seeds/traces.full.jsonl` holds only teacher carry notes, zero policy rows |
| set-down | arm A table-relay set-down + park | honest-negative — no per-phase telemetry in traces | NOT CITABLE | same as above |
| handover | arm B re-grip at the relay point | honest-negative — no per-phase telemetry in traces | NOT CITABLE | same as above |
| place | final mug in destination zone (end-to-end success) | 3/10 (seeds 1/7650, 4/7775, 7/7750 steps; seed 3 max 2457 mm over failing seeds) | CITABLE-TRANSCRIBED (raw rollout log outside Carril A lane) | `docs/submission_draft/SUBMISSION_TEXT.md` Sec 4 + `docs/submission_draft/EVIDENCE_INDEX.md` Sec 4 |

Reading the place row honestly:
- 3/10 is end-to-end place, not a per-leg phase pass. Seeds 1, 4, 7
  placed; seed 3 moved 2457 mm (max over failing seeds); seeds 0, 2, 5,
  6, 8, 9 moved (per-seed mm beyond the seed-3 max not provided in lane).
- Transcription check in lane:
  - seed1 7650: present in both docs
  - seed4 7775: present in both docs
  - seed7 7750: present in both docs
  - seed3 2457mm: present in both docs
  - 3/10: present in both docs
- Raw v3-100k rollout log with per-seed rows is outside Carril A
  (lane forbids `training/`, `scripts/eval_policy.py`). No raw path is
  cited for the 3/10; the two docs above are the citable source in lane.

## 5. Negative controls cited in lane

- v3-200k policy rollout seeds 0–9: 0/10, range 65-1620 mm —
  `out/gates/v3_200k_rollout.log` (`=== POLICY200K ===`, 10 rows
  `policy rollout: no place (mug moved Nmm)`, EXEC_STEPS=50).
- v3-200k random baseline: 0/10, 0 mm all seeds — same log
  (`=== RANDOM200K ===`).
- v3-200k swap control: 0/10, router refusal, 0 steps — same log
  (`=== SWAP200K ===`, `router refused (out-of-grammar)`).
- v3-100k random 0/10 (0 mm) + swap 0/10 (refusal, 0 steps) are filed as
  transcribed controls in `docs/submission_draft/SUBMISSION_TEXT.md` Sec 4
  and `docs/submission_draft/EVIDENCE_INDEX.md` Sec 6; their raw logs
  are outside Carril A and are NOT cited as raw paths here.

## 6. Logs cited (exact paths)

- `out/seeds/traces.jsonl` (10 rows, schema above)
- `out/seeds/traces.full.jsonl` (215 rows, teacher exploration archive)
- `out/seeds/seed_0/result.json` … `out/seeds/seed_9/result.json` (teacher 3-carry logs)
- `out/seeds/seed_hashes.json` (frozen input hashes 0–9)
- `docs/submission_draft/SUBMISSION_TEXT.md` Sec 4 (vehicle 3/10 transcription)
- `docs/submission_draft/EVIDENCE_INDEX.md` Sec 3–4, 6 (vehicle + floor + controls transcription)
- `out/gates/v3_200k_rollout.log` (v3-200k 0/10 + random + swap, raw in lane)
- `out/gates/v3_100k_export.log` (export parity max_err 1.192093e-06, IR 66.658 MB)
- `out/bench_intel_incoming/vehicle_20260916T135349Z/bench_fp32_LATENCY_sync.log` (Median 40.80 ms)
- `out/bench_intel_incoming/vehicle_20260916T135349Z/bench_fp32_THROUGHPUT_async.log` (Median 143.66 ms, Throughput 27.85 FPS)
- `out/bench_intel_incoming/acceptance_xeon.txt` (Xeon E-2386G, devices `['CPU']`)
- `out/bench_intel_incoming/cpu_matrix_20260916T140419Z/00_env.txt` (12 threads, Ubuntu 24.04, OpenVINO 2026.3.0)

## 7. Honest-negatives (explicitly NOT claimed)

- Vehicle pick / set-down / handover rates 0–9: no data in lane. Any
  per-phase numerator shown elsewhere without a new instrumented rollout
  is invented and must be rejected.
- Vehicle per-seed mm for failing seeds except the seed-3 max (2457 mm):
  not provided in lane.
- Vehicle per-seed rollout forward p50/p95: not in gate data
  (`EVIDENCE_INDEX.md` Sec 5 states this explicitly); `benchmark_app`
  medians are device latency, a different metric.
- OOD seeds 60–79, export IR bytes, training loss curve: raw paths
  (`out/ood/`, `out/export/`, `training/`) are outside Carril A and were
  not re-opened here. Numbers for those lines are reported from filed
  prose in lane, not re-verified by this script.
- No phase inference from teacher carries: teacher 10/10 never implies
  vehicle phases.

