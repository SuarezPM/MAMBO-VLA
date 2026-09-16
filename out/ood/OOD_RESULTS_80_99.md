# OOD Results — Seeds 80–99 (frozen OOD protocol)

Status: gate-verified transcription. Seeds 80–99 lie outside the training
distribution (frozen 0–9 plus extension 10–29 plus OOD 60–79); nothing below
reuses frozen-seed numbers. Runner: `scripts/run_seeds_ood.py`
(`seed_bundle_ood` reused verbatim; equivalent harness
`/tmp/opencode/eval_ood_80_99.py` imports
`rollout/load_policy/warmup_egl/EXEC_STEPS/parse_seeds` from
`scripts/eval_policy.py` without changing defaults; EXEC_STEPS=50,
stats-root `out/lerobot_v3`, checkpoint frozen v3-100k). Hashes:
`out/seeds_ood/seed_hashes_ood_80_99.json` + bundles
`out/seeds_ood/bundles_ood_80_99.json`. Movement in mm; `*` = fling
(>=1000 mm high-energy ejection, not a place attempt; same threshold as
60–79 where flings were 1848/4240 mm and max non-fling movement 427 mm).
No INFRA in the scored run (0/20); any infra error would be marked INFRA +
log, never FAIL.

## v3-100k (scored vehicle) — 1/20

Source per seed: `out/seeds_ood/seed_hashes_ood_80_99.json` + `out/seeds_ood/bundles_ood_80_99.json` + per-seed `out/seeds_ood/logs_ood_80_99/seed_XX.log` + aggregate `out/seeds_ood/logs_ood_80_99/eval_80_99_rerun.log` + `out/seeds_ood/logs_ood_80_99/results_80_99.json` (scored run). First full pass `out/seeds_ood/logs_ood_80_99/eval_80_99.log` kept as determinism check (same 1/20, seed 93).

| seed | result | steps / movement | note |
|------|--------|------------------|------|
| 80 | FAIL | 8 mm | movement |
| 81 | FAIL | 105 mm | movement |
| 82 | FAIL | 67 mm | movement |
| 83 | FAIL | 2493 mm* | fling |
| 84 | FAIL | 14 mm | movement |
| 85 | FAIL | 7 mm | movement |
| 86 | FAIL | 164 mm | movement |
| 87 | FAIL | 66 mm | movement |
| 88 | FAIL | 67 mm | movement |
| 89 | FAIL | 80 mm | movement |
| 90 | FAIL | 213 mm | movement |
| 91 | FAIL | 69 mm | movement |
| 92 | FAIL | 276 mm | movement |
| 93 | SUCCESS | place @ 7775 steps | place |
| 94 | FAIL | 71 mm | movement |
| 95 | FAIL | 121 mm | movement |
| 96 | FAIL | 161 mm | movement |
| 97 | FAIL | 260 mm | movement |
| 98 | FAIL | 59 mm | movement |
| 99 | FAIL | 86 mm | movement |
| mean | **1/20** | — | seed 93 places; fling at 83 |

## Verdict

v3-100k places once in twenty OOD seeds 80–99 (seed 93 @ 7775 steps), all others failing by low movement or fling (83); rerun-stable 1/20 with zero INFRA.
