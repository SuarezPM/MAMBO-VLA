# OOD Disturb — Seeds 100–119 (verify+repair, no retrain)

Status: new seeds 100–119 lie outside everything used (frozen 0–9,
extension 10–29, OOD 60–79 and 80–99); nothing below reuses those numbers.
Runner: `scripts/verify_repair.py` (new; imports `rollout/load_policy/
warmup_egl/EXEC_STEPS/parse_seeds` plus thresholds from
`scripts/eval_policy.py` WITHOUT changing defaults; EXEC_STEPS=50,
stats-root `out/lerobot_v3`, checkpoint frozen v3-100k
`out/checkpoints/act_full_v3/checkpoints/100000`). Hashes:
`out/seeds_disturb/seed_hashes_disturb.json` + bundles
`out/seeds_disturb/bundles_disturb.json` (same heavy generator as OOD:
see `out/ood/RANDOMIZATION.md`). Movement in mm; `*` = fling (>=1000 mm).
Scored gate for THESE runs only is P6-strict (3 cm + yaw + upright +
released); the frozen place gate is logged alongside for reference but
never decides here — past results (0–9, OOD) were judged by place and are
NOT redefined (explicit note in `out/ood/RANDOMIZATION.md`). 0 INFRA in
all three batteries; any infra error would be marked INFRA + log, never
FAIL/SUCCESS.

## disturb (scored recovery test) — 0/20

TEST SETUP per seed: at physics step 3000 the mug is shoved +0.05 m in x
via `set_free_body` (logged `TEST SETUP disturb`, queue cleared); the
repair path itself never teleports — recovery is re-perceive + re-queue
(MAX_RETRIES=2/phase) only. Source per seed:
`out/seeds_disturb/seed_hashes_disturb.json` +
`out/seeds_disturb/bundles_disturb.json` + per-seed
`out/seeds_disturb/logs_disturb/seed_XX.log` + aggregate
`out/seeds_disturb/logs_disturb/eval_disturb_100_119.log` +
`out/seeds_disturb/results_disturb.json`.

| seed | result | steps / movement | note |
|------|--------|------------------|------|
| 100 | FAIL | 86 mm | movement |
| 101 | FAIL | 99 mm | movement |
| 102 | FAIL | 43 mm | movement |
| 103 | FAIL | 156 mm | movement |
| 104 | FAIL | 67 mm | movement |
| 105 | FAIL | 194 mm | movement (pick/handover/set-down verified mid-episode, lost after shove) |
| 106 | FAIL | 81 mm | movement |
| 107 | FAIL | 438 mm | movement |
| 108 | FAIL | 198 mm | movement |
| 109 | FAIL | 48 mm | movement |
| 110 | FAIL | 70 mm | movement |
| 111 | FAIL | 46 mm | movement |
| 112 | FAIL | 46 mm | movement |
| 113 | FAIL | 182 mm | movement |
| 114 | FAIL | 217 mm | movement (pick verified pre-shove) |
| 115 | FAIL | 101 mm | movement (pick verified pre-shove) |
| 116 | FAIL | 312 mm | movement |
| 117 | FAIL | 40 mm | movement |
| 118 | FAIL | 55 mm | movement |
| 119 | FAIL | 60 mm | movement |
| mean | **0/20** | — | no STRICT recovery after mid-episode shove; 0 retries fired (shove stays on-table, knocked-off detector silent); 0/3 recovered among 3 with pick verified pre-shove (105/114/115 per `seed_*.log` grep, all VERIFIED steps <3000) |

## preplaced (skip control) — 20/20

TEST SETUP per seed: mug spawned at DEST at t=0 (logged `TEST SETUP
preplaced`); wrapper must verify place immediately and SKIP all four
phases with 0 policy steps. Source per seed: same hashes/bundles +
per-seed `out/seeds_disturb/logs_preplaced/seed_XX.log` + aggregate
`out/seeds_disturb/logs_preplaced/eval_preplaced_100_119.log` +
`out/seeds_disturb/results_preplaced.json`.

| seed | result | steps / movement | note |
|------|--------|------------------|------|
| 100–119 (each) | SUCCESS | place @ 0 steps | skip (all 4 phases verified at t=0, 0 retries) |
| mean | **20/20** | — | skip logic verified on all 20 seeds; wrapper control, not policy recovery |

## baseline control (same wrapper, no shove) — 0/20 STRICT (1/20 frozen)

Same wrapper + same seeds, no disturbance. Source per seed: same
hashes/bundles + per-seed `out/seeds_disturb/logs_baseline/seed_XX.log` +
aggregate `out/seeds_disturb/logs_baseline/eval_baseline_100_119.log` +
`out/seeds_disturb/results_baseline.json`.

| seed | result (STRICT) | steps / movement | note |
|------|-----------------|------------------|------|
| 100 | FAIL | 129 mm | movement |
| 101 | FAIL | 67 mm | movement |
| 102 | FAIL | 170 mm | movement |
| 103 | FAIL | 259 mm | movement |
| 104 | FAIL | 343 mm | movement |
| 105 | FAIL | 299 mm | frozen-place only (STRICT FAIL: yaw/released) |
| 106 | FAIL | 27 mm | movement |
| 107 | FAIL | 417 mm | movement |
| 108 | FAIL | 2723 mm* | fling (knocked-off, 2 retries exhausted, no recovery) |
| 109 | FAIL | 67 mm | movement |
| 110 | FAIL | 371 mm | movement |
| 111 | FAIL | 28 mm | movement |
| 112 | FAIL | 12 mm | movement |
| 113 | FAIL | 59 mm | movement |
| 114 | FAIL | 73 mm | movement |
| 115 | FAIL | 151 mm | movement (pick verified, never placed) |
| 116 | FAIL | 3103 mm* | fling (knocked-off, 2 retries exhausted, no recovery) |
| 117 | FAIL | 66 mm | movement |
| 118 | FAIL | 64 mm | movement |
| 119 | FAIL | 21 mm | movement |
| mean | **0/20** | — | 0 STRICT; 1/20 frozen (105); flings at 108, 116 with repair fired and exhausted |

## Verdict

Repair is an honest negative on policy recovery: 0/20 disturb recoveries (0/3 recovered among 3 with pick verified pre-shove: 105/114/115) and 0/20 baseline STRICT (repair fired only on the two flung seeds and exhausted without recovery), while the skip control passes 20/20 SKIP (wrapper control, 0 policy steps — not policy capability) — the wrapper verifies and skips correctly but adds no placing capability to the frozen policy.

Citation rule (Oracle gate): the preplaced aggregate MUST always carry its full label `20/20 SKIP (wrapper control, 0 policy steps — not policy capability)`; citing a bare `20/20 STRICT` for it in README/COPY/EVIDENCE is prohibited. Grep over README.md + docs/ today returns no bare `20/20 STRICT` aggregate (verified absent, no out-of-scope edits made).
