# OOD Results — Seeds 60–79 (frozen OOD protocol)

Status: gate-verified transcription. Seeds 60–79 lie outside the training
distribution (frozen 0–9 plus extension 10–29); nothing below reuses frozen-seed
numbers. Runner: `scripts/run_seeds_ood.py`. Hashes:
`out/seeds_ood/seed_hashes_ood.json` + bundles
`out/seeds_ood/bundles_ood.json`. Movement in mm; `*` = fling
(high-energy ejection, not a place attempt).
Note (held-out 10-29): `out/seeds_extra/seed_hashes.json` +
`out/seeds_extra/traces_extra.jsonl` exist (scripted traces); no policy
rollout table for seeds 10-29 exists — no 0/20 policy claim made here.

## v3-100k (scored vehicle) — 1/20

Source per seed: `out/seeds_ood/seed_hashes_ood.json` + `out/seeds_ood/bundles_ood.json`.

| seed | result | steps / movement | note |
|------|--------|------------------|------|
| 60 | FAIL | 110 mm | movement |
| 61 | FAIL | 203 mm | movement |
| 62 | FAIL | 237 mm | movement |
| 63 | FAIL | 63 mm | movement |
| 64 | FAIL | 112 mm | movement |
| 65 | SUCCESS | place @ 7850 steps | place |
| 66 | FAIL | 115 mm | movement |
| 67 | FAIL | 1848 mm* | fling |
| 68 | FAIL | 175 mm | movement |
| 69 | FAIL | 278 mm | movement |
| 70 | FAIL | 14 mm | movement |
| 71 | FAIL | 75 mm | movement |
| 72 | FAIL | 72 mm | movement |
| 73 | FAIL | 10 mm | movement |
| 74 | FAIL | 427 mm | movement |
| 75 | FAIL | 141 mm | movement |
| 76 | FAIL | 105 mm | movement |
| 77 | FAIL | 183 mm | movement |
| 78 | FAIL | 4240 mm* | fling |
| 79 | FAIL | 243 mm | movement |
| mean | **1/20** | — | seed 65 places; flings at 67, 78 |

## v1-100k (floor line) — 1/20

Source per seed: `out/seeds_ood/seed_hashes_ood.json` + `out/seeds_ood/bundles_ood.json`.

| seed | result | steps / movement | note |
|------|--------|------------------|------|
| 74 | SUCCESS | place @ 7725 steps | place |
| other 19 seeds | FAIL | per-seed log PENDING, agregado 0–483 mm no citable por seed | movement, near-static floor (6 seeds ≤ 5 mm per aggregate note; no per-seed log found) |
| mean | **1/20** | — | seed 74 places |

## Verdict

Ambas líneas colocan fuera de todo lo entrenado: the scored vehicle and the
floor line each place exactly once in twenty OOD seeds (v3-100k seed 65,
v1-100k seed 74), with all other seeds failing by low movement or fling.
No OOD seed re-rolled; failure notes retained.
