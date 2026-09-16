# Randomization — nominal / heavy (documented, harness unchanged)

This file DOCUMENTS ranges only. Nothing here changes the frozen harness:
`scripts/eval_policy.py` defaults untouched (imported, never edited),
`sim/` untouched, `training/` untouched, seeds 0–99 untouched (0–9 frozen,
10–29 filed, 60–99 OOD filed). New seeds 100–119 reuse the heavy generator
verbatim (`seed_bundle_ood` from `scripts/run_seeds_ood.py`, same RNG
`np.random.default_rng(seed ^ 0xD177E2)`).

## Nominal (frozen seeds 0–9; reference only, from `scripts/run_seeds.py`)

- mug xy = (-0.12, 0.02) + U(-0.009, +0.009) each axis
- plate xy = (0.18, 0.10) + U(-0.009, +0.009) each axis
- mug_color_R = 0.33 * U(0.9, 1.1) (G=0.73 B=0.69 A=1.0 fixed)
- plate_color_G = 0.51 * U(0.9, 1.1) (R=0.33 B=0.76 A=1.0 fixed)
- lighting = U(0.92, 1.06)
- layout / instruction / relay (0.0, -0.10, 0.76) / destination
  (0.18, -0.01, 0.76): identical everywhere.

## Heavy (seeds 10–29 extra, 60–99 OOD, 100–119 disturb; from `scripts/run_seeds_extra.py` / `scripts/run_seeds_ood.py`)

- mug xy = (-0.12, 0.02) + U(-0.020, +0.020) each axis
- plate xy = (0.18, 0.10) + U(-0.020, +0.020) each axis
- mug_color_R = 0.33 * U(0.8, 1.2) (G/B/A fixed as above)
- plate_color_G = 0.51 * U(0.8, 1.2) (R/B/A fixed as above)
- lighting = U(0.80, 1.20)
- layout / instruction / relay / destination: identical to nominal.

## F3 test setups on 100–119 (logged per run, not harness changes)

- disturb: at physics step 3000 the mug is shoved +0.05 m in x via
  `set_free_body` (logged `TEST SETUP disturb`, action queue cleared).
  Repair itself never teleports.
- preplaced: mug spawned at DEST (0.18, -0.01,
  TABLE_Z + MUG_HALFHEIGHT + SPAWN_CLEARANCE) at t=0 via `set_free_body`
  (logged `TEST SETUP preplaced`); expected outcome is full skip.
- wrapper: re-perceive before each phase (pick, set-down, handover,
  place); skip if already verified; knocked-off (z < TABLE_Z - 0.02 or
  |x|/|y| > 0.60) triggers re-queue (clear pending chunk, re-query policy
  from re-perceived state); MAX_RETRIES=2 per phase; everything to log.

## Strict success P6 — APPLICABLE ONLY TO THESE RUNS (100–119)

A new-run episode counts SUCCESS iff ALL hold at the same tick:

- |mug_xy - DEST_xy| <= 0.030 (3 cm),
- |mug yaw| <= 15 deg (about spawn yaw 0),
- mug up-axis > cos(15 deg) (same upright as frozen),
- |mug_z - rest_z| < 0.004 and no penetration below -0.001 (same as frozen),
- released: both gripper jaws open > 0.30 AND both grip sites > 0.040 m
  (xy) from the mug center.

The frozen place gate (1.5 cm radius, same upright/separation, no
yaw/released check) is logged alongside (`success_frozen`) for reference
but NEVER decides these runs.

EXPLICIT NON-REDEFINITION NOTE: frozen seeds 0–9 (vehicle 3/10: seeds 1,
4, 7) and all OOD seeds (60–79, 80–99) were judged by the frozen place
gate and are NOT redefined, rescored, or restated by P6. Any comparison of
0/20 STRICT here against 3/10 place there crosses both seeds AND gates and
is therefore not apples-to-apples; both numbers are reported side by side
with this caveat, and the past verdicts stand unchanged.
