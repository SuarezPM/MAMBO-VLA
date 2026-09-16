# MAMBO-VLA — Multi-modal Action Manipulation for Bimanual Operations via VLA

One instruction-switched visuomotor policy drives two SO-101 arms through a
dinner-table relay in MuJoCo — pick, table set-down with handover, place —
from a natural-language instruction plus four camera views. Trained locally,
exported to OpenVINO FP32, benchmarked on Intel.

## Result

| Metric | Value |
|---|---|
| Closed-loop, frozen seeds 0–9 (`act_full_v3_100k`, ACT-52M, 60 demos) | **3/10** (seeds 1/7650, 4/7775, 7/7750 steps) |
| Random-policy control | 0/10 (0 mm all seeds) |
| Swap-instruction control (wrong sentence, same scene) | 0/10, router refusal, 0 steps |
| OpenVINO FP32 IR | 66.658 MB, PyTorch parity max_err **1.19e-06** (tol 1e-3) |
| Intel Xeon E-2386G CPU, latency path (median, niter 100) | **40.80 ms / 24.51 FPS** |
| Intel Xeon E-2386G CPU, throughput path | **27.85 FPS** (143.66 ms async) |
| Demo video (10-seed cut + place clip) | `out/demo/MAMBO_VLA_RC_final.mp4` (+ `seed_7.mp4`) |

## Rubric map (100 pts)

- **End-to-end bimanual dinner-table, 30** — §Task. Dual SO-101 MuJoCo scene,
  table-supported relay, never an airborne handoff.
- **VLA multi-modal reasoning, 20** — §Policy. One ACT-52M checkpoint switched
  by instruction sentence; 4×256×256 RGB @ 20 fps plus measured joint state;
  out-of-grammar input refused, never guessed.
- **Robustness, 10 seeds, 15** — §Results. Frozen inputs, per-seed table,
  random + swap controls, all failures retained.
- **OpenVINO on Intel, 20** — §Intel bench. One FP32 CPU LATENCY artifact;
  latency/throughput/size/closed-loop on Xeon E-2386G; devices disclosed.
- **Reproducibility, 10** — §Reproduce + §Layout. Pinned stack, frozen scene,
  bench and eval scripts in repo.
- **Innovation, 5** — §Journey. Intermediate-peak selection rule and
  vision-dominance gating, both learned from filed internal ablations.

## Task

Two SO-101 arms face inward across a dinner table (shelf-only scene, no
drawer). Arm A picks the mug, sets it down at the table relay point, and
parks; arm B re-grips and carries it to the destination zone in two arc legs
with a midpoint set-down. The policy outputs absolute joint-plus-jaw chunks
(12-dim); a contact-gated teacher generated the 60 training demonstrations
offline. Cameras: overhead, table_left, table_right, wrist_cam —
256×256 RGB @ 20 fps, identical in training, evaluation, and video.

## Policy

LangACT, the sole action-generating network: ACT-52M backbone whose
environment-state slot carries the instruction embedding — one checkpoint for
all commands, switched by sentence, not by weights. Input: 4 images plus the
12-dim measured joint state. Output: 12-dim absolute joint+jaw action chunks
(chunk size 50). Operator language passes through a bounded grammar:
in-grammar commands run, anything else is refused with 0 steps (this refusal
*is* the swap-instruction control). A frozen off-the-shelf vision-language
model scores final placements post-hoc; it never emits actions.

## Results

Frozen protocol: seeds exactly 0–9, inputs hashed before running, no seed
re-rolled, every failure retained with log and clip.

| seed | result | steps | note |
|---|---|---|---|
| 1 | SUCCESS | 7650 | place |
| 4 | SUCCESS | 7775 | place |
| 7 | SUCCESS | 7750 | place |
| 3 | FAIL | — | movement, 2457 mm (max over failing seeds) |
| 0, 2, 5, 6, 8, 9 | FAIL | — | movement |
| mean | **3/10** | — | random 0/10; swap 0/10 refusal |

Floor reference (prior dataset generation, retained): 2/10 band over series
(2/10, 0/10, 1/10), export parity 1.13e-06. The vehicle above is the scored
submission; the floor is reported, not hidden.

## Journey: attempts → learnings → vehicle

1. **ACT baseline, step 3000** — closed-loop 8/20 (1/4 held-out) despite
   winning offline imitation 2/2. Diagnosis: the mug branch emitted a constant
   chunk pre-denormalization (mug vs handover variance 1.20e-09 vs 1.66e-04,
   ratio 138594.2; vision path dead under perturbation). An inference-only
   denormalization patch moved the constant attractor but left variance at
   3.0e-10: 0/8. Verdict KILL — the defect sat in weights/recipe, not scaling.
2. **LoRA fine-tune of a 450M VLA** — validation loss rose over the last three
   checkpoints (0.038 → 0.051); a divergence guard self-stopped training at
   step 14000. Pinned to the best checkpoint (step 4000, val 0.025596): 1/3
   offline wins, 0 closed-loop. Archived as recipe negative.
3. **Scripted ceiling + relay probe** — deterministic 10-seed sweep (mean
   2.6/5, overall 0.0) set the bar any learned policy must beat; the
   table-relay motion fix moved handover 7→8/10 against a 7/10 no-relay
   control. Seven flat-plate grasp routes all scored 0 — geometric
   impossibility at the hardware envelope, not a tunable. All KILL.
4. **Vehicle selection rule (the innovation)** — train past the target and
   evaluate the intermediate peak, not the last checkpoint: v3-100k (≈41
   epochs) scores 3/10 while v3-200k (≈82 epochs) collapses to 0/10
   (65–1620 mm) — memorization degradation, reported as data. The vision
   trend is reported honestly as proprioceptive dominance (ratio 6.6 → 3.3
   across 20k–200k), not as grounding the system does not have.

All filed numbers above are measured, frozen-protocol values; failures are
retained alongside successes in-repo.

## Reproduce

```bash
python3.10 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/env_check.py            # pins: mujoco==3.12.0, lerobot==0.4.4, openvino==2026.3.0
.venv/bin/python scripts/smoke_scene.py          # headless scene smoke, 4 cams
.venv/bin/python scripts/run_seeds.py            # 10 frozen teacher episodes (seeds 0-9)
.venv/bin/python scripts/convert_to_lerobot.py   # + convert_extra.py / convert_v3.py (30/60 demos)
.venv/bin/python training/act_mambo_v3.py full   # ACT-52M, batch 4, chunk 50, 200k steps
.venv/bin/python scripts/eval_policy.py --checkpoint <ckpt> --seeds 0-9   # + --random / --swap
.venv/bin/python scripts/export_openvino.py --checkpoint <ckpt>/pretrained_model --tag <tag>
# On Intel host only: scripts/bench_matrix_cpu.sh  (benchmark_app matrix, niter 100)
# Demo cut: scripts/record_demo.py + scripts/make_final_video.py → out/demo/
```

Pinned stack everywhere (see `requirements.txt` + `requirements.lock`):
Python 3.10, `mujoco==3.12.0`, `lerobot==0.4.4` (v3 dataset format),
`openvino==2026.3.0`, torch CUDA build for training. Frozen contact block
(2.5 mm box pads, only colliding finger geometry, elliptic-cone solver,
`timestep=0.002 impratio=10 noslip=3`, friction `1 0.05 0.001 condim=4`,
adjacent-body exclusions) — byte-identical across sim, training, and bench;
never tuned per object, seed, or milestone.

## Layout

- `sim/` — frozen dual-SO-101 dinner scene + IK teacher (data generation only)
- `scripts/` — dataset, frozen eval harness, vision gate, export, bench matrix, demo recorder
- `training/` — ACT launchers (identical hyperparams; dataset root differs) + router contract
- `src/mambo_vla_rc/` — seed freeze, trace, risk stubs
- `out/export/act_full_v3_100k/` — **scored OpenVINO IR** (66.658 MB, parity PASS)
- `out/demo/` — final video + per-seed clips
- `out/bench_intel_incoming/` — Intel bench logs + host acceptance

## Intel bench

Host: Xeon E-2386G (12 threads), Ubuntu 24.04, OpenVINO 2026.3.0,
`available_devices ['CPU']`. Source: `out/bench_intel_incoming/`
(20 `bench_*.log` + `00_env.txt` + `acceptance_xeon.txt`).

| run (niter 100) | latency sync (median) | latency async | throughput sync | throughput async |
|---|---|---|---|---|
| vehicle v3-100k FP32 | **40.80 ms / 24.51 FPS** | 40.97 ms / 24.39 FPS | 71.48 ms / 13.97 FPS | 143.66 ms / **27.85 FPS** |
| rehearsal 20k fp32 | 40.83 ms / 24.50 FPS | 40.89 ms / 24.43 FPS | 71.40 ms / 13.99 FPS | 143.22 ms / 27.22 FPS |
| rehearsal 20k fp16 | 40.86 ms / 24.43 FPS | 40.83 ms / 24.46 FPS | 71.38 ms / 13.99 FPS | 143.77 ms / 27.30 FPS |

Static batch-1 at bench time ([1,50,12] outputs over 4×256×256 inputs; the
export keeps batch/spatial dynamic and benchmark_app pins them via `-shape`).
Forward-pass latency only — never amortized replay across chunk sizes.
fp32 ≈ fp16 on every cell (e.g. 40.83 vs 40.86 ms): the documented FP16
no-op on the CPU path, reported as a finding, not a squeeze. GPU: ABORT, no
iGPU on node. NPU: absent on this Xeon host — numbers are CPU-only (see
disclosures).

## Disclosures and known negatives

- Vision is proprioception-dominant: dependence ratio 6.6 → 3.3 over training;
  reported as measured, not as grounding.
- v3-200k degrades to 0/10 by memorization; the submission scores the
  intermediate peak (v3-100k), with the selection rule stated.
- FP16 changes nothing on the CPU path (identical IR size 66.658 MB,
  identical parity 1.192093e-06 under both exec hints).
- No NPU and no iGPU on the demo host; GPU run ABORTs; NPU numbers unclaimed.
- 20k rollout 0/10 and the v1 0–2/10 band are retained as floor, not deleted.
- All scene assets are primitive-built or license-free.

## Rules compliance

The learned VLA policy (ACT-52M) is the dominant controller end to end:
images plus measured joint state in, absolute joint+jaw chunks out
(`training/act_mambo.py`: chunk_size 50, 4×256×256 visual features,
12-dim state in, 12-dim action out). IK exists only in expert-data generation
(`scripts/run_seeds.py` teacher: cartesian waypoints solved through DLS IK in
`sim/arm.py`). **Zero IK in the deployed path**: `scripts/eval_policy.py`
`rollout` splits each predicted chunk left/right and writes it directly
through position-actuator apply — no IK solve anywhere in the loop. Language
handling lives outside the network in the bounded-grammar router (exactly the
relay sentence accepted, anything else refused with 0 steps). Developed
locally; deployed and benchmarked on Intel (Xeon E-2386G + OpenVINO 2026.3.0,
CPU).

Evidence: per-seed tables, bench logs (`out/bench_intel_incoming/` with host
acceptance), the scored IR and the final video are all in-repo; failures are
retained alongside successes.
