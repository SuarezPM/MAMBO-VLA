"""M1-extension v3: scripted table relay over NEW seeds 30-59 only.

Teacher: REUSED via import from run_seeds (run_seed + INSTRUCTION). No
teacher logic is duplicated here. Same 4-step pick (above -> descend ->
close -> lift) plus table-supported relay (arm A places at relay, parks;
arm B re-grips in two arc legs and places at destination).

Staging variety (MIRROR of scripts/run_seeds_extra.py EXTRA range):
  V3 (30-59, this script):
    mug_x   = -0.12 + U(-0.020, +0.020)
    mug_y   =  0.02 + U(-0.020, +0.020)
    plate_x =  0.18 + U(-0.020, +0.020)
    plate_y =  0.10 + U(-0.020, +0.020)
    mug_color_R   = 0.33 * U(0.8, 1.2)  (G=0.73 B=0.69 A=1.0 fixed)
    plate_color_G = 0.51 * U(0.8, 1.2)  (R=0.33 B=0.76 A=1.0 fixed)
    lighting = U(0.80, 1.20)
  layout / instruction / relay_point / destination: identical to M1.
  RNG stream: np.random.default_rng(seed ^ 0xD177E2), same derivation as M1.

Integrity: seeds 0-29 + out/lerobot + out/lerobot_v2 + out/seeds/ +
out/seeds_extra/ + out/dataset/ + out/dataset_extra/ are FROZEN and never
touched here. Bundles are hashed with seed_freeze.hash_bundle BEFORE
running; outputs go only to out/seeds_v3/ (bundles, results,
screenshots, seed_hashes.json, sample render) and out/dataset_v3/
(episodes seed_<s>.npz, same schema as M1). Traces go to
out/seeds_v3/traces_v3.jsonl (never the frozen traces files). Gate
requires 30/30 success; nonzero exit otherwise.

Runtime: MuJoCo CPU headless only. Same backend as the v2 lane
(MUJOCO_GL from env, repo default egl headless). Physics is CPU either
way. No torch import, no CUDA compute, no training-process contact.
Launch with CUDA_VISIBLE_DEVICES="" to keep this lane off the GPU that
trains the v2 run.
"""

import json
import os
import sys
from pathlib import Path

os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "osmesa")

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_seeds as M1
from mambo_vla_rc.bench_stub import print_table
from mambo_vla_rc.seed_freeze import hash_bundle
from mambo_vla_rc.trace import append_trace

V3_SEEDS = list(range(30, 60))
RUN_TAG_V3 = "e2r-v3"

OUT_SEEDS_V3 = ROOT / "out" / "seeds_v3"
OUT_DATASET_V3 = ROOT / "out" / "dataset_v3"
TRACE_V3 = OUT_SEEDS_V3 / "traces_v3.jsonl"


def seed_bundle_v3(seed: int) -> dict:
    rng = np.random.default_rng(seed ^ 0xD177E2)
    mug_xy = [-0.12 + float(rng.uniform(-0.020, 0.020)), 0.02 + float(rng.uniform(-0.020, 0.020))]
    plate_xy = [0.18 + float(rng.uniform(-0.020, 0.020)), 0.10 + float(rng.uniform(-0.020, 0.020))]
    teal = [0.33 * float(rng.uniform(0.8, 1.2)), 0.73, 0.69, 1.0]
    blue = [0.33, 0.51 * float(rng.uniform(0.8, 1.2)), 0.76, 1.0]
    return {
        "layout": "dual_so101_dinner shelf-only, no drawer",
        "mug_xy": mug_xy,
        "plate_xy": plate_xy,
        "mug_color": [round(float(c), 4) for c in teal],
        "plate_color": [round(float(c), 4) for c in blue],
        "lighting": round(float(rng.uniform(0.80, 1.20)), 4),
        "instruction": M1.INSTRUCTION,
        "relay_point": list(M1.scene.RELAY_POINT),
        "destination": list(M1.scene.DEST_POINT),
    }


def main() -> int:
    only = [int(a) for a in sys.argv[1:] if a.lstrip("-").isdigit()]
    seeds = [s for s in V3_SEEDS if not only or s in only]
    bundles = {s: seed_bundle_v3(s) for s in V3_SEEDS}
    hashes = {s: hash_bundle(bundles[s]) for s in V3_SEEDS}
    OUT_SEEDS_V3.mkdir(parents=True, exist_ok=True)
    OUT_DATASET_V3.mkdir(parents=True, exist_ok=True)
    (OUT_SEEDS_V3 / "seed_hashes.json").write_text(
        json.dumps({str(k): v for k, v in hashes.items()}, indent=2) + "\n"
    )

    rows = []
    first_success_shot = None
    for seed in seeds:
        result = M1.run_seed(seed, bundles[seed], hashes[seed])
        seed_dir = OUT_SEEDS_V3 / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "result.json").write_text(
            json.dumps({k: v for k, v in result.items()
                        if k not in ("frames", "frame_site", "frame_state", "frame_time",
                                     "frame_active", "frame_grip", "frame_jaw_cap",
                                     "frame_ctrl", "frame_mug", "frame_ori",
                                     "step_qpos", "step_jaw", "step_jaw_meas",
                                     "step_jaw_cap", "step_ctrl",
                                     "step_force", "step_time")}, indent=2) + "\n"
        )
        last = result["frames"]["overhead"][-1] if result["frames"]["overhead"] else None
        if last is not None:
            Image.fromarray(last).save(seed_dir / "screenshot.png")
            if result["success"] and first_success_shot is None:
                first_success_shot = last
        failure_log = seed_dir / "failure.log"
        if not result["success"]:
            failure_log.write_text(
                f"seed {seed} FAILED: {result['cause']}\nfinal={result['final_pos']}\n"
                f"hash={hashes[seed]}\nlog={' | '.join(result['log'])}\n"
            )
        elif failure_log.exists():
            failure_log.unlink()
        frames = {k: np.asarray(v, dtype=np.uint8) for k, v in result["frames"].items()}
        np.savez_compressed(
            OUT_DATASET_V3 / f"seed_{seed}.npz",
            **{f"frames_{k}": v for k, v in frames.items()},
            frame_site_left=result["frame_site"]["left"],
            frame_site_right=result["frame_site"]["right"],
            frame_state=result["frame_state"],
            frame_time=result["frame_time"],
            frame_active=result["frame_active"],
            frame_grip=result["frame_grip"],
            frame_jaw_cap=result["frame_jaw_cap"],
            frame_ctrl=result["frame_ctrl"],
            frame_mug=result["frame_mug"],
            frame_ori=result["frame_ori"],
            step_qpos=result["step_qpos"],
            step_jaw=result["step_jaw"],
            step_jaw_meas=result["step_jaw_meas"],
            step_jaw_cap=result["step_jaw_cap"],
            step_ctrl=result["step_ctrl"],
            step_force=result["step_force"],
            step_time=result["step_time"],
            instruction=np.array(M1.INSTRUCTION),
            seed=np.array(seed),
            poses_hash=np.array(hashes[seed]),
            success=np.array(result["success"]),
            bundle=np.array(json.dumps(bundles[seed], sort_keys=True)),
        )
        append_trace(
            TRACE_V3,
            {
                "skill": "table_relay",
                "seed": seed,
                "run": RUN_TAG_V3,
                "instruction": M1.INSTRUCTION,
                "poses_hash": hashes[seed],
                "outcome": "success" if result["success"] else "fail",
                "note": result["note"],
                **({"failure_cause": result["cause"]} if result["cause"] else {}),
            },
        )
        rows.append(
            {
                "seed": seed,
                "success": result["success"],
                "steps": result["steps"],
                "forward_p50_ms": result["forward_p50_ms"],
                "note": result["note"],
            }
        )
        print(f"seed {seed}: success={result['success']} steps={result['steps']} "
              f"xy_err={result['final_xy_err_m'] * 1000:.1f}mm | {' | '.join(result['log'])}",
              flush=True)

    if first_success_shot is not None:
        Image.fromarray(first_success_shot).save(OUT_SEEDS_V3 / "sample_render.png")

    print_table(rows, M1.INSTRUCTION)
    wins = sum(1 for r in rows if r["success"])
    print(f"v3 gate: {wins}/{len(rows)} seeds 30-59")
    print(f"hashes: {OUT_SEEDS_V3 / 'seed_hashes.json'} traces: {TRACE_V3} dataset: {OUT_DATASET_V3}")
    if wins != len(rows) or len(rows) != 30:
        print(f"V3 GATE FAILED: {wins}/{len(rows)} (required 30/30)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
