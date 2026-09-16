"""OOD held-out: frozen bundles for NEW seeds 60-79 (vehicle generalization test).

Mirrors the run_seeds_extra pattern exactly:
- Teacher constants REUSED via import from run_seeds (INSTRUCTION,
  RELAY_POINT, DEST_POINT) so strings/points cannot drift. No teacher
  logic duplicated.
- SAME EXTRA randomization ranges as seeds 10-29 (wider than 0-9):
    mug/plate xy  = base + U(-0.020, +0.020)
    mug_color_R   = 0.33 * U(0.8, 1.2)  (G=0.73 B=0.69 A=1.0 fixed)
    plate_color_G = 0.51 * U(0.8, 1.2)  (R=0.33 B=0.76 A=1.0 fixed)
    lighting = U(0.80, 1.20)
  layout / instruction / relay / destination identical; RNG stream
  np.random.default_rng(seed ^ 0xD177E2), same derivation.
- Bundles hashed with seed_freeze.hash_bundle BEFORE any use; outputs go
  ONLY to out/seeds_ood/ (new dir): seed_hashes_ood.json + bundles_ood.json.

Teacher episodes are deliberately NOT run here: eval_policy loads initial
conditions procedurally from these bundles and never reads teacher
datasets, so no dataset_ood is needed and none is created.

Runtime: pure numpy RNG + hashing. CPU-only by construction (no torch, no
mujoco import of our own; for the generation step run with
CUDA_VISIBLE_DEVICES="" on the command line — never baked into the module,
so importing this file from the eval harness cannot hide the GPU).
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_seeds as M1
from mambo_vla_rc.seed_freeze import hash_bundle

OOD_SEEDS = list(range(60, 80))

OUT_SEEDS_OOD = ROOT / "out" / "seeds_ood"


def seed_bundle_ood(seed: int) -> dict:
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
    bundles = {s: seed_bundle_ood(s) for s in OOD_SEEDS}
    hashes = {s: hash_bundle(bundles[s]) for s in OOD_SEEDS}
    OUT_SEEDS_OOD.mkdir(parents=True, exist_ok=True)
    (OUT_SEEDS_OOD / "seed_hashes_ood.json").write_text(
        json.dumps({str(k): v for k, v in hashes.items()}, indent=2) + "\n"
    )
    (OUT_SEEDS_OOD / "bundles_ood.json").write_text(
        json.dumps({str(k): v for k, v in bundles.items()}, indent=2) + "\n"
    )
    # Self-verify 20/20: recompute from the generator and compare.
    saved = json.loads((OUT_SEEDS_OOD / "seed_hashes_ood.json").read_text())
    bad = [s for s in OOD_SEEDS if hash_bundle(seed_bundle_ood(s)) != saved[str(s)]]
    print(f"ood freeze: {len(OOD_SEEDS) - len(bad)}/{len(OOD_SEEDS)} seeds 60-79")
    if bad:
        print(f"MISMATCH: {bad}")
        return 1
    print(f"hashes: {OUT_SEEDS_OOD / 'seed_hashes_ood.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
