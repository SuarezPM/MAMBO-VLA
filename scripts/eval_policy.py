"""Policy rollout eval in the frozen sim (keys: checkpoint + seeds).

The router accepts exactly one in-grammar instruction (the relay
sentence, imported from the teacher so the strings cannot drift). Any
other sentence is refused before any physics runs: the arms never move
and the episode scores as a failure. That refusal IS the
swap-instruction control, measured by this same harness.

The rollout drives the sim through the same actuator path as the
teacher: each predicted 12-dim action chunk splits into left/right
(5 joints + jaw) and is written with the same position-actuator apply
call the teacher uses. No teleporting, no state edits. Jaw authority is
held constant at the caged value (1.0 Nm, documented in training/) for
the whole rollout.

Why constant-caged authority is conservative (document only; the
controller is untouched): the teacher travels WIDE open (3.35 Nm, fast
rigid moves) and only firms to 1.0 Nm once caged, while the rollout
never exceeds 1.0 Nm. That can only under-drive the jaw, never ram it:
free-space slews are slower and travel holds are gentler than the
demos, so any rollout that succeeds does so despite weaker jaw
authority, not because of stronger hardware than the teacher had. The
matching risk (a swinging load ratcheting the jaw open under the lower
ceiling) fails loudly through the same gate, never silently.

Gate protocol: thresholds are the teacher's (radius imported, upright
cos(15deg) / separation +-0.004 / penetration -0.001 mirrored);
chunk execution is frozen at EXEC_STEPS with the value logged per row;
--random runs the uniform joint-range baseline through the same path.
F-stats: --stats-root selects the dataset root used ONLY to load the
normalization statistics for the pre/post processors (default out/lerobot,
frozen behavior preserved); checkpoints trained on recomputed stats (v2/v3,
different visual scale) must pass their own root or the policy is denormalized
through the wrong statistics.

Usage:
  .venv/bin/python scripts/eval_policy.py --checkpoint out/checkpoints/act_smoke --seeds 0-1
  .venv/bin/python scripts/eval_policy.py --checkpoint out/checkpoints/act_smoke --seeds 0 --swap
  .venv/bin/python scripts/eval_policy.py --checkpoint X --seeds 0-9 --random
  .venv/bin/python scripts/eval_policy.py --checkpoint X --seeds 0 --exec_steps 1
  .venv/bin/python scripts/eval_policy.py --checkpoint Y --seeds 0-9 --stats-root out/lerobot_v3
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import load as scene
from arm import ArmIK
from mambo_vla_rc.bench_stub import print_table
from mambo_vla_rc.seed_freeze import SEEDS, freeze_table
from run_seeds import FRAME_EVERY, HOME, INSTRUCTION, JAW_OPEN, SUCCESS_RADIUS, seed_bundle

from lerobot.policies.act.modeling_act import ACTPolicy

JAW_ROLLOUT_NM = 1.0
MAX_STEPS = 9000
# Gate thresholds: SUCCESS_RADIUS is imported from the teacher so the
# gate can never drift from it. The upright/penetration bounds mirror the
# teacher's run_seed success rule exactly (up-axis above cos(15deg),
# separation inside +-0.004 with penetration below -0.001 failing):
# eval must clear the same bar the demos cleared, not a looser copy.
UPRIGHT_MIN = float(np.cos(np.radians(15)))
SEPARATION_TOL = 0.004
PENETRATION_TOL = -0.001
# M2 gate protocol freezes ONE chunk-execution value (F3): native chunk
# execution matching the checkpoint's own n_action_steps. The CLI flag
# stays for ablations, but gate numbers always run the frozen default
# and every result row logs the value actually used.
EXEC_STEPS = 50
SWAP_SENTENCE = "Transfer the plate from right staging to the left zone via the table relay."


def route(instruction: str) -> bool:
    """Bounded-grammar router: exactly the relay sentence is accepted."""
    return instruction.strip() == INSTRUCTION


def parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def warmup_egl(bundle: dict):
    """Create one throwaway renderer BEFORE torch touches CUDA: on this
    box an EGL display initialized after the CUDA context fails at
    make-current, while an already-live display keeps working. The
    warmup renderer stays open for the whole run to pin that state."""
    model = scene.load_model(
        light_multiplier=bundle["lighting"],
        mug_rgba=tuple(bundle["mug_color"]),
        plate_rgba=tuple(bundle["plate_color"]),
    )
    data = scene.make_data(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=256, width=256)
    renderer.update_scene(data, camera="overhead")
    renderer.render()
    return renderer


def load_policy(ckpt: Path, device: str | None, stats_root: str):
    from dataclasses import replace as _replace

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors

    pre_dir = ckpt / "pretrained_model"
    if not pre_dir.is_dir():
        pre_dir = ckpt
    if device:
        base = ACTPolicy.config_class.from_pretrained(pre_dir)
        policy = ACTPolicy.from_pretrained(pre_dir, config=_replace(base, device=device))
    else:
        policy = ACTPolicy.from_pretrained(pre_dir)
    ds = LeRobotDataset(repo_id="mambo/rc-dinner", root=str(stats_root))
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=str(pre_dir), dataset_stats=ds.meta.stats
    )
    return policy, pre, post


def rollout(seed: int, bundle: dict, policy, pre, post, exec_steps: int,
            random_mode: bool = False) -> dict:
    model = scene.load_model(
        light_multiplier=bundle["lighting"],
        mug_rgba=tuple(bundle["mug_color"]),
        plate_rgba=tuple(bundle["plate_color"]),
    )
    data = scene.make_data(model)
    scene.set_free_body(
        model, data, "mug",
        (bundle["mug_xy"][0], bundle["mug_xy"][1],
         scene.TABLE_Z + scene.MUG_HALFHEIGHT + scene.SPAWN_CLEARANCE),
    )
    arms = {side: ArmIK(model, data, side) for side in ("left", "right")}
    for side in ("left", "right"):
        model.actuator_forcerange[arms[side].act_ids[5]] = [-JAW_ROLLOUT_NM, JAW_ROLLOUT_NM]
        arms[side].set_qpos(HOME[side])
        data.qpos[arms[side].jaw_qadr] = JAW_OPEN
        arms[side].apply(HOME[side], JAW_OPEN)
    mujoco.mj_forward(model, data)
    for _ in range(50):
        mujoco.mj_step(model, data)
    renderer = mujoco.Renderer(model, height=256, width=256)
    mug_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mug")
    dest = np.array(scene.DEST_POINT)
    query_ms: list[float] = []
    queue: list[np.ndarray] = []
    steps = 0
    success = False
    mug0 = np.array(data.xpos[mug_body], dtype=float)[:2].copy()
    # Seeded per episode so the random baseline is reproducible; consumed
    # only in random_mode, inert otherwise.
    rng = np.random.default_rng(seed)
    lim_lo = lim_hi = None
    if random_mode:
        # Random-action baseline: independent uniform samples over the
        # actuated joint ranges, seeded per episode so the baseline is
        # reproducible. Same cadence, same actuator path, no network.
        rng = np.random.default_rng(seed)
        lo, hi = [], []
        for side in ("left", "right"):
            for jid in arms[side].joint_ids + [arms[side].jaw_id]:
                a, b = model.jnt_range[jid]
                lo.append(float(a))
                hi.append(float(b))
        lim_lo, lim_hi = np.array(lo), np.array(hi)
    while steps < MAX_STEPS:
        mujoco.mj_step(model, data)
        steps += 1
        if steps % FRAME_EVERY != 0:
            continue
        if not queue:
            t0 = time.perf_counter()
            if random_mode:
                queue = [rng.uniform(lim_lo, lim_hi).astype(np.float64)
                         for _ in range(exec_steps)]
                query_ms.append((time.perf_counter() - t0) * 1000.0)
            else:
                frames = {}
                for cam in scene.CAMERAS:
                    renderer.update_scene(data, camera=cam)
                    img = renderer.render().copy()
                    # Same tensor the dataset serves the trainer: float32 CHW
                    # in [0,1]. (Feeding raw uint8 poisons the normalizer: it
                    # adapts all stats to the first-seen dtype and the negative
                    # state means then overflow uint8.)
                    frames[f"observation.images.{cam}"] = (
                        torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                    )
                state = torch.from_numpy(np.concatenate([
                    arms["left"].qpos(), [arms["left"].jaw()],
                    arms["right"].qpos(), [arms["right"].jaw()],
                ]).astype(np.float32))
                batch = dict(frames)
                batch["observation.state"] = state
                normed = pre(batch)
                with torch.no_grad():
                    chunk = policy.predict_action_chunk(normed)
                out = post(chunk)
                first = out[0]
                if torch.is_tensor(first):
                    first = first.cpu()
                actions = np.asarray(first)
                query_ms.append((time.perf_counter() - t0) * 1000.0)
                queue = [np.asarray(a, dtype=np.float64) for a in actions[:exec_steps]]
        act = queue.pop(0)
        arms["left"].apply(act[0:5], float(act[5]))
        arms["right"].apply(act[6:11], float(act[11]))
        mug = np.array(data.xpos[mug_body], dtype=float)
        up = float(data.xmat[mug_body, 8])
        rest_z = scene.TABLE_Z + scene.MUG_HALFHEIGHT
        separation = float(mug[2]) - rest_z
        penetrates = separation < PENETRATION_TOL
        if (float(np.linalg.norm(mug[:2] - dest[:2])) <= SUCCESS_RADIUS
                and abs(separation) < SEPARATION_TOL and up > UPRIGHT_MIN
                and not penetrates):
            success = True
            break
    renderer.close()
    mug_end = np.array(data.xpos[mug_body], dtype=float)
    mug_moved_mm = float(np.linalg.norm(mug_end[:2] - mug0) * 1000.0)
    if random_mode:
        note = (f"random-action baseline via actuator apply" if success
                else f"random baseline: no place (mug moved {mug_moved_mm:.0f}mm)")
    else:
        note = ("policy rollout via actuator apply" if success
                else f"policy rollout: no place (mug moved {mug_moved_mm:.0f}mm)")
    return {
        "seed": seed,
        "success": success,
        "steps": steps,
        "forward_p50_ms": round(float(np.median(query_ms)), 4) if query_ms else 0.0,
        "forward_p95_ms": round(float(np.percentile(query_ms, 95)), 4) if query_ms else 0.0,
        "exec_steps": exec_steps,
        "note": note,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--swap", action="store_true")
    ap.add_argument("--exec_steps", type=int, default=EXEC_STEPS)
    ap.add_argument("--device", default=None)
    ap.add_argument("--random", action="store_true",
                    help="random-action baseline: uniform joint samples, no network")
    ap.add_argument("--stats-root", default=str(ROOT / "out" / "lerobot"),
                    help="dataset root for normalization stats only (F-stats); "
                    "default preserves frozen v1 behavior")
    args = ap.parse_args()
    ckpt = Path(args.checkpoint)
    pre_dir = ckpt / "pretrained_model"
    if not pre_dir.is_dir():
        pre_dir = ckpt
    instruction = SWAP_SENTENCE if args.swap else INSTRUCTION
    rows = []
    if not route(instruction):
        for seed in parse_seeds(args.seeds):
            rows.append({"seed": seed, "success": False, "steps": 0,
                         "forward_p50_ms": 0.0, "forward_p95_ms": 0.0,
                         "exec_steps": args.exec_steps,
                         "note": "router refused (out-of-grammar)"})
        print_table(rows, instruction)
        print("swap control: success collapses to 0 (refusal, arms never moved)")
        return 0
    bundles = {s: seed_bundle(s) for s in SEEDS}
    hashes = freeze_table(bundles)
    saved = {s: h for s, h in json_hashes().items()}
    for s in parse_seeds(args.seeds):
        assert hashes[s] == saved[s], f"seed {s} hash mismatch"
    seeds = parse_seeds(args.seeds)
    print(f"stats_root={args.stats_root}", flush=True)
    _warm = warmup_egl(bundles[seeds[0]])
    if args.random:
        policy = pre = post = None
    else:
        policy, pre, post = load_policy(ckpt, args.device, args.stats_root)
    for seed in seeds:
        rows.append(rollout(seed, bundles[seed], policy, pre, post,
                            args.exec_steps, random_mode=args.random))
        r = rows[-1]
        print(f"seed {seed}: success={r['success']} steps={r['steps']} "
              f"fwd_p50={r['forward_p50_ms']}ms fwd_p95={r['forward_p95_ms']}ms "
              f"exec={r['exec_steps']} | {r['note']}", flush=True)
    print_table(rows, instruction)
    print("seed | success | steps | forward_p50_ms | forward_p95_ms | exec_steps | note")
    for r in rows:
        print(f"{r['seed']:>4} | {r['success']} | {r['steps']} | "
              f"{r['forward_p50_ms']} | {r['forward_p95_ms']} | "
              f"{r['exec_steps']} | {r['note']}")
    return 0


def json_hashes() -> dict:
    import json as _json
    return {int(k): v for k, v in _json.loads(
        (ROOT / "out" / "seeds" / "seed_hashes.json").read_text()).items()}


if __name__ == "__main__":
    raise SystemExit(main())

