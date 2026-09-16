"""F3 verify+repair wrapper (no retrain, no harness edits).

Imports the frozen harness WITHOUT changing defaults:
  from eval_policy import (EXEC_STEPS, MAX_STEPS, SUCCESS_RADIUS,
    UPRIGHT_MIN, SEPARATION_TOL, PENETRATION_TOL, JAW_ROLLOUT_NM,
    load_policy, warmup_egl, parse_seeds, rollout)
  from run_seeds import (FRAME_EVERY, HOME, JAW_OPEN, INSTRUCTION)
  from sim/load.py (as scene): TABLE_Z, MUG_HALFHEIGHT, DEST_POINT,
    RELAY_POINT, body_pos, set_free_body (TEST SETUP ONLY, never repair)
  from sim/arm.py: ArmIK (qpos/jaw/site_pos/apply state queries)

What this wrapper does (all to log):
  - Before each relay phase (pick, set-down, handover, place) it
    RE-PERCEIVES state (mug pos/up/yaw/separation + jaws + sites).
  - If the phase predicate is already true it SKIPS the phase
    (0 policy steps for that phase, 0 retries consumed).
  - If the mug is KNOCKED-OFF (below table / far outside workspace)
    it RE-QUEUES the policy (clears the pending action chunk queue so
    the next frame re-queries the policy from the re-perceived state).
    No teleport, no state edits in the repair path.
  - MAX_RETRIES=2 per phase (queue-clear retries). Exceeding it marks
    the phase failed but the episode still runs to MAX_STEPS (honest fail).

Test modes on NEW seeds 100-119 (never 0-99):
  - baseline : plain phased rollout, no injected disturbance.
  - disturb  : TEST SETUP displaces the mug mid-episode by a fixed
    +0.05 m x offset via set_free_body (logged as TEST SETUP, not repair),
    then the wrapper must re-perceive + re-queue and try to recover.
  - preplaced: TEST SETUP spawns the mug already at DEST via set_free_body
    (logged); the wrapper must verify place immediately and SKIP all phases.

Scoring for THESE runs only (P6-strict, new runs only):
  strict = |xy-dest|<=0.030 AND |yaw|<=15deg AND up>cos(15deg)
           AND released (both jaws open>0.30 AND both grip sites>0.040 m
           from mug center).
  The frozen place gate (1.5 cm, no yaw/released) is ALSO logged per run
  for reference but NEVER decides these runs. Past results (0-9 frozen,
  OOD 60-99) were judged by place and are NOT redefined here.

Usage (each invocation tee-s its stdout to a file: H12 discipline):
  .venv/bin/python scripts/verify_repair.py --mode disturb --seeds 100-119
  .venv/bin/python scripts/verify_repair.py --mode preplaced --seeds 100-119
  .venv/bin/python scripts/verify_repair.py --mode baseline --seeds 100-109
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

# Frozen harness imports: values reused, never reassigned here.
import eval_policy as EP
from run_seeds import FRAME_EVERY, HOME, JAW_OPEN, INSTRUCTION

# Re-exported frozen constants (read-only aliases; the source of truth
# stays in eval_policy / run_seeds / sim/load).
EXEC_STEPS = EP.EXEC_STEPS
MAX_STEPS = EP.MAX_STEPS
SUCCESS_RADIUS = EP.SUCCESS_RADIUS
UPRIGHT_MIN = EP.UPRIGHT_MIN
SEPARATION_TOL = EP.SEPARATION_TOL
PENETRATION_TOL = EP.PENETRATION_TOL
JAW_ROLLOUT_NM = EP.JAW_ROLLOUT_NM

assert EXEC_STEPS == 50, "frozen EXEC_STEPS drifted"

MAX_RETRIES = 2
PHASES = ("pick", "set-down", "handover", "place")

# P6-strict (NEW RUNS ONLY, see module docstring).
STRICT_RADIUS = 0.030
STRICT_YAW_TOL = float(np.radians(15))
RELEASED_JAW_MIN = 0.30
RELEASED_DIST_MIN = 0.040

# Knocked-off detector (repair trigger, not a success gate).
KNOCK_Z_BELOW = 0.020   # mug z more than 2 cm below table surface
KNOCK_XY_FAR = 0.60     # |x| or |y| beyond 60 cm from origin

# Disturb test setup (logged, not repair): fixed +x shove mid-episode.
DISTURB_DX = 0.05
DISTURB_STEP = 3000  # physics steps (of MAX_STEPS=9000)

OUT_DISTURB = ROOT / "out" / "seeds_disturb"


def mug_yaw(data, body_id: int) -> float:
    rmat = np.array(data.xmat[body_id], dtype=float).reshape(3, 3)
    return float(np.arctan2(rmat[1, 0], rmat[0, 0]))


def perceive(model, data, mug_body: int, arms: dict) -> dict:
    mug = np.array(data.xpos[mug_body], dtype=float)
    up = float(data.xmat[mug_body, 8])
    yaw = mug_yaw(data, mug_body)
    rest_z = scene.TABLE_Z + scene.MUG_HALFHEIGHT
    separation = float(mug[2]) - rest_z
    jaws = {s: arms[s].jaw() for s in ("left", "right")}
    sites = {s: arms[s].site_pos() for s in ("left", "right")}
    site_dist = {s: float(np.linalg.norm(sites[s][:2] - mug[:2])) for s in sites}
    return {
        "mug": mug, "up": up, "yaw": yaw, "separation": separation,
        "jaws": jaws, "site_dist": site_dist,
    }


def is_knocked_off(st: dict) -> bool:
    mug = st["mug"]
    if float(mug[2]) < scene.TABLE_Z - KNOCK_Z_BELOW:
        return True
    if abs(float(mug[0])) > KNOCK_XY_FAR or abs(float(mug[1])) > KNOCK_XY_FAR:
        return True
    return False


def frozen_place(st: dict) -> bool:
    # NOTE (F3-remediation): this 4-line boolean is intentionally duplicated,
    # not imported: scripts/eval_policy.py exposes NO importable place
    # predicate (its gate is inline in rollout(), lines ~242-244), and
    # extracting it would require editing that frozen file (forbidden).
    # Every constant below IS imported from the frozen harness
    # (SUCCESS_RADIUS, SEPARATION_TOL, UPRIGHT_MIN, PENETRATION_TOL plus
    # scene.DEST_POINT), so the gate cannot drift — only the boolean
    # composition is mirrored here.
    dest = np.array(scene.DEST_POINT, dtype=float)
    mug = st["mug"]
    penetrates = st["separation"] < PENETRATION_TOL
    return bool(
        float(np.linalg.norm(mug[:2] - dest[:2])) <= SUCCESS_RADIUS
        and abs(st["separation"]) < SEPARATION_TOL
        and st["up"] > UPRIGHT_MIN
        and not penetrates
    )


def strict_place(st: dict) -> bool:
    dest = np.array(scene.DEST_POINT, dtype=float)
    mug = st["mug"]
    penetrates = st["separation"] < PENETRATION_TOL
    yaw_ok = abs(float((st["yaw"] + np.pi) % (2 * np.pi) - np.pi)) <= STRICT_YAW_TOL
    released = (
        st["jaws"]["left"] > RELEASED_JAW_MIN
        and st["jaws"]["right"] > RELEASED_JAW_MIN
        and st["site_dist"]["left"] > RELEASED_DIST_MIN
        and st["site_dist"]["right"] > RELEASED_DIST_MIN
    )
    return bool(
        float(np.linalg.norm(mug[:2] - dest[:2])) <= STRICT_RADIUS
        and yaw_ok
        and st["up"] > UPRIGHT_MIN
        and abs(st["separation"]) < SEPARATION_TOL
        and not penetrates
        and released
    )


def phase_verified(phase: str, st: dict, bundle: dict) -> bool:
    """Phase predicates from re-perceived state (no policy signal)."""
    mug = st["mug"]
    if phase == "pick":
        # Lifted off staging: separation clearly above rest.
        return bool(st["separation"] > 0.010 and st["up"] > UPRIGHT_MIN)
    if phase == "set-down":
        relay = np.array(bundle["relay_point"] if "relay_point" in bundle else scene.RELAY_POINT, dtype=float)
        return bool(float(np.linalg.norm(mug[:2] - relay[:2])) <= 0.030 and abs(st["separation"]) < 0.020)
    if phase == "handover":
        # Re-gripped at relay and lifted again.
        relay = np.array(bundle["relay_point"] if "relay_point" in bundle else scene.RELAY_POINT, dtype=float)
        return bool(float(np.linalg.norm(mug[:2] - relay[:2])) <= 0.030 and st["separation"] > 0.010)
    if phase == "place":
        return strict_place(st)
    raise ValueError(phase)


def phased_rollout(seed: int, bundle: dict, policy, pre, post,
                   exec_steps: int, mode: str, log) -> dict:
    """Run one episode with verify+repair hooks. All decisions via log()."""
    assert exec_steps == EXEC_STEPS == 50
    model = scene.load_model(
        light_multiplier=bundle["lighting"],
        mug_rgba=tuple(bundle["mug_color"]),
        plate_rgba=tuple(bundle["plate_color"]),
    )
    data = scene.make_data(model)
    if mode == "preplaced":
        dest = list(scene.DEST_POINT)
        scene.set_free_body(
            model, data, "mug",
            (dest[0], dest[1], scene.TABLE_Z + scene.MUG_HALFHEIGHT + scene.SPAWN_CLEARANCE),
        )
        log(f"TEST SETUP preplaced: mug spawned at DEST {dest[:2]} (skip test)")
    else:
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
    mug0 = np.array(data.xpos[mug_body], dtype=float)[:2].copy()

    # Pre-phase re-perception: skip anything already verified.
    retries = {p: 0 for p in PHASES}
    exhausted_logged: set[str] = set()  # F3 cap: 1 post-exhaustion line per phase
    skipped = []
    st0 = perceive(model, data, mug_body, arms)
    log(f"re-perceive t=0 mug_xy={np.round(st0['mug'][:2], 4).tolist()} z={st0['mug'][2]:.4f} "
        f"up={st0['up']:.4f} yaw={st0['yaw']:.3f} sep={st0['separation']:.4f} "
        f"frozen_place={frozen_place(st0)} strict_place={strict_place(st0)}")
    if mode == "preplaced" and strict_place(st0):
        # Full skip: already placed under the strict gate.
        for p in PHASES:
            skipped.append(p)
            log(f"phase {p}: SKIP (already verified at t=0)")
        renderer.close()
        return {
            "seed": seed, "mode": mode, "success_strict": True,
            "success_frozen": True, "steps": 0,
            "forward_p50_ms": 0.0, "forward_p95_ms": 0.0,
            "exec_steps": exec_steps, "retries": retries,
            "skipped": skipped, "disturbed": False,
            "note": "preplaced skip: mug already at DEST strict (0 policy steps)",
        }

    query_ms: list[float] = []
    queue: list[np.ndarray] = []
    steps = 0
    disturbed = False
    phase_done = {p: False for p in PHASES}
    # Entry skips for phases already true at t=0 (e.g. none in normal spawn).
    for p in PHASES:
        if phase_verified(p, st0, bundle):
            phase_done[p] = True
            skipped.append(p)
            log(f"phase {p}: SKIP (already verified at t=0)")

    success_strict = False
    success_frozen = False
    while steps < MAX_STEPS:
        mujoco.mj_step(model, data)
        steps += 1
        # TEST SETUP disturbance (logged, not repair).
        if mode == "disturb" and not disturbed and steps >= DISTURB_STEP:
            cur = np.array(data.xpos[mug_body], dtype=float)
            scene.set_free_body(
                model, data, "mug",
                (float(cur[0]) + DISTURB_DX, float(cur[1]),
                 scene.TABLE_Z + scene.MUG_HALFHEIGHT + scene.SPAWN_CLEARANCE),
            )
            disturbed = True
            queue.clear()  # disturbed state => must re-query policy
            st = perceive(model, data, mug_body, arms)
            log(f"TEST SETUP disturb @step={steps}: mug shoved +{DISTURB_DX} m x "
                f"-> xy={np.round(st['mug'][:2], 4).tolist()} (queue cleared)")
        if steps % FRAME_EVERY != 0:
            continue
        # Re-perceive before acting (phase hooks every frame tick).
        st = perceive(model, data, mug_body, arms)
        if is_knocked_off(st):
            # Find first unfinished phase to charge the retry.
            target = next((p for p in PHASES if not phase_done[p]), "place")
            if retries[target] < MAX_RETRIES:
                retries[target] += 1
                queue.clear()
                log(f"REPAIR knocked-off @step={steps}: mug z={st['mug'][2]:.4f} "
                    f"xy={np.round(st['mug'][:2], 4).tolist()} -> re-queue "
                    f"(phase {target} retry {retries[target]}/{MAX_RETRIES})")
            else:
                # Capped: one line per exhausted phase (was one per frame tick).
                if target not in exhausted_logged:
                    exhausted_logged.add(target)
                    log(f"REPAIR knocked-off @step={steps}: phase {target} retries "
                        f"exhausted ({MAX_RETRIES}), continuing without re-queue")
        # Mark phases as they verify (skip remaining work for that phase).
        for p in PHASES:
            if not phase_done[p] and phase_verified(p, st, bundle):
                phase_done[p] = True
                log(f"phase {p}: VERIFIED @step={steps} "
                    f"xy={np.round(st['mug'][:2], 4).tolist()} sep={st['separation']:.4f}")
        if strict_place(st):
            success_strict = True
            success_frozen = frozen_place(st)
            log(f"place STRICT @step={steps} (frozen_place={success_frozen})")
            break
        if not queue:
            import time as _t
            t0 = _t.perf_counter()
            frames = {}
            for cam in scene.CAMERAS:
                renderer.update_scene(data, camera=cam)
                img = renderer.render().copy()
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
            query_ms.append((_t.perf_counter() - t0) * 1000.0)
            queue = [np.asarray(a, dtype=np.float64) for a in actions[:exec_steps]]
        act = queue.pop(0)
        arms["left"].apply(act[0:5], float(act[5]))
        arms["right"].apply(act[6:11], float(act[11]))
    renderer.close()
    st_end = perceive(model, data, mug_body, arms)
    if not success_strict:
        success_frozen = frozen_place(st_end)
    mug_end = np.array(st_end["mug"], dtype=float)
    mug_moved_mm = float(np.linalg.norm(mug_end[:2] - mug0) * 1000.0)
    total_retries = sum(retries.values())
    if success_strict:
        note = (f"repair rollout {mode}: STRICT place @ {steps} steps "
                f"(retries={total_retries}, skipped={skipped or 'none'})")
    else:
        note = (f"repair rollout {mode}: no STRICT place "
                f"(mug moved {mug_moved_mm:.0f}mm, retries={total_retries}, "
                f"frozen_place={success_frozen})")
    return {
        "seed": seed, "mode": mode, "success_strict": success_strict,
        "success_frozen": success_frozen, "steps": steps,
        "forward_p50_ms": round(float(np.median(query_ms)), 4) if query_ms else 0.0,
        "forward_p95_ms": round(float(np.percentile(query_ms, 95)), 4) if query_ms else 0.0,
        "exec_steps": exec_steps, "retries": retries,
        "skipped": skipped, "disturbed": disturbed,
        "mug_moved_mm": round(mug_moved_mm, 1),
        "note": note,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["baseline", "disturb", "preplaced"])
    ap.add_argument("--seeds", default="100-119")
    ap.add_argument("--checkpoint", default=str(ROOT / "out" / "checkpoints" / "act_full_v3" / "checkpoints" / "100000"))
    ap.add_argument("--stats-root", default=str(ROOT / "out" / "lerobot_v3"))
    ap.add_argument("--hash-file", default=str(OUT_DISTURB / "seed_hashes_disturb.json"))
    ap.add_argument("--exec-steps", type=int, default=EXEC_STEPS)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    assert args.exec_steps == 50, "EXEC_STEPS frozen at 50"
    assert args.exec_steps == EP.EXEC_STEPS

    from mambo_vla_rc.seed_freeze import hash_bundle
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_seeds_ood import seed_bundle_ood

    seeds = EP.parse_seeds(args.seeds)
    OUT_DISTURB.mkdir(parents=True, exist_ok=True)
    import json as _json
    saved = {int(k): v for k, v in _json.loads(Path(args.hash_file).read_text()).items()}
    bundles = {}
    for s in seeds:
        b = seed_bundle_ood(s)
        assert hash_bundle(b) == saved[s], f"seed {s} hash mismatch"
        bundles[s] = b
    print(f"stats_root={args.stats_root}", flush=True)
    print(f"checkpoint={args.checkpoint}", flush=True)
    print(f"exec_steps={args.exec_steps} (frozen, logged per row)", flush=True)
    print(f"mode={args.mode} hash_file={args.hash_file}", flush=True)
    print(f"MAX_RETRIES={MAX_RETRIES} per phase; STRICT=3cm+yaw15+upright+released (NEW RUNS ONLY)", flush=True)

    _warm = EP.warmup_egl(bundles[seeds[0]])
    policy, pre, post = EP.load_policy(Path(args.checkpoint), args.device, args.stats_root)
    rows = []
    t_all0 = time.time()
    for seed in seeds:
        log_path = OUT_DISTURB / f"logs_{args.mode}" / f"seed_{seed}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []

        def log(msg: str):
            line = f"seed {seed} [{args.mode}]: {msg}"
            print(line, flush=True)
            lines.append(line)

        t0 = time.time()
        try:
            r = phased_rollout(seed, bundles[seed], policy, pre, post,
                               args.exec_steps, args.mode, log)
            dt = time.time() - t0
            lines.append(f"wall_s={dt:.1f} result={r}")
            log_path.write_text("\n".join(lines) + "\n")
            rows.append(r)
            print(f"seed {seed}: strict={r['success_strict']} frozen={r['success_frozen']} "
                  f"steps={r['steps']} exec={r['exec_steps']} | {r['note']}", flush=True)
        except Exception as e:  # INFRA, never FAIL
            import traceback as _tb
            dt = time.time() - t0
            tb = _tb.format_exc()
            print(f"seed {seed}: INFRA {type(e).__name__}: {e}", flush=True)
            log_path.write_text("\n".join(lines) + f"\nINFRA {type(e).__name__}: {e}\n{tb}\nwall_s={dt:.1f}\n")
            rows.append({"seed": seed, "mode": args.mode, "success_strict": False,
                         "success_frozen": False, "steps": -1,
                         "forward_p50_ms": 0.0, "forward_p95_ms": 0.0,
                         "exec_steps": args.exec_steps,
                         "note": f"INFRA: {type(e).__name__}: {e}"})
    dt_all = time.time() - t_all0
    print(f"total_wall_s={dt_all:.1f}", flush=True)
    print("seed | strict | frozen | steps | exec | note", flush=True)
    for r in rows:
        print(f"{r['seed']:>4} | {r.get('success_strict')} | {r.get('success_frozen')} | "
              f"{r.get('steps')} | {r.get('exec_steps')} | {r.get('note')}", flush=True)
    (OUT_DISTURB / f"results_{args.mode}.json").write_text(
        __import__("json").dumps(rows, indent=2) + "\n")
    ok = sum(1 for r in rows if r.get("success_strict"))
    infra = sum(1 for r in rows if str(r.get("note", "")).startswith("INFRA"))
    print(f"summary {args.mode}: {ok}/{len(rows)} STRICT, {infra} INFRA", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
