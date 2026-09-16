"""Record headless demo videos of the final vehicle policy (delivery artifact).

Reuses the frozen rollout pipeline from eval_policy.py by import (that file
is NOT modified): same actuator path, same gate thresholds, same
MAX_STEPS=9000 / EXEC_STEPS=50, same stats-root handling. The only addition
is video capture: the overhead render already produced for the policy input
at each FRAME_EVERY step is overlaid with seed + instruction and appended
to a per-seed mp4 (imageio, 20 fps, 256x256).

Render is offscreen only: MUJOCO_GL defaults to egl (proven on this box),
falling back to osmesa if EGL init fails. No display is ever touched.

Usage:
  .venv/bin/python scripts/record_demo.py
  .venv/bin/python scripts/record_demo.py --seeds 1,4,7 --stats-root out/lerobot_v3
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
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import load as scene
from arm import ArmIK
from eval_policy import (
    EXEC_STEPS,
    JAW_ROLLOUT_NM,
    MAX_STEPS,
    PENETRATION_TOL,
    SEPARATION_TOL,
    SUCCESS_RADIUS,
    UPRIGHT_MIN,
    json_hashes,
    load_policy,
    parse_seeds,
    warmup_egl,
)
from mambo_vla_rc.seed_freeze import SEEDS, freeze_table
from run_seeds import FRAME_EVERY, HOME, INSTRUCTION, JAW_OPEN, seed_bundle

import imageio.v2 as imageio

FPS = 20
WIDTH = HEIGHT = 256


def overlay(img: np.ndarray, seed: int) -> np.ndarray:
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    d.rectangle([0, 0, WIDTH, 30], fill=(0, 0, 0))
    d.text((6, 4), f"seed {seed}", fill=(255, 255, 0))
    d.text((6, 17), INSTRUCTION[:48], fill=(255, 255, 255))
    return np.asarray(pil)


def record(seed: int, bundle: dict, policy, pre, post, exec_steps: int,
           out_path: Path) -> dict:
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
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    mug_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mug")
    dest = np.array(scene.DEST_POINT)
    queue: list[np.ndarray] = []
    steps = 0
    success = False
    mug0 = np.array(data.xpos[mug_body], dtype=float)[:2].copy()
    writer = imageio.get_writer(str(out_path), fps=FPS, macro_block_size=1)
    nframes = 0
    try:
        while steps < MAX_STEPS:
            mujoco.mj_step(model, data)
            steps += 1
            if steps % FRAME_EVERY != 0:
                continue
            # Video frame at every FRAME_EVERY step (policy cadence = 20 fps).
            renderer.update_scene(data, camera="overhead")
            writer.append_data(overlay(renderer.render().copy(), seed))
            nframes += 1
            if not queue:
                frames = {}
                for cam in scene.CAMERAS:
                    renderer.update_scene(data, camera=cam)
                    img = renderer.render().copy()
                    frames[f"observation.images.{cam}"] = (
                        torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                    )
                # Overhead view doubles as the video frame (already rendered
                # above; zero extra renders).
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
    finally:
        writer.close()
        renderer.close()
    mug_end = np.array(data.xpos[mug_body], dtype=float)
    mug_moved_mm = float(np.linalg.norm(mug_end[:2] - mug0) * 1000.0)
    return {"seed": seed, "success": success, "steps": steps,
            "frames": nframes, "mug_moved_mm": round(mug_moved_mm, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default=str(ROOT / "out" / "checkpoints" / "act_full_v3"
                                / "checkpoints" / "100000"))
    ap.add_argument("--seeds", default="1,4,7")
    ap.add_argument("--exec_steps", type=int, default=EXEC_STEPS)
    ap.add_argument("--stats-root", default=str(ROOT / "out" / "lerobot_v3"))
    ap.add_argument("--out-dir", default=str(ROOT / "out" / "demo"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundles = {s: seed_bundle(s) for s in SEEDS}
    hashes = freeze_table(bundles)
    saved = {s: h for s, h in json_hashes().items()}
    seeds = parse_seeds(args.seeds)
    for s in seeds:
        assert hashes[s] == saved[s], f"seed {s} hash mismatch"
    print(f"stats_root={args.stats_root}", flush=True)

    try:
        _warm = warmup_egl(bundles[seeds[0]])
    except Exception as e:
        print(f"EGL warmup failed ({e}); falling back to osmesa", flush=True)
        os.environ["MUJOCO_GL"] = "osmesa"
        _warm = warmup_egl(bundles[seeds[0]])

    policy, pre, post = load_policy(Path(args.checkpoint), None, args.stats_root)
    for seed in seeds:
        out_path = out_dir / f"seed_{seed}.mp4"
        t0 = time.perf_counter()
        r = record(seed, bundles[seed], policy, pre, post,
                   args.exec_steps, out_path)
        dt = time.perf_counter() - t0
        size = out_path.stat().st_size
        print(f"seed {seed}: success={r['success']} steps={r['steps']} "
              f"frames={r['frames']} moved={r['mug_moved_mm']}mm "
              f"-> {out_path} ({size} bytes, {dt:.0f}s wall)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
