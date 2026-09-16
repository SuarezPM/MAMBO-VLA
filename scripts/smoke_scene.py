"""M0 headless smoke: load, step 500, render 4 cams, assert rest physics.

Render backend: MUJOCO_GL env wins; otherwise try egl, then osmesa.
Saves 256x256 frames to out/smoke/<camera>.png.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
import load as scene

OUT = Path(__file__).resolve().parents[1] / "out" / "smoke"
WIDTH = HEIGHT = 256
STEPS = 500
TOL_PENETRATION = 0.001


def render_all(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    frames = {}
    try:
        renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    except Exception as exc:  # noqa: BLE001 - report backend, try fallback
        fallback = "osmesa" if os.environ.get("MUJOCO_GL") != "osmesa" else "egl"
        print(f"renderer backend {os.environ.get('MUJOCO_GL')} failed ({exc}); retry {fallback}")
        os.environ["MUJOCO_GL"] = fallback
        renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    for cam in scene.CAMERAS:
        renderer.update_scene(data, camera=cam)
        frames[cam] = renderer.render().copy()
    renderer.close()
    return frames


def check_rest(name: str, halfheight: float, data: mujoco.MjData) -> float:
    pos = scene.body_pos(data, name)
    separation = float(pos[2] - (scene.TABLE_Z + halfheight))
    print(f"{name}: z={pos[2]:.6f} separation={separation * 1000:.2f}mm")
    assert separation >= -TOL_PENETRATION, f"{name} penetrates the table"
    return separation


def main() -> int:
    scene.assert_frozen_contact_block(scene.load_model())
    print("frozen contact block: OK")
    model = scene.load_model()
    data = scene.make_data(model)
    # Spawn heights: center = table_top + halfheight + clearance.
    plate_z0 = scene.TABLE_Z + scene.PLATE_HALFHEIGHT + scene.SPAWN_CLEARANCE
    mug_z0 = scene.TABLE_Z + scene.MUG_HALFHEIGHT + scene.SPAWN_CLEARANCE
    plate = scene.body_pos(data, "plate")
    mug = scene.body_pos(data, "mug")
    assert abs(plate[2] - plate_z0) < 1e-9, f"plate spawn {plate[2]} != {plate_z0}"
    assert abs(mug[2] - mug_z0) < 1e-9, f"mug spawn {mug[2]} != {mug_z0}"
    print(f"spawn heights OK (plate {plate_z0:.4f}, mug {mug_z0:.4f})")
    for _ in range(STEPS):
        mujoco.mj_step(model, data)
    sep_plate = check_rest("plate", scene.PLATE_HALFHEIGHT, data)
    sep_mug = check_rest("mug", scene.MUG_HALFHEIGHT, data)
    assert sep_plate < 0.003, "plate did not settle onto the table"
    assert sep_mug < 0.003, "mug did not settle onto the table"
    frames = render_all(model, data)
    OUT.mkdir(parents=True, exist_ok=True)
    for cam, img in frames.items():
        assert img.shape == (HEIGHT, WIDTH, 3), f"{cam} bad shape {img.shape}"
        assert img.std() > 1.0, f"{cam} frame looks blank"
        Image.fromarray(img).save(OUT / f"{cam}.png")
        print(f"{cam}: saved + variance {img.std():.1f}")
    print(f"SMOKE GREEN: {STEPS} steps, 4 cams in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
