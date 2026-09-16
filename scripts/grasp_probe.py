"""Negative control for the contact gate (Gate E2R validation).

Drives the left arm to the mug with a lateral X offset, closes the jaw,
and reports whether sim/grasp.py gates attachment. Offset 0 must ATTACH;
offset 20mm must FAIL (pads cannot straddle the 50mm mug). Exits 0 when
both expectations hold, 1 otherwise.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "src"))
import load as scene
from arm import ArmIK
from grasp import GraspGate

HOME = np.array([0.0, -0.3, -0.4, 0.5, 0.0])
JAW_OPEN = 0.6


def try_grasp(offset_x: float) -> tuple[bool, str]:
    model = scene.load_model()
    data = scene.make_data(model)
    scene.set_free_body(model, data, "mug",
                        (-0.12, 0.02, scene.TABLE_Z + scene.SPAWN_CLEARANCE))
    arm = ArmIK(model, data, "left")
    # Teacher jaw schedule: WIDE while traveling, NARROW once gated.
    model.actuator_forcerange[arm.act_ids[5]] = [-3.35, 3.35]
    gate = GraspGate(model, data, "left")
    mug_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mug")
    arm.set_qpos(HOME)
    data.qpos[arm.jaw_qadr] = JAW_OPEN  # preset open (reset, no contacts yet)
    arm.apply(HOME, JAW_OPEN)
    for _ in range(50):
        mujoco.mj_step(model, data)
    mug = np.array(data.xpos[mug_body], dtype=float)

    def servo(target: np.ndarray, cap: int, gain: float = 0.5,
              max_step: float = 0.30) -> None:
        good = 0
        for _ in range(cap):
            qpos = arm.servo_step(np.asarray(target, dtype=float), gain=gain,
                                  max_step=max_step)
            arm.apply(qpos, JAW_OPEN)
            arm.q_ref = 0.8 * arm.q_ref + 0.2 * qpos  # same leaky ref as teacher
            mujoco.mj_step(model, data)
            err = float(np.linalg.norm(np.asarray(target) - arm.site_pos()))
            good = good + 1 if err < 0.002 else 0
            if good >= 10:
                break

    def pads_mid():
        mujoco.mj_forward(model, data)
        s = np.array(data.geom_xpos[gate.static_id], dtype=float)
        m = np.array(data.geom_xpos[gate.moving_id], dtype=float)
        return (s + m) / 2

    def center_xy():
        for _ in range(4):
            want = mug + np.array([offset_x, 0.0, -0.012])
            have = pads_mid()
            err_vec = np.array([want[0] - have[0], want[1] - have[1], 0.0])
            if float(np.linalg.norm(err_vec)) < 0.003:
                break
            servo(arm.site_pos() + err_vec, 80, gain=0.2, max_step=0.02)

    servo(mug + np.array([offset_x, 0.0, 0.12]), 250)
    center_xy()
    dz = (mug[2] - 0.012) - pads_mid()[2]  # vertical descent onto centered cage
    servo(arm.site_pos() + np.array([0, 0, dz]), 250)
    for _ in range(2):  # safety net only; over-servoing walks the mug
        mujoco.mj_forward(model, data)
        live = np.array(data.xpos[mug_body], dtype=float)
        err = live + np.array([offset_x, 0.0, -0.012]) - pads_mid()
        if float(np.linalg.norm(err)) < 0.003:
            break
        target = arm.site_pos() + err
        servo(target, 80, gain=0.2, max_step=0.02)
    mujoco.mj_forward(model, data)
    s = np.array(data.geom_xpos[gate.static_id], dtype=float)
    m = np.array(data.geom_xpos[gate.moving_id], dtype=float)
    mug_now = np.array(data.xpos[mug_body], dtype=float)
    print(f"offset {offset_x * 1000:.0f}mm: centered pads-mid err "
          f"{np.linalg.norm(mug_now - (s + m) / 2) * 1000:.1f}mm")
    jaw = arm.jaw()
    verdict: tuple[bool, str] = (False, "timeout")
    taps = 0
    touched = False
    for _ in range(60):
        jaw = max(0.05, jaw - 0.02)
        for _ in range(200):  # capped jaw slews slowly: wait for arrival
            arm.apply(arm.qpos(), jaw)
            mujoco.mj_step(model, data)
            if abs(arm.jaw() - jaw) < 0.02:
                break
        probe = gate.scan()
        if probe["static"]["touch"] or probe["moving"]["touch"]:
            if not touched:
                touched = True
                model.actuator_forcerange[arm.act_ids[5]] = [-0.15, 0.15]
        gated, why = gate.gated()
        if gated and jaw <= 0.50:
            model.actuator_forcerange[arm.act_ids[5]] = [-1.0, 1.0]
            verdict = (True, f"{why} @jaw {jaw:.2f}")
            break
        verdict = (False, f"{why} @jaw {jaw:.2f}")
        if jaw <= 0.05 + 1e-9:
            break
        taps += 1
        if taps % 5 == 0:  # re-symmetrize while closing (teacher mirrors this)
            mujoco.mj_forward(model, data)
            s = np.array(data.geom_xpos[gate.static_id], dtype=float)
            m = np.array(data.geom_xpos[gate.moving_id], dtype=float)
            err = mug + np.array([offset_x, 0.0, -0.012]) - (s + m) / 2
            servo(arm.site_pos() + err, 25)
    return verdict


def main() -> int:
    ok0, why0 = try_grasp(0.0)
    ok20, why20 = try_grasp(0.02)
    print(f"offset 0mm:  attached={ok0} ({why0})")
    print(f"offset 20mm: attached={ok20} ({why20})")
    good = ok0 and not ok20
    print("NEGATIVE CONTROL " + ("PASS" if good else "FAIL"))
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
