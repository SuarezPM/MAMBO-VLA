"""M1 scripted table relay over seeds 0-9, articulated teacher + dataset.

Teacher: cartesian site waypoints solved through DLS IK (sim/arm.py) to
joint targets on position actuators. The 4-step pick (above -> descend ->
close -> lift) plus table-supported relay (arm A places at the relay point
and parks; arm B re-grips and places at the destination). Carry is
friction-only through the ARCH pads: sim/grasp.py gates attachment on
both-pad contact + squeeze force + opposition, and every lift is verified
(mug rises with the cage) before transport. The object is NEVER teleported:
set_free_body is spawn placement only. One re-grasp retry per carry leg;
second loss fails the seed honestly.

Seed bundles are hashed with seed_freeze BEFORE running; physics is never
edited post-freeze. Traces append (run tag e2r); stale failure.log is
removed on success. Raw episodes go to out/dataset/seed_<s>.npz;
scripts/convert_to_lerobot.py converts them to LeRobot v3. Control-intent
streams (jaw forcerange schedule + actuator ctrl targets) are logged every
step and every frame alongside measured state (F1/F2).

Per-seed table columns follow src/mambo_vla_rc/bench_stub.py.
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "src"))
import load as scene
from arm import ArmIK
from grasp import HOLD_MIN_N, GraspGate
from mambo_vla_rc.bench_stub import print_table
from mambo_vla_rc.seed_freeze import SEEDS, freeze_table
from mambo_vla_rc.trace import append_trace

INSTRUCTION = "Transfer the mug from left staging to the right zone via the table relay."
RUN_TAG = "e2r"

OUT_SEEDS = ROOT / "out" / "seeds"
OUT_DATASET = ROOT / "out" / "dataset"
TRACE_PATH = OUT_SEEDS / "traces.jsonl"

FRAME_EVERY = 25  # 20 fps equivalent at dt=0.002
SETTLE_STEPS = 150
LIFT = 0.10
# Pinch ABOVE the mug center (aerial carry): lag centers the load,
# gravity pendulums it upright, and both pads bear symmetrically in a
# vertical lift (push-carries are inherently single-sided and cannot
# satisfy the both-pads carry gate).
GRASP_DZ = 0.020
CRUISE_H = 0.10
HOVER = 0.008
# Extra jaw travel past first gate: DISABLED (0.0). The gated angle already
# holds squeeze force via the position servo; any extra travel ejects the
# load (seed-squeeze) instead of adding friction.
SQUEEZE_EXTRA = 0.0
JAW_OPEN = 0.6
JAW_CLOSED_FLOOR = 0.05
JAW_STEP = 0.02
# Teacher jaw force schedule (runtime control params, constant all seeds):
# WIDE (±3.35Nm, spec value) while traveling and approaching (fast, rigid);
# FIRM (±1.0Nm) once caged (shallow penetration stays gentle, and the firm
# ceiling stops the hanging load ratcheting the jaw open). NARROW (±0.15Nm)
# is defined as the gentle-touch authority but this teacher never commands
# it: the logged schedule below carries exactly the caps that were set.
# The per-frame cap stream is recorded (frame_jaw_cap) and exported, so the
# schedule in the dataset is the executed one, not this comment.
JAW_WIDE_NM = 3.35
JAW_NARROW_NM = 0.15
JAW_FIRM_NM = 1.0
# Arm push authority (runtime control params, constant all seeds): WIDE for
# free/reach motion; MEDIUM (+-1.0Nm) while sliding a caged load: enough to
# break table stiction decisively (a 0.4Nm cap starves into honest stalls),
# still capped far below snap energies (uncapped first touch hits ~8N).
ARM_WIDE_NM = 3.35
ARM_PUSH_NM = 1.0
SUCCESS_RADIUS = 0.015
# Site sits at pads-mid height; open-jaw pads-mid sits +x of the site.
SITE_TO_MID = np.array([0.0224, 0.0094, 0.0])

HOME = {
    "left": np.array([0.0, -0.3, -0.4, 0.5, 0.0]),
    "right": np.array([0.0, -0.3, -0.4, 0.5, 0.0]),
}
# Tool-mount orientation (pads straight down, pinch horizontal): the grasp
# approach biases toward it so the carry attitude needs only small
# corrections (large mid-carry attitude corrections swing the 10cm tool
# lever and drag the load off).
MOUNT_Q = np.array([0.707107, 0.707107, 0.0, 0.0])
PARK = {"left": np.array([-0.18, 0.0895, 1.2439]), "right": np.array([0.18, 0.0895, 1.2439])}


def seed_bundle(seed: int) -> dict:
    rng = np.random.default_rng(seed ^ 0xD177E2)
    # Staging sits well inside the left arm's healthy workspace (bent
    # elbow, far from singularity) and closer to the relay: shorter,
    # kinder carries. Still left-of-center ("mugs left").
    mug_xy = [-0.12 + float(rng.uniform(-0.009, 0.009)), 0.02 + float(rng.uniform(-0.009, 0.009))]
    plate_xy = [0.18 + float(rng.uniform(-0.009, 0.009)), 0.10 + float(rng.uniform(-0.009, 0.009))]
    teal = [0.33 * float(rng.uniform(0.9, 1.1)), 0.73, 0.69, 1.0]
    blue = [0.33, 0.51 * float(rng.uniform(0.9, 1.1)), 0.76, 1.0]
    return {
        "layout": "dual_so101_dinner shelf-only, no drawer",
        "mug_xy": mug_xy,
        "plate_xy": plate_xy,
        "mug_color": [round(float(c), 4) for c in teal],
        "plate_color": [round(float(c), 4) for c in blue],
        "lighting": round(float(rng.uniform(0.92, 1.06)), 4),
        "instruction": INSTRUCTION,
        "relay_point": list(scene.RELAY_POINT),
        "destination": list(scene.DEST_POINT),
    }


class Episode:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.arms = {side: ArmIK(model, data, side) for side in ("left", "right")}
        self.ctrl_ids = self.arms["left"].act_ids + self.arms["right"].act_ids
        self.jaw_cap_nm = {"left": JAW_WIDE_NM, "right": JAW_WIDE_NM}
        for side in ("left", "right"):
            self.jaw_cap(side, JAW_WIDE_NM)
        self.gates = {side: GraspGate(model, data, side) for side in ("left", "right")}
        self.mug_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mug")
        self.frame_mug: list = []
        self.loss_info: dict = {}
        self.renderer = mujoco.Renderer(model, height=256, width=256)
        self.step_times: list[float] = []
        self.steps = 0
        self.frames: dict[str, list] = {c: [] for c in scene.CAMERAS}
        self.frame_site = {"left": [], "right": []}
        self.frame_state: list = []
        self.frame_time: list = []
        self.frame_active: list = []
        self.frame_grip: list = []
        self.frame_ori: list = []
        self.step_qpos: list = []
        self.step_jaw: list = []
        self.step_jaw_meas: list = []
        self.step_force: list = []
        self.step_time: list = []
        # F1/F2 control-intent streams: jaw forcerange schedule (WIDE/FIRM
        # authority caps set by jaw_cap) and full actuator ctrl targets,
        # logged every step and every frame alongside measured state.
        self.step_jaw_cap: list = []
        self.frame_jaw_cap: list = []
        self.step_ctrl: list = []
        self.frame_ctrl: list = []
        self.hold = {side: (HOME[side].copy(), JAW_OPEN) for side in ("left", "right")}
        self.q_above: dict[str, np.ndarray] = {}
        self.q_grasp_pose: dict[str, np.ndarray] = {}
        # Compliant jaw hold per side: once gated, the jaw target tracks the
        # measured angle (+hair preload) instead of a fixed overshoot, so a
        # position servo against a rigid contact cannot ramp force unbounded.
        self.compliant = {"left": False, "right": False}
        self._prev_mug: np.ndarray | None = None
        self._last_bow = float("nan")
        self.active = "left"

    def jaw_cap(self, side: str, nm: float) -> None:
        self.model.actuator_forcerange[self.arms[side].act_ids[5]] = [-nm, nm]
        self.jaw_cap_nm[side] = float(nm)

    def arm_cap(self, side: str, nm: float) -> None:
        for act in self.arms[side].act_ids[:5]:
            self.model.actuator_forcerange[act] = [-nm, nm]

    def mug_pos(self) -> np.ndarray:
        return np.array(self.data.xpos[self.mug_body], dtype=float)

    def pads_mid(self, side: str) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        gate = self.gates[side]
        s = np.array(self.data.geom_xpos[gate.static_id], dtype=float)
        m = np.array(self.data.geom_xpos[gate.moving_id], dtype=float)
        return (s + m) / 2

    def do_step(self) -> None:
        t0 = time.perf_counter()
        mujoco.mj_step(self.model, self.data)
        self.step_times.append((time.perf_counter() - t0) * 1000.0)
        self.steps += 1
        q12 = np.concatenate([self.arms["left"].qpos(), [self.arms["left"].jaw()],
                                self.arms["right"].qpos(), [self.arms["right"].jaw()]])
        jaws = [self.hold["left"][1], self.hold["right"][1]]
        self.step_qpos.append(q12.astype(np.float32))
        self.step_jaw.append(np.array(jaws, dtype=np.float32))
        self.step_time.append(float(self.data.time))
        self.step_jaw_meas.append(
            np.array([self.arms["left"].jaw(), self.arms["right"].jaw()],
                     dtype=np.float32))
        jaw_cap_now = np.array([self.jaw_cap_nm["left"], self.jaw_cap_nm["right"]],
                               dtype=np.float32)
        ctrl_now = np.array([self.data.ctrl[i] for i in self.ctrl_ids],
                            dtype=np.float32)
        self.step_jaw_cap.append(jaw_cap_now)
        self.step_ctrl.append(ctrl_now)
        scan = self.gates[self.active].scan()
        if scan["static"]["touch"] and scan["moving"]["touch"]:
            self.step_force.append(
                float(min(scan["static"]["force"], scan["moving"]["force"])))
        else:
            self.step_force.append(0.0)
        if self.steps % FRAME_EVERY == 0:
            for cam in scene.CAMERAS:
                self.renderer.update_scene(self.data, camera=cam)
                self.frames[cam].append(self.renderer.render().copy())
            self.frame_site["left"].append(self.arms["left"].site_pos())
            self.frame_site["right"].append(self.arms["right"].site_pos())
            self.frame_state.append(q12.astype(np.float32))
            self.frame_time.append(float(self.data.time))
            self.frame_active.append(self.active)
            active_jaw = self.hold[self.active][1] if self.active in self.hold else 0.0
            self.frame_grip.append(float(active_jaw))
            self.frame_jaw_cap.append(jaw_cap_now.copy())
            self.frame_ctrl.append(ctrl_now.copy())
            mug_xpos = np.array(self.data.xpos[self.mug_body], dtype=float)
            mug_up = float(self.data.xmat[self.mug_body, 8])
            self.frame_mug.append(np.concatenate([mug_xpos, [mug_up]]).astype(np.float32))
            self.frame_ori.append(np.concatenate(
                [self.arms["left"].site_quat(), self.arms["right"].site_quat()]
            ).astype(np.float32))

    def servo(self, side: str, target: np.ndarray, cap: int, jaw: float,
              tol: float = 0.002, hold: int = 10, gain: float = 0.5,
              max_step: float = 0.30, watch: bool = False,
              exact: bool = True, stall_escape: bool = False,
              lim_avoid: float = 0.0, yaw_target: float | None = None,
              yaw_weight: float = 0.0, ori_target: np.ndarray | None = None,
              ori_weight: float = 0.0,
              ) -> tuple[float, str]:
        """Drive one arm to a site target; the other holds. Early-exits on
        convergence (honest per-seed step variance). With watch=True the
        cage is monitored: contact-gate streak, load sanity (fell/tipping),
        cage-slip guard, stall escape. Returns (final err, loss or '')."""
        self.active = side
        self.hold[side] = (self.hold[side][0], float(jaw))
        good = 0
        err = float("inf")
        loss = ""
        bad_streak = 0
        held, why = True, "init"
        best_err = float("inf")
        stall_count = 0
        escaped = False
        aim = np.asarray(target, dtype=float).copy()
        rel0: np.ndarray | None = None
        rel_prev: np.ndarray | None = None
        for step in range(cap):
            arm = self.arms[side]
            qpos = arm.servo_step(np.asarray(aim, dtype=float), gain=gain,
                                  max_step=max_step, lim_avoid=lim_avoid,
                                  yaw_target=yaw_target, yaw_weight=yaw_weight,
                                  ori_target=ori_target, ori_weight=ori_weight)
            arm.apply(qpos, float(jaw))
            self.hold[side] = (self.hold[side][0], float(jaw))
            arm.q_ref = 0.8 * arm.q_ref + 0.2 * qpos
            other = "right" if side == "left" else "left"
            self.arms[other].apply(*self.hold[other])
            self.do_step()
            err = float(np.linalg.norm(aim - arm.site_pos()))
            good = good + 1 if err < tol else 0
            if watch and step % 5 == 0 and step > 0:
                mug_now = self.mug_pos()
                rel = mug_now - self.pads_mid(side)
                rel = np.array([rel[0], rel[1], 0.0])
                if rel0 is None:
                    rel0 = rel.copy()
                    rel_prev = rel.copy()
                else:
                    jump = float(np.linalg.norm(rel - rel_prev))
                    drift = float(np.linalg.norm(rel - rel0))
                    rel_prev = rel.copy()
                    if jump > 0.004 or drift > 0.025:
                        loss = (f"cage slip (jump {jump * 1000:.0f}mm, "
                                f"drift {drift * 1000:.0f}mm)")
                        break
                if step % 20 == 0:
                    held, why = self.gates[side].gated(strict=False)
                    if mug_now[2] < scene.TABLE_Z - 0.05:
                        loss = f"mug escaped (z={mug_now[2]:.3f})"
                        break
                    if float(self.data.xmat[self.mug_body, 8]) < 0.75:
                        loss = "load tipping"
                        break
                    if float(np.linalg.norm(mug_now - arm.site_pos())) > 0.30:
                        loss = "mug out of cage (>0.3m)"
                        break
                    if not held:
                        bad_streak += 1
                        if bad_streak >= 3:
                            mug_now = self.mug_pos()
                            loss = (
                                f"cage lost en route ({why}) mug={np.round(mug_now, 3).tolist()} "
                                f"up={float(self.data.xmat[self.mug_body, 8]):.2f} "
                                f"jaw={float(jaw):.2f}"
                            )
                            if not self.loss_info:
                                self.loss_info = {"step": self.steps, "side": side,
                                                  "mug": mug_now.tolist(), "why": why}
                            break
                    else:
                        bad_streak = 0
            if exact and good >= hold:
                break
            if err < best_err - 0.002:
                best_err = err
                stall_count = 0
            else:
                stall_count += 1
            if watch and stall_escape and stall_count >= 150 and not escaped:
                escaped = True
                stall_count = 0
                best_err = float("inf")
                away = arm.site_pos() - aim
                if float(np.linalg.norm(away)) > 1e-6:
                    away = away / float(np.linalg.norm(away)) * 0.025
                else:
                    away = np.array([0, 0, 0.025])
                back = arm.site_pos() + away
                err_before = float(np.linalg.norm(aim - arm.site_pos()))
                for _ in range(100):
                    qpos = arm.servo_step(back, gain=0.15, max_step=0.01)
                    arm.apply(qpos, jaw)
                    other = "right" if side == "left" else "left"
                    self.arms[other].apply(*self.hold[other])
                    self.do_step()
                arm.q_ref = arm.qpos().copy()
                err_after = float(np.linalg.norm(aim - arm.site_pos()))
                if err_after > err_before - 0.005:
                    loss = (f"persistent stall (err {err_after * 1000:.0f}mm, "
                            f"escape inert)")
                    break
            elif stall_count >= 300:
                loss = f"stall (err {err * 1000:.0f}mm, no progress)"
                break
        arm = self.arms[side]
        arm.q_ref = arm.qpos().copy()
        self.hold[side] = (arm.qpos().copy(), self.hold[side][1])
        return err, loss

    def settle(self, n: int = SETTLE_STEPS) -> None:
        for _ in range(n):
            for side in ("left", "right"):
                qpos, jaw = self.hold[side]
                if self.compliant[side]:
                    jaw = self.arms[side].jaw() + 0.008
                    self.hold[side] = (qpos, float(jaw))
                self.arms[side].apply(qpos, jaw)
            self.do_step()

    def center_xy(self, side: str) -> float:
        """Center pads-mid over the mug in X/Y at the current height."""
        err = float("inf")
        for _ in range(6):
            want = self.mug_pos() + np.array([0, 0, GRASP_DZ])
            have = self.pads_mid(side)
            err_vec = np.array([want[0] - have[0], want[1] - have[1], 0.0])
            err = float(np.linalg.norm(err_vec))
            if err < 0.002:
                break
            self.servo(side, self.arms[side].site_pos() + err_vec, 150,
                       self.hold[side][1], gain=0.2, max_step=0.02,
                       stall_escape=True)
        return err

    def center_on_mug(self, side: str) -> float:
        # Safety net: Z trim ONLY. XY is already centered and re-servoing it
        # in full 3D around contacts walks the cage off (27mm residuals).
        err = float("inf")
        for _ in range(3):
            err_z = (self.mug_pos()[2] + GRASP_DZ) - self.pads_mid(side)[2]
            err = abs(float(err_z))
            if err < 0.002:
                break
            self.servo(side, self.arms[side].site_pos() + np.array([0, 0, err_z]), 150,
                       self.hold[side][1], gain=0.2, max_step=0.02,
                       stall_escape=True)
        have = self.pads_mid(side)
        want = self.mug_pos() + np.array([0, 0, GRASP_DZ])
        return float(np.linalg.norm(want - have))

    def close_and_gate(self, side: str) -> tuple[bool, str, float]:
        self.compliant[side] = False
        jaw = self.arms[side].jaw()
        why = "timeout"
        gated = False
        accepted = False
        taps = 0
        for _ in range(60):
            jaw = max(JAW_CLOSED_FLOOR, jaw - JAW_STEP)
            for _ in range(200):
                self.arms[side].apply(self.arms[side].qpos(), jaw)
                other = "right" if side == "left" else "left"
                self.arms[other].apply(*self.hold[other])
                self.do_step()
                if abs(self.arms[side].jaw() - jaw) < 0.02:
                    break
            self.hold[side] = (self.hold[side][0], jaw)
            gated, why = self.gates[side].gated()
            if gated and jaw <= 0.50:
                accepted = True
                break
            if jaw <= JAW_CLOSED_FLOOR + 1e-9:
                why = f"jaw floor without cage ({why})"
                break
            taps += 1
            if taps % 5 == 0:
                # Re-symmetrize periodically while closing (not every tap:
                # the cage must be allowed to settle onto the contacts).
                err_vec = ((self.mug_pos() + np.array([0, 0, GRASP_DZ]))
                           - self.pads_mid(side))
                self.servo(side, self.arms[side].site_pos() + err_vec, 25,
                           self.hold[side][1], gain=0.2, max_step=0.02)
                gated, why = self.gates[side].gated()
                if gated and jaw <= 0.50:
                    accepted = True
                    break
        if not accepted:
            self.jaw_cap(side, JAW_WIDE_NM)
            return False, f"no cage ({why})", jaw
        # Firm the jaw authority now that the cage is closed (see schedule):
        # the shallow first-touch penetration keeps steady force gentle while
        # the firm ceiling stops the hanging load ratcheting the jaw open.
        self.jaw_cap(side, JAW_FIRM_NM)
        jaw_hold = float(jaw)
        self.hold[side] = (self.arms[side].qpos().copy(), jaw_hold)
        gated, why = self.gates[side].gated()
        if gated:
            scan = self.gates[side].scan()
            held = min(scan["static"]["force"], scan["moving"]["force"])
            self.q_grasp_pose[side] = self.arms[side].qpos().copy()
            return True, f"{why} hold {held:.3f}N", jaw_hold
        return False, f"hold broke gate ({why})", jaw_hold

    def grasp(self, side: str, obj: np.ndarray) -> tuple[bool, str]:
        self.compliant[side] = False
        self.jaw_cap(side, JAW_WIDE_NM)
        if float(self.data.xmat[self.mug_body, 8]) < 0.9:
            return False, "mug tipped, unrecoverable"
        obj = np.asarray(obj, dtype=float)
        self.servo(side, obj + np.array([0, 0, 0.12]), 250, JAW_OPEN)
        self.q_above[side] = self.arms[side].qpos().copy()
        xy_res = self.center_xy(side)
        # Descend vertically onto the centered cage (no sideways push):
        # gentle servo, or the lunge undoes the centering.
        dz = (obj[2] + GRASP_DZ) - self.pads_mid(side)[2]
        self.servo(side, self.arms[side].site_pos() + np.array([0, 0, dz]), 250,
                   JAW_OPEN, gain=0.2, max_step=0.02, stall_escape=True)
        residual = self.center_on_mug(side)
        ok, detail, _ = self.close_and_gate(side)
        return ok, f"dz {GRASP_DZ:+.3f} xy {xy_res * 1000:.1f}mm center {residual * 1000:.1f}mm; {detail}"

    def lift_verify(self, side: str) -> tuple[bool, str]:
        # Cartesian vertical lift: retrace the proven descend line upward.
        # The old joint-space lift (interpolating back toward q_above)
        # twisted the tool for the right arm's geometry and halved its
        # pinch (3.26N at grasp -> 1.29N after lift, vs 3.08N -> 2.56N on
        # the left); the weakened cage then tipped at the first cruise
        # perturbation. A vertical lift preserves tool attitude, so the
        # grasp-time opposition and squeeze survive into the carry.
        z0 = float(self.mug_pos()[2])
        yaw0 = self.arms[side].site_yaw()
        yaw_w = 0.30 if side == "right" else 0.0
        quat0 = self.arms[side].site_quat()
        ori_w = 0.0
        _, lost = self.servo(side, self.arms[side].site_pos() + np.array([0, 0, 0.055]),
                             600, self.hold[side][1], tol=0.004,
                             gain=0.15, max_step=0.006,
                             watch=True, stall_escape=True, exact=True,
                             yaw_target=yaw0, yaw_weight=yaw_w,
                             ori_target=quat0, ori_weight=ori_w)
        if lost:
            return False, f"lift lost cage ({lost})"
        self.arms[side].q_ref = self.arms[side].qpos().copy()
        self.hold[side] = (self.arms[side].qpos().copy(), self.hold[side][1])
        rise = float(self.mug_pos()[2]) - z0
        # Flicker-tolerant verdict: best of 3 scans over 30 steps.
        verdicts = []
        for _ in range(3):
            gated, why = self.gates[side].gated()
            verdicts.append((gated, why))
            for _ in range(10):
                self.arms[side].apply(self.arms[side].qpos(), self.hold[side][1])
                other = "right" if side == "left" else "left"
                self.arms[other].apply(*self.hold[other])
                self.do_step()
        gated, why = next(((g, w) for g, w in verdicts if g), verdicts[-1])
        if rise >= 0.015 and gated:
            return True, f"rose {rise * 1000:.1f}mm, {why}"
        return False, f"slip (rose {rise * 1000:.1f}mm, {why})"

    def execute_joints(self, side: str, q_to: np.ndarray, jaw: float,
                       line_from: np.ndarray, line_to: np.ndarray,
                       label: str = "") -> str:
        """Joint-interpolated carry with cartesian trim. The interpolation
        base guarantees progress (stall-proof); one small DLS trim per step
        toward the straight cartesian line kills joint-space bows (an
        untrimmed bow once flipped the caged mug 90 deg in one frame)."""
        arm = self.arms[side]
        q_from = arm.qpos().copy()
        start = np.asarray(line_from, dtype=float)
        end = np.asarray(line_to, dtype=float)
        dist = float(np.linalg.norm(np.asarray(q_to) - q_from))
        n = max(200, int(dist / 0.001))
        loss = ""
        for k in range(1, n + 1):
            s = k / n
            e = s * s * (3.0 - 2.0 * s)
            q_base = q_from + (np.asarray(q_to) - q_from) * e
            arm.set_qpos(q_base)
            mujoco.mj_forward(self.model, self.data)
            line_pt = start + (end - start) * e
            q_trim = arm.servo_step(line_pt, gain=0.3, max_step=0.002)
            blend = 0.0  # trim disabled: it vibrates against the base
            q_apply = q_base + (q_trim - q_base) * blend
            arm.apply(q_apply, jaw)
            other = "right" if side == "left" else "left"
            self.arms[other].apply(*self.hold[other])
            self.do_step()
            if k % 20 == 0:
                held, why = self.gates[side].gated(strict=False)
                mug_now = self.mug_pos()
                if mug_now[2] < scene.TABLE_Z - 0.05:
                    loss = f"mug escaped (z={mug_now[2]:.3f})"
                    break
                if float(np.linalg.norm(mug_now - arm.site_pos())) > 0.30:
                    loss = "mug out of cage (>0.3m)"
                    break
                if not held:
                    loss = f"cage lost ({why})"
                    break
        arm.q_ref = arm.qpos().copy()
        self.hold[side] = (arm.qpos().copy(), self.hold[side][1])
        if loss:
            return f"{label}: {loss}" if label else loss
        return ""

    def touchdown(self, side: str) -> tuple[bool, str]:
        """Lower the verified pinch back onto the table (joint interp to the
        recorded grasp pose, eased). The carry then slides at table level."""
        q_from = self.arms[side].qpos().copy()
        q_to = self.q_grasp_pose[side].copy()
        jaw = self.hold[side][1]
        n = 300
        for k in range(1, n + 1):
            s = k / n
            e = s * s * (3.0 - 2.0 * s)
            self.arms[side].apply(q_from + (q_to - q_from) * e, jaw)
            other = "right" if side == "left" else "left"
            self.arms[other].apply(*self.hold[other])
            self.do_step()
            if k % 20 == 0:
                held, why = self.gates[side].gated(strict=False)
                if not held:
                    return False, f"touchdown lost cage ({why})"
        self.arms[side].q_ref = self.arms[side].qpos().copy()
        self.hold[side] = (self.arms[side].qpos().copy(), self.hold[side][1])
        return True, "touchdown"

    def _veer(self, a: np.ndarray, b: np.ndarray) -> float:
        """Perpendicular distance of the live mug from segment a->b."""
        a = np.asarray(a, dtype=float)[:2]
        b = np.asarray(b, dtype=float)[:2]
        p = self.mug_pos()[:2]
        ab = b - a
        denom = float(np.linalg.norm(ab))
        if denom < 1e-9:
            return float(np.linalg.norm(p - a))
        t = float(np.clip(np.dot(p - a, ab) / (denom * denom), 0.0, 1.0))
        return float(np.linalg.norm(p - (a + ab * t)))

    def set_down(self, side: str) -> None:
        """Abort recovery: if the load is still near-upright, lower it
        blind (like the grasp descend) to table height for a safe rest.
        If it already leans badly, a vertical descend would shove it over,
        so hold still instead and let the swing damp: it may re-seat.
        Releases only when the cage is lost or the load stays leaned; on
        recovery the cage stays shut and the attempt loop resumes cruise."""
        yaw_sd = self.arms[side].site_yaw()
        yaw_w = 0.30 if side == "right" else 0.0
        quat_sd = self.arms[side].site_quat()
        ori_w = 0.0
        # Calm first: the abort may have caught a swing mid-stroke, and a
        # still load often re-seats itself upright in the cage.
        self.settle(300)
        held, _ = self.gates[side].gated(strict=False)
        up_now = float(self.data.xmat[self.mug_body, 8])
        if held and up_now > 0.90:
            return
        up_now = float(self.data.xmat[self.mug_body, 8])
        if up_now > 0.85:
            for _ in range(2):
                dz = ((scene.TABLE_Z + scene.MUG_HALFHEIGHT + 0.010)
                      - float(self.mug_pos()[2]))
                if dz >= -0.005:
                    break
                # Blind descend (no watch): like the grasp descend, this must
                # reach table height before release. A watch abort here would
                # release from height and drop-tip the load it tries to save.
                self.servo(side, self.arms[side].site_pos() + np.array([0, 0, dz]),
                           600, self.hold[side][1], tol=0.004,
                           gain=0.15, max_step=0.006,
                           watch=False, stall_escape=True, exact=True,
                           yaw_target=yaw_sd, yaw_weight=yaw_w,
                           ori_target=quat_sd, ori_weight=ori_w)
        else:
            # Already leaning: a fast descend would shove it over, but a
            # 100mm drop is worse. Creep down gently, then release.
            for _ in range(2):
                dz = ((scene.TABLE_Z + scene.MUG_HALFHEIGHT + 0.010)
                      - float(self.mug_pos()[2]))
                if dz >= -0.005:
                    break
                self.servo(side, self.arms[side].site_pos() + np.array([0, 0, dz]),
                           900, self.hold[side][1], tol=0.004,
                           gain=0.10, max_step=0.003,
                           watch=False, stall_escape=True, exact=True,
                           yaw_target=yaw_sd, yaw_weight=yaw_w,
                           ori_target=quat_sd, ori_weight=ori_w)
        held, _ = self.gates[side].gated(strict=False)
        up_now = float(self.data.xmat[self.mug_body, 8])
        if held and up_now > 0.90:
            return
        self.release(side)

    def traverse_joints(self, side: str, site_target: np.ndarray,
                          quat_hold: np.ndarray, jaw: float,
                          segments: int = 2, seg_steps: int = 400) -> str:
        """Joint-space traverse with cage watch. Offline ori-biased IK plans
        start->mid->end near the grasp-time attitude; slow smoothstep joint
        interpolation then executes without the online-DLS pitching (which
        scooped the right arm's load out sideways). Returns loss or ''."""
        arm = self.arms[side]
        start_site = arm.site_pos()
        end_site = np.asarray(site_target, dtype=float)
        qs = [arm.qpos().copy()]
        res = float("inf")
        for k in range(1, segments + 1):
            tgt = start_site + (end_site - start_site) * (k / segments)
            qk, res = arm.plan_to(tgt, qs[-1],
                                  ori_target=np.asarray(quat_hold, dtype=float),
                                  ori_weight=0.0)
            qs.append(qk)
        if res > 0.010:
            return f"no IK (residual {res * 1000:.0f}mm)"
        other = "right" if side == "left" else "left"
        rel0: np.ndarray | None = None
        rel_prev: np.ndarray | None = None
        bad_streak = 0
        step = 0
        for s in range(segments):
            q_from, q_to = qs[s], qs[s + 1]
            for k in range(1, seg_steps + 1):
                e = k / seg_steps
                e = e * e * (3.0 - 2.0 * e)
                arm.apply(q_from + (q_to - q_from) * e, float(jaw))
                self.arms[other].apply(*self.hold[other])
                self.do_step()
                step += 1
                if step % 5 == 0:
                    mug_now = self.mug_pos()
                    rel = mug_now - self.pads_mid(side)
                    rel = np.array([rel[0], rel[1], 0.0])
                    if rel0 is None:
                        rel0 = rel.copy()
                        rel_prev = rel.copy()
                    else:
                        jump = float(np.linalg.norm(rel - rel_prev))
                        drift = float(np.linalg.norm(rel - rel0))
                        rel_prev = rel.copy()
                        if jump > 0.004 or drift > 0.025:
                            return (f"cage slip (jump {jump * 1000:.0f}mm, "
                                    f"drift {drift * 1000:.0f}mm)")
                    if step % 20 == 0:
                        held, why = self.gates[side].gated(strict=False)
                        if mug_now[2] < scene.TABLE_Z - 0.05:
                            return f"mug escaped (z={mug_now[2]:.3f})"
                        if float(self.data.xmat[self.mug_body, 8]) < 0.75:
                            return "load tipping"
                        if float(np.linalg.norm(mug_now - arm.site_pos())) > 0.30:
                            return "mug out of cage (>0.3m)"
                        if not held:
                            bad_streak += 1
                            if bad_streak >= 3:
                                return f"cage lost en route ({why})"
                        else:
                            bad_streak = 0
        arm.q_ref = arm.qpos().copy()
        self.hold[side] = (arm.qpos().copy(), self.hold[side][1])
        return ""

    def carry_to(self, side: str, dest: np.ndarray,
                 tol_xy: float = 0.015, min_up: float = 0.94) -> tuple[bool, str]:
        """Aerial relay: grasp+lift once, then ONE direct servo to above
        dest (no intermediate waypoints: every retarget kicked the caged
        load). Slow position-only tracking with cage monitoring; up to 2
        full recoveries (release, re-grasp, re-lift); then fail honestly.
        tol_xy/min_up gate the touchdown: a loose first leg (midpoint
        set-down) only needs the load upright nearby; the final leg
        places precisely."""
        dest = np.asarray(dest, dtype=float)
        cruise_z = scene.TABLE_Z + scene.MUG_HALFHEIGHT + CRUISE_H
        hover_z = scene.TABLE_Z + scene.MUG_HALFHEIGHT + HOVER
        self.settle(40)
        lost = "attempts exhausted"
        attempts: list[str] = []
        for attempt in range(3):
            gated, _ = self.gates[side].gated()
            if not gated:
                if attempt > 0:
                    # Recovery grasps must start from the same known-good
                    # stance as initial grasps: centering from a leftover
                    # abort stance stalled at 19mm and rim-shoved the mug.
                    self.park(side)
                ok, gdetail = self.grasp(side, self.mug_pos())
                if not ok:
                    return False, f"attempt{attempt} grasp failed ({gdetail})"
                ok, detail = self.lift_verify(side)
                if not ok:
                    return False, f"attempt{attempt} lift failed ({detail})"
                print(f"carry {side} attempt{attempt} caged: [{gdetail}] [{detail}]",
                      flush=True)
            # Calm before cruise: the lift can leave residual swing, and
            # accelerating into the traverse with a swinging load pumps it.
            # Hold still ~2 pendulum periods so cruise starts from rest.
            self.settle(300)
            # Climb to cruise height VERTICALLY before traversing: the old
            # diagonal climb-traverse pitched the tool 28deg at cruise start
            # and scooped the load sideways out of the cage. Verticals keep
            # attitude (proven by the lift); the traverse then runs level.
            yaw_cv = self.arms[side].site_yaw()
            yaw_cw = 0.30 if side == "right" else 0.0
            _, lost_cv = self.servo(side, self.arms[side].site_pos()
                                    + np.array([0, 0, cruise_z - float(self.mug_pos()[2])]),
                                     600, self.hold[side][1], tol=0.004,
                                     gain=0.15, max_step=0.006,
                                     watch=True, stall_escape=True, exact=True,
                                     yaw_target=yaw_cv, yaw_weight=yaw_cw)
            if lost_cv:
                print(f"carry {side} attempt{attempt} climb lost: {lost_cv}", flush=True)
                self.set_down(side)
                attempts.append(f"a{attempt}:climb:{lost_cv}")
                continue
            mug0 = self.mug_pos()
            off = self.arms[side].site_pos() - mug0
            # Yaw-hold: the long lateral traverse otherwise sweeps the base
            # yaw joint through ~70deg, twisting the caged load until it
            # leans out of the cage. Pin the grasp-time tool yaw (weak 1D
            # task: positions keep priority) for cruise + hover.
            yaw0 = self.arms[side].site_yaw()
            # Right leg only: its straight traverse crosses the base line
            # (a ~70deg yaw sweep that twists the load out of the cage).
            # The left leg's diagonal trip places perfectly without it.
            # Strong weight: over a short (<=110mm) leg the wrist can
            # counter-rotate the sweep, so the pin is feasible; a weak
            # weight lost 27deg and the load leaned out.
            yaw_w = 0.30 if side == "right" else 0.0
            # Full attitude hold (right leg): yaw alone held to 4deg yet the
            # load still walked out, so the uncontrolled pitch/roll must be
            # rocking the pads across the mug. Bias the whole grasp-time
            # attitude (weak 6D: positions keep priority).
            quat0 = self.arms[side].site_quat()
            # Full-attitude bias only where the path is short and vertical
            # (hover): on the long cruise the 5 joints cannot serve 6D and
            # the compromise stalls 85mm short, so cruise pins yaw only.
            ori_w = 0.0
            print(f"carry {side} cruise start: yaw0={np.degrees(yaw0):.1f}deg w={yaw_w}",
                  flush=True)
            cruise = np.array([dest[0], dest[1], cruise_z]) + off
            lost = self.traverse_joints(side, cruise, quat0, self.hold[side][1])
            if lost.startswith("no IK"):
                print(f"carry {side} attempt{attempt} joint plan failed: {lost}; "
                      f"falling back to online servo", flush=True)
                lost = ""
                for _ in range(5):
                    _, lost = self.servo(side, cruise, 800, self.hold[side][1],
                                         tol=0.004, gain=0.10, max_step=0.006,
                                         watch=True, stall_escape=True,
                                         exact=True,
                                         yaw_target=yaw0, yaw_weight=yaw_w,
                                         ori_target=quat0, ori_weight=ori_w)
                    if lost:
                        break
                # Re-anchor: tool attitude drift rotates the site-mug offset;
                # a stale offset aims 10s of mm off. Recompute and resume.
                off = self.arms[side].site_pos() - self.mug_pos()
                cruise = np.array([dest[0], dest[1], cruise_z]) + off
                if float(np.linalg.norm(self.arms[side].site_pos() - cruise)) < 0.006:
                    lost = ""
                    break
                lost = ""
            if lost:
                print(f"carry {side} attempt{attempt} cruise lost: {lost} "
                      f"(yaw {np.degrees(self.arms[side].site_yaw()):.1f}deg)",
                      flush=True)
                self.set_down(side)
                attempts.append(f"a{attempt}:cruise:{lost}")
                continue
            off = self.arms[side].site_pos() - self.mug_pos()
            hover = np.array([dest[0], dest[1], hover_z]) + off
            # Joint-interp descend: the online DLS descend pitched the tool
            # 41deg and dragged the load 29mm off. Same chord treatment as
            # the traverse (which held pitch at zero with the load at 1.00).
            lost = self.traverse_joints(side, hover, quat0, self.hold[side][1],
                                        segments=1, seg_steps=400)
            if lost:
                print(f"carry {side} attempt{attempt} hover lost: {lost}", flush=True)
                self.set_down(side)
                attempts.append(f"a{attempt}:hover:{lost}")
                continue
            final_gap = float(np.linalg.norm(self.mug_pos()[:2] - dest[:2]))
            held, why = self.gates[side].gated(strict=False)
            self.release(side)
            up = float(self.data.xmat[self.mug_body, 8])
            print(f"carry {side} attempt{attempt}: gap {final_gap * 1000:.0f}mm up {up:.2f} gate {why}",
                  flush=True)
            if final_gap <= tol_xy and up > min_up:
                return True, f"attempt{attempt} placed ({why})"
            lost = f"missed by {final_gap * 1000:.0f}mm up {up:.2f} ({why})"
            attempts.append(f"a{attempt}:{lost}")
        return False, f"carry lost ({' | '.join(attempts)})"

    def release(self, side: str) -> None:
        self.compliant[side] = False
        self.jaw_cap(side, JAW_WIDE_NM)
        for _ in range(800):
            self.arms[side].apply(self.arms[side].qpos(), JAW_OPEN)
            other = "right" if side == "left" else "left"
            self.arms[other].apply(*self.hold[other])
            self.do_step()
            if abs(self.arms[side].jaw() - JAW_OPEN) < 0.02:
                break
        self.hold[side] = (self.hold[side][0], JAW_OPEN)
        self.settle()

    def park(self, side: str, loaded: bool = False) -> None:
        if loaded:
            # Gentle fold-back with a caged load (never the fast default).
            self.servo(side, PARK[side], 600, self.hold[side][1],
                       tol=0.006, gain=0.10, max_step=0.005,
                       watch=True, stall_escape=False)
        else:
            self.servo(side, PARK[side], 400, JAW_OPEN, stall_escape=True)

    def close(self) -> None:
        self.renderer.close()


def run_seed(seed: int, bundle: dict, poses_hash: str) -> dict:
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
    ep = Episode(model, data)
    for side in ("left", "right"):
        ep.arms[side].set_qpos(HOME[side])
        ep.arms[side].q_ref = HOME[side].copy()
        # Preset the jaw open as a reset (no contacts yet): under the
        # capped hold force it would otherwise take ~1200 steps to slew.
        data.qpos[ep.arms[side].jaw_qadr] = JAW_OPEN
        ep.arms[side].apply(HOME[side], JAW_OPEN)
    mujoco.mj_forward(model, data)
    ep.settle(50)

    relay = np.array(scene.RELAY_POINT)
    dest = np.array(scene.DEST_POINT)
    log: list[str] = []

    ok, detail = ep.carry_to("left", relay)
    log.append(f"A carry: {detail}")
    if not ok:
        return finish(ep, seed, False, f"A carry failed ({detail})", log, np.nan)
    ep.park("left")
    at_relay = ep.mug_pos()
    relay_err = float(np.linalg.norm(at_relay[:2] - relay[:2]))

    # B (right arm) runs in two arc legs: its relay->dest traverse slews
    # the base yaw ~60deg, and the wrist cannot counter-rotate that far,
    # so the tool twists the caged load out. Each ~30deg leg holds yaw to
    # a few degrees (proven) with a fresh cage per leg.
    arc_mid = np.array([0.08, -0.03, scene.TABLE_Z])
    ok, detail = ep.carry_to("right", arc_mid, tol_xy=0.060, min_up=0.90)
    log.append(f"B1 carry: {detail}")
    if not ok:
        return finish(ep, seed, False, f"B1 carry failed ({detail})", log, relay_err)
    ep.park("right")

    ok, detail = ep.carry_to("right", dest)
    log.append(f"B carry: {detail}")
    if not ok:
        return finish(ep, seed, False, f"B carry failed ({detail})", log, relay_err)
    ep.park("right")
    ep.settle(60)

    final = ep.mug_pos()
    rest_z = scene.TABLE_Z + scene.MUG_HALFHEIGHT
    separation = float(final[2] - rest_z)
    xy_err = float(np.linalg.norm(final[:2] - dest[:2]))
    upright = bool(data.xmat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mug"), 8]
                   > np.cos(np.radians(15)))
    penetrates = separation < -0.001
    success = bool(xy_err <= SUCCESS_RADIUS and abs(separation) < 0.004
                   and upright and not penetrates)
    cause = ""
    if penetrates:
        cause = "penetration"
    elif xy_err > SUCCESS_RADIUS:
        cause = f"missed destination by {xy_err * 1000:.1f}mm"
    elif not upright:
        cause = "mug tipped"
    return finish(ep, seed, success,
                  "scripted teacher; physics-step p50, no NN forward" if success else f"FAILED: {cause}",
                  log, relay_err, cause, xy_err, separation, final)


def finish(ep: Episode, seed: int, success: bool, note: str, log: list,
           relay_err: float, cause: str = "", xy_err: float = float("nan"),
           separation: float = float("nan"), final=None) -> dict:
    forward_p50 = float(np.median(ep.step_times)) if ep.step_times else 0.0
    result = {
        "seed": seed,
        "success": success,
        "steps": ep.steps,
        "forward_p50_ms": round(forward_p50, 4),
        "note": note,
        "final_xy_err_m": xy_err,
        "relay_err_m": relay_err,
        "separation_m": separation,
        "cause": cause,
        "log": log,
        "frames": ep.frames,
        "frame_site": {k: np.asarray(v) for k, v in ep.frame_site.items()},
        "frame_state": np.asarray(ep.frame_state, dtype=np.float32),
        "frame_time": np.asarray(ep.frame_time, dtype=np.float64),
        "frame_active": np.asarray(ep.frame_active),
        "frame_grip": np.asarray(ep.frame_grip, dtype=np.float32),
        "frame_jaw_cap": np.asarray(ep.frame_jaw_cap, dtype=np.float32),
        "frame_ctrl": np.asarray(ep.frame_ctrl, dtype=np.float32),
        "frame_mug": np.asarray(ep.frame_mug, dtype=np.float32),
        "frame_ori": np.asarray(ep.frame_ori, dtype=np.float32),
        "step_qpos": np.asarray(ep.step_qpos, dtype=np.float32),
        "step_jaw": np.asarray(ep.step_jaw, dtype=np.float32),
        "step_jaw_meas": np.asarray(ep.step_jaw_meas, dtype=np.float32),
        "step_jaw_cap": np.asarray(ep.step_jaw_cap, dtype=np.float32),
        "step_ctrl": np.asarray(ep.step_ctrl, dtype=np.float32),
        "step_force": np.asarray(ep.step_force, dtype=np.float32),
        "step_time": np.asarray(ep.step_time, dtype=np.float64),
        "final_pos": (final.tolist() if final is not None else ep.mug_pos().tolist()),
    }
    ep.close()
    return result


def main() -> int:
    bundles = {s: seed_bundle(s) for s in SEEDS}
    hashes = freeze_table(bundles)
    only = [int(a) for a in sys.argv[1:] if a.lstrip("-").isdigit()]
    seeds = [s for s in SEEDS if not only or s in only]
    OUT_SEEDS.mkdir(parents=True, exist_ok=True)
    OUT_DATASET.mkdir(parents=True, exist_ok=True)
    (OUT_SEEDS / "seed_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")

    rows = []
    for seed in seeds:
        result = run_seed(seed, bundles[seed], hashes[seed])
        seed_dir = OUT_SEEDS / f"seed_{seed}"
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
            OUT_DATASET / f"seed_{seed}.npz",
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
            instruction=np.array(INSTRUCTION),
            seed=np.array(seed),
            poses_hash=np.array(hashes[seed]),
            success=np.array(result["success"]),
            bundle=np.array(json.dumps(bundles[seed], sort_keys=True)),
        )
        append_trace(
            TRACE_PATH,
            {
                "skill": "table_relay",
                "seed": seed,
                "run": RUN_TAG,
                "instruction": INSTRUCTION,
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

    print_table(rows, INSTRUCTION)
    wins = sum(1 for r in rows if r["success"])
    print(f"scripted gate: {wins}/10 (need >=9/10)")
    print(f"hashes: {OUT_SEEDS / 'seed_hashes.json'} traces: {TRACE_PATH} dataset: {OUT_DATASET}")
    return 0 if wins >= 9 else 2


if __name__ == "__main__":
    raise SystemExit(main())
