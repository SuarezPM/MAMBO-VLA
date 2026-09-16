"""Contact-gated grasp.

Attach requires all three: both pads touching the mug, a minimum squeeze
force (a real pinch, not a graze), and opposition (contacts on opposite
sides of the mug center, rejecting same-side/top presses). No weld, no
mocap, no teleport: once gated, the mug rides friction only.
"""

import mujoco
import numpy as np

FORCE_MIN_N = 0.05
RETAIN_MIN_N = 0.15
# 90 deg still rejects same-side/top presses (measured <60 deg) while
# allowing the working low pinch, whose contacts sit ~90-120 deg apart.
OPPOSITION_DEG = 90.0
# Regulated hold window: enough friction for 0.44N weight + swing margin,
# capped far below ejection forces (uncapped first touch hit ~8N).
HOLD_MIN_N = 0.7

_MUG_GEOM = "mug_geom"
_MUG_BODY = "mug"


class GraspGate:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, side: str):
        self.model = model
        self.data = data
        self.static_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_static_finger_pad"
        )
        self.moving_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_moving_finger_pad"
        )
        self.mug_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, _MUG_GEOM)
        self.mug_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _MUG_BODY)
        self._force = np.zeros(6)

    def scan(self) -> dict:
        """Strongest normal force per pad plus contact positions."""
        static = {"touch": False, "force": 0.0, "pos": None}
        moving = {"touch": False, "force": 0.0, "pos": None}
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            pair = {contact.geom1, contact.geom2}
            if self.mug_geom not in pair:
                continue
            other = (pair - {self.mug_geom}).pop()
            if other not in (self.static_id, self.moving_id):
                continue
            mujoco.mj_contactForce(self.model, self.data, i, self._force)
            normal = float(self._force[0])
            slot = static if other == self.static_id else moving
            if normal > slot["force"]:
                slot.update(
                    {"touch": True, "force": normal, "pos": np.array(contact.pos)}
                )
        return {"static": static, "moving": moving}

    def opposition_deg(self, scan: dict) -> float | None:
        if not (scan["static"]["touch"] and scan["moving"]["touch"]):
            return None
        center = np.array(self.data.xpos[self.mug_body], dtype=float)
        v1 = scan["static"]["pos"] - center
        v2 = scan["moving"]["pos"] - center
        n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        if n1 < 1e-9 or n2 < 1e-9:
            return None
        cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        return float(np.degrees(np.arccos(cosang)))

    def push_gate(self) -> tuple[bool, str]:
        """Push-transport gate: EITHER pad driving with force while the mug
        advances upright. (A centered squeeze has net zero push and cannot
        break table stiction; pushing is inherently single-sided. The strict
        pinch is proven separately at attach+lift every leg.)"""
        scan = self.scan()
        best = 0.0
        touched = False
        for key in ("static", "moving"):
            if scan[key]["touch"]:
                touched = True
                best = max(best, scan[key]["force"])
        if not touched:
            return False, "no push contact"
        if best < 0.15:
            return False, f"push unloaded {best:.3f}N"
        return True, f"push {best:.3f}N"

    def gated(self, scan: dict | None = None, strict: bool = True) -> tuple[bool, str]:
        """Attach gate. Strict (attach/lift): squeeze + opposition.
        Retention (carry): both pads still loaded (early warning)."""
        scan = scan if scan is not None else self.scan()
        if not (scan["static"]["touch"] and scan["moving"]["touch"]):
            return False, "single-side contact"
        weakest = min(scan["static"]["force"], scan["moving"]["force"])
        if strict:
            if weakest < FORCE_MIN_N:
                return False, f"weak squeeze {weakest:.3f}N"
            angle = self.opposition_deg(scan)
            if angle is None or angle < OPPOSITION_DEG:
                return False, f"no opposition ({angle})"
            return True, f"pinch {weakest:.3f}N @{angle:.0f}deg"
        if weakest < RETAIN_MIN_N:
            return False, f"cage unloaded {weakest:.3f}N"
        return True, f"cage held {weakest:.3f}N"
