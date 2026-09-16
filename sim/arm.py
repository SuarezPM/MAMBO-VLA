"""Damped least-squares position servo for one articulated arm.

Tracks cartesian site targets with a position-primary solve plus a weak
pull toward a reference posture (keeps the solver on sane branches).
Orientation readout is for monitoring; the grasp gate enforces pad
opposition instead of an exact 6D pose (overconstrained for five joints).
"""

import mujoco
import numpy as np

ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
JAW_JOINT = "gripper"


class ArmIK:
    """Damped least-squares position servo for one arm."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        side: str,
        damping: float = 0.08,
        max_step: float = 0.30,
        ref_pull: float = 0.0,
    ):
        if side not in ("left", "right"):
            raise ValueError("side must be left or right")
        self.model = model
        self.data = data
        self.side = side
        self.damping = damping
        self.max_step = max_step
        self.ref_pull = ref_pull
        self.joint_names = [f"{side}_{name}" for name in ARM_JOINTS]
        self.jaw_name = f"{side}_{JAW_JOINT}"
        self.joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.joint_names
        ]
        if any(j < 0 for j in self.joint_ids):
            raise ValueError(f"Arm joints missing for {side}")
        self.jaw_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, self.jaw_name)
        self.qadr = [model.jnt_qposadr[j] for j in self.joint_ids]
        self.dofs = [model.jnt_dofadr[j] for j in self.joint_ids]
        self.jaw_qadr = model.jnt_qposadr[self.jaw_id]
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_grip")
        if self.site_id < 0:
            raise ValueError(f"Site {side}_grip missing")
        actuator_names = [
            f"{side}_pan",
            f"{side}_lift",
            f"{side}_elbow",
            f"{side}_wrist",
            f"{side}_roll",
            f"{side}_jaw_act",
        ]
        self.act_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in actuator_names
        ]
        if any(a < 0 for a in self.act_ids):
            raise ValueError(f"Actuators missing for {side}")
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
        self.q_ref = self.qpos().copy()
        self._qbuf = np.zeros(4)

    def qpos(self) -> np.ndarray:
        return np.array([self.data.qpos[a] for a in self.qadr])

    def set_qpos(self, q: np.ndarray) -> None:
        for adr, value in zip(self.qadr, q):
            self.data.qpos[adr] = float(value)

    def site_pos(self) -> np.ndarray:
        return np.array(self.data.site_xpos[self.site_id], dtype=float)

    def site_quat(self) -> np.ndarray:
        mujoco.mju_mat2Quat(self._qbuf, self.data.site_xmat[self.site_id])
        return self._qbuf.copy()

    def site_yaw(self) -> float:
        rmat = self.data.site_xmat[self.site_id].reshape(3, 3)
        return float(np.arctan2(rmat[1, 0], rmat[0, 0]))

    def jaw(self) -> float:
        return float(self.data.qpos[self.jaw_qadr])

    def servo_step(self, target: np.ndarray, gain: float = 0.5,
                   max_step: float | None = None, ori_target: np.ndarray | None = None,
                   ori_weight: float = 0.0, yaw_target: float | None = None,
                   yaw_weight: float = 0.0, lim_avoid: float = 0.0) -> np.ndarray:
        """DLS position step with reference pull; optionally holds a tool
        orientation (weak 6D) and/or a tool-X yaw angle (1D, cheap: keeps
        the pinch axis aligned without fighting positioning)."""
        mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.site_id)
        cols = self.dofs
        err_pos = np.asarray(target, dtype=float) - self.site_pos()
        err_ref = (self.q_ref - self.qpos()) * self.ref_pull
        if ori_target is None or ori_weight <= 0.0:
            err = np.concatenate([err_pos, err_ref])
            jac = np.vstack([self.jacp[:, cols], np.eye(len(cols)) * self.ref_pull])
        else:
            quat = self.site_quat()
            w1, x1, y1, z1 = (float(v) for v in ori_target)
            w2, x2, y2, z2 = quat[0], -quat[1], -quat[2], -quat[3]
            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
            if w < 0.0:
                x, y, z, w = -x, -y, -z, -w
            ang = 2.0 * float(np.arccos(min(1.0, w)))
            spread = float(np.sqrt(max(0.0, 1.0 - w * w)))
            axis = (np.array([x, y, z]) / spread * ang) if spread > 1e-6 else np.zeros(3)
            err = np.concatenate([err_pos, err_ref, axis * ori_weight])
            jac = np.vstack(
                [self.jacp[:, cols], np.eye(len(cols)) * self.ref_pull,
                 self.jacr[:, cols] * ori_weight]
            )
        if yaw_target is not None and yaw_weight > 0.0:
            rmat = self.data.site_xmat[self.site_id].reshape(3, 3)
            yaw_now = float(np.arctan2(rmat[1, 0], rmat[0, 0]))
            yaw_err = (float(yaw_target) - yaw_now + np.pi) % (2.0 * np.pi) - np.pi
            err = np.concatenate([err, np.array([yaw_err * yaw_weight])])
            jac = np.vstack([jac, self.jacr[2:3, cols] * yaw_weight])
        # Joint-limit avoidance (opt-in): repulsive pull inside 10% margins
        # keeps travel reserve on long cruises (DLS otherwise folds into
        # limits and sticks 39mm short).
        lim_err = []
        lim_jac = []
        if lim_avoid > 0.0:
            for i, joint in enumerate(self.joint_ids):
                lo, hi = self.model.jnt_range[joint]
                if hi <= lo:
                    continue
                span = hi - lo
                margin = 0.1 * span
                qi = float(self.data.qpos[self.qadr[i]])
                row = np.zeros(len(cols))
                row[i] = 1.0
                if qi < lo + margin:
                    lim_err.append(lim_avoid * (lo + margin - qi) / margin)
                    lim_jac.append(row * lim_avoid)
                elif qi > hi - margin:
                    lim_err.append(-lim_avoid * (qi - (hi - margin)) / margin)
                    lim_jac.append(row * lim_avoid)
        if lim_err:
            err = np.concatenate([err, np.array(lim_err)])
            jac = np.vstack([jac] + [r.reshape(1, -1) for r in lim_jac])
        dq, *_ = np.linalg.lstsq(
            jac.T @ jac + self.damping**2 * np.eye(len(cols)), jac.T @ err, rcond=None
        )
        limit = self.max_step if max_step is None else max_step
        dq = np.clip(dq * gain, -limit, limit)
        qpos = self.qpos() + dq
        for i, joint in enumerate(self.joint_ids):
            lo, hi = self.model.jnt_range[joint]
            if hi > lo:
                qpos[i] = float(np.clip(qpos[i], lo, hi))
        return qpos

    def plan_to(self, target: np.ndarray, q_start: np.ndarray,
                iters: int = 500, ori_target: np.ndarray | None = None,
                ori_weight: float = 0.0) -> tuple[np.ndarray, float]:
        """Offline static IK: solve a joint config for a site target without
        stepping physics (planning only; execution interpolates joints, which
        cannot stall mid-path the way online DLS tracking can). A weak
        orientation bias keeps consecutive plans on similar tool attitudes so
        joint interpolation between them does not whip the cage."""
        saved = self.qpos().copy()
        saved_ref = self.q_ref.copy()
        self.set_qpos(np.asarray(q_start, dtype=float))
        self.q_ref = np.asarray(q_start, dtype=float).copy()
        mujoco.mj_forward(self.model, self.data)
        qpos = self.qpos().copy()
        for _ in range(iters):
            dq_target = self.servo_step(np.asarray(target, dtype=float), gain=0.5,
                                        ori_target=ori_target, ori_weight=ori_weight)
            qpos = dq_target
            self.set_qpos(qpos)
            self.q_ref = 0.8 * self.q_ref + 0.2 * qpos
            mujoco.mj_forward(self.model, self.data)
            if float(np.linalg.norm(np.asarray(target) - self.site_pos())) < 0.002:
                break
        residual = float(np.linalg.norm(np.asarray(target) - self.site_pos()))
        solution = self.qpos().copy()
        self.set_qpos(saved)
        self.q_ref = saved_ref
        mujoco.mj_forward(self.model, self.data)
        return solution, residual

    def apply(self, qpos: np.ndarray, jaw: float) -> None:
        for act, value in zip(self.act_ids[:5], qpos):
            self.data.ctrl[act] = float(value)
        self.data.ctrl[self.act_ids[5]] = float(jaw)
