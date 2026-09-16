"""M0 scene loader: resolve the MJCF, apply frozen seed visuals, query helpers.

Only the seed bundle may change visuals (object colors, light level);
physics (option block, pads, bodies, table, shelf) is never edited here.
set_free_body is spawn/reset placement ONLY (never for carry).
"""

from pathlib import Path

import mujoco
import numpy as np

SCENE_PATH = Path(__file__).with_name("dual_so101_dinner.xml")

TABLE_Z = 0.76
CAMERAS = ("overhead", "table_left", "table_right", "wrist_cam")
RELAY_POINT = (0.0, -0.10, TABLE_Z)
DEST_POINT = (0.18, -0.01, TABLE_Z)

# Frozen contact values mirrored from ARCH Section 3.
ARCH_PAD_SIZE = (0.00125, 0.00125, 0.00125)
ARCH_PAD_FRICTION = (1.0, 0.05, 0.001)
ARCH_PAD_CONDIM = 4
ARCH_STATIC_POS = (-0.008875, 0.0, -0.100)
# Moving-pad local offset lives in the hinge-jaw frame (same numbers as the
# ARCH block); at jaw angle 0 its tool-frame pose must equal the composed
# fingertip pose below (SO-101 jaw geometry).
ARCH_MOVING_POS = (-0.01136, -0.076, 0.019)
ARCH_MOVING_COMPOSED = (0.00884, -0.0002, -0.0994)
ARCH_TIMESTEP = 0.002
# Pinned scene constants (NOT part of the ARCH contact block): object
# contact stiffness, asserted frozen, never tuned per seed.
ARCH_SOLREF = (0.012, 1.0)

MUG_HALFHEIGHT = 0.032
PLATE_HALFHEIGHT = 0.012
SPAWN_CLEARANCE = 0.002

EXCLUDE_PAIRS = (
    ("base", "pan"),
    ("pan", "upper"),
    ("upper", "fore"),
    ("fore", "wrist"),
    ("wrist", "tool"),
    ("tool", "jaw"),
)


def load_model(
    light_multiplier: float = 1.0,
    mug_rgba: tuple | None = None,
    plate_rgba: tuple | None = None,
) -> mujoco.MjModel:
    """Load the frozen scene; only light level and object colors vary."""
    if not SCENE_PATH.exists():
        raise FileNotFoundError(f"Scene MJCF missing: {SCENE_PATH}")
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "key")
    if key_id >= 0:
        base = np.array([0.75, 0.77, 0.80], dtype=float) * float(light_multiplier)
        model.light_diffuse[key_id] = base
    if mug_rgba is not None:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "mug_geom")
        model.geom_rgba[geom_id] = np.asarray(mug_rgba, dtype=float)
    if plate_rgba is not None:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "plate_geom")
        model.geom_rgba[geom_id] = np.asarray(plate_rgba, dtype=float)
    return model


def make_data(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return data


def set_free_body(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    xyz: tuple | list,
    yaw: float = 0.0,
) -> None:
    """Spawn/reset placement for a freejoint body. NEVER for carry."""
    joint_name = f"{body_name}_free"
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jnt_id < 0:
        raise ValueError(f"No freejoint for body {body_name}")
    adr = model.jnt_qposadr[jnt_id]
    data.qpos[adr : adr + 3] = np.asarray(xyz, dtype=float)
    half = float(yaw) / 2.0
    data.qpos[adr + 3 : adr + 7] = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
    mujoco.mj_forward(model, data)


def body_pos(data: mujoco.MjData, body_name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return np.array(data.xpos[body_id], dtype=float)


def assert_frozen_contact_block(model: mujoco.MjModel) -> None:
    """Fail loudly if the scene contact block drifts from ARCH Section 3."""
    assert abs(model.opt.timestep - ARCH_TIMESTEP) < 1e-12, "timestep drifted"
    assert model.opt.impratio == 10, "impratio drifted"
    assert model.opt.noslip_iterations == 3, "noslip_iterations drifted"
    assert model.opt.cone == 1, "cone must be elliptic"
    for side in ("left", "right"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_static_finger_pad")
        assert gid >= 0, f"missing pad {side}_static_finger_pad"
        assert tuple(model.geom_size[gid]) == ARCH_PAD_SIZE, "static pad size drift"
        assert tuple(model.geom_friction[gid]) == ARCH_PAD_FRICTION, "static pad friction drift"
        assert int(model.geom_condim[gid]) == ARCH_PAD_CONDIM, "static pad condim drift"
        assert np.allclose(model.geom_pos[gid], ARCH_STATIC_POS, atol=1e-9), "static pad pos drift"
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_moving_finger_pad")
        assert gid >= 0, f"missing pad {side}_moving_finger_pad"
        assert tuple(model.geom_size[gid]) == ARCH_PAD_SIZE, "moving pad size drift"
        assert tuple(model.geom_friction[gid]) == ARCH_PAD_FRICTION, "moving pad friction drift"
        assert int(model.geom_condim[gid]) == ARCH_PAD_CONDIM, "moving pad condim drift"
        assert np.allclose(model.geom_pos[gid], ARCH_MOVING_POS, atol=1e-9), "moving pad jaw-local drift"
    # Pinned object solref (frozen scene constants, not the contact block).
    for geom in ("mug_geom", "plate_geom"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
        assert tuple(model.geom_solref[gid]) == ARCH_SOLREF, f"{geom} solref drift"
    # Adjacent-body exclusion surface (ARCH Sec 3 para 4 pattern).
    text = SCENE_PATH.read_text()
    for side in ("left", "right"):
        for proximal, distal in EXCLUDE_PAIRS:
            tag = f'<exclude body1="{side}_{proximal}" body2="{side}_{distal}"/>'
            assert tag in text, f"missing exclusion {tag}"


def assert_moving_pad_composed(model: mujoco.MjModel) -> None:
    """At jaw angle 0 the moving pad must sit at the ARCH fingertip pose."""
    data = mujoco.MjData(model)
    for side in ("left", "right"):
        jaw = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_gripper")
        data.qpos[model.jnt_qposadr[jaw]] = 0.0
    mujoco.mj_forward(model, data)
    for side in ("left", "right"):
        tool = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_tool")
        pad = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_moving_finger_pad")
        rel_world = np.array(data.geom_xpos[pad]) - np.array(data.xpos[tool])
        axis = np.array(data.xmat[tool]).reshape(3, 3)
        rel_local = axis.T @ rel_world  # tool mount is pitched; compare in tool frame
        assert np.allclose(rel_local, ARCH_MOVING_COMPOSED, atol=1e-4), (
            f"{side} moving pad composed {tuple(np.round(rel_local, 5))} != ARCH fingertip"
        )
