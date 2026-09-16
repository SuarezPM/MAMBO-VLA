"""P4-lite: back-projection geometrica de depth (cero ML, cero detector entrenado).

Metodo: intrinsecos pinhole desde sim/dual_so101_dinner.xml (fovy, SOLO
LECTURA via sim/load.py) + depth del renderer MuJoCo + mascara GT del
simulador (segmentacion, geom mug_geom) DECLARADA STAND-IN DE DETECTOR:
mide solo la cadena geometrica (intrinsecos + back-projection), NUNCA se
presenta como detector. Sin pesos, sin red, sin entrenamiento.

Spawn: defecto del XML (mug en staging izquierdo del propio MJCF). NO usa
bundles de seeds congeladas (0-119 prohibidas); solo lectura de formatos.
No escribe ni lee ficheros de imagen: todo numerico en memoria.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sim"))
import load as scene  # noqa: E402  (solo lectura: load_model/make_data)

CAM = "overhead"
W = H = 256  # estandar del repo (renderer 256x256)
OBJ_GEOM = "mug_geom"
OBJ_BODY = "mug"
TABLE_CENTER_XY = (0.0, 0.07)  # centro geometrico de la mesa (XML, solo lectura)


def main() -> int:
    model = scene.load_model()  # default XML, sin seed bundle
    data = scene.make_data(model)
    camid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAM)
    fovy = float(model.cam_fovy[camid])  # 45.0, del XML
    f = (W / 2.0) / np.tan(np.radians(fovy / 2.0))
    cx = cy = (W - 1) / 2.0
    print(f"intrinsics: cam={CAM} fovy={fovy}deg f={f:.3f}px cx=cy={cx}")

    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, OBJ_GEOM)
    assert gid >= 0, "mug_geom ausente"
    geom_enum = int(mujoco.mjtObj.mjOBJ_GEOM)

    rd = mujoco.Renderer(model, height=H, width=W)
    rd.enable_depth_rendering()
    rd.update_scene(data, camera=CAM)
    depth = np.asarray(rd.render(), dtype=np.float64)
    rd.close()

    rs = mujoco.Renderer(model, height=H, width=W)
    rs.enable_segmentation_rendering()
    rs.update_scene(data, camera=CAM)
    seg = np.asarray(rs.render())
    rs.close()

    mask = (seg[:, :, 0] == gid) & (seg[:, :, 1] == geom_enum)
    n = int(mask.sum())
    print(f"oracle-mask: geom={OBJ_GEOM} id={gid} pixels={n} (STAND-IN de detector, GT del sim)")
    if n < 10:
        print("verdict: NEGATIVO (mascara vacia/ocluida, no corre)")
        return 2

    R = np.asarray(data.cam_xmat[camid], dtype=np.float64).reshape(3, 3)
    t = np.asarray(data.cam_xpos[camid], dtype=np.float64)
    vs, us = np.nonzero(mask)
    z = depth[mask]
    z = z[z > 0]
    ok = z.size >= 10
    if not ok:
        print("verdict: NEGATIVO (depth invalida en mascara)")
        return 2
    # Recorta us/vs a los pixeles con depth valida (mismo orden row-major).
    valid = depth[mask] > 0
    us, vs = us[valid], vs[valid]
    xc = (us - cx) * z / f
    yc = -(vs - cy) * z / f
    zc = -z
    pts = (R @ np.stack([xc, yc, zc])) + t[:, None]
    est = pts.mean(axis=1)

    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJ_BODY)
    gt = np.asarray(data.xpos[body], dtype=np.float64)
    err_xy = float(np.linalg.norm(est[:2] - gt[:2])) * 1000.0
    err_3d = float(np.linalg.norm(est - gt)) * 1000.0

    # Autochequeo: re-proyecta el GT y compara con el centroide de la mascara.
    pcam = R.T @ (gt - t)
    d = -pcam[2]
    up, vp = f * pcam[0] / d + cx, cy - f * pcam[1] / d
    reproj = float(np.hypot(up - us.mean(), vp - vs.mean()))
    print(f"gt=[{gt[0]:.4f},{gt[1]:.4f},{gt[2]:.4f}] "
          f"est=[{est[0]:.4f},{est[1]:.4f},{est[2]:.4f}]")
    print(f"err_est_xy={err_xy:.2f}mm err_est_3d={err_3d:.2f}mm reproj_selfcheck={reproj:.2f}px")

    base = np.array([TABLE_CENTER_XY[0], TABLE_CENTER_XY[1]])
    err_base = float(np.linalg.norm(gt[:2] - base)) * 1000.0
    print(f"baseline ingenuo centro-mesa xy={TABLE_CENTER_XY} err_base_xy={err_base:.2f}mm")
    if reproj <= 3.0 and err_xy <= err_base:
        print("verdict: SHIP (stride honesto: <= baseline con mascara oracle declarada)")
        return 0
    print("verdict: NEGATIVO (no supera baseline o autochequeo falla; se archiva y se para)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
