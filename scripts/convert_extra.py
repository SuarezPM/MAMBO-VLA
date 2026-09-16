"""Merge lane: original 10 + extra 20 teacher episodes -> LeRobot v3 at out/lerobot_v2.

FOR A POTENTIAL FUTURE RELAUNCH ONLY - do not start any training from this
script. This dataset exists so a future run can relaunch from 30 episodes
without touching the frozen M1 bundle (out/lerobot) or the detached full
training. This script performs CPU-only conversion and validation only.

Inputs (read-only, never written):
- out/dataset/seed_0..9.npz (original 10 episodes)
- out/dataset_extra/seed_10..29.npz (extra 20 episodes)
- out/seeds_extra/seed_hashes.json (frozen bundle hashes for seeds 10-29,
  copied verbatim, never regenerated)

Output (only writable target besides this script file):
- out/lerobot_v2/ (rebuilt from scratch on every run)

Frame/action contract (same as scripts/convert_to_lerobot.py, frozen M1 path):
- 1 frame = 25 physics steps at dt=0.002, so fps=20. Frame i covers
  simulator steps [25*i, 25*i+25); all frame streams share this grid.
- observation.state (12, float32): both arms' joint qpos plus jaw position
  as recorded ([left 5 joints, left jaw, right 5 joints, right jaw]).
  The jaw channels (5 and 11) move in every episode (asserted below):
  WIDE while traveling, NARROW through first touch, FIRM once caged.
- observation.site_left / observation.site_right (3, float32): cartesian
  tool sites per frame, so E3 can derive cartesian deltas from the same
  rows as the joint targets without a second export.
- observation.jaw_forcerange (2, float32): the commanded jaw authority cap
  [left_Nm, right_Nm] active at this frame. Teacher schedule: 3.35 while
  traveling/approaching, 1.00 once caged (logged live from the actuator
  forcerange the teacher sets; same grid as all frame streams).
- observation.commanded (12, float32): actuator ctrl targets
  ([left 5 joints, left jaw, right 5 joints, right jaw], same layout as
  observation.state) as commanded at this frame.
- E3 training contract: train on measured-next (action = next frame's
  measured state) PLUS commanded intent (observation.commanded at the
  current frame and observation.jaw_forcerange as the authority context
  the intent was issued under). Exact fields: observation.state (measured),
  action (measured-next), observation.commanded (intent),
  observation.jaw_forcerange (authority).
- action (12, float32): the NEXT frame's state (absolute joint+jaw
  targets, ACT-style). The final frame repeats its own state (zero-delta
  hold), so every frame has exactly one action and any chunk size slices
  cleanly: chunk [i, i+k) predicts states [i+1, i+k+1).
- observation.images.{overhead,table_left,table_right,wrist_cam}: 256x256
  RGB uint8, stored per frame (no video encoding).
- task: the episode instruction string, identical for every frame.
- Only episodes with success=True are written; each npz seed file maps to
  exactly one LeRobot episode, in seed order (0..29).

Delta vs scripts/convert_to_lerobot.py (M1, frozen - NOT modified):
- NONE to feature schema, dtypes, shapes, fps, or frame/action alignment.
  FEATURES below is an exact copy of the M1 FEATURES dict.
- ONLY differences: input set is 10 original + 20 extra npz files merged
  in seed order; output root is out/lerobot_v2 instead of out/lerobot;
  dataset stats are recomputed over all 30 episodes by finalize(); and
  the frozen seed_hashes.json for seeds 10-29 is copied verbatim to
  out/lerobot_v2/seed_hashes_extra.json for provenance (read-only copy,
  byte-identical, never regenerated).

No hub upload happens in this lane.
"""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path(__file__).resolve().parents[1]
ORIG_DATASET = ROOT / "out" / "dataset"
EXTRA_DATASET = ROOT / "out" / "dataset_extra"
LEROBOT_V1_ROOT = ROOT / "out" / "lerobot"
LEROBOT_V2_ROOT = ROOT / "out" / "lerobot_v2"
HASHES_SRC = ROOT / "out" / "seeds_extra" / "seed_hashes.json"
HASHES_DST = LEROBOT_V2_ROOT / "seed_hashes_extra.json"
REPO_ID = "mambo/rc-dinner"
FPS = 20

CAMERAS = ("overhead", "table_left", "table_right", "wrist_cam")

# Exact copy of the M1 FEATURES dict from scripts/convert_to_lerobot.py.
FEATURES = {
    f"observation.images.{cam}": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    }
    for cam in CAMERAS
} | {
    "observation.state": {"dtype": "float32", "shape": (12,), "names": ["state"]},
    "observation.site_left": {"dtype": "float32", "shape": (3,), "names": ["site"]},
    "observation.site_right": {"dtype": "float32", "shape": (3,), "names": ["site"]},
    "observation.jaw_forcerange": {"dtype": "float32", "shape": (2,), "names": ["jaw_authority_Nm"]},
    "observation.commanded": {"dtype": "float32", "shape": (12,), "names": ["commanded"]},
    "action": {"dtype": "float32", "shape": (12,), "names": ["action"]},
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_episode(dataset, path: Path) -> int:
    with np.load(path, allow_pickle=True) as episode:
        if not bool(episode["success"]):
            print(f"{path.name}: skipped (success=False)")
            return 0
        states = np.asarray(episode["frame_state"], dtype=np.float32)
        sites = {
            "left": np.asarray(episode["frame_site_left"], dtype=np.float32),
            "right": np.asarray(episode["frame_site_right"], dtype=np.float32),
        }
        jaw_caps = np.asarray(episode["frame_jaw_cap"], dtype=np.float32)
        commanded = np.asarray(episode["frame_ctrl"], dtype=np.float32)
        images = {
            cam: np.asarray(episode[f"frames_{cam}"]) for cam in CAMERAS
        }
        n_frames = states.shape[0]
        assert all(images[cam].shape[0] == n_frames for cam in CAMERAS), path.name
        assert jaw_caps.shape == (n_frames, 2), path.name
        assert commanded.shape == (n_frames, 12), path.name
        uniq = np.unique(jaw_caps).tolist()
        assert all(
            any(abs(float(u) - v) < 1e-4 for v in (3.35, 1.0)) for u in uniq
        ), f"{path.name}: unexpected jaw schedule values {uniq}"
        jaw_travel = float(
            np.abs(states[:, 5] - states[0, 5]).max()
            + np.abs(states[:, 11] - states[0, 11]).max()
        )
        assert jaw_travel > 0.1, f"{path.name}: jaws never moved"
        task = str(episode["instruction"])
        for i in range(n_frames):
            frame = {
                f"observation.images.{cam}": images[cam][i]
                for cam in CAMERAS
            } | {
                "observation.state": states[i],
                "observation.site_left": sites["left"][i],
                "observation.site_right": sites["right"][i],
                "observation.jaw_forcerange": jaw_caps[i],
                "observation.commanded": commanded[i],
                "action": states[i + 1] if i + 1 < n_frames else states[i],
                "task": task,
            }
            dataset.add_frame(frame)
        dataset.save_episode()
        print(f"{path.name}: wrote {n_frames} frames")
        return n_frames


def main() -> int:
    orig_files = sorted(ORIG_DATASET.glob("seed_*.npz"))
    extra_files = sorted(EXTRA_DATASET.glob("seed_10.npz")) + sorted(
        p for p in EXTRA_DATASET.glob("seed_*.npz") if p.name != "seed_10.npz"
    )
    # Re-sort to guarantee seed order 10..29.
    extra_files = sorted(extra_files, key=lambda p: int(p.stem.split("_")[1]))
    assert len(orig_files) == 10, f"expected 10 original files, got {len(orig_files)}"
    assert len(extra_files) == 20, f"expected 20 extra files, got {len(extra_files)}"
    files = orig_files + extra_files
    seeds = [int(p.stem.split("_")[1]) for p in files]
    assert seeds == list(range(30)), f"expected seeds 0..29 in order, got {seeds}"
    if LEROBOT_V2_ROOT.exists():
        shutil.rmtree(LEROBOT_V2_ROOT)
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        root=LEROBOT_V2_ROOT,
        fps=FPS,
        features=FEATURES,
        use_videos=False,
    )
    expected_frames = 0
    written = 0
    for path in files:
        n_frames = _write_episode(dataset, path)
        if n_frames:
            written += 1
            expected_frames += n_frames
    assert written == 30, f"expected 30 episodes, wrote {written}"
    dataset.finalize()
    # Provenance: verbatim read-only copy of frozen hashes (never regenerate).
    shutil.copyfile(HASHES_SRC, HASHES_DST)
    assert _sha256(HASHES_SRC) == _sha256(HASHES_DST), "hash copy mismatch"
    hashes = json.loads(HASHES_DST.read_text())
    assert sorted(int(k) for k in hashes) == list(range(10, 30)), "hash keys must be 10..29"
    print(f"hashes copied verbatim: {HASHES_SRC} -> {HASHES_DST} "
          f"sha256={_sha256(HASHES_DST)[:16]}...")
    # Validation: API reload.
    reloaded = LeRobotDataset(repo_id=REPO_ID, root=LEROBOT_V2_ROOT)
    print(f"episodes={reloaded.num_episodes} frames={reloaded.num_frames} "
          f"fps={reloaded.fps}")
    sample = reloaded[0]
    print(f"sample keys={sorted(sample)} "
          f"state={np.asarray(sample['observation.state']).shape} "
          f"action={np.asarray(sample['action']).shape}")
    assert reloaded.num_episodes == 30, reloaded.num_episodes
    assert reloaded.num_frames == expected_frames, (
        reloaded.num_frames, expected_frames)
    # Validation: schema match vs frozen v1 (user features only; timestamp /
    # frame_index / episode_index / index / task_index are auto-added).
    v1_info = json.loads((LEROBOT_V1_ROOT / "meta" / "info.json").read_text())
    v2_info = json.loads((LEROBOT_V2_ROOT / "meta" / "info.json").read_text())
    auto = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    v1_user = {k: v for k, v in v1_info["features"].items() if k not in auto}
    v2_user = {k: v for k, v in v2_info["features"].items() if k not in auto}
    assert v1_user == v2_user, "schema delta vs out/lerobot (must be none)"
    assert v2_info["fps"] == v1_info["fps"] == FPS
    assert v2_info["total_episodes"] == 30, v2_info["total_episodes"]
    assert v2_info["total_frames"] == expected_frames, v2_info["total_frames"]
    print(f"schema match vs out/lerobot: OK ({len(v2_user)} user features, "
          f"delta=none, fps={FPS})")
    # Validation: one npz -> parquet row-count cross-check (seed_10).
    import pandas as pd

    with np.load(EXTRA_DATASET / "seed_10.npz", allow_pickle=True) as z:
        npz_rows = int(np.asarray(z["frame_state"]).shape[0])
    parquet_files = sorted((LEROBOT_V2_ROOT / "data").rglob("*.parquet"))
    assert parquet_files, "no parquet files in out/lerobot_v2/data"
    parquet_rows = sum(int(pd.read_parquet(f).shape[0]) for f in parquet_files)
    assert parquet_rows == expected_frames == reloaded.num_frames, (
        parquet_rows, expected_frames)
    print(f"cross-check: npz total={expected_frames} parquet total={parquet_rows} "
          f"reloaded frames={reloaded.num_frames} (seed_10 npz rows={npz_rows})")
    print(f"lerobot_v2 dataset OK: {written} episodes, {expected_frames} frames "
          f"at {LEROBOT_V2_ROOT}")
    print("NOTE: FOR A POTENTIAL FUTURE RELAUNCH ONLY - no training started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
