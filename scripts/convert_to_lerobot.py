"""Teacher episodes (npz) -> local LeRobot v3 dataset (lerobot 0.4.4).

Frame/action contract (fixed for E3 training):
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
  exactly one LeRobot episode, in seed order.

The output root is rebuilt from scratch on every run so re-recorded seeds
can never mix with stale episodes. No hub upload happens in this lane.
"""

import shutil
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "out" / "dataset"
LEROBOT_ROOT = ROOT / "out" / "lerobot"
REPO_ID = "mambo/rc-dinner"
FPS = 20

CAMERAS = ("overhead", "table_left", "table_right", "wrist_cam")

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


def main() -> int:
    files = sorted(DATASET.glob("seed_*.npz"))
    if not files:
        print(f"no episodes in {DATASET}")
        return 1
    if LEROBOT_ROOT.exists():
        shutil.rmtree(LEROBOT_ROOT)
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        root=LEROBOT_ROOT,
        fps=FPS,
        features=FEATURES,
        use_videos=False,
    )
    written = 0
    for path in files:
        with np.load(path, allow_pickle=True) as episode:
            if not bool(episode["success"]):
                print(f"{path.name}: skipped (success=False)")
                continue
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
            written += 1
            print(f"{path.name}: wrote {n_frames} frames")
    dataset.finalize()
    reloaded = LeRobotDataset(repo_id=REPO_ID, root=LEROBOT_ROOT)
    print(f"episodes={reloaded.num_episodes} frames={reloaded.num_frames} "
          f"fps={reloaded.fps}")
    sample = reloaded[0]
    print(f"sample keys={sorted(sample)} "
          f"state={np.asarray(sample['observation.state']).shape} "
          f"action={np.asarray(sample['action']).shape}")
    assert reloaded.num_episodes == written and written > 0
    print(f"lerobot dataset OK: {written} episodes at {LEROBOT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
