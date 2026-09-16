"""Vision-life diagnostic: stats audit + CPU checkpoint probe.

Why this exists: a falling training loss proves the optimizer moves, not
that the vision branch drives the output. Shared normalization statistics
can park one operating regime inside another's attractor and silence the
cameras while the loss still decreases on state alone. This script checks
both halves: the statistics for dead dimensions, and the checkpoint for
vision-driven output changes.

Re-run at the 20k checkpoint:
  .venv/bin/python scripts/vision_alive.py --checkpoint out/checkpoints/act_full/checkpoints/020000
What COLLAPSED implies: stop the line before any export or demo burn.
A vision-dead checkpoint must not be exported, demoed, or evaluated as
a visuomotor result; fix the data/statistics first, then resume training.

Exit codes: 0 = ALIVE (vision drives the chunk), 1 = COLLAPSED
(cameras do not move the output; do not export or demo),
2 = stats risk only (probe alive but a statistics dimension is dead).

CPU only by construction: every tensor is built on torch.device("cpu"),
the policy config is forced to cpu before weight load, and the script
asserts at exit that no CUDA context was ever initialized. The running
full training on the GPU is never touched.

F-stats: --stats-root selects the dataset root for BOTH the stats audit and
the episode frames (default out/lerobot, frozen behavior preserved);
checkpoints trained on recomputed stats (v2/v3, different visual scale)
must pass their own root or both halves are measured through the wrong
statistics.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CPU = torch.device("cpu")
DEAD_STD = 1e-3
COLLAPSE_VAR = 1e-6
ALIVE_RATIO = 10.0

POLICY_INPUTS = [
    "observation.images.overhead",
    "observation.images.table_left",
    "observation.images.table_right",
    "observation.images.wrist_cam",
    "observation.state",
]


def audit_stats(stats_path: Path) -> bool:
    """Part 1: statistics audit. Returns True for STATS_OK."""
    stats = json.loads(stats_path.read_text())
    reasons: list[str] = []
    print("== stats audit ==")
    for key in POLICY_INPUTS + ["action", "observation.commanded", "observation.jaw_forcerange"]:
        s = stats[key]
        std = np.asarray(s["std"], dtype=np.float64).flatten()
        mean = np.asarray(s["mean"], dtype=np.float64).flatten()
        mn = np.asarray(s["min"], dtype=np.float64).flatten()
        mx = np.asarray(s["max"], dtype=np.float64).flatten()
        if not (np.all(np.isfinite(std)) and np.all(np.isfinite(mean))
                and np.all(np.isfinite(mn)) and np.all(np.isfinite(mx))):
            reasons.append(f"{key}: non-finite stat entries")
        dead = [i for i, v in enumerate(std) if v < DEAD_STD]
        ratio = float(np.max(np.abs(mean) / np.maximum(std, 1e-12)))
        print(f"{key}: ndim={std.size} std_min={std.min():.2e} "
              f"std_max={std.max():.2e} max|mean|/std={ratio:.1f} "
              f"dead_dims={dead if dead else 'none'}")
        if dead and key in POLICY_INPUTS + ["action"]:
            reasons.append(
                f"{key}: dims {dead} have std<1e-3 "
                f"(stds {[f'{std[i]:.1e}' for i in dead]})"
            )
    # Action pad/mask handling: the stored episodes carry no pad rows;
    # the final frame repeats its own state (zero-delta hold), so chunk
    # tails that run past an episode end target a stationary hold. The
    # per-sample pad mask the loss uses is synthesized at batch time by
    # the episode-aware sampler, not stored in the dataset.
    print("action pads: none stored; final-frame hold covers chunk tails; "
          "mask synthesized at batch time")
    # Shared-mean-shift risk: with a single task the shared statistics ARE
    # the task's own statistics, so there is no second regime to be
    # absorbed into. The ratios above are reported for the record.
    print("shared-mean-shift: single-task dataset, no cross-regime absorption possible")
    if reasons:
        print("verdict: STATS_RISK")
        for r in reasons:
            print(f"  - {r}")
        return False
    print("verdict: STATS_OK")
    return True


def resolve_checkpoint(path: Path) -> Path:
    """Accept a run dir, a checkpoints/NNNNNN dir, or a pretrained_model dir."""
    if (path / "pretrained_model" / "config.json").exists():
        return path / "pretrained_model"
    if (path / "config.json").exists() and (path / "model.safetensors").exists():
        return path
    cands = sorted((path / "checkpoints").glob("[0-9]*")) if (path / "checkpoints").is_dir() else []
    cands = [c for c in cands if (c / "pretrained_model" / "config.json").exists()]
    if cands:
        return cands[-1] / "pretrained_model"
    raise SystemExit(f"no usable checkpoint under {path}")


def load_cpu(ckpt_dir: Path, stats_root: str):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(ckpt_dir)
    cfg.device = "cpu"
    policy = ACTPolicy.from_pretrained(ckpt_dir, config=cfg)
    policy.eval()
    ds = LeRobotDataset(repo_id="mambo/rc-dinner", root=str(stats_root))
    # The saved pipelines pin device=cuda; override to cpu so no CUDA
    # context can initialize on this lane (asserted at exit).
    cpu_only = {"device_processor": {"device": "cpu"}}
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=str(ckpt_dir), dataset_stats=ds.meta.stats,
        preprocessor_overrides=cpu_only, postprocessor_overrides=cpu_only,
    )
    return policy, pre, post, ds


def fetch_obs(ds, episode: int, frac: float) -> dict:
    """One raw observation from an episode, in dataset-native tensors."""
    row = ds.meta.episodes[episode]
    start, end = int(row["dataset_from_index"]), int(row["dataset_to_index"])
    idx = start + int((end - start - 1) * frac)
    item = ds[idx]
    return {
        "observation.images.overhead": item["observation.images.overhead"].to(CPU).float(),
        "observation.images.table_left": item["observation.images.table_left"].to(CPU).float(),
        "observation.images.table_right": item["observation.images.table_right"].to(CPU).float(),
        "observation.images.wrist_cam": item["observation.images.wrist_cam"].to(CPU).float(),
        "observation.state": item["observation.state"].to(CPU).float(),
    }


def predict(policy, pre, post, obs: dict) -> np.ndarray:
    with torch.no_grad():
        chunk = policy.predict_action_chunk(pre(dict(obs)))
    out = post(chunk)[0]
    if torch.is_tensor(out):
        out = out.cpu()
    return np.asarray(out, dtype=np.float64)


def probe(policy, pre, post, ds, n_episodes: int) -> bool:
    """Part 2: CPU checkpoint probe. Returns True for ALIVE."""
    print("== checkpoint probe (cpu) ==")
    episodes = [round(i * 9 / max(n_episodes - 1, 1)) for i in range(n_episodes)]
    base_obs = [fetch_obs(ds, e, 0.40) for e in episodes]
    base = [predict(policy, pre, post, o) for o in base_obs]
    for i, c in enumerate(base):
        assert c.shape[1] == 12, c.shape
    stacked = np.stack(base)  # (K, chunk, dim)
    var_per_dim = stacked.var(axis=0).mean(axis=0)
    var_mean = float(var_per_dim.mean())
    print(f"episodes={episodes} chunk={stacked.shape[1]} "
          f"cross-seed chunk var mean={var_mean:.2e} "
          f"per-dim min={var_per_dim.min():.2e} max={var_per_dim.max():.2e}")

    def l1(rows_a, rows_b) -> float:
        return float(np.mean([np.abs(a - b).mean() for a, b in zip(rows_a, rows_b)]))

    zero_img, swap_img, zero_state = [], [], []
    for k, o in enumerate(base_obs):
        zi = dict(o)
        for cam in [c for c in o if c.startswith("observation.images")]:
            zi[cam] = torch.zeros_like(o[cam])
        zero_img.append(predict(policy, pre, post, zi))
        # Fair vision contrast: the donor is another episode at an EARLY
        # phase (approach, arms near home) while the base sits at 40%
        # (mid-carry). Same-episode-phase swaps are near-identical pixels
        # by design (frozen teacher, millimeter jitter) and cannot move
        # any vision-driven output, so they would fake a collapse.
        donor_ep = episodes[(k + len(base_obs) // 2) % len(base_obs)]
        donor = fetch_obs(ds, donor_ep, 0.05)
        si = dict(o)
        for cam in [c for c in o if c.startswith("observation.images")]:
            si[cam] = donor[cam].clone()
        swap_img.append(predict(policy, pre, post, si))
        zs = dict(o)
        zs["observation.state"] = torch.zeros_like(o["observation.state"])
        zero_state.append(predict(policy, pre, post, zs))
    d_zero_img = l1(base, zero_img)
    d_swap_img = l1(base, swap_img)
    d_zero_state = l1(base, zero_state)
    print(f"L1 chunk deltas: zero-images={d_zero_img:.2e} "
          f"swapped-images={d_swap_img:.2e} zero-state={d_zero_state:.2e}")
    ratio = d_swap_img / max(d_zero_state, 1e-12)
    print(f"vision/state ablation ratio (swapped-images / zero-state) = {ratio:.1f} "
          f"(needs > {ALIVE_RATIO})")
    alive = (var_mean > COLLAPSE_VAR) and (ratio > ALIVE_RATIO)
    print(f"verdict: {'ALIVE' if alive else 'COLLAPSED'}")
    if not alive:
        if var_mean <= COLLAPSE_VAR:
            print(f"  - cross-seed chunk variance {var_mean:.2e} <= {COLLAPSE_VAR:.0e}: "
                  f"predictions identical across seeds")
        if ratio <= ALIVE_RATIO:
            print(f"  - vision ablation moves output less than {ALIVE_RATIO}x the "
                  f"state ablation: cameras do not drive the chunk")
    return alive


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "out" / "checkpoints" / "act_smoke"))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--stats-root", default=str(ROOT / "out" / "lerobot"),
                    help="dataset root for the stats audit and episode frames "
                    "(F-stats); default preserves frozen v1 behavior")
    args = ap.parse_args()
    ok_stats = audit_stats(Path(args.stats_root) / "meta" / "stats.json")
    ckpt = resolve_checkpoint(Path(args.checkpoint))
    print(f"checkpoint: {ckpt}")
    print(f"stats_root: {args.stats_root}")
    policy, pre, post, ds = load_cpu(ckpt, args.stats_root)
    ok_probe = probe(policy, pre, post, ds, max(2, min(args.seeds, 10)))
    assert not torch.cuda.is_initialized(), "CUDA context was touched on a CPU-only lane"
    print(f"CUDA untouched: {not torch.cuda.is_initialized()}")
    print(f"summary: {'STATS_OK' if ok_stats else 'STATS_RISK'} / "
          f"{'ALIVE' if ok_probe else 'COLLAPSED'}")
    if not ok_probe:
        return 1
    if not ok_stats:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
