"""Bounded CPU-only OpenVINO FP32 export rehearsal for the ACT chunk predictor.

20k re-run command (mechanical, CPU only, run from repo root):
  CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/export_openvino.py \
    --checkpoint out/checkpoints/act_full/checkpoints/020000/pretrained_model \
    --tag act_full_20k

Smoke rehearsal command (this task):
  CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/export_openvino.py \
    --checkpoint out/checkpoints/act_smoke/checkpoints/000300/pretrained_model \
    --tag smoke

Artifact spec (docs/INTEL_STRATEGY.md): single FP32 CPU LATENCY artifact,
PERFORMANCE_HINT=LATENCY, INFERENCE_PRECISION_HINT=f32, PyTorch tol 1e-3.
No fp16 compression. Outputs: out/export/<tag>/model.xml + model.bin,
parity.json {max_err, pass, ir_size_mb}, export.log {times per stage},
calibration.npz (fixed seeded batch, saved for determinism).

Notes:
  - CPU only: checkpoint loads with device="cpu"; never touches CUDA.
  - No dataset or physics access: calibration batch is synthetic, seeded.
  - Re-run determinism: same command twice uses the same seed and eval
    mode, so the parity verdict is stable.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

CALIB_SEED = 0
CALIB_BATCH = 1
CALIB_IMAGE_SHAPE = (3, 256, 256)
CALIB_STATE_DIM = 12
PARITY_TOL = 1e-3

CAM_ORDER = ("overhead", "table_left", "table_right", "wrist_cam")


def resolve_pretrained(checkpoint: Path) -> Path:
    if (checkpoint / "config.json").is_file():
        return checkpoint
    if (checkpoint / "pretrained_model" / "config.json").is_file():
        return checkpoint / "pretrained_model"
    ckpts = checkpoint / "checkpoints"
    if ckpts.is_dir():
        cands = sorted([d for d in ckpts.iterdir() if d.is_dir() and d.name[0].isdigit()])
        if cands:
            latest = cands[-1] / "pretrained_model"
            if (latest / "config.json").is_file():
                return latest
        last = ckpts / "last" / "pretrained_model"
        if (last / "config.json").is_file():
            return last
    raise FileNotFoundError(f"no pretrained_model found under {checkpoint}")


class ACTChunkWrapper(torch.nn.Module):
    """Net-only adapter: 4 images (B,3,256,256) + state (B,12) -> chunk (B,S,12)."""

    def __init__(self, act_module: torch.nn.Module):
        super().__init__()
        self.net = act_module

    def forward(self, overhead, table_left, table_right, wrist_cam, state):
        from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

        batch = {
            OBS_IMAGES: [overhead, table_left, table_right, wrist_cam],
            OBS_STATE: state,
        }
        actions, _ = self.net(batch)
        return actions


def build_calibration_batch():
    g = torch.Generator().manual_seed(CALIB_SEED)
    images = [
        torch.randn((CALIB_BATCH, *CALIB_IMAGE_SHAPE), generator=g, dtype=torch.float32)
        for _ in CAM_ORDER
    ]
    state = torch.randn((CALIB_BATCH, CALIB_STATE_DIM), generator=g, dtype=torch.float32)
    return images, state


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    import openvino as ov

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.configuration_act import ACTConfig  # noqa: F401 (registers policy choice)
    from lerobot.policies.act.modeling_act import ACTPolicy

    ckpt_in = Path(args.checkpoint)
    pre_dir = resolve_pretrained(ckpt_in)
    out_dir = ROOT / "out" / "export" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    xml_path = out_dir / "model.xml"
    bin_path = out_dir / "model.bin"

    stages: dict[str, float] = {}
    t0 = time.perf_counter()
    cfg = PreTrainedConfig.from_pretrained(str(pre_dir))
    cfg.device = "cpu"
    policy = ACTPolicy.from_pretrained(str(pre_dir), config=cfg)
    policy.eval()
    policy.model.eval()
    stages["load_s"] = time.perf_counter() - t0

    wrapper = ACTChunkWrapper(policy.model)
    wrapper.eval()

    images, state = build_calibration_batch()
    example_input = (*images, state)
    np.savez(
        out_dir / "calibration.npz",
        **{f"image_{c}": images[i].numpy() for i, c in enumerate(CAM_ORDER)},
        state=state.numpy(),
        seed=np.array(CALIB_SEED),
    )

    t0 = time.perf_counter()
    with torch.no_grad():
        ov_model = ov.convert_model(wrapper, example_input=example_input)
    stages["convert_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ov.save_model(ov_model, str(xml_path))
    stages["save_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    core = ov.Core()
    compiled = core.compile_model(
        ov_model,
        "CPU",
        {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f32"},
    )
    stages["compile_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    with torch.no_grad():
        torch_out = wrapper(*images, state).detach().cpu().numpy()
    ov_out = compiled([a.numpy() for a in (*images, state)])[0]
    ov_out = np.asarray(ov_out)
    stages["parity_infer_s"] = time.perf_counter() - t0

    max_err = float(np.max(np.abs(torch_out - ov_out)))
    passed = bool(max_err <= PARITY_TOL)
    ir_size_mb = float(
        (xml_path.stat().st_size + bin_path.stat().st_size) / (1024 * 1024)
    )

    parity = {
        "max_err": max_err,
        "pass": passed,
        "ir_size_mb": ir_size_mb,
        "tol": PARITY_TOL,
        "tag": args.tag,
        "checkpoint": str(pre_dir),
        "calib_seed": CALIB_SEED,
        "torch_shape": list(torch_out.shape),
        "ov_shape": list(ov_out.shape),
    }
    (out_dir / "parity.json").write_text(json.dumps(parity, indent=2))

    total = sum(stages.values())
    lines = [
        f"tag={args.tag} checkpoint={pre_dir}",
        f"calib_seed={CALIB_SEED} batch={CALIB_BATCH} tol={PARITY_TOL}",
        *[f"{k}={v:.3f}s" for k, v in stages.items()],
        f"total_tracked_s={total:.3f}s",
        f"max_err={max_err:.6e} pass={passed} ir_size_mb={ir_size_mb:.3f}",
        f"torch_shape={list(torch_out.shape)} ov_shape={list(ov_out.shape)}",
    ]
    (out_dir / "export.log").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
