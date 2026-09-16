"""FP16-weight export rehearsal for the ACT chunk predictor (CPU-only).

Companion to scripts/export_openvino.py (NOT modified): same net-only adapter,
same seeded calibration batch, same conversion call, same parity-vs-torch check
(tol 1e-3). The precision knob under test is EXECUTION precision
(INFERENCE_PRECISION_HINT=f16 vs f32), because in openvino 2026.3 neither
ov.convert_model nor ovc.convert_model exposes compress_to_fp16, and the
converter already stores weights FP16-majority by default (census of the FP32
rehearsal IR: 272 f16 / 35 f32 / 199 i64 constants) — so a separate
"FP16 storage" artifact would be near-identical bytes, not a real optimization.
The FP32-vs-FP16 delta reported here is therefore real and attributable to
execution precision alone.

20k rehearsal command (mechanical, CPU only, run from repo root):
  CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/export_openvino_fp16.py \
    --checkpoint out/checkpoints/act_full/checkpoints/020000/pretrained_model \
    --tag act_full_20k_fp16

Outputs (NEW directory, never overwrites the FP32 rehearsal):
  out/export/<tag>/model.xml + model.bin (FP16 weights),
  parity_fp16.json {max_err_f16, pass_f16, max_err_f32exec, pass_f32exec,
    ir_size_mb, tol, tag, checkpoint, ...},
  export.log {times per stage}, calibration.npz (same fixed seed as FP32).

Readout contract: if the f16 parity FAILS tol 1e-3, that is reported as data
(pass_f16=false) and the script still exits 0 — a failed squeeze candidate must
never block or rewrite the FP32 primary artifact (docs/INTEL_STRATEGY.md Sec 1).

Notes:
  - CPU only: checkpoint loads with device="cpu"; asserts at exit that no CUDA
    context was ever initialized (the live GPU training run is never touched).
  - No dataset or physics access: calibration batch is synthetic, seeded.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import export_openvino as E

PARITY_TOL = E.PARITY_TOL


def infer_numpy(compiled, inputs) -> np.ndarray:
    out = compiled([a.numpy() for a in inputs])[0]
    return np.asarray(out)


def constant_census(xml_path: Path) -> dict[str, int]:
    """Count stored constant dtypes in the saved IR (text scan of model.xml).

    Documents how the converter actually stored weights under openvino 2026.3
    defaults, so the FP32-vs-FP16 comparison cannot be misread as a storage
    change when it is an execution-precision change.
    """
    import re

    text = xml_path.read_text()
    return {
        m: len(re.findall(rf'element_type="{m}"', text))
        for m in ("f16", "f32", "f64", "i32", "i64", "u8", "boolean")
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", default="act_full_20k_fp16")
    args = ap.parse_args()

    import openvino as ov

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.configuration_act import ACTConfig  # noqa: F401 (registers policy choice)
    from lerobot.policies.act.modeling_act import ACTPolicy

    ckpt_in = Path(args.checkpoint)
    pre_dir = E.resolve_pretrained(ckpt_in)
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

    wrapper = E.ACTChunkWrapper(policy.model)
    wrapper.eval()

    images, state = E.build_calibration_batch()
    example_input = (*images, state)
    np.savez(
        out_dir / "calibration.npz",
        **{f"image_{c}": images[i].numpy() for i, c in enumerate(E.CAM_ORDER)},
        state=state.numpy(),
        seed=np.array(E.CALIB_SEED),
    )

    t0 = time.perf_counter()
    with torch.no_grad():
        # Identical conversion call to the FP32 rehearsal: openvino 2026.3
        # offers no compress_to_fp16 flag (checked on ov.convert_model and
        # ovc.convert_model), and its defaults already store FP16-majority
        # weights. Any measured delta is therefore execution precision, and
        # the census below proves the storage claim instead of assuming it.
        ov_model = ov.convert_model(wrapper, example_input=example_input)
    stages["convert_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ov.save_model(ov_model, str(xml_path))
    stages["save_s"] = time.perf_counter() - t0
    census = constant_census(xml_path)

    core = ov.Core()
    t0 = time.perf_counter()
    compiled_f16 = core.compile_model(
        ov_model,
        "CPU",
        {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f16"},
    )
    stages["compile_f16_s"] = time.perf_counter() - t0

    # Secondary datum: the SAME IR executed in fp32. This separates execution
    # rounding (f16 arithmetic) from everything else, so a parity failure can
    # be attributed correctly. It should reproduce the FP32 rehearsal (~1e-6).
    t0 = time.perf_counter()
    compiled_f32 = core.compile_model(
        ov_model,
        "CPU",
        {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f32"},
    )
    stages["compile_f32exec_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    with torch.no_grad():
        torch_out = wrapper(*images, state).detach().cpu().numpy()
    ov_f16 = infer_numpy(compiled_f16, (*images, state))
    ov_f32 = infer_numpy(compiled_f32, (*images, state))
    stages["parity_infer_s"] = time.perf_counter() - t0

    max_err_f16 = float(np.max(np.abs(torch_out - ov_f16)))
    max_err_f32exec = float(np.max(np.abs(torch_out - ov_f32)))
    passed_f16 = bool(max_err_f16 <= PARITY_TOL)
    passed_f32exec = bool(max_err_f32exec <= PARITY_TOL)
    ir_size_mb = float(
        (xml_path.stat().st_size + bin_path.stat().st_size) / (1024 * 1024)
    )

    parity = {
        "max_err_f16": max_err_f16,
        "pass_f16": passed_f16,
        "max_err_f32exec": max_err_f32exec,
        "pass_f32exec": passed_f32exec,
        "ir_size_mb": ir_size_mb,
        "tol": PARITY_TOL,
        "tag": args.tag,
        "checkpoint": str(pre_dir),
        "calib_seed": E.CALIB_SEED,
        "conversion": "ov.convert_model defaults (no compress flag in 2026.3)",
        "constant_census": census,
        "exec_hint_primary": "LATENCY+f16",
        "torch_shape": list(torch_out.shape),
        "ov_shape": list(ov_f16.shape),
    }
    (out_dir / "parity_fp16.json").write_text(json.dumps(parity, indent=2))

    assert not torch.cuda.is_initialized(), "CUDA context was touched on a CPU-only lane"

    total = sum(stages.values())
    lines = [
        f"tag={args.tag} checkpoint={pre_dir}",
        f"calib_seed={E.CALIB_SEED} batch={E.CALIB_BATCH} tol={PARITY_TOL}",
        *[f"{k}={v:.3f}s" for k, v in stages.items()],
        f"total_tracked_s={total:.3f}s",
        f"max_err_f16={max_err_f16:.6e} pass_f16={passed_f16} "
        f"max_err_f32exec={max_err_f32exec:.6e} pass_f32exec={passed_f32exec} "
        f"ir_size_mb={ir_size_mb:.3f}",
        f"constant_census={census}",
        f"torch_shape={list(torch_out.shape)} ov_shape={list(ov_f16.shape)}",
        f"CUDA untouched: {not torch.cuda.is_initialized()}",
    ]
    (out_dir / "export.log").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
