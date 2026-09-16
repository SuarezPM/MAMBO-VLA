"""Local CPU bench for the frozen FP32 IR (Carril A / Deepwork F1).

Scope: text/code only. No training, no re-eval of seeds 0-9, no PDF/PNG/MP4 reads.
Reads the existing IR at out/export/act_full_v3_100k/ (never re-exports).

Two paths, mirroring benchmark_app cells:
  - latency-sync: PERFORMANCE_HINT=LATENCY, 1 infer request, sync infer().
    Reports p50/p95 latency over niter iters after warmup.
  - throughput-async: PERFORMANCE_HINT=THROUGHPUT, AsyncInferQueue jobs=4,
    reports aggregate + per-stream FPS over niter iters after warmup.

Inputs: static batch-1 shapes pinned at bench time (export keeps them
dynamic, same as benchmark_app -shape):
  overhead/table_left/table_right/wrist_cam [1,3,256,256] f32, state [1,12] f32.
Values: deterministic seeded synthetic (rng seed 0) unless --calib points at
the export calibration.npz (same seed 0 batch used for parity). Default uses
--calib (honest: same batch the parity check used).

Output JSON (default out/bench_local/bench_local_cpu.json):
  {cpu_model, device:'CPU', precision:'FP32', p50_ms, p95_ms,
   fps_agg, fps_per_stream, niter, warmup, timestamp}
Plus nested detail (latency_path / throughput_path) for audit. Required keys
are always present; detail is extra, never a substitute.

Host labelling: cpu_model is read from /proc/cpuinfo on THIS host.
Never claim Xeon here: Xeon E-2386G numbers live in frozen logs under
out/bench_intel_incoming/ (other host, cited separately by results_table.py).

Usage:
  .venv/bin/python scripts/bench_openvino_local.py
  .venv/bin/python scripts/bench_openvino_local.py --niter 100 --warmup 10
"""

import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = ROOT / "out" / "export" / "act_full_v3_100k" / "model.xml"
DEFAULT_CALIB = ROOT / "out" / "export" / "act_full_v3_100k" / "calibration.npz"
DEFAULT_OUT = ROOT / "out" / "bench_local" / "bench_local_cpu.json"

CAM_ORDER = ("overhead", "table_left", "table_right", "wrist_cam")
IMG_SHAPE = (1, 3, 256, 256)
STATE_SHAPE = (1, 12)
NSTREAMS = 4  # matches benchmark_app async cell (4 infer requests)


def read_cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.strip().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown-cpu (/proc/cpuinfo unreadable)"


def load_inputs(calib_path: Path | None, seed: int = 0) -> dict:
    if calib_path is not None and calib_path.is_file():
        z = np.load(str(calib_path))
        out = {}
        for c in CAM_ORDER:
            key = f"image_{c}"
            arr = np.asarray(z[key], dtype=np.float32)
            out[c] = arr
        out["state"] = np.asarray(z["state"], dtype=np.float32)
        return out
    rng = np.random.default_rng(seed)
    out = {c: rng.standard_normal(IMG_SHAPE).astype(np.float32) for c in CAM_ORDER}
    out["state"] = rng.standard_normal(STATE_SHAPE).astype(np.float32)
    return out


def percentile_ms(samples: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(samples, dtype=np.float64), q))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_XML))
    ap.add_argument("--calib", default=str(DEFAULT_CALIB))
    ap.add_argument("--niter", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    import openvino as ov

    xml_path = Path(args.model)
    if not xml_path.is_file():
        print(f"missing IR: {xml_path}")
        return 1
    niter: int = args.niter
    warmup: int = args.warmup
    calib = Path(args.calib) if args.calib else None
    inputs = load_inputs(calib if calib and calib.is_file() else None)
    if calib is not None and not calib.is_file():
        print(f"note: calib not found at {calib}, using seeded synthetic (seed 0)")

    static_shapes: dict[str, list[int]] = {c: list(IMG_SHAPE) for c in CAM_ORDER}
    static_shapes["state"] = list(STATE_SHAPE)

    cpu_model = read_cpu_model()
    ov_version = ov.__version__
    print(f"host cpu_model: {cpu_model}")
    print(f"openvino: {ov_version}")
    print(f"model: {xml_path}")
    print(f"niter={niter} warmup={warmup} precision=FP32 device=CPU")
    print(f"shapes: 4x[1,3,256,256] + [1,12]; async streams={NSTREAMS}")

    core = ov.Core()

    # ---- latency-sync path ----
    model_lat = core.read_model(str(xml_path))
    try:
        model_lat.reshape({k: v for k, v in static_shapes.items()})
    except Exception as exc:  # noqa: BLE001 - report and continue with dynamic
        print(f"reshape LATENCY skipped ({exc})")
    compiled_lat = core.compile_model(model_lat, "CPU", {"PERFORMANCE_HINT": "LATENCY"})
    req = compiled_lat.create_infer_request()
    for _ in range(warmup):
        req.infer(inputs)
    lat_ms: list[float] = []
    for _ in range(niter):
        t0 = time.perf_counter()
        req.infer(inputs)
        lat_ms.append((time.perf_counter() - t0) * 1000.0)
    p50 = percentile_ms(lat_ms, 50)
    p95 = percentile_ms(lat_ms, 95)
    avg = float(np.mean(lat_ms))
    print(f"latency-sync: p50={p50:.2f}ms p95={p95:.2f}ms avg={avg:.2f}ms "
          f"min={min(lat_ms):.2f}ms max={max(lat_ms):.2f}ms")

    # ---- throughput-async path ----
    model_tput = core.read_model(str(xml_path))
    try:
        model_tput.reshape({k: v for k, v in static_shapes.items()})
    except Exception as exc:  # noqa: BLE001
        print(f"reshape THROUGHPUT skipped ({exc})")
    compiled_tput = core.compile_model(
        model_tput, "CPU", {"PERFORMANCE_HINT": "THROUGHPUT"}
    )
    queue = ov.AsyncInferQueue(compiled_tput, NSTREAMS)
    for _ in range(warmup):
        queue.start_async(inputs, userdata=None)
        queue.wait_all()
    t0 = time.perf_counter()
    for _ in range(niter):
        queue.start_async(inputs, userdata=None)
    queue.wait_all()
    total_s = time.perf_counter() - t0
    fps_agg = float(niter / total_s) if total_s > 0 else 0.0
    fps_per_stream = float(fps_agg / NSTREAMS)
    async_median_est = float((total_s * 1000.0 / niter) * NSTREAMS)
    print(f"throughput-async: total={total_s*1000.0:.1f}ms for {niter} iters "
          f"fps_agg={fps_agg:.2f} fps_per_stream={fps_per_stream:.2f} "
          f"(est median per-request ~{async_median_est:.2f}ms)")

    payload = {
        "cpu_model": cpu_model,
        "device": "CPU",
        "precision": "FP32",
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "fps_agg": round(fps_agg, 2),
        "fps_per_stream": round(fps_per_stream, 2),
        "niter": niter,
        "warmup": warmup,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "detail": {
            "model": str(xml_path.relative_to(ROOT)) if xml_path.is_relative_to(ROOT) else str(xml_path),
            "openvino_version": ov_version,
            "input_source": "calibration.npz" if (calib and calib.is_file()) else "seeded-synthetic-seed0",
            "latency_path": {
                "hint": "LATENCY", "api": "sync", "jobs": 1,
                "avg_ms": round(avg, 2),
                "min_ms": round(float(min(lat_ms)), 2),
                "max_ms": round(float(max(lat_ms)), 2),
            },
            "throughput_path": {
                "hint": "THROUGHPUT", "api": "async", "jobs": NSTREAMS,
                "total_ms": round(total_s * 1000.0, 1),
            },
            "note": "This host only. Xeon E-2386G numbers are frozen in "
                    "out/bench_intel_incoming/ and never merged here.",
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
