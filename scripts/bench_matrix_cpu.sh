#!/usr/bin/env bash
# CPU benchmark_app matrix: FP32 vs FP16 x LATENCY vs THROUGHPUT x sync vs async.
#
# WHERE TO RUN: the contracted Intel validation box (Core Ultra, G5-gated per
# docs/INTEL_STRATEGY.md Sec 4) — NOT the local AMD+NVIDIA training box. No
# OpenVINO number leaves non-Intel hardware (INTEL_STRATEGY Sec 3). This script
# was prepared text-only on the local box and has NOT been executed there.
#
# PREREQUISITES ON THE INTEL BOX:
#   - Same pinned stack (requirements.lock): openvino==2026.3.0 provides
#     `benchmark_app` on PATH (or .venv/bin/benchmark_app).
#   - IRs present: out/export/act_full_20k/model.xml (FP32 rehearsal) and
#     out/export/act_full_20k_fp16/model.xml (FP16 rehearsal).
#   - VPS acceptance evidence captured BEFORE paying (G5): /dev/accel (NPU),
#     /dev/dri (iGPU), device list showing CPU/GPU/NPU.
#
# WHAT IT RUNS (CPU device only; GPU/NPU/AUTO rows belong to the full device
# matrix in SUBMISSION_TEXT.md Sec 5 and are out of scope for this script):
#   for MODEL in fp32 fp16; for HINT in LATENCY THROUGHPUT; for API in sync async:
#     benchmark_app -m <xml> -d CPU -niter 100 -api <api> -hint <hint>
# -niter 100 per INTEL_STRATEGY Sec 2. Forward-pass latency only — never
# amortized replay latency across chunk sizes.
#
# LOGS TO ATTACH (every file under out/bench/cpu_matrix_<ts>/):
#   00_env.txt ......... uname, lscpu, openvino version, available devices,
#                        /dev/dri + /dev/accel presence (paste ALL of it,
#                        including absent-device lines).
#   bench_<model>_<hint>_<api>.log ... one per matrix cell (8 total), full
#                        benchmark_app stdout including the latency/throughput
#                        summary lines. A failing cell is data: keep the log,
#                        record the error verbatim, do not delete.
#   intel_gpu_top.txt .. ONLY if /dev/dri exists (iGPU present). If there is
#                        no iGPU, do NOT install/fake it: CPU-only logs are the
#                        honest result and the submission falls back to the
#                        G5 disclosure ("NPU unavailable on the demo host;
#                        numbers are CPU/GPU-only").
# After the run, paste the 8 latency/throughput rows into SUBMISSION_TEXT.md
# Sec 5 (CPU row) and flip the corresponding EVIDENCE_INDEX.md Sec 5 cells
# from PENDING-VPS to PRESENT with the log paths.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$ROOT/out/bench/cpu_matrix_$TS"
BENCH="${BENCHMARK_APP:-benchmark_app}"

FP32_XML="$ROOT/out/export/act_full_20k/model.xml"
FP16_XML="$ROOT/out/export/act_full_20k_fp16/model.xml"

echo "== preflight =="
command -v "$BENCH" >/dev/null 2>&1 || { echo "missing: $BENCH (install openvino==2026.3.0)"; exit 1; }
test -f "$FP32_XML" || { echo "missing FP32 IR: $FP32_XML"; exit 1; }
test -f "$FP16_XML" || { echo "missing FP16 IR: $FP16_XML"; exit 1; }
mkdir -p "$OUT"

echo "== env capture -> $OUT/00_env.txt =="
{
  echo "date_utc=$TS"
  uname -a
  echo "--- lscpu ---"
  lscpu
  echo "--- openvino version ---"
  "$ROOT/.venv/bin/python" -c "import openvino as ov; print(ov.__version__)" 2>/dev/null || python3 -c "import openvino as ov; print(ov.__version__)"
  echo "--- available devices ---"
  "$ROOT/.venv/bin/python" -c "import openvino as ov; print(ov.Core().available_devices)" 2>/dev/null || python3 -c "import openvino as ov; print(ov.Core().available_devices)"
  echo "--- /dev/dri (iGPU?) ---"
  ls -la /dev/dri 2>&1 || true
  echo "--- /dev/accel (NPU?) ---"
  ls -la /dev/accel 2>&1 || true
  echo "--- benchmark_app version ---"
  "$BENCH" --version 2>&1 || true
} | tee "$OUT/00_env.txt"

echo "== matrix: FP32 vs FP16 x LATENCY vs THROUGHPUT x sync vs async (-d CPU -niter 100) =="
for MODEL in "fp32:$FP32_XML" "fp16:$FP16_XML"; do
  NAME="${MODEL%%:*}"; XML="${MODEL##*:}"
  for HINT in LATENCY THROUGHPUT; do
    for API in sync async; do
      LOG="$OUT/bench_${NAME}_${HINT}_${API}.log"
      echo "-- $NAME $HINT $API -> $LOG"
      # benchmark_app>=2026 requires lowercase perf hints; filenames keep UPPER labels.
      # IR exports keep batch+spatial dynamic -> static -shape (batch 1, runtime geometry).
      hint="$(printf '%s' "$HINT" | tr '[:upper:]' '[:lower:]')"
      SHAPE="overhead[1,3,256,256],table_left[1,3,256,256],table_right[1,3,256,256],wrist_cam[1,3,256,256],state[1,12]"
      "$BENCH" -m "$XML" -d CPU -niter 100 -api "$API" -hint "$hint" -shape "$SHAPE" 2>&1 | tee "$LOG"
    done
  done
done

echo "== done: attach the whole directory =="
ls -la "$OUT"
echo "If /dev/dri exists, also capture: intel_gpu_top -l > $OUT/intel_gpu_top.txt"
