"""Results table generator (read-only sources, no invention).

Reads (never writes except README markers block + stdout):
  - out/bench_local/bench_local_cpu.json (this host, AMD)
  - out/export/act_full_v3_100k/parity.json (frozen export)
  - out/bench_intel_incoming/vehicle_20260916T135349Z/bench_fp32_LATENCY_sync.log
  - out/bench_intel_incoming/vehicle_20260916T135349Z/bench_fp32_THROUGHPUT_async.log
  - out/ood/OOD_RESULTS_60_79.md
  - out/ood/OOD_RESULTS_80_99.md

Missing/unparseable source -> cell says honest-negative, never a guess.

Updates README.md ONLY between:
  <!-- results:begin --> ... <!-- results:end -->
Creates the markers block at end of file if absent. Nothing else touched.

Usage:
  .venv/bin/python scripts/results_table.py
  .venv/bin/python scripts/results_table.py --check   # no write, exit 1 if stale
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
BENCH_LOCAL = ROOT / "out" / "bench_local" / "bench_local_cpu.json"
PARITY = ROOT / "out" / "export" / "act_full_v3_100k" / "parity.json"
XEON_LAT = ROOT / "out" / "bench_intel_incoming" / "vehicle_20260916T135349Z" / "bench_fp32_LATENCY_sync.log"
XEON_TPUT = ROOT / "out" / "bench_intel_incoming" / "vehicle_20260916T135349Z" / "bench_fp32_THROUGHPUT_async.log"
OOD_60 = ROOT / "out" / "ood" / "OOD_RESULTS_60_79.md"
OOD_80 = ROOT / "out" / "ood" / "OOD_RESULTS_80_99.md"

BEGIN = "<!-- results:begin -->"
END = "<!-- results:end -->"

MED_RE = re.compile(r"Median:\s+([\d.]+)\s*ms")
TPUT_RE = re.compile(r"Throughput:\s+([\d.]+)\s*FPS")


def parse_bench_log(path: Path) -> tuple[str, str]:
    try:
        text = path.read_text()
    except OSError:
        return ("honest-negative (log no leido)", "honest-negative (log no leido)")
    med = MED_RE.search(text)
    tput = TPUT_RE.search(text)
    return (
        med.group(1) + " ms" if med else "honest-negative (sin Median en log)",
        tput.group(1) + " FPS" if tput else "honest-negative (sin Throughput en log)",
    )


def parse_ood_mean(path: Path) -> str:
    try:
        text = path.read_text()
    except OSError:
        return f"honest-negative ({path.name} no leido)"
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("| mean |") and "1/20" in s:
            cells = [c.strip() for c in s.strip("|").split("|")]
            # cells = [mean, **1/20**, —, note]
            if len(cells) >= 4:
                return f"{cells[1]} — {cells[3]}"
            return s.replace("|", "/")
    return f"honest-negative (sin fila mean 1/20 en {path.name})"


def build_table() -> str:
    # Local bench (this host)
    try:
        b = json.loads(BENCH_LOCAL.read_text())
        local_lat = f"{b['p50_ms']} / {b['p95_ms']} ms (p50/p95, niter {b['niter']}+warmup {b['warmup']})"
        local_tput = f"{b['fps_agg']} agg / {b['fps_per_stream']} per-stream FPS"
        local_host = f"{b.get('cpu_model', 'unknown')} (este host, {b.get('device')}/{b.get('precision')})"
        local_cite = "`out/bench_local/bench_local_cpu.json` + `out/bench_local/bench_local_run.log`"
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        local_host = "este host (bench_local no leido)"
        local_lat = f"honest-negative ({exc})"
        local_tput = f"honest-negative ({exc})"
        local_cite = "`out/bench_local/bench_local_cpu.json` (ausente o ilegible)"

    # Frozen export parity
    try:
        p = json.loads(PARITY.read_text())
        parity_val = f"{p['ir_size_mb']:.3f} MB, max_err {p['max_err']:.6e} (tol {p['tol']}, pass={p['pass']})"
        parity_cite = "`out/export/act_full_v3_100k/parity.json` + `export.log`"
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        parity_val = f"honest-negative ({exc})"
        parity_cite = "`out/export/act_full_v3_100k/parity.json` (no leido)"

    xeon_lat_med, xeon_lat_tput = parse_bench_log(XEON_LAT)
    xeon_tput_med, xeon_tput_tput = parse_bench_log(XEON_TPUT)

    ood60 = parse_ood_mean(OOD_60)
    ood80 = parse_ood_mean(OOD_80)

    lines = [
        "| Fuente (host) | Metrica | Valor | Log citado |",
        "|---|---|---|---|",
        f"| Local CPU — {local_host} | latency-sync p50/p95 | {local_lat} | {local_cite} |",
        f"| Local CPU — {local_host} | throughput-async agg/per-stream | {local_tput} | {local_cite} |",
        f"| Xeon E-2386G congelado (otro host, no este) | latency-sync median / throughput | {xeon_lat_med} / {xeon_lat_tput} | `out/bench_intel_incoming/vehicle_20260916T135349Z/bench_fp32_LATENCY_sync.log` |",
        f"| Xeon E-2386G congelado (otro host, no este) | throughput-async median / throughput agg (6.96/stream = 27.85/4) | {xeon_tput_med} / {xeon_tput_tput} | `out/bench_intel_incoming/vehicle_20260916T135349Z/bench_fp32_THROUGHPUT_async.log` |",
        f"| Export FP32 IR `act_full_v3_100k` | tamano / paridad PyTorch | {parity_val} | {parity_cite} |",
        f"| OOD 60-79 v3-100k (filed, no re-abierto) | mean | ` {ood60} ` | `out/ood/OOD_RESULTS_60_79.md` |",
        f"| OOD 80-99 v3-100k (filed, no re-abierto) | mean | ` {ood80} ` | `out/ood/OOD_RESULTS_80_99.md` |",
        "| Seeds 0-9 (congeladas, PROHIBIDO re-evaluar en Carril A) | vehicle/random/swap | ver README Sec Result (3/10 vs 0/10 vs 0/10, transcrito) — no re-run aqui | `docs/submission_draft/SUBMISSION_TEXT.md` Sec 4 + `docs/submission_draft/EVIDENCE_INDEX.md` Sec 4 |",
        "",
        "_Local = medido en este host (ver `cpu_model` en JSON). Xeon = cifras congeladas de otro host, citadas no mezcladas. OOD/seeds 0-9 no re-evaluados en este carril._",
    ]
    return "\n".join(lines) + "\n"


def render_block(table: str) -> str:
    return f"{BEGIN}\n{table}{END}\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    table = build_table()
    block = render_block(table)
    if args.check:
        try:
            cur = README.read_text()
        except OSError as exc:
            print(f"README no leido: {exc}")
            return 1
        if BEGIN not in cur or END not in cur:
            print("markers ausentes")
            return 1
        inside = cur.split(BEGIN, 1)[1].split(END, 1)[0].strip()
        if inside == table.strip():
            print("results table: FRESH")
            return 0
        print("results table: STALE")
        return 1
    try:
        cur = README.read_text()
    except OSError as exc:
        print(f"README no leido: {exc}")
        return 1
    if BEGIN in cur and END in cur:
        assert cur.count(BEGIN) == 1 and cur.count(END) == 1, "README debe tener un unico par BEGIN/END"
        pre, rest = cur.split(BEGIN, 1)
        _, post = rest.split(END, 1)
        new = pre + block + post
    else:
        new = cur.rstrip() + "\n\n" + block
    if new != cur:
        README.write_text(new)
        print("README markers actualizados")
    else:
        print("README ya estaba al dia")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
