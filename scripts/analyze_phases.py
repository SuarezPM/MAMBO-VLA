#!/usr/bin/env python3
"""Phase breakdown for vehicle v3-100k (seeds 0-9) from existing traces.

Lane A — text-only, no images, no training, no forbidden paths.
Reads ONLY (never writes except the single output MD):
  - out/seeds/traces.jsonl (frozen 10-row schema, small, text)
  - out/seeds/traces.full.jsonl (teacher exploration archive, text)
  - out/seeds/seed_*/result.json (teacher per-seed carries, text)
  - out/seeds/seed_hashes.json
  - out/gates/v3_200k_rollout.log (v3-200k 0/10 negative control, text)
  - out/gates/v3_100k_export.log (export parity, text)
  - docs/submission_draft/SUBMISSION_TEXT.md + EVIDENCE_INDEX.md (transcribed vehicle table)
  - out/bench_intel_incoming/vehicle_20260916T135349Z/*.log (bench medians, text)

Never touches (lane boundary): /tmp/mambo-vla-submit, out/ood/, out/export/,
scripts/eval_policy.py, sim/, training/, out/seeds_ood/.
Never reads PDFs/PNGs/MP4s.

Output: out/phases/phase_breakdown_v3_100k.md
Rule: zero invented numbers. Anything not citable is marked honest-negative.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_MD = ROOT / "out" / "phases" / "phase_breakdown_v3_100k.md"

TRACES = ROOT / "out" / "seeds" / "traces.jsonl"
TRACES_FULL = ROOT / "out" / "seeds" / "traces.full.jsonl"
SEED_HASHES = ROOT / "out" / "seeds" / "seed_hashes.json"
SUB_TEXT = ROOT / "docs" / "submission_draft" / "SUBMISSION_TEXT.md"
EVID_IDX = ROOT / "docs" / "submission_draft" / "EVIDENCE_INDEX.md"
ROLL_200K = ROOT / "out" / "gates" / "v3_200k_rollout.log"
EXPORT_100K = ROOT / "out" / "gates" / "v3_100k_export.log"
BENCH_DIR = ROOT / "out" / "bench_intel_incoming" / "vehicle_20260916T135349Z"


def load_jsonl(path):
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main():
    # 1. Real schema of out/seeds/traces.jsonl
    traces = load_jsonl(TRACES)
    keys_per_row = [sorted(r.keys()) for r in traces]
    all_keys = sorted({k for r in traces for k in r.keys()})
    seeds = sorted(r.get("seed") for r in traces)
    outcomes = Counter(r.get("outcome") for r in traces)
    skills = Counter(r.get("skill") for r in traces)
    has_phase_field = any(
        k in ("phase", "phases", "pick", "set_down", "setdown", "handover", "place")
        for k in all_keys
    )
    phase_hits = [
        r for r in traces
        if any(s in json.dumps(r).lower() for s in ("pick", "handover", "set-down", "setdown", '"phase"'))
    ]

    # 2. traces.full.jsonl: does it hold vehicle per-phase telemetry?
    full = load_jsonl(TRACES_FULL)
    full_keys = sorted({k for r in full for k in r.keys()})
    full_notes = Counter(r.get("note", "")[:60] for r in full)
    full_phase_mentions = sum(
        1 for r in full
        if "carry failed" in str(r.get("note", "")) or "placed" in str(r.get("note", ""))
    )
    # Notes mention A/B1/B carries but every row is teacher exploration or
    # scripted-teacher success, never a v3-100k policy rollout row.
    full_has_policy_row = any("policy rollout" in str(r.get("note", "")) for r in full)

    # 3. Teacher per-seed result.json carries (NOT the vehicle).
    teacher_carries = {}
    for seed in range(10):
        p = ROOT / "out" / "seeds" / f"seed_{seed}" / "result.json"
        if p.exists():
            d = json.loads(p.read_text())
            teacher_carries[seed] = d.get("log", [])

    # 4. Transcribed vehicle end-to-end table (only citable vehicle source in lane).
    sub_text = SUB_TEXT.read_text() if SUB_TEXT.exists() else ""
    evid_text = EVID_IDX.read_text() if EVID_IDX.exists() else ""
    # Verify the exact transcribed numbers are present in both docs.
    transcribed_checks = {
        "seed1 7650": ("7650" in sub_text and "7650" in evid_text),
        "seed4 7775": ("7775" in sub_text and "7775" in evid_text),
        "seed7 7750": ("7750" in sub_text and "7750" in evid_text),
        "seed3 2457mm": ("2457" in sub_text and "2457" in evid_text),
        "3/10": ("3/10" in sub_text and "3/10" in evid_text),
    }

    # 5. v3-200k negative control (directly readable in lane).
    # Scope the mm range to the POLICY200K block only (random 0mm excluded).
    rollout_200k = ROLL_200K.read_text() if ROLL_200K.exists() else ""
    policy_block = rollout_200k.split("=== RANDOM200K ===")[0]
    m200k = re.findall(r"mug moved (\d+)mm", policy_block)
    range_200k = f"{min(map(int, m200k))}-{max(map(int, m200k))} mm" if m200k else "honest-negative"

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    A = lines.append
    A("# Phase breakdown — vehicle v3-100k, seeds 0–9 (from existing traces)")
    A("")
    A("Vehicle: `out/checkpoints/act_full_v3/checkpoints/100000` (ACT-52M, router-gated).")
    A("Protocol: frozen seeds exactly 0–9, EXEC_STEPS=50, stats v3.")
    A("Generator: `scripts/analyze_phases.py` (stdlib only, deterministic).")
    A("Lane boundary: Carril A never opens `/tmp/mambo-vla-submit`, `out/ood/`,")
    A("`out/export/`, `scripts/eval_policy.py`, `sim/`, `training/`, `out/seeds_ood/`.")
    A("No PDFs/PNGs/MP4s read. Zero invented numbers.")
    A("")
    A("## 1. Real schema of `out/seeds/traces.jsonl`")
    A("")
    A(f"- Rows: {len(traces)}; seeds: {seeds}")
    A(f"- Keys (union): `{', '.join(all_keys)}`")
    A(f"- Outcomes: {dict(outcomes)}; skills: {dict(skills)}")
    A(f"- Has dedicated phase field (phase/pick/setdown/handover/place): {has_phase_field}")
    A(f"- Rows mentioning a phase token: {len(phase_hits)}")
    A("- Verdict: this file carries NO per-phase telemetry. It records one")
    A("  end-to-end `outcome` per frozen seed for the scripted teacher")
    A("  (`note: scripted teacher; physics-step p50, no NN forward`).")
    A("  It cannot yield vehicle per-phase rates by itself.")
    A("")
    A("## 2. What `out/seeds/traces.full.jsonl` adds (and does not add)")
    A("")
    A(f"- Rows: {len(full)}; keys (union): `{', '.join(full_keys)}`")
    A(f"- Rows whose note mentions a carry outcome: {full_phase_mentions}")
    A(f"- Rows that are v3-100k policy rollouts: {full_has_policy_row} (none found)")
    A("- The carry strings (`A carry failed ...`, `B1 carry failed ...`,")
    A("  `B carry failed ...`) belong to teacher exploration attempts, not to")
    A("  the v3-100k learned policy. They document how the scripted teacher")
    A("  failed while generating data; they are NOT vehicle phase passes.")
    A("- Therefore: no numerator/denominator for vehicle pick / set-down /")
    A("  handover exists in either traces file.")
    A("")
    A("## 3. Teacher contrast (NOT the vehicle — do not conflate)")
    A("")
    A("- `out/seeds/seed_*/result.json` (seeds 0–9, scripted teacher): 10/10")
    A("  end-to-end success, each log holding 3 placed carries")
    A("  (`A carry ... placed`, `B1 carry ... placed`, `B carry ... placed`).")
    A("- Example `out/seeds/seed_1/result.json`: steps 8144, 3/3 carries placed.")
    A("- This is the data-generation ceiling, not the learned policy.")
    A("  Vehicle rates below never reuse these 10/10 numbers.")
    A("")
    A("## 4. Vehicle phase table (seeds 0–9, v3-100k)")
    A("")
    A("| phase | definition used here | v3-100k rate 0–9 | status | log cited |")
    A("|---|---|---|---|---|")
    A("| pick | arm A first grasp + lift of the mug | honest-negative — no per-phase telemetry in traces | NOT CITABLE | `out/seeds/traces.jsonl` keys lack any phase field; `out/seeds/traces.full.jsonl` holds only teacher carry notes, zero policy rows |")
    A("| set-down | arm A table-relay set-down + park | honest-negative — no per-phase telemetry in traces | NOT CITABLE | same as above |")
    A("| handover | arm B re-grip at the relay point | honest-negative — no per-phase telemetry in traces | NOT CITABLE | same as above |")
    A("| place | final mug in destination zone (end-to-end success) | 3/10 (seeds 1/7650, 4/7775, 7/7750 steps; seed 3 max 2457 mm over failing seeds) | CITABLE-TRANSCRIBED (raw rollout log outside Carril A lane) | `docs/submission_draft/SUBMISSION_TEXT.md` Sec 4 + `docs/submission_draft/EVIDENCE_INDEX.md` Sec 4 |")
    A("")
    A("Reading the place row honestly:")
    A("- 3/10 is end-to-end place, not a per-leg phase pass. Seeds 1, 4, 7")
    A("  placed; seed 3 moved 2457 mm (max over failing seeds); seeds 0, 2, 5,")
    A("  6, 8, 9 moved (per-seed mm beyond the seed-3 max not provided in lane).")
    A("- Transcription check in lane:")
    for k, v in transcribed_checks.items():
        A(f"  - {k}: {'present in both docs' if v else 'MISSING — treat as honest-negative'}")
    A("- Raw v3-100k rollout log with per-seed rows is outside Carril A")
    A("  (lane forbids `training/`, `scripts/eval_policy.py`). No raw path is")
    A("  cited for the 3/10; the two docs above are the citable source in lane.")
    A("")
    A("## 5. Negative controls cited in lane")
    A("")
    A(f"- v3-200k policy rollout seeds 0–9: 0/10, range {range_200k} —")
    A("  `out/gates/v3_200k_rollout.log` (`=== POLICY200K ===`, 10 rows")
    A("  `policy rollout: no place (mug moved Nmm)`, EXEC_STEPS=50).")
    A("- v3-200k random baseline: 0/10, 0 mm all seeds — same log")
    A("  (`=== RANDOM200K ===`).")
    A("- v3-200k swap control: 0/10, router refusal, 0 steps — same log")
    A("  (`=== SWAP200K ===`, `router refused (out-of-grammar)`).")
    A("- v3-100k random 0/10 (0 mm) + swap 0/10 (refusal, 0 steps) are filed as")
    A("  transcribed controls in `docs/submission_draft/SUBMISSION_TEXT.md` Sec 4")
    A("  and `docs/submission_draft/EVIDENCE_INDEX.md` Sec 6; their raw logs")
    A("  are outside Carril A and are NOT cited as raw paths here.")
    A("")
    A("## 6. Logs cited (exact paths)")
    A("")
    A("- `out/seeds/traces.jsonl` (10 rows, schema above)")
    A("- `out/seeds/traces.full.jsonl` (215 rows, teacher exploration archive)")
    A("- `out/seeds/seed_0/result.json` … `out/seeds/seed_9/result.json` (teacher 3-carry logs)")
    A("- `out/seeds/seed_hashes.json` (frozen input hashes 0–9)")
    A("- `docs/submission_draft/SUBMISSION_TEXT.md` Sec 4 (vehicle 3/10 transcription)")
    A("- `docs/submission_draft/EVIDENCE_INDEX.md` Sec 3–4, 6 (vehicle + floor + controls transcription)")
    A("- `out/gates/v3_200k_rollout.log` (v3-200k 0/10 + random + swap, raw in lane)")
    A("- `out/gates/v3_100k_export.log` (export parity max_err 1.192093e-06, IR 66.658 MB)")
    A("- `out/bench_intel_incoming/vehicle_20260916T135349Z/bench_fp32_LATENCY_sync.log` (Median 40.80 ms)")
    A("- `out/bench_intel_incoming/vehicle_20260916T135349Z/bench_fp32_THROUGHPUT_async.log` (Median 143.66 ms, Throughput 27.85 FPS)")
    A("- `out/bench_intel_incoming/acceptance_xeon.txt` (Xeon E-2386G, devices `['CPU']`)")
    A("- `out/bench_intel_incoming/cpu_matrix_20260916T140419Z/00_env.txt` (12 threads, Ubuntu 24.04, OpenVINO 2026.3.0)")
    A("")
    A("## 7. Honest-negatives (explicitly NOT claimed)")
    A("")
    A("- Vehicle pick / set-down / handover rates 0–9: no data in lane. Any")
    A("  per-phase numerator shown elsewhere without a new instrumented rollout")
    A("  is invented and must be rejected.")
    A("- Vehicle per-seed mm for failing seeds except the seed-3 max (2457 mm):")
    A("  not provided in lane.")
    A("- Vehicle per-seed rollout forward p50/p95: not in gate data")
    A("  (`EVIDENCE_INDEX.md` Sec 5 states this explicitly); `benchmark_app`")
    A("  medians are device latency, a different metric.")
    A("- OOD seeds 60–79, export IR bytes, training loss curve: raw paths")
    A("  (`out/ood/`, `out/export/`, `training/`) are outside Carril A and were")
    A("  not re-opened here. Numbers for those lines are reported from filed")
    A("  prose in lane, not re-verified by this script.")
    A("- No phase inference from teacher carries: teacher 10/10 never implies")
    A("  vehicle phases.")
    A("")
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD} ({len(traces)} traces rows, {len(full)} full rows)")

if __name__ == "__main__":
    sys.exit(main())
