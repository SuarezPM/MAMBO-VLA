"""Determinism check — ARTEFACTOS, jamais outcomes closed-loop.

Verifica determinismo de artefactos (hashes, tabla, IR+paridad).
NO verifica outcomes closed-loop: documentado no-determinista
(seed 93: run1 7750 steps en out/seeds_ood/logs_ood_80_99/eval_80_99.log
vs rerun 7775 en out/seeds_ood/logs_ood_80_99/eval_80_99_rerun.log y
out/seeds_ood/logs_ood_80_99/results_80_99.json).

Checks (solo lectura, sin entrenar/evaluar, sin cargar pesos):
  (a) smoke de 1 seed: regenerar bundle seed 60 con el generador real
      (scripts/run_seeds_ood.py:seed_bundle_ood) y comparar hash con hash-file
  (b) scripts/results_table.py --check FRESH
  (c) re-sha de IR (model.xml/model.bin) + parity match

Uso:
  .venv/bin/python scripts/determinism_check.py
Reporta en out/determinism/DETERMINISM_REPORT.md con PASS/FAIL por check.
"""

import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "out" / "determinism"
REPORT = OUT_DIR / "DETERMINISM_REPORT.md"

SEED_A = 60
HASH_FILE_A = ROOT / "out" / "seeds_ood" / "seed_hashes_ood.json"
FROZEN = ROOT / "out" / "frozen-selection.json"
PARITY = ROOT / "out" / "export" / "act_full_v3_100k" / "parity.json"
IR_XML = ROOT / "out" / "export" / "act_full_v3_100k" / "model.xml"
IR_BIN = ROOT / "out" / "export" / "act_full_v3_100k" / "model.bin"
RESULTS_TABLE = ROOT / "scripts" / "results_table.py"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_a() -> tuple[str, str]:
    """Smoke de 1 seed: generador real seed_bundle_ood(60) vs hash-file."""
    try:
        if str(ROOT / "scripts") not in sys.path:
            sys.path.insert(0, str(ROOT / "scripts"))
        if str(ROOT / "src") not in sys.path:
            sys.path.insert(0, str(ROOT / "src"))
        from run_seeds_ood import seed_bundle_ood  # type: ignore
        from mambo_vla_rc.seed_freeze import hash_bundle as real_hash  # type: ignore
    except ImportError as exc:
        return ("FAIL", f"generador real no importable: {exc}")
    try:
        saved = json.loads(HASH_FILE_A.read_text())
        expected = saved[str(SEED_A)]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        return ("FAIL", f"hash-file no leido: {exc}")
    try:
        bundle = seed_bundle_ood(SEED_A)
        got = real_hash(bundle)
    except Exception as exc:
        return ("FAIL", f"regeneracion con generador real fallo: {exc}")
    if got == expected:
        return ("PASS", f"smoke 1 seed: seed {SEED_A} generador real hash match {got[:16]}... ({HASH_FILE_A.name})")
    return ("FAIL", f"smoke 1 seed: seed {SEED_A} mismatch got={got} expected={expected}")


def check_b() -> tuple[str, str]:
    """results_table.py --check FRESH (solo lectura)."""
    try:
        proc = subprocess.run(
            [sys.executable, str(RESULTS_TABLE), "--check"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception as exc:
        return ("FAIL", f"subprocess fallo: {exc}")
    out = (proc.stdout + proc.stderr).strip()
    if proc.returncode == 0 and "FRESH" in out:
        return ("PASS", f"results_table --check FRESH ({out})")
    return ("FAIL", f"exit={proc.returncode} out={out}")


def check_c() -> tuple[str, str]:
    """Re-sha IR + parity match contra frozen-selection."""
    try:
        frozen = json.loads(FROZEN.read_text())
        parity = json.loads(PARITY.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return ("FAIL", f"json no leido: {exc}")
    try:
        xml_sha = sha256_file(IR_XML)
        bin_sha = sha256_file(IR_BIN)
    except OSError as exc:
        return ("FAIL", f"sha lectura fallo: {exc}")
    exp_xml = frozen["model"]["ir_xml_sha"]
    exp_bin = frozen["model"]["ir_bin_sha"]
    exp_err = frozen["model"]["parity_max_err"]
    ok = (xml_sha == exp_xml) and (bin_sha == exp_bin)
    ok = ok and (parity.get("max_err") == exp_err) and (parity.get("pass") is True)
    detail = f"xml {xml_sha[:16]}...=={exp_xml[:16]}... ; bin {bin_sha[:16]}...=={exp_bin[:16]}... ; max_err {parity.get('max_err')}=={exp_err} pass={parity.get('pass')}"
    return (("PASS", detail) if ok else ("FAIL", detail))


def main() -> int:
    utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sta, deta = check_a()
    stb, detb = check_b()
    stc, detc = check_c()
    overall = "PASS" if (sta == stb == stc == "PASS") else "FAIL"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = f"""# Determinism Report — artefactos (jamás outcomes closed-loop)

Fecha (UTC): {utc}
Alcance: determinismo de ARTEFACTOS. Outcomes closed-loop documentados no-deterministas y excluidos: seed 93 run1 7750 steps (`out/seeds_ood/logs_ood_80_99/eval_80_99.log`) vs rerun 7775 (`out/seeds_ood/logs_ood_80_99/eval_80_99_rerun.log`, `out/seeds_ood/logs_ood_80_99/results_80_99.json`).

| Check | Resultado | Detalle |
|---|---|---|
| (a) bundle seed 60 vs hash-file | {sta} | {deta} |
| (b) results_table --check | {stb} | {detb} |
| (c) re-sha IR + parity | {stc} | {detc} |

Overall: {overall}
"""
    REPORT.write_text(report)
    print(report)
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
