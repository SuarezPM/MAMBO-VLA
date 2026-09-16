#!/usr/bin/env bash
# MAMBO-VLA quickstart (Carril A / Deepwork F1) — texto/codigo solo.
# Cadena: setup -> self-check -> export(check) -> bench -> eval(smoke 1 seed).
# Lane boundaries (NO tocar): /tmp/mambo-vla-submit, out/ood/, out/export/,
# out/seeds_ood/, scripts/eval_policy.py (defaults), sim/, training/, seeds 0-9.
# PROHIBIDO entrenar y re-evaluar 0-9: el paso eval usa --swap con seed 60
# (router refusal, 0 steps, sin fisica ni red, sin tocar 0-9 ni re-abrir OOD).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
CKPT="$ROOT/out/checkpoints/act_full_v3/checkpoints/100000"
IR="$ROOT/out/export/act_full_v3_100k/model.xml"

echo "== [1/5] setup =="
test -x "$PY" || { echo "honest-negative: falta $PY (crear .venv con Python 3.10 e instalar requirements.txt)"; exit 1; }
"$PY" --version
test -f "$ROOT/requirements.txt" || { echo "honest-negative: falta requirements.txt"; exit 1; }
echo "setup: OK (.venv presente, sin reinstalar por defecto)"

echo "== [2/5] self-check =="
"$PY" "$ROOT/scripts/env_check.py"

echo "== [3/5] export(check, sin re-exportar) =="
test -f "$IR" || { echo "honest-negative: falta IR $IR (fuera del carril: no re-exportar)"; exit 1; }
test -f "$ROOT/out/export/act_full_v3_100k/parity.json" || { echo "honest-negative: falta parity.json"; exit 1; }
"$PY" -c "import json,sys; p=json.load(open('$ROOT/out/export/act_full_v3_100k/parity.json')); print(f\"parity max_err={p['max_err']:.6e} pass={p['pass']} ir_size_mb={p['ir_size_mb']:.3f}\"); sys.exit(0 if p['pass'] else 1)"
echo "export(check): OK (IR existente verificado, no re-exportado)"

echo "== [4/5] bench (warmup10+n100, CPU FP32, este host) =="
mkdir -p "$ROOT/out/bench_local"
"$PY" "$ROOT/scripts/bench_openvino_local.py" 2>&1 | tee "$ROOT/out/bench_local/bench_local_run.log"
"$PY" "$ROOT/scripts/results_table.py"
echo "bench: OK (JSON + tabla README actualizada)"

echo "== [5/5] eval smoke 1 seed (swap refusal, seed 60, sin tocar 0-9) =="
test -d "$CKPT" || { echo "honest-negative: falta checkpoint $CKPT (clon sin checkpoints: el smoke swap cubre router y el bench cubre inferencia, pero sin CKPT no hay GREEN)"; exit 1; }
# --swap = control de router: frase fuera de gramatica -> 0 steps, sin fisica,
# sin red, sin verificacion de hashes; nunca toca seeds 0-9 ni re-abre OOD filed.
"$PY" "$ROOT/scripts/eval_policy.py" --checkpoint "$CKPT" --seeds 60 --swap
echo "eval-smoke: OK (swap refusal, 0 steps, carril respetado)"

echo "QUICKSTART GREEN"
