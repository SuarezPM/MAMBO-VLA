#!/bin/bash
# night_watch.sh — hook nocturno MAMBO-VLA-RC. Solo lectura + probes CPU-only.
# NUNCA toca el proceso de entreno, ni GPU de cómputo, ni scripts congelados,
# ni datasets, ni checkpoints. Rollout/export (GPU) quedan para el orquestador.
set -u
P=/home/thelinconx/Proyectos/MAMBO-VLA-RC
VENV="$P/.venv/bin/python"
LOG="$P/out/night_watch.log"
mkdir -p "$P/out/gates"
ts() { date '+%m-%d %H:%M:%S'; }
STEP=$(grep -a "ot_train.py:435" "$P/out/training/v3_60ep.log" 2>/dev/null | tail -n 1 | tr -d '\000' | grep -a -o "step:[0-9]*K.*loss:[0-9.]*" | tail -n 1)
ALIVE="MUERTO"
ps --no-headers -p "$(cat "$P/out/training/v3.pid" 2>/dev/null)" >/dev/null 2>&1 && ALIVE="VIVO"
echo "$(ts) heartbeat v3=$ALIVE $STEP" >> "$LOG"
for RUNG in 040000 060000 080000 100000; do
  CKPT="$P/out/checkpoints/act_full_v3/checkpoints/$RUNG"
  MARK="$P/out/gates/v3_${RUNG}_vision.done"
  if [ -f "$CKPT/training_state/training_step.json" ] && [ ! -f "$MARK" ]; then
    echo "$(ts) probe vision $RUNG (CPU-only)" >> "$LOG"
    CUDA_VISIBLE_DEVICES="" "$VENV" "$P/scripts/vision_alive.py" \
      --checkpoint "$CKPT" --stats-root out/lerobot_v3 \
      >> "$LOG" 2>&1 <<< "" || true
    # tail -n 6 equivalent: el script ya imprime summary; marcamos igual para no reintentar en bucle
    echo "$(ts) probe $RUNG done marker" >> "$LOG"
    touch "$MARK"
  fi
done
