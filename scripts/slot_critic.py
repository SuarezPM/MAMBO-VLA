"""P1-lite slot critic: parser determinista de reglas para slots {fase, brazo, objeto}.

Metodo: REGLAS (expresiones regulares + listas cerradas). Cero ML, cero pesos,
cero red, cero embeddings, cero llamadas a modelos. Solo stdlib (re/json/sys).

Gramatica aceptada (minusculas, guiones/espacios equivalentes):
  fase:   pick (pick, picks, picked, picking, pick up, grasp, grasps, grasped,
                grasping)
          set-down (set down, setdown, set-down, sets down)
          handover (handover, hand over, handoff, hand off, hands off)
          place (place, places, placed, placing)
  brazo:  A (arm a, left arm) | B (arm b, right arm)
          NOTA: "left"/"right" sueltos son LUGARES (left staging, right zone),
          no brazos. "A"/"B" sueltos no cuentan (colision con el articulo "a").
  objeto: mug (mug, mugs) | plate (plate, plates). Objetos del relay.

Rechazos:
  - ambiguo: 0 o >1 valores distintos en algun slot (incluye pronombres
    "it/them" sin nombre explicito -> objeto ausente).
  - fuera de gramatica: sustantivos (peg, socket, drawer, spoon, fork, knife,
    cup, glass, bowl, dish, pan, pot, bottle, box, key, slider) o verbos
    (insert, pour, stir, cut, open, close, cook, wash) fuera del plan.
  - negacion (not, n't, never, do/does not, without): sin alcance de
    negacion en la gramatica -> se rechaza.
  - verbos de tarea completa (transfer, move, carry, bring, fetch, slide,
    push, pull, lift): no nombran una fase -> fase ausente (rechazo).
    La frase de despliegue del relay se acepta en el ROUTER
    (scripts/eval_policy.py:route, igualdad exacta), no aqui: este critico
    valida ordenes a nivel de fase.

Adjuntos no listados (p. ej. "at the relay") se IGNORAN: limitacion honesta
documentada en out/grounding/CRITIC_CHECK.md. Excepcion: adjuntos
safety-relevantes (_SAFETY_ADJUNCTS, hoy solo "park") emiten WARN por stderr
y en el campo "warn" ANTES de seguir (precondicion de cableado futuro:
aparcar el brazo es safety-relevante; el outcome accept/reject no cambia).
Idioma: ingles solo.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PHASES = ("pick", "set-down", "handover", "place")
ARMS = ("A", "B")
OBJECTS = ("mug", "plate")

PLAN_RELAY = {"phases": list(PHASES), "arms": list(ARMS), "objects": list(OBJECTS)}

_PHASE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("pick", (r"\bpicks?\b", r"\bpicked\b", r"\bpicking\b",
              r"\bpicks?\s+up\b", r"\bpicking\s+up\b",
              r"\bgrasps?\b", r"\bgrasped\b", r"\bgrasping\b")),
    ("set-down", (r"\bsets?\s+down\b", r"\bsetdown\b")),
    ("handover", (r"\bhandover\b", r"\bhand\s+over\b",
                  r"\bhandoffs?\b", r"\bhands?\s+off\b")),
    ("place", (r"\bplaces?\b", r"\bplaced\b", r"\bplacing\b")),
)

_ARM_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("A", (r"\barm\s+a\b", r"\bleft\s+arm\b")),
    ("B", (r"\barm\s+b\b", r"\bright\s+arm\b")),
)

_OBJECT_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("mug", (r"\bmugs?\b",)),
    ("plate", (r"\bplates?\b",)),
)

OOG_NOUNS = ("peg", "socket", "drawer", "spoon", "fork", "knife", "cup",
             "glass", "bowl", "dish", "pan", "pot", "bottle", "box", "key",
             "slider")
OOG_VERBS = ("insert", "pour", "stir", "cut", "open", "close", "cook", "wash")
NEGATIONS = (r"\bnot\b", r"n['\u2019]t\b", r"\bnever\b",
             r"\bdo\s+not\b", r"\bdoes\s+not\b", r"\bwithout\b")
WHOLE_TASK_VERBS = ("transfer", "move", "carry", "bring", "fetch", "slide",
                    "push", "pull", "lift")
# Adjuntos safety-relevantes que la gramatica actual IGNORA: cada hit emite
# WARN (stderr + campo "warn") ANTES de seguir. Precondicion de cableado
# futuro: "park" aparca el brazo y es safety-relevante; hoy solo se avisa.
_SAFETY_ADJUNCTS = ("park",)


def _normalize(text: str) -> str:
    t = text.lower().replace("\u2019", "'")
    t = re.sub(r"[-_]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _hits(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(p, text) for p in patterns)


def parse(text: str) -> dict:
    """Extrae menciones por slot. Determinista, sin ML."""
    t = _normalize(text)
    phases = sorted(p for p, ps in _PHASE_PATTERNS if _hits(t, ps))
    arms = sorted(a for a, ps in _ARM_PATTERNS if _hits(t, ps))
    objects = sorted(o for o, ps in _OBJECT_PATTERNS if _hits(t, ps))
    oog = sorted({w for w in OOG_NOUNS + OOG_VERBS
                  if re.search(r"\b" + re.escape(w) + r"s?\b", t)})
    neg = sorted({p for p in NEGATIONS if re.search(p, t)})
    whole = sorted({w for w in WHOLE_TASK_VERBS
                    if re.search(r"\b" + re.escape(w) + r"s?\b", t)})
    warn = sorted({w for w in _SAFETY_ADJUNCTS
                   if re.search(r"\b" + re.escape(w) + r"s?\b", t)})
    return {"phases": phases, "arms": arms, "objects": objects,
            "oog": oog, "negation": neg, "whole_task": whole, "warn": warn}


def critique(text: str, plan: dict | None = None) -> dict:
    """Valida una frase contra el plan. Devuelve {ok, reason, slots, mentions}.

    ok=True solo si hay exactamente 1 fase, 1 brazo y 1 objeto del plan,
    sin palabras fuera de gramatica ni negacion.
    """
    plan = plan or PLAN_RELAY
    m = parse(text)
    if m["warn"]:
        # Warning ANTES de seguir: adjunct-drop safety-relevante, precondicion
        # de cableado futuro. No altera el outcome accept/reject.
        print(f"WARN adjunct-drop safety-relevante "
              f"({','.join(m['warn'])}) ignorado en: {text.strip()} "
              f"[precondicion de cableado futuro]",
              file=sys.stderr)
    problems: list[str] = []
    if m["negation"]:
        problems.append("negation out-of-grammar")
    if m["oog"]:
        problems.append("out-of-grammar: " + ",".join(m["oog"]))
    for slot, key in (("fase", "phases"), ("brazo", "arms"), ("objeto", "objects")):
        vals = m[key]
        if len(vals) == 0:
            problems.append(f"missing {slot}")
        elif len(vals) > 1:
            problems.append(f"ambiguous {slot}: " + ",".join(vals))
    slots = {"fase": None, "brazo": None, "objeto": None}
    if not problems:
        slots = {"fase": m["phases"][0], "brazo": m["arms"][0],
                 "objeto": m["objects"][0]}
        for slot, key in (("fase", "phases"), ("brazo", "arms"), ("objeto", "objects")):
            allowed = set(plan[{"fase": "phases", "brazo": "arms",
                                "objeto": "objects"}[slot]])
            if m[key][0] not in allowed:
                problems.append(f"{slot} '{m[key][0]}' not in plan")
                slots = {"fase": None, "brazo": None, "objeto": None}
                break
    ok = not problems
    if ok:
        reason = "accept: " + ",".join(
            f"{k}={v}" for k, v in slots.items())
    else:
        reason = "reject: " + "; ".join(problems)
        if m["whole_task"] and not m["phases"]:
            reason += f" (whole-task verb, no phase: {','.join(m['whole_task'])})"
    return {"ok": ok, "reason": reason, "slots": slots, "mentions": m,
            "warn": m["warn"]}


def check_file(path: str | Path) -> dict:
    """Pasa el critico por un .jsonl {text, expect} y compara."""
    acc_ok = acc_tot = rej_ok = rej_tot = 0
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            res = critique(item["text"])
            got = "accept" if res["ok"] else "reject"
            match = got == item["expect"]
            if item["expect"] == "accept":
                acc_tot += 1
                acc_ok += match
            else:
                rej_tot += 1
                rej_ok += match
            rows.append({"id": item.get("id", i), "expect": item["expect"],
                         "got": got, "match": match, "reason": res["reason"],
                         "warn": res["warn"], "text": item["text"]})
    return {"aceptadasOK": acc_ok, "aceptadasTot": acc_tot,
            "rechazadasOK": rej_ok, "rechazadasTot": rej_tot,
            "all_ok": acc_ok == acc_tot and rej_ok == rej_tot, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Slot critic determinista (reglas).")
    ap.add_argument("--check", help="path a paraphrases.jsonl {text,expect}")
    ap.add_argument("--parse", help="frase suelta a criticar")
    args = ap.parse_args(argv)
    if args.parse is not None:
        print(json.dumps({"text": args.parse, **critique(args.parse)},
                         ensure_ascii=False))
        return 0
    if args.check:
        rep = check_file(args.check)
        for r in rep["rows"]:
            flag = "OK " if r["match"] else "MISS"
            w = f" WARN={','.join(r['warn'])}" if r["warn"] else ""
            print(f"[{flag}] {r['id']} expect={r['expect']} got={r['got']}{w} :: "
                  f"{r['reason']} :: {r['text']}")
        print(f"aceptadasOK/total: {rep['aceptadasOK']}/{rep['aceptadasTot']}")
        print(f"rechazadasOK/total: {rep['rechazadasOK']}/{rep['rechazadasTot']}")
        return 0 if rep["all_ok"] else 1
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
