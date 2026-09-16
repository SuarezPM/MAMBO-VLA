"""Bench stub (stdlib only): per-seed table printer + swap-control placeholder.

No torch, no MuJoCo import. Real rollouts replace `run_table` internals;
the printed columns (success/fail, steps, forward p50, note) stay frozen.
"""

from typing import Dict, List

from mambo_vla_rc.seed_freeze import SEEDS

CORRECT_INSTRUCTION = "Pick up the mug with the right arm and transfer it to the left arm."
SWAP_INSTRUCTION = "Insert the peg into the socket."


def empty_table() -> List[Dict[str, object]]:
    return [
        {"seed": s, "success": None, "steps": None, "forward_p50_ms": None, "note": ""}
        for s in SEEDS
    ]


def print_table(rows: List[Dict[str, object]], label: str) -> None:
    print(f"instruction: {label}")
    print("seed | success | steps | forward_p50_ms | note")
    for row in rows:
        print(
            f"{row['seed']:>4} | {row['success']} | {row['steps']} | "
            f"{row['forward_p50_ms']} | {row['note']}"
        )


def main() -> int:
    print_table(empty_table(), CORRECT_INSTRUCTION)
    print("swap-control (placeholder): rerun with instruction below; success must collapse.")
    print(f"swap instruction: {SWAP_INSTRUCTION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
