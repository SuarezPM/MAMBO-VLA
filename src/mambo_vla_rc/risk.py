"""Risk table helper (stdlib only): count traces per (skill, outcome).

Offline consolidation for the memory-lite stretch lane: clusters retained
episode traces into per-(skill, outcome) counts with one representative note.
"""

from typing import Any, Dict, List, Tuple

RiskTable = Dict[Tuple[str, str], Dict[str, Any]]


def build_risk_table(traces: List[Dict[str, Any]]) -> RiskTable:
    table: RiskTable = {}
    for trace in traces:
        key = (str(trace.get("skill")), str(trace.get("outcome")))
        entry = table.setdefault(key, {"count": 0, "note": ""})
        entry["count"] += 1
        if not entry["note"]:
            entry["note"] = str(trace.get("note", ""))
    return table


def render_text(table: RiskTable) -> str:
    lines = ["skill | outcome | count | note"]
    for (skill, outcome) in sorted(table):
        entry = table[(skill, outcome)]
        lines.append(f"{skill} | {outcome} | {entry['count']} | {entry['note']}")
    return "\n".join(lines)
