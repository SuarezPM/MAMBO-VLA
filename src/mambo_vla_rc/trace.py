"""Trace log helper (stdlib only): append/read JSONL episode traces.

Each trace record is one episode outcome. Required keys: skill, seed,
instruction, poses_hash, outcome, note. Extra keys (e.g. failure_cause)
are allowed and preserved.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Union

REQUIRED_KEYS = ("skill", "seed", "instruction", "poses_hash", "outcome", "note")


def append_trace(path: Union[str, Path], record: Dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_KEYS if key not in record]
    if missing:
        raise ValueError(f"Missing trace keys: {missing}")
    line = json.dumps(record, sort_keys=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def read_traces(path: Union[str, Path]) -> List[Dict[str, Any]]:
    traces = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    return traces
