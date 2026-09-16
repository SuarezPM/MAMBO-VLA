"""Seed freeze helper (stdlib only): canonical JSON + SHA-256 per seed bundle.

Each seed input bundle covers scene layout, object poses, object
colors/materials, lighting, and instruction strings. Hash before running;
publish the hash list with the results (see docs/SDD_PLAN.md).
"""

import hashlib
import json
from typing import Any, Dict, List

SEEDS: List[int] = list(range(10))


def hash_bundle(bundle: Dict[str, Any]) -> str:
    canonical = json.dumps(bundle, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def freeze_table(bundles: Dict[int, Dict[str, Any]]) -> Dict[int, str]:
    missing = [seed for seed in SEEDS if seed not in bundles]
    if missing:
        raise ValueError(f"Missing seed bundles: {missing}")
    return {seed: hash_bundle(bundles[seed]) for seed in SEEDS}
