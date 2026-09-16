"""Environment check: assert the frozen E1 pins, never install.

Asserts exact versions (mujoco 3.12.0, lerobot 0.4.4, openvino 2026.3.0,
torch 2.10.x CUDA build) plus torch.cuda.is_available(). Exits non-zero
on any mismatch so CI and phase gates fail loudly.
"""

import importlib.metadata
import sys

EXPECTED = {
    "mujoco": "3.12.0",
    "lerobot": "0.4.4",
    "openvino": "2026.3.0",
}

TORCH_PREFIX = "2.10."


def main() -> int:
    print(f"python: {sys.version.split()[0]}")
    if not sys.version.startswith("3.10."):
        print("MISMATCH: Python 3.10 is required")
        return 1
    failures = []
    for name, pinned in EXPECTED.items():
        try:
            found = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            print(f"{name}: MISSING")
            failures.append(f"{name} missing")
            continue
        status = "OK" if found == pinned else "MISMATCH"
        print(f"{name}: {status} (found {found}, pinned {pinned})")
        if found != pinned:
            failures.append(f"{name} found {found}, pinned {pinned}")
    try:
        import torch

        print(f"torch: found {torch.__version__}")
        if not torch.__version__.startswith(TORCH_PREFIX):
            failures.append(f"torch found {torch.__version__}, pinned {TORCH_PREFIX}x")
        cuda = torch.cuda.is_available()
        print(f"torch.cuda.is_available(): {cuda}")
        if not cuda:
            failures.append("torch CUDA not available")
    except ImportError:
        print("torch: MISSING")
        failures.append("torch missing")
    if failures:
        print(f"ENV CHECK FAILED: {'; '.join(failures)}")
        return 1
    print("ALL PRESENT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
