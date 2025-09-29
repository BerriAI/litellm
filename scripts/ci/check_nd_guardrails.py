#!/usr/bin/env python3
"""
ND Guardrails — fail CI if we accidentally neuter non-deterministic smokes.

Checks:
1) Harness must contain an ND_REAL gate in scripts/mvp_check.py
2) No test under tests/ndsmoke/ may reference MINI_AGENT_ALLOW_DUMMY
3) At least one ND-real test must be gated on ND_REAL=1 and import os

Exit 0 if all good, else non-zero with a clear message.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def main() -> int:
    errors: list[str] = []

    # (1) Harness must contain ND_REAL gate
    mvp = ROOT / "scripts" / "mvp_check.py"
    mvp_s = read_text(mvp)
    if "ND_REAL" not in mvp_s or "ollama_chat/" not in mvp_s:
        errors.append("scripts/mvp_check.py is missing ND_REAL gating for ND lane (and ollama_chat selection)")

    # (2) No dummy in ND smokes
    nd_dir = ROOT / "tests" / "ndsmoke"
    if nd_dir.exists():
        for p in nd_dir.rglob("*.py"):
            s = read_text(p)
            if "MINI_AGENT_ALLOW_DUMMY" in s:
                errors.append(f"{p.as_posix()}: references MINI_AGENT_ALLOW_DUMMY — not allowed in ND lane")

    # (3) At least one ND-real test gated on ND_REAL
    nd_real_ok = False
    if nd_dir.exists():
        for p in nd_dir.rglob("*.py"):
            s = read_text(p)
            if "ND_REAL" in s and "skipif(os.getenv(\"ND_REAL\"" in s:
                nd_real_ok = True
                break
    if not nd_real_ok:
        errors.append("No ND-real test found (expected a test in tests/ndsmoke/ with @pytest.mark.skipif(os.getenv('ND_REAL')!='1', ...))")

    if errors:
        print("ND guardrails failed:\n - " + "\n - ".join(errors))
        return 1
    print("ND guardrails: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

