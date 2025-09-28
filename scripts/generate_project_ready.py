#!/usr/bin/env python3
"""
Generate PROJECT_READY.md from MVP/matrix checks.

Inputs:
  - local/artifacts/mvp/mvp_report.json (from scripts/mvp_check.py)

Outputs:
  - PROJECT_READY.md at repo root with a summary + status table

Usage:
  python scripts/generate_project_ready.py
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict, Any, List


NAMES = {
    "deterministic_local": "Deterministic Local (agent core, response utils)",
    "mini_agent_e2e_low": "Mini-Agent E2E (local shim)",
    "codex_agent_router_shim": "codex-agent via Router (shim)",
    "docker_smokes": "Docker Readiness (readiness + loopback gated)",
}


def short_status(ok: bool | None, skipped: bool | None = None) -> str:
    if skipped:
        return "SKIP"
    if ok is True:
        return "PASS"
    if ok is False:
        return "FAIL"
    return "UNK"


def build_table(checks: List[Dict[str, Any]]) -> str:
    rows = []
    for c in checks:
        name = c.get("name")
        label = NAMES.get(name, name)
        st = short_status(c.get("ok"), c.get("skipped"))
        note = "" if st == "PASS" else (c.get("details", "").strip()[:200] or "")
        rows.append((label, st, note))

    # Markdown table
    out = ["| Capability | Status | Notes |", "|---|---|---|"]
    for label, st, note in rows:
        out.append(f"| {label} | {st} | {note.replace('|','/')} |")
    return "\n".join(out)


def overall(checks: List[Dict[str, Any]]) -> str:
    # Overall PASS if every non-skipped check is ok
    non_skipped = [c for c in checks if not c.get("skipped")]
    if non_skipped and all(c.get("ok") for c in non_skipped):
        return "READY"
    return "NOT READY"


def main() -> int:
    path = "local/artifacts/mvp/mvp_report.json"
    if not os.path.exists(path):
        print(f"Missing {path}. Run scripts/mvp_check.py first.")
        return 1
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    checks = data.get("checks", [])
    # Use timezone-aware UTC to avoid deprecation warnings
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    ov = overall(checks)
    table = build_table(checks)

    md = f"""
# Project Readiness — {stamp}

**Summary**: {ov}

**How this is computed**
- Deterministic + local low E2E + docker readiness (gated) checks
- Source: `scripts/mvp_check.py` → `local/artifacts/mvp/mvp_report.json`
- Status meanings: PASS (green), FAIL (red), SKIP (intentionally not exercised)

{table}

**Notes**
- Docker checks are skip‑friendly by design. Enable them locally with: `make docker-up && make ndsmoke-docker`.
- Heavy E2E flows (tools/parallel/repair) are available as optional tests and can be added to this matrix later as needed.

""".strip()

    with open("PROJECT_READY.md", "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print("Wrote PROJECT_READY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
