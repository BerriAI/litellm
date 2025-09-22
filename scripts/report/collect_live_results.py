#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect non-deterministic (live) smoke test results into JSON without external pytest plugins.

Targets a validated minimal live set that should SKIP gracefully if env/prereqs are missing.
Defaults focus on reliable connectivity + basic local-model checks. A heavier repair-loop test can be
opted-in via NDSMOKE_INCLUDE_REPAIR=1.

Writes:
- local/artifacts/smoke_results_nd.json

Usage:
  python3 scripts/report/collect_live_results.py
"""

from __future__ import annotations

import json
import os
import sys
import socket
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any
from types import ModuleType

import pytest


ARTIFACTS_DIR = Path("local/artifacts")
OUT_JSON = ARTIFACTS_DIR / "smoke_results_nd.json"

# Minimal validated set (node IDs / file paths)
# Expanded to include core Ollama + Chutes live validations. Each test will self-skip
# if required services/keys are unavailable, so it's safe to include broadly here.
LIVE_TARGETS: List[str] = [
    # Core reachability / local-model checks
    "tests/smoke/test_mini_agent_ollama_optional.py",
    "tests/smoke/test_ollama_live_optional.py",
    # Chutes routing checks (will self-skip if creds absent)
    "tests/smoke/test_chutes_live_optional.py",
    "tests/smoke/test_escalation_chutes_ndsmoke.py",
]


@dataclass
class Rec:
    nodeid: str
    suite: str  # "live"
    status: str  # "passed" | "failed" | "skipped"
    duration_s: float
    model: str | None = None
    iters: int | None = None
    notes: str | None = None
    markers: List[str] | None = None
    error: str | None = None  # first-line error summary for failed tests


class CollectorPlugin:
    def __init__(self) -> None:
        self._markers_by_nodeid: Dict[str, List[str]] = {}
        self._results: Dict[str, Rec] = {}

        # Ensure repo root is importable early (before test collection)
        cwd = os.getcwd()
        if cwd not in sys.path:
            sys.path.insert(0, cwd)

        # Make import of litellm safe in environments without optional extras
        fastuuid_mod = ModuleType("fastuuid")
        setattr(fastuuid_mod, "uuid4", staticmethod(lambda: "0" * 32))  # type: ignore[attr-defined]
        sys.modules.setdefault("fastuuid", fastuuid_mod)

        # Stub MCP import used by experimental_mcp_client package init
        mcp_mod = ModuleType("mcp")
        setattr(mcp_mod, "ClientSession", object)
        sys.modules.setdefault("mcp", mcp_mod)

        mcp_types_mod = ModuleType("mcp.types")
        setattr(mcp_types_mod, "CallToolRequestParams", type("CallToolRequestParams", (), {}))
        setattr(mcp_types_mod, "CallToolResult", type("CallToolResult", (), {}))
        setattr(mcp_types_mod, "Tool", type("Tool", (), {}))
        sys.modules.setdefault("mcp.types", mcp_types_mod)

        # Conditionally stub chutes import if not present and creds absent
        # This avoids hard-failing import on modules that are optional in local runs
        if not (os.getenv("CHUTES_API_KEY") or os.getenv("CHUTES_API_TOKEN")):
            try:
                __import__("litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent")
            except Exception:
                mod = ModuleType("litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent")
                def _stub_arouter_call(*_a, **_k):
                    raise RuntimeError("chutes arouter_call stubbed (no credentials); test will self-skip")
                setattr(mod, "arouter_call", _stub_arouter_call)
                sys.modules["litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent"] = mod

    def pytest_sessionstart(self, session) -> None:  # type: ignore
        # Sensible defaults (caller can override with env)
        os.environ.setdefault("NDSMOKE_MAX_ITERS", "6")
        os.environ.setdefault("NDSMOKE_TOOL_TIMEOUT", "120")
        os.environ.setdefault("NDSMOKE_TIMEOUT", "360")

    def pytest_collection_modifyitems(self, session, config, items) -> None:  # type: ignore
        for item in items:
            nodeid = item.nodeid
            # capture marker names
            marks = [m.name for m in getattr(item, "iter_markers", lambda: [])()]
            self._markers_by_nodeid[nodeid] = marks

    def pytest_runtest_logreport(self, report) -> None:  # type: ignore
        nodeid = report.nodeid
        # We record call-phase and setup-skip
        if report.when == "call":
            status = "passed" if report.passed else "failed" if report.failed else "skipped"
            duration = float(getattr(report, "duration", 0.0) or 0.0)
            marks = self._markers_by_nodeid.get(nodeid, [])
            err_text: str | None = None
            if status == "failed":
                err_text = getattr(report, "longreprtext", None)
                if not err_text:
                    try:
                        err_text = str(getattr(report, "longrepr", "")) or None
                    except Exception:
                        err_text = None
                if err_text:
                    first_non_empty = ""
                    for _line in err_text.splitlines():
                        s = _line.strip()
                        if s:
                            first_non_empty = s
                            break
                    err_text = (first_non_empty or err_text.splitlines()[0]).strip()
                    if len(err_text) > 300:
                        err_text = err_text[:300] + "…"
            self._results[nodeid] = Rec(
                nodeid=nodeid,
                suite="live",
                status=status,
                duration_s=duration,
                markers=marks,
                error=err_text,
            )
        elif report.when == "setup" and report.skipped:
            duration = float(getattr(report, "duration", 0.0) or 0.0)
            marks = self._markers_by_nodeid.get(nodeid, [])
            if nodeid not in self._results:
                self._results[nodeid] = Rec(
                    nodeid=nodeid,
                    suite="live",
                    status="skipped",
                    duration_s=duration,
                    markers=marks,
                )

    def write_json(self, path: Path) -> None:
        tests = [asdict(r) for r in self._results.values()]
        # Normalize optional None -> exclude for compactness
        norm_tests: List[Dict[str, Any]] = []
        for t in tests:
            t = {k: v for k, v in t.items() if v is not None}
            norm_tests.append(t)
        data = {
            "run_meta": {
                "author": os.getenv("USER", "auto"),
                "repo_branch": os.getenv("GIT_BRANCH", ""),
            },
            "tests": norm_tests,
        }
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def _load_env_file(path: Path) -> None:
    """Lightweight .env loader (avoids external deps).
    - Supports KEY=VALUE with optional quotes
    - Ignores comments and empty lines
    - Does not override variables that are already set in the environment
    """
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        val = v.strip().strip('"').strip("'")
        # don't override explicitly-set env
        os.environ.setdefault(key, val)


def _can_connect(host: str, port: int, timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _sanitize_ollama_env() -> None:
    """Normalize OLLAMA_HOST/OLLAMA_PORT so host does not contain a port.
    Prevents patterns like "localhost:11434:11434" in skip messages.
    """
    host = os.getenv("OLLAMA_HOST")
    if host and ":" in host:
        try:
            h, p = host.rsplit(":", 1)
            if h:
                os.environ["OLLAMA_HOST"] = h
            if p.isdigit() and not os.getenv("OLLAMA_PORT"):
                os.environ["OLLAMA_PORT"] = p
        except Exception:
            pass


def main() -> int:
    # Ensure tests import from repo root BEFORE plugin is constructed
    os.environ.setdefault("PYTHONPATH", os.getcwd())
    plugin = CollectorPlugin()

    # Load .env if present to make API keys/model aliases available to tests
    env_path = Path(".env")
    if env_path.exists():
        _load_env_file(env_path)

    # If the mini-agent exec RPC is reachable locally, implicitly enable DOCKER_MINI_AGENT
    # so live tests that require it won't skip due to a missing flag.
    if not os.getenv("DOCKER_MINI_AGENT") and _can_connect("127.0.0.1", 8790):
        os.environ["DOCKER_MINI_AGENT"] = "1"

    # Normalize Ollama env if user specified host:port in OLLAMA_HOST
    _sanitize_ollama_env()

    # Build pytest args for live targets; opt-in heavy repair loop via env
    targets = list(LIVE_TARGETS)
    if os.getenv("NDSMOKE_INCLUDE_REPAIR") == "1":
        targets.append("tests/smoke/test_mini_agent_compress_runs_ndsmoke.py::test_mini_agent_compress_runs_iterates_live_optional")

    args = ["-q"] + targets
    exit_code = pytest.main(args=args, plugins=[plugin])

    # Always write JSON to aid triage, even if failures occurred or everything skipped
    plugin.write_json(OUT_JSON)
    return int(exit_code)


if __name__ == "__main__":
    sys.exit(main())