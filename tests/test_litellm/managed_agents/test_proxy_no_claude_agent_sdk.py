"""Architecture guard: the proxy never imports ``claude_agent_sdk``.

Phase 2 boundary (LIT-2879): the agent runtime runs ON the per-Session
VM (in ``litellm/managed_agents/daemon/runtime/``), not inside the
proxy process. Any import of ``claude_agent_sdk`` from
``litellm/proxy/`` would mean someone reintroduced the in-proxy runtime
shape that Phase 1 had to roll back.

This test scans every ``.py`` file under ``litellm/proxy/`` and fails if
``claude_agent_sdk`` appears in an ``import`` or ``from ... import``
statement.

If this test fails:

* Move the import into ``litellm/managed_agents/daemon/runtime/`` —
  that is the only place ``claude_agent_sdk`` is allowed to live.
* The proxy talks to the daemon over the
  ``/v1/internal/sessions/{sid}/...`` control-plane endpoints; it
  never instantiates the SDK directly.
"""

import pathlib
import re

import pytest

# tests/test_litellm/managed_agents/ → repo root is parents[3]
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
PROXY_DIR = REPO_ROOT / "litellm" / "proxy"

# Match either `import claude_agent_sdk` or `from claude_agent_sdk ...`.
_IMPORT_RE = re.compile(
    r"^\s*(?:import\s+claude_agent_sdk|from\s+claude_agent_sdk(?:\s|\.))",
    re.MULTILINE,
)


def _iter_proxy_python_files() -> list[pathlib.Path]:
    return [p for p in PROXY_DIR.rglob("*.py") if p.is_file()]


def test_proxy_does_not_import_claude_agent_sdk() -> None:
    offenders: list[str] = []
    for path in _iter_proxy_python_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        if _IMPORT_RE.search(text):
            offenders.append(str(path.relative_to(REPO_ROOT)))

    if offenders:
        pretty = "\n  ".join(offenders)
        pytest.fail(
            "claude_agent_sdk must not be imported from litellm/proxy/.\n"
            "Move the import into litellm/managed_agents/daemon/runtime/.\n"
            f"Offending files:\n  {pretty}"
        )
