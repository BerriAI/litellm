"""Tests for litellm/harness/__init__.py: the public API surface."""

from __future__ import annotations

import subprocess
import sys

from litellm import harness
from litellm.harness.adapters import ADAPTER_PATHS

PUBLIC_NAMES = [
    "Harness",
    "Gateway",
    "run",
    "arun",
    "stream",
    "astream",
    "session",
    "asession",
    "resume",
    "aresume",
    "capabilities",
    "Result",
    "Usage",
    "State",
    "Capabilities",
    "Session",
    "EventStream",
    "Text",
    "Reasoning",
    "ToolCall",
    "ToolResult",
    "FileChange",
    "Compaction",
    "Approval",
    "Done",
    "Event",
    "ClaudeCodeOptions",
    "CodexOptions",
    "OpenCodeOptions",
    "DeepAgentsOptions",
    "HarnessError",
    "CapabilityUnsupported",
    "OptionsMismatch",
    "HarnessInstallFailed",
    "SandboxError",
    "SessionClosed",
    "StateIncompatible",
    "OutputInvalid",
]
ERROR_NAMES = [
    "CapabilityUnsupported",
    "OptionsMismatch",
    "HarnessInstallFailed",
    "SandboxError",
    "SessionClosed",
    "StateIncompatible",
    "OutputInvalid",
]
LAZY_IMPORT_CHECK = (
    "import sys, litellm\n"
    "assert 'litellm.harness' not in sys.modules\n"
    "h = litellm.harness\n"
    "assert h.Harness.CODEX.value == 'codex'\n"
    "assert 'starlette' not in sys.modules and 'uvicorn' not in sys.modules\n"
    "print('ok')\n"
)


def test_public_api_names_exported():
    missing = [name for name in PUBLIC_NAMES if not hasattr(harness, name)]
    assert missing == []
    assert set(PUBLIC_NAMES) <= set(harness.__all__)


def test_errors_share_base_class():
    for name in ERROR_NAMES:
        assert issubclass(getattr(harness, name), harness.HarnessError)


def test_litellm_harness_attribute_is_lazy():
    out = subprocess.run(
        [sys.executable, "-c", LAZY_IMPORT_CHECK],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


def test_adapter_registry_paths_cover_every_harness():
    assert set(ADAPTER_PATHS) == set(harness.Harness)
