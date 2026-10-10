"""Tests for litellm/harness/__init__.py: the public API surface."""

from __future__ import annotations


from litellm import harness
from tests.test_litellm_rust.support.child_interpreter import run_child_interpreter
from litellm.utils import ProviderConfigManager

PUBLIC_NAMES = [
    "Harness",
    "agent",
    "aagent",
    "agent_session",
    "aagent_session",
    "agent_resume",
    "aagent_resume",
    "agent_capabilities",
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
    "ToolLoopOptions",
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
    out = run_child_interpreter(LAZY_IMPORT_CHECK, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


def test_adapter_registry_paths_cover_every_harness():
    for member in harness.Harness:
        config = ProviderConfigManager.get_provider_harness_config(member)
        assert config is not None and config.harness is member


def test_litellm_agent_is_top_level_and_lazy():
    code = (
        "import sys, litellm; assert 'litellm.harness' not in sys.modules; "
        "assert litellm.agent is litellm.harness.agent; "
        "assert litellm.Harness.CODEX.value == 'codex'; "
        "assert litellm.ToolLoopOptions is litellm.harness.ToolLoopOptions"
    )
    out = run_child_interpreter(code, timeout=120)
    assert out.returncode == 0, out.stderr
