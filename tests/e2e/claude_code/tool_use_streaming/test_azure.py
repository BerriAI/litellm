"""tool_use_streaming x Microsoft Foundry (Azure).

Drive the real `claude` CLI in headless `--output-format stream-json`
mode against a running LiteLLM proxy that routes Claude requests to
Microsoft Foundry's Anthropic deployments on Azure, ask Claude to
invoke a built-in tool (`Bash`), and assert that the upstream (a)
emitted a `tool_use` content block and (b) actually streamed events
incrementally.

Bash is restricted to the exact command `echo pong` plus
`--permission-mode dontAsk`; see `claude_code/_tool_use.py` for the
security rationale.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/tool_use_streaming/test_azure.py
                       ^^^^^^^^^^^^^^^^^^      ^^^^^
                       feature_id              provider
"""

from __future__ import annotations

import pytest

from claude_code._tool_use import run_tool_use_cell


AZURE_MODELS = [
    "claude-haiku-4-5-azure",
    "claude-sonnet-4-5-azure",
    "claude-opus-4-7-azure",
]


@pytest.mark.covers("llm.messages.azure_foundry.tool_use.stream.works")
def test_tool_use_streaming_azure(compat_result):
    run_tool_use_cell(
        compat_result=compat_result,
        models=AZURE_MODELS,
        verify_streaming=True,
    )
