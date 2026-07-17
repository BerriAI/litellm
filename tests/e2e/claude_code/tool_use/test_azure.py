"""tool_use x Azure (Microsoft Foundry).

Drive the real `claude` CLI against a running LiteLLM proxy that routes
Claude requests to Anthropic's models hosted in Microsoft Foundry on
Azure, ask Claude to invoke a built-in tool (`Bash`), and assert that
the upstream returned a `tool_use` content block.

Foundry's Anthropic deployments support function/tool calling
identically to anthropic.com; LiteLLM's `azure_ai/claude-*` route
inherits the full Anthropic tool-use transformation.

Bash is restricted to the exact command `echo pong` plus
`--permission-mode dontAsk`; see `claude_code/_tool_use.py` for the
security rationale.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/tool_use/test_azure.py
                       ^^^^^^^^      ^^^^^
                       feature_id    provider
"""

from __future__ import annotations

import pytest

from claude_code._tool_use import run_tool_use_cell


AZURE_MODELS = [
    "claude-haiku-4-5-azure",
    "claude-sonnet-4-5-azure",
    "claude-opus-4-7-azure",
]


@pytest.mark.covers("llm.messages.azure_foundry.tool_use.nonstream.works")
def test_tool_use_azure(compat_result):
    """Drive the `claude` CLI against the LiteLLM proxy and assert a
    tool call was emitted on the wire."""
    run_tool_use_cell(compat_result=compat_result, models=AZURE_MODELS)
