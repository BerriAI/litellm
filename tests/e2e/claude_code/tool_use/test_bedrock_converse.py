"""tool_use x Bedrock (Converse).

Drive the real `claude` CLI against a running LiteLLM proxy that routes
Claude requests to AWS Bedrock via the unified `Converse` API path, ask
Claude to invoke a built-in tool (`Bash`), and assert that the upstream
returned a `tool_use` content block.

Bash is restricted to the exact command `echo pong` plus
`--permission-mode dontAsk`; see `claude_code/_tool_use.py` for the
security rationale.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/tool_use/test_bedrock_converse.py
                       ^^^^^^^^      ^^^^^^^^^^^^^^^^
                       feature_id    provider
"""

from __future__ import annotations

import pytest

from claude_code._tool_use import run_tool_use_cell


BEDROCK_CONVERSE_MODELS = [
    "claude-haiku-4-5-bedrock-converse",
    "claude-sonnet-4-5-bedrock-converse",
    "claude-opus-4-7-bedrock-converse",
]


@pytest.mark.covers("llm.messages.bedrock_converse.tool_use.nonstream.works")
def test_tool_use_bedrock_converse(compat_result):
    """Drive the `claude` CLI against the LiteLLM proxy and assert a
    tool call was emitted on the wire."""
    run_tool_use_cell(compat_result=compat_result, models=BEDROCK_CONVERSE_MODELS)
