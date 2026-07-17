"""tool_use x Anthropic.

Drive the real `claude` CLI against a running LiteLLM proxy that routes
to Anthropic, ask Claude to invoke a built-in tool (`Bash`), and assert
that the upstream returned a `tool_use` content block. This proves the
proxy preserves Claude Code's tool-call wire shape end-to-end.

Bash is restricted to the exact command `echo pong` plus
`--permission-mode dontAsk`; the security rationale lives in
`claude_code/_tool_use.py` alongside the shared cell body.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/tool_use/test_anthropic.py
                       ^^^^^^^^      ^^^^^^^^^
                       feature_id    provider
"""

from __future__ import annotations

import pytest

from claude_code._tool_use import run_tool_use_cell


ANTHROPIC_MODELS = [
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-opus-4-7",
]


@pytest.mark.covers("llm.messages.anthropic.tool_use.nonstream.works")
def test_tool_use_anthropic(compat_result):
    """Drive the `claude` CLI against the LiteLLM proxy and assert a
    tool call was emitted on the wire."""
    run_tool_use_cell(compat_result=compat_result, models=ANTHROPIC_MODELS)
