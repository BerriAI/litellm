"""tool_use x Vertex AI.

Drive the real `claude` CLI against a running LiteLLM proxy that routes
Claude requests to Anthropic's models on Google Cloud Vertex AI, ask
Claude to invoke a built-in tool (`Bash`), and assert that the upstream
returned a `tool_use` content block.

Bash is restricted to the exact command `echo pong` plus
`--permission-mode dontAsk`; see `claude_code/_tool_use.py` for the
security rationale.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/tool_use/test_vertex_ai.py
                       ^^^^^^^^      ^^^^^^^^^
                       feature_id    provider
"""

from __future__ import annotations

import pytest

from claude_code._tool_use import run_tool_use_cell


VERTEX_AI_MODELS = [
    "claude-haiku-4-5-vertex",
    "claude-sonnet-4-5-vertex",
    "claude-opus-4-7-vertex",
]


@pytest.mark.covers("llm.messages.vertex.tool_use.nonstream.works")
def test_tool_use_vertex_ai(compat_result):
    """Drive the `claude` CLI against the LiteLLM proxy and assert a
    tool call was emitted on the wire."""
    run_tool_use_cell(compat_result=compat_result, models=VERTEX_AI_MODELS)
