"""tool_use_streaming x Vertex AI.

Drive the real `claude` CLI in headless `--output-format stream-json`
mode against a running LiteLLM proxy that routes Claude requests to
GCP Vertex AI, ask Claude to invoke a built-in tool (`Bash`), and
assert that the upstream (a) emitted a `tool_use` content block and
(b) actually streamed events incrementally.

Vertex AI exposes Anthropic models via `:streamRawPredict`; this cell
catches gateway regressions where the proxy buffers the response or
strips the streaming beta header on the way to Vertex.

Bash is restricted to the exact command `echo pong` plus
`--permission-mode dontAsk`; see `claude_code/_tool_use.py` for the
security rationale.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/tool_use_streaming/test_vertex_ai.py
                       ^^^^^^^^^^^^^^^^^^      ^^^^^^^^^
                       feature_id              provider
"""

from __future__ import annotations

import pytest

from claude_code._tool_use import run_tool_use_cell


VERTEX_AI_MODELS = [
    "claude-haiku-4-5-vertex",
    "claude-sonnet-4-5-vertex",
    "claude-opus-4-7-vertex",
]


@pytest.mark.covers("llm.messages.vertex.tool_use.stream.works")
def test_tool_use_streaming_vertex_ai(compat_result):
    run_tool_use_cell(
        compat_result=compat_result,
        models=VERTEX_AI_MODELS,
        verify_streaming=True,
    )
