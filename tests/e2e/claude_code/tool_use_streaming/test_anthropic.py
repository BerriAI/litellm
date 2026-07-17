"""tool_use_streaming x Anthropic.

Drive the real `claude` CLI in headless `--output-format stream-json`
mode (with `--include-partial-messages`) against a running LiteLLM
proxy that routes to Anthropic, ask Claude to invoke a built-in tool
(`Bash`), and assert that the upstream (a) emitted a `tool_use` content
block and (b) actually streamed the tool input incrementally, i.e.
`input_json_delta` stream events were observed for the block.

This is the "fine-grained tool streaming" path. Historically gateways
break it in two ways: they either buffer/collapse the streamed tool
input into a single complete block (no `input_json_delta` records
reach the client) or they strip the
`fine-grained-tool-streaming-2025-05-14` beta header and the upstream
falls back to non-streaming tool_use. Both regressions are caught by
the shared cell body in `claude_code/_tool_use.py`.

Bash is restricted to the exact command `echo pong` plus
`--permission-mode dontAsk`; see `claude_code/_tool_use.py` for the
security rationale.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/tool_use_streaming/test_anthropic.py
                       ^^^^^^^^^^^^^^^^^^      ^^^^^^^^^
                       feature_id              provider
"""

from __future__ import annotations

import pytest

from claude_code._tool_use import run_tool_use_cell


ANTHROPIC_MODELS = [
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-opus-4-7",
]


@pytest.mark.covers("llm.messages.anthropic.tool_use.stream.works")
def test_tool_use_streaming_anthropic(compat_result):
    """Drive the `claude` CLI against the LiteLLM proxy and assert the
    proxy preserves fine-grained tool streaming end-to-end."""
    run_tool_use_cell(
        compat_result=compat_result,
        models=ANTHROPIC_MODELS,
        verify_streaming=True,
    )
