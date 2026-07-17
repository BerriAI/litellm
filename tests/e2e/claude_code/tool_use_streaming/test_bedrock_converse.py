"""tool_use_streaming x Bedrock (Converse).

Drive the real `claude` CLI in headless `--output-format stream-json`
mode against a running LiteLLM proxy that routes Claude requests to
AWS Bedrock via the `Converse` API (`ConverseStream`), ask Claude to
invoke a built-in tool (`Bash`), and assert that the upstream (a)
emitted a `tool_use` content block and (b) actually streamed events
incrementally.

Bedrock Converse has its own tool-streaming envelope (`toolUse` blocks
with `delta` chunks); this cell catches gateway regressions where the
proxy buffers the response or fails to translate Converse's streaming
envelope back to the Anthropic `message_*` event shape Claude Code
expects.

Bash is restricted to the exact command `echo pong` plus
`--permission-mode dontAsk`; see `claude_code/_tool_use.py` for the
security rationale.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/tool_use_streaming/test_bedrock_converse.py
                       ^^^^^^^^^^^^^^^^^^      ^^^^^^^^^^^^^^^^
                       feature_id              provider
"""

from __future__ import annotations

import pytest

from claude_code._tool_use import run_tool_use_cell


BEDROCK_CONVERSE_MODELS = [
    "claude-haiku-4-5-bedrock-converse",
    "claude-sonnet-4-5-bedrock-converse",
    "claude-opus-4-7-bedrock-converse",
]


@pytest.mark.covers("llm.messages.bedrock_converse.tool_use.stream.works")
def test_tool_use_streaming_bedrock_converse(compat_result):
    run_tool_use_cell(
        compat_result=compat_result,
        models=BEDROCK_CONVERSE_MODELS,
        verify_streaming=True,
    )
