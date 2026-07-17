"""tool_use x OpenAI (GPT-5.6).

Drive the real `claude` CLI against a running LiteLLM proxy that
routes Anthropic Messages requests to OpenAI's GPT-5.6 family (Sol,
Terra, Luna), ask the model to invoke a built-in tool (`Bash`), and
assert that a `tool_use` content block came back over the wire.

Claude Code declares its tools in Anthropic `tools` format; LiteLLM's
`openai/gpt-*` route translates them to OpenAI function calling and
maps the returned `tool_calls` back to Anthropic `tool_use` blocks, so
this cell exercises the tool-schema translation in both directions.

Bash is restricted to the exact command `echo pong` plus
`--permission-mode dontAsk`; see `claude_code/_tool_use.py` for the
security rationale.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/tool_use/test_openai.py
                       ^^^^^^^^      ^^^^^^
                       feature_id    provider
"""

from __future__ import annotations

from claude_code._tool_use import run_tool_use_cell

OPENAI_MODELS = [
    "gpt-5-6-sol-openai",
    "gpt-5-6-terra-openai",
    "gpt-5-6-luna-openai",
]


def test_tool_use_openai(compat_result):
    """Drive the `claude` CLI against the LiteLLM proxy and assert a
    tool call was emitted on the wire by each GPT-5.6 tier."""
    run_tool_use_cell(compat_result=compat_result, models=OPENAI_MODELS)
