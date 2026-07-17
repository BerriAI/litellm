"""tool_use_streaming x OpenAI (GPT-5.6).

Drive the real `claude` CLI in headless `--output-format stream-json`
mode against a running LiteLLM proxy that routes Anthropic Messages
requests to OpenAI's GPT-5.6 family (Sol, Terra, Luna), ask the model
to invoke a built-in tool (`Bash`), and assert that the upstream (a)
emitted a `tool_use` content block and (b) streamed the tool input
incrementally as `input_json_delta` events.

OpenAI streams tool arguments as incremental `tool_calls` argument
fragments; LiteLLM must re-emit them as Anthropic `input_json_delta`
deltas rather than buffering the full input into one complete block.

Bash is restricted to the exact command `echo pong` plus
`--permission-mode dontAsk`; see `claude_code/_tool_use.py` for the
security rationale.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/tool_use_streaming/test_openai.py
                       ^^^^^^^^^^^^^^^^^^      ^^^^^^
                       feature_id              provider
"""

from __future__ import annotations

from claude_code._tool_use import run_tool_use_cell

OPENAI_MODELS = [
    "gpt-5-6-sol-openai",
    "gpt-5-6-terra-openai",
    "gpt-5-6-luna-openai",
]


def test_tool_use_streaming_openai(compat_result):
    run_tool_use_cell(
        compat_result=compat_result,
        models=OPENAI_MODELS,
        verify_streaming=True,
    )
