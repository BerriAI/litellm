"""tool_use_streaming x AWS Bedrock Mantle (GPT-5.6).

Drive the real `claude` CLI in headless `--output-format stream-json`
mode against a running LiteLLM proxy that routes Anthropic Messages
requests to OpenAI's GPT-5.6 family (Sol, Terra, Luna) on AWS
Bedrock's Mantle endpoint, ask the model to invoke a built-in tool
(`Bash`), and assert that the upstream (a) emitted a `tool_use`
content block and (b) streamed the tool input incrementally as
`input_json_delta` events.

Mantle streams OpenAI Responses API `function_call_arguments.delta`
events over SigV4-signed SSE; LiteLLM must re-emit them as Anthropic
`input_json_delta` deltas rather than buffering the full input into
one complete block.

Bash is restricted to the exact command `echo pong` plus
`--permission-mode dontAsk`; see `claude_code/_tool_use.py` for the
security rationale.

Mantle cells are opt-in via COMPAT_MANTLE_CELLS=1 (see
`claude_code._gpt_cells`).

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/tool_use_streaming/test_bedrock_mantle.py
                       ^^^^^^^^^^^^^^^^^^      ^^^^^^^^^^^^^^
                       feature_id              provider
"""

from __future__ import annotations

from claude_code._gpt_cells import skip_unless_mantle_cells_enabled
from claude_code._tool_use import run_tool_use_cell

BEDROCK_MANTLE_MODELS = [
    "gpt-5-6-sol-bedrock-mantle",
    "gpt-5-6-terra-bedrock-mantle",
    "gpt-5-6-luna-bedrock-mantle",
]


def test_tool_use_streaming_bedrock_mantle(compat_result):
    skip_unless_mantle_cells_enabled()
    run_tool_use_cell(
        compat_result=compat_result,
        models=BEDROCK_MANTLE_MODELS,
        verify_streaming=True,
    )
