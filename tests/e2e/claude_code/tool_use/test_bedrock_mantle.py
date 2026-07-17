"""tool_use x AWS Bedrock Mantle (GPT-5.6).

Drive the real `claude` CLI against a running LiteLLM proxy that
routes Anthropic Messages requests to OpenAI's GPT-5.6 family (Sol,
Terra, Luna) on AWS Bedrock's Mantle endpoint, ask the model to invoke
a built-in tool (`Bash`), and assert that a `tool_use` content block
came back over the wire.

Mantle speaks the OpenAI Responses API, whose tool declarations and
`function_call` outputs differ from both Anthropic Messages and
chat completions; LiteLLM's `bedrock_mantle/openai.gpt-*` route
translates Anthropic `tools` into Responses tool declarations and maps
the emitted function calls back to `tool_use` blocks.

Bash is restricted to the exact command `echo pong` plus
`--permission-mode dontAsk`; see `claude_code/_tool_use.py` for the
security rationale.

Mantle cells are opt-in via COMPAT_MANTLE_CELLS=1 (see
`claude_code._gpt_cells`).

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/tool_use/test_bedrock_mantle.py
                       ^^^^^^^^      ^^^^^^^^^^^^^^
                       feature_id    provider
"""

from __future__ import annotations

from claude_code._gpt_cells import skip_unless_mantle_cells_enabled
from claude_code._tool_use import run_tool_use_cell

BEDROCK_MANTLE_MODELS = [
    "gpt-5-6-sol-bedrock-mantle",
    "gpt-5-6-terra-bedrock-mantle",
    "gpt-5-6-luna-bedrock-mantle",
]


def test_tool_use_bedrock_mantle(compat_result):
    """Drive the `claude` CLI against the LiteLLM proxy and assert a
    tool call was emitted on the wire by each GPT-5.6 tier."""
    skip_unless_mantle_cells_enabled()
    run_tool_use_cell(compat_result=compat_result, models=BEDROCK_MANTLE_MODELS)
