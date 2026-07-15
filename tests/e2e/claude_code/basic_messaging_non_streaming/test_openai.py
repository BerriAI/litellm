"""basic_messaging_non_streaming x OpenAI (GPT-5.6).

Drive the real `claude` CLI in headless mode against a running LiteLLM
proxy that routes Anthropic Messages requests to OpenAI's GPT-5.6
family (Sol, Terra, Luna), and report the outcome via `compat_result`.

Claude Code only speaks the Anthropic Messages API; LiteLLM's
`openai/gpt-*` route translates the request to OpenAI chat completions
and maps the response back, so this cell exercises the full
cross-provider translation layer in both directions.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/basic_messaging_non_streaming/test_openai.py
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^      ^^^^^^
                       feature_id                         provider

Every GPT cell exercises the three GPT-5.6 tiers; the cell only goes
green if all three pass. Cells are opt-in via COMPAT_GPT_CELLS=1 (see
`claude_code._gpt_cells`).
"""

from __future__ import annotations

from claude_code._basic_messaging import run_basic_messaging_cell
from claude_code._gpt_cells import skip_unless_gpt_cells_enabled

OPENAI_MODELS = [
    "gpt-5-6-sol-openai",
    "gpt-5-6-terra-openai",
    "gpt-5-6-luna-openai",
]


def test_basic_messaging_non_streaming_openai(compat_result):
    """Drive the `claude` CLI against the LiteLLM proxy and assert a
    non-empty reply from each GPT-5.6 tier."""
    skip_unless_gpt_cells_enabled()
    run_basic_messaging_cell(
        compat_result=compat_result,
        models=OPENAI_MODELS,
        prompt="Reply with the single word 'pong' and nothing else.",
    )
