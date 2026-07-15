"""basic_messaging_streaming x AWS Bedrock Mantle (GPT-5.6).

Drive the real `claude` CLI in headless `--output-format stream-json`
mode against a running LiteLLM proxy that routes Anthropic Messages
requests to OpenAI's GPT-5.6 family (Sol, Terra, Luna) on AWS
Bedrock's Mantle endpoint, and report the outcome via `compat_result`.

Mantle streams OpenAI Responses API events over SigV4-signed SSE;
LiteLLM re-emits them as Anthropic stream events, and the
`verify_streaming=True` assertion (via `--include-partial-messages`)
proves the events arrived incrementally rather than as one buffered
response.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/basic_messaging_streaming/test_bedrock_mantle.py
                       ^^^^^^^^^^^^^^^^^^^^^^^^^      ^^^^^^^^^^^^^^
                       feature_id                     provider

Every GPT cell exercises the three GPT-5.6 tiers; the cell only goes
green if all three pass. Cells are opt-in via COMPAT_GPT_CELLS=1 (see
`claude_code._gpt_cells`).
"""

from __future__ import annotations

from claude_code._basic_messaging import run_basic_messaging_cell
from claude_code._gpt_cells import skip_unless_gpt_cells_enabled

BEDROCK_MANTLE_MODELS = [
    "gpt-5-6-sol-bedrock-mantle",
    "gpt-5-6-terra-bedrock-mantle",
    "gpt-5-6-luna-bedrock-mantle",
]


def test_basic_messaging_streaming_bedrock_mantle(compat_result):
    """Drive the `claude` CLI against the LiteLLM proxy and assert a
    non-empty streamed reply from each GPT-5.6 tier."""
    skip_unless_gpt_cells_enabled()
    run_basic_messaging_cell(
        compat_result=compat_result,
        models=BEDROCK_MANTLE_MODELS,
        prompt="Count from 1 to 5, one number per line.",
        verify_streaming=True,
    )
