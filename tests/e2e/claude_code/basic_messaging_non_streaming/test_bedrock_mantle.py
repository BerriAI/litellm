"""basic_messaging_non_streaming x AWS Bedrock Mantle (GPT-5.6).

Drive the real `claude` CLI in headless mode against a running LiteLLM
proxy that routes Anthropic Messages requests to OpenAI's GPT-5.6
family (Sol, Terra, Luna) hosted on AWS Bedrock, and report the
outcome via `compat_result`.

Bedrock exposes the GPT-5.6 models through the Mantle endpoint, which
speaks the OpenAI Responses API rather than Converse/Invoke; LiteLLM's
`bedrock_mantle/openai.gpt-*` route signs the request with SigV4 and
translates Anthropic Messages to Responses, so this cell exercises a
translation path no other column covers.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/basic_messaging_non_streaming/test_bedrock_mantle.py
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^      ^^^^^^^^^^^^^^
                       feature_id                         provider

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


def test_basic_messaging_non_streaming_bedrock_mantle(compat_result):
    """Drive the `claude` CLI against the LiteLLM proxy and assert a
    non-empty reply from each GPT-5.6 tier."""
    skip_unless_gpt_cells_enabled()
    run_basic_messaging_cell(
        compat_result=compat_result,
        models=BEDROCK_MANTLE_MODELS,
        prompt="Reply with the single word 'pong' and nothing else.",
    )
