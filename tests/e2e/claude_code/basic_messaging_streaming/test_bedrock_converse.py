"""basic_messaging_streaming x Bedrock (Converse).

Drive the real `claude` CLI in headless `--output-format stream-json`
mode against a running LiteLLM proxy that routes Claude requests to AWS
Bedrock via the unified `Converse` API path, and report the outcome via
`compat_result`.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/basic_messaging_streaming/test_bedrock_converse.py
                       ^^^^^^^^^^^^^^^^^^^^^^^^^      ^^^^^^^^^^^^^^^^
                       feature_id                     provider
"""

from __future__ import annotations

from claude_code._basic_messaging import run_basic_messaging_cell

BEDROCK_CONVERSE_MODELS = [
    "claude-haiku-4-5-bedrock-converse",
    "claude-sonnet-4-6-bedrock-converse",
    "claude-opus-4-7-bedrock-converse",
]


def test_basic_messaging_streaming_bedrock_converse(compat_result):
    """Drive the `claude` CLI against the LiteLLM proxy and assert a
    non-empty streamed reply (one row per Claude tier).
    """
    run_basic_messaging_cell(
        compat_result=compat_result,
        models=BEDROCK_CONVERSE_MODELS,
        prompt="Count from 1 to 5, one number per line.",
        verify_streaming=True,
    )
