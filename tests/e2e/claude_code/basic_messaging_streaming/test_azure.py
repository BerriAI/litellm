"""basic_messaging_streaming x Azure (Microsoft Foundry).

Drive the real `claude` CLI in headless `--output-format stream-json`
mode against a running LiteLLM proxy that routes Claude requests to
Anthropic's models hosted in Microsoft Foundry on Azure, and report the
outcome via `compat_result`.

Foundry exposes Claude on an Anthropic-shape `/anthropic/v1/messages`
endpoint with native SSE streaming; LiteLLM forwards stream events
through the `azure_ai/claude-*` provider unchanged.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/basic_messaging_streaming/test_azure.py
                       ^^^^^^^^^^^^^^^^^^^^^^^^^      ^^^^^
                       feature_id                     provider
"""

from __future__ import annotations

import pytest
from claude_code._basic_messaging import run_basic_messaging_cell

AZURE_MODELS = [
    "claude-haiku-4-5-azure",
    "claude-sonnet-4-5-azure",
    "claude-opus-4-7-azure",
]


@pytest.mark.covers("llm.messages.azure_foundry.basic.stream.works")
def test_basic_messaging_streaming_azure(compat_result):
    """Drive the `claude` CLI against the LiteLLM proxy and assert a
    non-empty streamed reply (one row per Claude tier).
    """
    run_basic_messaging_cell(
        compat_result=compat_result,
        models=AZURE_MODELS,
        prompt="Count from 1 to 5, one number per line.",
        verify_streaming=True,
    )
