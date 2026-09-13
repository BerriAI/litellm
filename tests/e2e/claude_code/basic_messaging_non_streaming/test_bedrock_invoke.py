"""basic_messaging_non_streaming x Bedrock (Invoke).

Drive the real `claude` CLI in headless mode against a running LiteLLM
proxy that routes Claude requests to AWS Bedrock via the legacy
`InvokeModel` API path, and report the outcome via `compat_result`.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/basic_messaging_non_streaming/test_bedrock_invoke.py
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^      ^^^^^^^^^^^^^^
                       feature_id                          provider

Per the PRD, every cell exercises Claude Haiku 4.5, Sonnet 4.6, and Opus
4.7; the cell only goes green if all three pass. The shared
`run_basic_messaging_cell` helper fans the three model runs out in
parallel and reports one `compat_result.add(...)` entry per model so
the matrix builder still sees three rows for this (feature, provider).
"""

from __future__ import annotations

import pytest
from claude_code._basic_messaging import run_basic_messaging_cell

# Per-model aliases registered in the LiteLLM proxy's routing config to
# point at Bedrock's legacy InvokeModel endpoint. The driver only sends
# the alias; the proxy is the one that knows the upstream model id and
# routing strategy.
BEDROCK_INVOKE_MODELS = [
    "claude-haiku-4-5-bedrock-invoke",
    "claude-sonnet-4-5-bedrock-invoke",
    "claude-opus-4-7-bedrock-invoke",
]


@pytest.mark.covers("llm.messages.bedrock_invoke.basic.nonstream.works")
def test_basic_messaging_non_streaming_bedrock_invoke(compat_result):
    """Drive the `claude` CLI against the LiteLLM proxy and assert a reply."""
    run_basic_messaging_cell(
        compat_result=compat_result,
        models=BEDROCK_INVOKE_MODELS,
        prompt="Reply with the single word 'pong' and nothing else.",
    )
