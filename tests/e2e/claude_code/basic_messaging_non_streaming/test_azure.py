"""basic_messaging_non_streaming x Azure (Microsoft Foundry).

Drive the real `claude` CLI in headless mode against a running LiteLLM
proxy that routes Claude requests to Anthropic's models hosted in
Microsoft Foundry on Azure, and report the outcome via `compat_result`.

Anthropic announced Claude Haiku 4.5, Sonnet 4.5/4.6, and Opus 4.1/4.6/4.7
in Microsoft Foundry on 2025-11-18; LiteLLM exposes them via the
`azure_ai/claude-*` provider prefix, which talks to Foundry's
Anthropic-shape `/anthropic/v1/messages` endpoint.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/basic_messaging_non_streaming/test_azure.py
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^      ^^^^^
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
# point at Microsoft Foundry's Anthropic deployments. The driver only
# sends the alias; the proxy is the one that knows the upstream Foundry
# resource URL and API key.
AZURE_MODELS = [
    "claude-haiku-4-5-azure",
    "claude-sonnet-4-6-azure",
    "claude-opus-4-7-azure",
]


@pytest.mark.covers("llm.messages.azure_foundry.basic.nonstream.works")
def test_basic_messaging_non_streaming_azure(compat_result):
    """Drive the `claude` CLI against the LiteLLM proxy and assert a reply.

    "Basic messaging" means: send a single user prompt, receive any
    non-empty assistant text reply, no tools, no streaming, no thinking.
    The whole point of this slice is to prove the path works at all —
    so the assertion is intentionally lenient on the reply contents.
    """
    run_basic_messaging_cell(
        compat_result=compat_result,
        models=AZURE_MODELS,
        prompt="Reply with the single word 'pong' and nothing else.",
    )
