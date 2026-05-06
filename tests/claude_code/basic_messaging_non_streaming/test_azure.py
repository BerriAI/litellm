"""basic_messaging_non_streaming x Azure (Microsoft Foundry).

Drive the real `claude` CLI in headless mode against a running LiteLLM
proxy that routes Claude requests to Anthropic's models hosted in
Microsoft Foundry on Azure, and report the outcome via `compat_result`.

Anthropic announced Claude Haiku 4.5, Sonnet 4.5/4.6, and Opus 4.1/4.6/4.7
in Microsoft Foundry on 2025-11-18; LiteLLM exposes them via the
`azure_ai/claude-*` provider prefix, which talks to Foundry's
Anthropic-shape `/anthropic/v1/messages` endpoint.

The (feature, provider) for this cell is inferred from the file path by
`tests/claude_code/conftest.py`:

    tests/claude_code/basic_messaging_non_streaming/test_azure.py
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^      ^^^^^
                       feature_id                          provider

Per the PRD, every cell exercises Claude Haiku 4.5, Sonnet 4.6, and Opus
4.7; the cell only goes green if all three pass. We fan the three model
runs out in parallel inside this single test and report one
`compat_result.add(...)` entry per model so the matrix builder still sees
three rows for this (feature, provider).
"""

from __future__ import annotations

import os

import pytest

from tests.claude_code.cli_driver import (
    ClaudeCLIError,
    failure_diagnostic,
    run_claude_models_parallel,
)

PROXY_BASE_URL_ENV = "LITELLM_PROXY_BASE_URL"
PROXY_API_KEY_ENV = "LITELLM_PROXY_API_KEY"

# Per-model aliases registered in the LiteLLM proxy's routing config to
# point at Microsoft Foundry's Anthropic deployments. The driver only
# sends the alias; the proxy is the one that knows the upstream Foundry
# resource URL and API key.
AZURE_MODELS = [
    "claude-haiku-4-5-azure",
    "claude-sonnet-4-6-azure",
    "claude-opus-4-7-azure",
]


def test_basic_messaging_non_streaming_azure(compat_result):
    """Drive the `claude` CLI against the LiteLLM proxy and assert a reply.

    "Basic messaging" means: send a single user prompt, receive any
    non-empty assistant text reply, no tools, no streaming, no thinking.
    The whole point of this slice is to prove the path works at all —
    so the assertion is intentionally lenient on the reply contents.
    """
    base_url = os.environ.get(PROXY_BASE_URL_ENV)
    api_key = os.environ.get(PROXY_API_KEY_ENV)
    if not base_url or not api_key:
        compat_result.set(
            {
                "status": "fail",
                "error": (
                    f"missing required env: set {PROXY_BASE_URL_ENV} and "
                    f"{PROXY_API_KEY_ENV} to point at a running LiteLLM proxy"
                ),
            }
        )
        pytest.fail(
            f"{PROXY_BASE_URL_ENV} / {PROXY_API_KEY_ENV} not configured", pytrace=False
        )

    outcomes = run_claude_models_parallel(
        models=AZURE_MODELS,
        prompt="Reply with the single word 'pong' and nothing else.",
        base_url=base_url,
        api_key=api_key,
    )

    failures = []
    for model in AZURE_MODELS:
        outcome = outcomes[model]
        if isinstance(outcome, ClaudeCLIError):
            error = f"[{model}] {outcome}"
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        if outcome.exit_code != 0:
            error = f"[{model}] claude CLI failed: {failure_diagnostic(outcome)}"
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        if not outcome.text.strip():
            error = f"[{model}] claude returned empty assistant text"
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        compat_result.add({"status": "pass"})

    if failures:
        pytest.fail("; ".join(failures), pytrace=False)
