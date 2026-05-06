"""basic_messaging_non_streaming x Bedrock (Converse).

Drive the real `claude` CLI in headless mode against a running LiteLLM
proxy that routes Claude requests to AWS Bedrock via the unified
`Converse` API path, and report the outcome via `compat_result`.

The (feature, provider) for this cell is inferred from the file path by
`tests/claude_code/conftest.py`:

    tests/claude_code/basic_messaging_non_streaming/test_bedrock_converse.py
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^      ^^^^^^^^^^^^^^^^
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
# point at Bedrock's Converse endpoint. The driver only sends the alias;
# the proxy is the one that knows the upstream model id and routing
# strategy.
BEDROCK_CONVERSE_MODELS = [
    "claude-haiku-4-5-bedrock-converse",
    "claude-sonnet-4-6-bedrock-converse",
    "claude-opus-4-7-bedrock-converse",
]


def test_basic_messaging_non_streaming_bedrock_converse(compat_result):
    """Drive the `claude` CLI against the LiteLLM proxy and assert a reply."""
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
        models=BEDROCK_CONVERSE_MODELS,
        prompt="Reply with the single word 'pong' and nothing else.",
        base_url=base_url,
        api_key=api_key,
    )

    failures = []
    for model in BEDROCK_CONVERSE_MODELS:
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
