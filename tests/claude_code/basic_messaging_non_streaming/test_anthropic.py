"""basic_messaging_non_streaming × Anthropic.

The thinnest end-to-end path through every layer of the matrix: drive the
real `claude` CLI in headless mode against a running LiteLLM proxy that
routes to Anthropic, and report the outcome via `compat_result`.

The (feature, provider) for this cell is inferred from the file path by
`tests/claude_code/conftest.py`:

    tests/claude_code/basic_messaging_non_streaming/test_anthropic.py
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^      ^^^^^^^^^
                       feature_id                         provider

Per the PRD, every cell exercises Claude Haiku 4.5, Sonnet 4.6, and Opus
4.7; the cell only goes green if all three pass. We fan the three model
runs out in parallel inside this single test so the per-cell wall time is
bounded by the slowest model rather than the sum, and report one
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

# Per the PRD: each cell is exercised against three Claude tiers via the
# Anthropic provider. Aliases are configured in the LiteLLM proxy's
# routing config; the driver only sends the alias.
ANTHROPIC_MODELS = [
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
]


def test_basic_messaging_non_streaming_anthropic(compat_result):
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
        models=ANTHROPIC_MODELS,
        prompt="Reply with the single word 'pong' and nothing else.",
        base_url=base_url,
        api_key=api_key,
    )

    failures = []
    for model in ANTHROPIC_MODELS:
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
