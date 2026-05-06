"""basic_messaging_streaming x Anthropic.

Drive the real `claude` CLI in headless `--output-format stream-json`
mode against a running LiteLLM proxy that routes to Anthropic, and
report the outcome via `compat_result`.

The CLI is run with `--print --output-format stream-json`, which streams
incremental events as the upstream produces tokens. The cell goes green
only when every Claude tier returns a non-empty reply over a streamed
wire (i.e. at least one stream-json event is observed). This catches
regressions where the proxy buffers the full response before flushing,
silently degrading the streaming experience customers rely on.

The (feature, provider) for this cell is inferred from the file path by
`tests/claude_code/conftest.py`:

    tests/claude_code/basic_messaging_streaming/test_anthropic.py
                       ^^^^^^^^^^^^^^^^^^^^^^^^^      ^^^^^^^^^
                       feature_id                     provider

The three Claude tiers run in parallel inside this single test, with
one `compat_result.add(...)` entry per model so the matrix builder
still sees three rows for this (feature, provider).
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

ANTHROPIC_MODELS = [
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
]


def test_basic_messaging_streaming_anthropic(compat_result):
    """Drive the `claude` CLI against the LiteLLM proxy and assert a
    non-empty streamed reply (at least one stream-json event observed).
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
        prompt="Count from 1 to 5, one number per line.",
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

        if not outcome.events:
            error = f"[{model}] no stream-json events emitted; streaming wire silent"
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
