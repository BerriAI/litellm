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
4.7; the cell only goes green if all three pass. We parametrize over the
three models and the conftest aggregator produces one cell from the three
results.
"""

from __future__ import annotations

import os

import pytest

from tests.claude_code.cli_driver import ClaudeCLIError, failure_diagnostic, run_claude

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


@pytest.mark.parametrize("model", ANTHROPIC_MODELS)
def test_basic_messaging_non_streaming_anthropic(compat_result, model):
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

    print(f"base_url: {base_url}")
    print(f"api_key: {api_key}")

    try:
        result = run_claude(
            prompt="Reply with the single word 'pong' and nothing else.",
            model=model,
            base_url=base_url,
            api_key=api_key,
        )
    except ClaudeCLIError as exc:
        compat_result.set({"status": "fail", "error": f"[{model}] {exc}"})
        pytest.fail(str(exc), pytrace=False)
        return

    if result.exit_code != 0:
        compat_result.set(
            {
                "status": "fail",
                "error": f"[{model}] claude CLI failed: {failure_diagnostic(result)}",
            }
        )
        pytest.fail(
            f"[{model}] claude CLI failed: {failure_diagnostic(result)}", pytrace=False
        )
        return

    if not result.text.strip():
        compat_result.set(
            {
                "status": "fail",
                "error": f"[{model}] claude returned empty assistant text",
            }
        )
        pytest.fail(f"empty reply for {model}", pytrace=False)
        return

    compat_result.set({"status": "pass"})
