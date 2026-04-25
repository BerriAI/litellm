"""basic_messaging_non_streaming x Bedrock (Invoke).

Drive the real `claude` CLI in headless mode against a running LiteLLM
proxy that routes Claude requests to AWS Bedrock via the legacy
`InvokeModel` API path, and report the outcome via `compat_result`.

The (feature, provider) for this cell is inferred from the file path by
`tests/claude_code/conftest.py`:

    tests/claude_code/basic_messaging_non_streaming/test_bedrock_invoke.py
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^      ^^^^^^^^^^^^^^
                       feature_id                          provider

Per the PRD, every cell exercises Claude Haiku 4.5, Sonnet 4.6, and Opus
4.7; the cell only goes green if all three pass. We parametrize over the
three models and the conftest aggregator produces one cell from the
three results, naming the failing model in the error string when any
model fails.
"""

from __future__ import annotations

import os

import pytest

from tests.claude_code.cli_driver import ClaudeCLIError, run_claude

PROXY_BASE_URL_ENV = "LITELLM_PROXY_BASE_URL"
PROXY_API_KEY_ENV = "LITELLM_PROXY_API_KEY"

# Per-model aliases registered in the LiteLLM proxy's routing config to
# point at Bedrock's legacy InvokeModel endpoint. The driver only sends
# the alias; the proxy is the one that knows the upstream model id and
# routing strategy.
BEDROCK_INVOKE_MODELS = [
    "claude-haiku-4-5-bedrock-invoke",
    "claude-sonnet-4-6-bedrock-invoke",
    "claude-opus-4-7-bedrock-invoke",
]


@pytest.mark.parametrize("model", BEDROCK_INVOKE_MODELS)
def test_basic_messaging_non_streaming_bedrock_invoke(compat_result, model):
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
                "error": f"[{model}] claude CLI exited {result.exit_code}: {result.stderr.strip()}",
            }
        )
        pytest.fail(f"claude CLI exited {result.exit_code} for {model}", pytrace=False)
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
