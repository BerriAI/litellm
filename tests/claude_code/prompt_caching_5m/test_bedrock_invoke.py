"""prompt_caching_5m x Bedrock (Invoke).

Drive the real `claude` CLI against a running LiteLLM proxy that routes
Claude requests to AWS Bedrock via the legacy `InvokeModel` API path,
and assert that the upstream's usage block reports either
`cache_creation_input_tokens` or `cache_read_input_tokens` > 0.

The (feature, provider) for this cell is inferred from the file path by
`tests/claude_code/conftest.py`:

    tests/claude_code/prompt_caching_5m/test_bedrock_invoke.py
                       ^^^^^^^^^^^^^^^^^      ^^^^^^^^^^^^^^
                       feature_id             provider
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional

import pytest

from tests.claude_code.cli_driver import ClaudeCLIError, failure_diagnostic, run_claude

PROXY_BASE_URL_ENV = "LITELLM_PROXY_BASE_URL"
PROXY_API_KEY_ENV = "LITELLM_PROXY_API_KEY"

BEDROCK_INVOKE_MODELS = [
    "claude-haiku-4-5-bedrock-invoke",
    "claude-sonnet-4-6-bedrock-invoke",
    "claude-opus-4-7-bedrock-invoke",
]


def _cache_tokens(usage: Optional[Mapping[str, Any]]) -> int:
    if not isinstance(usage, Mapping):
        return 0
    creation = usage.get("cache_creation_input_tokens") or 0
    read = usage.get("cache_read_input_tokens") or 0
    try:
        return int(creation) + int(read)
    except (TypeError, ValueError):
        return 0


@pytest.mark.parametrize("model", BEDROCK_INVOKE_MODELS)
def test_prompt_caching_5m_bedrock_invoke(compat_result, model):
    """Drive the `claude` CLI against the LiteLLM proxy and assert the
    upstream usage block surfaces a non-zero cache token count."""
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
                "error": f"[{model}] claude CLI failed: {failure_diagnostic(result)}",
            }
        )
        pytest.fail(
            f"[{model}] claude CLI failed: {failure_diagnostic(result)}", pytrace=False
        )
        return

    if _cache_tokens(result.usage) <= 0:
        compat_result.set(
            {
                "status": "fail",
                "error": (
                    f"[{model}] usage block reported zero cache tokens; "
                    "expected cache_control on the system prompt to produce a non-zero "
                    "cache_creation_input_tokens or cache_read_input_tokens"
                ),
            }
        )
        pytest.fail(f"no cache tokens for {model}", pytrace=False)
        return

    compat_result.set({"status": "pass"})
