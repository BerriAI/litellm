"""prompt_caching_5m x Azure (Microsoft Foundry).

Drive the real `claude` CLI against a running LiteLLM proxy that routes
Claude requests to Anthropic's models hosted in Microsoft Foundry on
Azure, and assert that the upstream's usage block reports either
`cache_creation_input_tokens` or `cache_read_input_tokens` > 0.

Foundry's Anthropic deployments honor the default 5-minute ephemeral
`cache_control` exactly like anthropic.com. The 1-hour `scope: "global"`
variant is *not* supported on Foundry — LiteLLM strips that field
before forwarding (see `_remove_scope_from_cache_control` in
`litellm/llms/azure_ai/anthropic/messages_transformation.py`) — but
this row exercises the 5-minute TTL only, so that quirk does not apply.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/prompt_caching_5m/test_azure.py
                       ^^^^^^^^^^^^^^^^^      ^^^^^
                       feature_id             provider
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional

import pytest

from claude_code.cli_driver import (
    ClaudeCLIError,
    failure_diagnostic,
    run_claude_models_parallel,
)

PROXY_BASE_URL_ENV = "LITELLM_PROXY_BASE_URL"
PROXY_API_KEY_ENV = "LITELLM_PROXY_API_KEY"

AZURE_MODELS = [
    "claude-haiku-4-5-azure",
    "claude-sonnet-4-6-azure",
    "claude-opus-4-8-azure",
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


def test_prompt_caching_5m_azure(compat_result):
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

        if _cache_tokens(outcome.usage) <= 0:
            error = (
                f"[{model}] usage block reported zero cache tokens; "
                "expected cache_control on the system prompt to produce a non-zero "
                "cache_creation_input_tokens or cache_read_input_tokens"
            )
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        compat_result.add({"status": "pass"})

    if failures:
        pytest.fail("; ".join(failures), pytrace=False)
