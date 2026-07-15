"""prompt_caching_1h x Microsoft Foundry (Azure).

Drive the real `claude` CLI against a running LiteLLM proxy that routes
Claude requests to Microsoft Foundry's Anthropic deployments on Azure,
opt into the 1-hour cache TTL via `ENABLE_PROMPT_CACHING_1H`, and
assert the upstream's usage block reports a non-zero cache token count.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/prompt_caching_1h/test_azure.py
                       ^^^^^^^^^^^^^^^^^      ^^^^^
                       feature_id             provider
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import pytest

from claude_code._env import require_compat_cli_credentials
from claude_code.conftest import _compat_cli_key_provider
from claude_code.cli_driver import (
    ClaudeCLIError,
    failure_diagnostic,
    run_claude_models_parallel,
)


AZURE_MODELS = [
    "claude-haiku-4-5-azure",
    "claude-sonnet-4-6-azure",
    "claude-opus-4-7-azure",
]

CACHE_1H_ENV = {"ENABLE_PROMPT_CACHING_1H": "1"}


def _cache_tokens(usage: Optional[Mapping[str, Any]]) -> int:
    if not isinstance(usage, Mapping):
        return 0
    creation = usage.get("cache_creation_input_tokens") or 0
    read = usage.get("cache_read_input_tokens") or 0
    try:
        return int(creation) + int(read)
    except (TypeError, ValueError):
        return 0


def test_prompt_caching_1h_azure(compat_result):
    base_url, api_key = require_compat_cli_credentials(
        compat_result, cli_key_provider=_compat_cli_key_provider
    )

    outcomes = run_claude_models_parallel(
        models=AZURE_MODELS,
        prompt="Reply with the single word 'pong' and nothing else.",
        base_url=base_url,
        api_key=api_key,
        extra_env=CACHE_1H_ENV,
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
                f"[{model}] usage block reported zero cache tokens with "
                "ENABLE_PROMPT_CACHING_1H=1"
            )
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        compat_result.add({"status": "pass"})

    if failures:
        pytest.fail("; ".join(failures), pytrace=False)
