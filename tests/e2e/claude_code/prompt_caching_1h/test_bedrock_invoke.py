"""prompt_caching_1h x Bedrock (Invoke).

Drive the real `claude` CLI against a running LiteLLM proxy that routes
Claude requests to AWS Bedrock via the legacy `InvokeModel` API path,
opt into the 1-hour cache TTL via `ENABLE_PROMPT_CACHING_1H`, and
assert the upstream's usage block reports a non-zero cache token count.

Bedrock historically gated 1h prompt caching behind a separate
`ENABLE_PROMPT_CACHING_1H_BEDROCK` env var (see 2.1.108: deprecated but
still honored). The proxy must accept either env var and forward an
appropriate `cache_control` shape to the Bedrock InvokeModel endpoint;
this cell catches regressions where the TTL is silently downgraded to
5 minutes on the way through.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/prompt_caching_1h/test_bedrock_invoke.py
                       ^^^^^^^^^^^^^^^^^      ^^^^^^^^^^^^^^
                       feature_id             provider
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import pytest

from claude_code._env import require_proxy
from claude_code.cli_driver import (
    ClaudeCLIError,
    failure_diagnostic,
    run_claude_models_parallel,
)


BEDROCK_INVOKE_MODELS = [
    "claude-haiku-4-5-bedrock-invoke",
    "claude-sonnet-4-5-bedrock-invoke",
    "claude-opus-4-7-bedrock-invoke",
]

# Set both the modern and the deprecated-but-honored Bedrock var so we
# match whichever code path the proxy is following.
CACHE_1H_ENV = {
    "ENABLE_PROMPT_CACHING_1H": "1",
    "ENABLE_PROMPT_CACHING_1H_BEDROCK": "1",
}


def _cache_tokens(usage: Optional[Mapping[str, Any]]) -> int:
    if not isinstance(usage, Mapping):
        return 0
    creation = usage.get("cache_creation_input_tokens") or 0
    read = usage.get("cache_read_input_tokens") or 0
    try:
        return int(creation) + int(read)
    except (TypeError, ValueError):
        return 0


def test_prompt_caching_1h_bedrock_invoke(compat_result):
    base_url, api_key = require_proxy(compat_result)

    outcomes = run_claude_models_parallel(
        models=BEDROCK_INVOKE_MODELS,
        prompt="Reply with the single word 'pong' and nothing else.",
        base_url=base_url,
        api_key=api_key,
        extra_env=CACHE_1H_ENV,
    )

    failures = []
    for model in BEDROCK_INVOKE_MODELS:
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
                "1h-TTL opt-in env vars set; the proxy likely stripped or "
                "downgraded the cache_control TTL"
            )
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        compat_result.add({"status": "pass"})

    if failures:
        pytest.fail("; ".join(failures), pytrace=False)
