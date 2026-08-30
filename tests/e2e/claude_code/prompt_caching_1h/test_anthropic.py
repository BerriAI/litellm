"""prompt_caching_1h x Anthropic.

Drive the real `claude` CLI against a running LiteLLM proxy that routes
to Anthropic, opt into the 1-hour cache TTL via `ENABLE_PROMPT_CACHING_1H`,
and assert that the upstream's usage block reports either
`cache_creation_input_tokens` or `cache_read_input_tokens` > 0 — i.e.
the proxy preserved Claude Code's `cache_control: { ttl: "1h" }`
annotations end-to-end and the upstream actually honored them.

This is the 1-hour-TTL companion to `prompt_caching_5m/`. It exists as
its own cell because the 1h TTL travels through the proxy with a
distinct `cache_control` shape (and a distinct beta-header gate on
some providers); a regression that strips or downgrades the TTL on the
way through is invisible to the 5m cell, which would still see cache
hits with a default-TTL annotation.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/prompt_caching_1h/test_anthropic.py
                       ^^^^^^^^^^^^^^^^^      ^^^^^^^^^
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


ANTHROPIC_MODELS = [
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-opus-4-7",
]

# Per the changelog (2.1.108): `ENABLE_PROMPT_CACHING_1H` flips Claude
# Code from the default 5-minute cache TTL to a 1-hour TTL on the
# `cache_control` annotations it adds to the system prompt and the
# most recent user turn. Setting it here is what we are validating
# the proxy faithfully forwards to the upstream.
CACHE_1H_ENV = {"ENABLE_PROMPT_CACHING_1H": "1"}


def _cache_tokens(usage: Optional[Mapping[str, Any]]) -> int:
    """Return cache_creation_input_tokens + cache_read_input_tokens from
    the upstream usage block, or 0 if the keys are missing."""
    if not isinstance(usage, Mapping):
        return 0
    creation = usage.get("cache_creation_input_tokens") or 0
    read = usage.get("cache_read_input_tokens") or 0
    try:
        return int(creation) + int(read)
    except (TypeError, ValueError):
        return 0


@pytest.mark.covers("llm.messages.anthropic.prompt_cache_1h.nonstream.works")
def test_prompt_caching_1h_anthropic(compat_result):
    """Drive the `claude` CLI against the LiteLLM proxy with the 1h
    TTL opt-in env var set, and assert the upstream usage block
    surfaces a non-zero cache token count."""
    base_url, api_key = require_proxy(compat_result)

    outcomes = run_claude_models_parallel(
        models=ANTHROPIC_MODELS,
        prompt="Reply with the single word 'pong' and nothing else.",
        base_url=base_url,
        api_key=api_key,
        extra_env=CACHE_1H_ENV,
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

        if _cache_tokens(outcome.usage) <= 0:
            error = (
                f"[{model}] usage block reported zero cache tokens with "
                "ENABLE_PROMPT_CACHING_1H=1; the proxy likely stripped the "
                "1h TTL beta header or rejected the cache_control shape"
            )
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        compat_result.add({"status": "pass"})

    if failures:
        pytest.fail("; ".join(failures), pytrace=False)
