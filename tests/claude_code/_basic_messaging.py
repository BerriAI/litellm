"""Shared body for the `basic_messaging_*` × <provider> compat cells.

Every basic_messaging cell follows the same skeleton:

  1. Read the proxy base URL + API key from env, fail-early if missing.
  2. Fan the three Claude tiers out via `run_claude_models_parallel`.
  3. Inspect each model's outcome and report one `compat_result` row per
     model — `ClaudeCLIError`, non-zero exit, missing stream events
     (streaming variant only), and empty assistant text are all per-model
     fails; everything else is a per-model pass.
  4. Surface a joined failure message via `pytest.fail(...)` so the
     pytest run also goes red.

The conftest infers `(feature_id, provider)` purely from the test file
path, so each per-provider file just declares its model list and calls
`run_basic_messaging_cell(...)`. This keeps all cell logic in one place
— a future tweak to the env-missing guard, the failure-loop shape, or
the stream-events check now propagates to every cell automatically.

The leading underscore in the filename is what keeps pytest from
collecting this module as a test file.
"""

from __future__ import annotations

import os
from typing import Sequence

import pytest

from tests.claude_code.cli_driver import (
    ClaudeCLIError,
    failure_diagnostic,
    run_claude_models_parallel,
)

PROXY_BASE_URL_ENV = "LITELLM_PROXY_BASE_URL"
PROXY_API_KEY_ENV = "LITELLM_PROXY_API_KEY"


def run_basic_messaging_cell(
    *,
    compat_result,
    models: Sequence[str],
    prompt: str,
    require_stream_events: bool = False,
) -> None:
    """Run the shared `basic_messaging_*` × <provider> cell body.

    `require_stream_events=True` adds the streaming-variant assertion
    that at least one stream-json event is observed for each model —
    the regression check that catches a proxy buffering the full
    response before flushing.
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
            f"{PROXY_BASE_URL_ENV} / {PROXY_API_KEY_ENV} not configured",
            pytrace=False,
        )

    outcomes = run_claude_models_parallel(
        models=models,
        prompt=prompt,
        base_url=base_url,
        api_key=api_key,
    )

    failures = []
    for model in models:
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

        if require_stream_events and not outcome.events:
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
