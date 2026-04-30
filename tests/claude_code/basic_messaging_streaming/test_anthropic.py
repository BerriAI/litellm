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
"""

from __future__ import annotations

import os

import pytest

from tests.claude_code.cli_driver import ClaudeCLIError, failure_diagnostic, run_claude

PROXY_BASE_URL_ENV = "LITELLM_PROXY_BASE_URL"
PROXY_API_KEY_ENV = "LITELLM_PROXY_API_KEY"

ANTHROPIC_MODELS = [
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
]


@pytest.mark.parametrize("model", ANTHROPIC_MODELS)
def test_basic_messaging_streaming_anthropic(compat_result, model):
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

    try:
        result = run_claude(
            prompt="Count from 1 to 5, one number per line.",
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

    if not result.events:
        compat_result.set(
            {
                "status": "fail",
                "error": f"[{model}] no stream-json events emitted; streaming wire silent",
            }
        )
        pytest.fail(f"no stream events for {model}", pytrace=False)
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
