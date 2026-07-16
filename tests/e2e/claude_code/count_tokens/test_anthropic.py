"""count_tokens x Anthropic.

HTTP-probe row. Unlike the CLI-driven rows, this test never invokes
the `claude` CLI: it `POST`s directly to
`{proxy}/v1/messages/count_tokens` for each Claude tier and asserts
the response is shaped `{"input_tokens": <positive int>}`.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/count_tokens/test_anthropic.py
                       ^^^^^^^^^^^^      ^^^^^^^^^
                       feature_id        provider

Why HTTP probe instead of CLI:

Claude Code calls `count_tokens` internally to compute budget /
context-window usage display, but the result is consumed by the CLI
in-process and never appears in stream-json events. There is no CLI
flag that emits the count to stdout in a way our existing
stream-json parser can pick up, so we can't test the endpoint round
trip through the CLI surface.

The proxy *is* expected to expose `/v1/messages/count_tokens` for
every Claude-style provider it routes to -- LiteLLM has historically
had provider-specific bugs in this endpoint (Vertex AI `count_tokens`
returned 400 to proxy gateways; see Claude Code release notes 2.1.121).
Treating it as a matrix row keeps regressions in the cron's daily
diff.

The cell goes red if *any* tier's probe fails the minimal shape
check; the matrix's per-cell aggregator handles that automatically.
Three tiers run sequentially because count_tokens is cheap (<100ms
per request typical) and the parallelization that matters for the
CLI rows isn't useful here.
"""

from __future__ import annotations

import os

import pytest

from claude_code.http_probe import (
    assert_count_tokens_shape,
    probe_count_tokens,
)

PROXY_BASE_URL_ENV = "LITELLM_PROXY_BASE_URL"
PROXY_API_KEY_ENV = "LITELLM_PROXY_API_KEY"

ANTHROPIC_MODELS = [
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
]


@pytest.mark.covers("llm.messages.anthropic.count_tokens.nonstream.works")
def test_count_tokens_anthropic(compat_result):
    """Probe `/v1/messages/count_tokens` for each Anthropic tier and
    assert the response shape."""
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

    failures = []
    for model in ANTHROPIC_MODELS:
        result = probe_count_tokens(
            base_url=base_url, api_key=api_key, model=model
        )
        shape_error = assert_count_tokens_shape(result)
        if shape_error is not None:
            error = f"[{model}] count_tokens probe failed: {shape_error}"
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        compat_result.add({"status": "pass"})

    if failures:
        pytest.fail("; ".join(failures), pytrace=False)
