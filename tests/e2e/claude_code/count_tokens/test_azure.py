"""count_tokens x Azure (Microsoft Foundry).

HTTP-probe row. Unlike the CLI-driven rows, this test never invokes
the `claude` CLI: it `POST`s directly to
`{proxy}/v1/messages/count_tokens` for each Claude tier and asserts
the response is shaped `{"input_tokens": <positive int>}`.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/count_tokens/test_azure.py
                       ^^^^^^^^^^^^      ^^^^^
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
The three tiers are probed concurrently via `run_probe_cell`: each
probe first waits on the shared per-provider token bucket, so a
sequential cell paid that wait three times before reporting.
"""

from __future__ import annotations

import pytest

from claude_code._env import require_proxy_client
from claude_code._probe_cell import run_probe_cell
from claude_code.http_probe import (
    assert_count_tokens_shape,
    probe_count_tokens,
)


AZURE_MODELS = [
    "claude-haiku-4-5-azure",
    "claude-sonnet-4-5-azure",
    "claude-opus-4-7-azure",
]


@pytest.mark.covers("llm.messages.azure_foundry.count_tokens.nonstream.works")
def test_count_tokens_azure(compat_result):
    """Probe `/v1/messages/count_tokens` for each Azure (Microsoft Foundry) tier and
    assert the response shape."""
    client, api_key = require_proxy_client(compat_result)

    run_probe_cell(
        compat_result=compat_result,
        models=AZURE_MODELS,
        probe=lambda model: probe_count_tokens(
            client=client, api_key=api_key, model=model
        ),
        check_shape=assert_count_tokens_shape,
        probe_name="count_tokens",
    )
