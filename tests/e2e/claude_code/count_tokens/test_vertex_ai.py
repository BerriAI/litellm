"""count_tokens x Vertex AI.

HTTP-probe row. Unlike the CLI-driven rows, this test never invokes
the `claude` CLI: it `POST`s directly to
`{proxy}/v1/messages/count_tokens` for each Claude tier and asserts
the response is shaped `{"input_tokens": <positive int>}`.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/count_tokens/test_vertex_ai.py
                       ^^^^^^^^^^^^      ^^^^^^^^^
                       feature_id        provider

Vertex Claude currently returns 400 "is not supported for token counting"
for several tiers (haiku/sonnet at least). Those rows are recorded as
`not_applicable` so the matrix does not treat an upstream capability gap
as a LiteLLM regression. A real proxy transform bug (5xx, wrong shape)
still fails the cell.
"""

from __future__ import annotations

import pytest

from claude_code._env import require_proxy
from claude_code.http_probe import (
    assert_count_tokens_shape,
    probe_count_tokens,
)


VERTEX_AI_MODELS = [
    "claude-haiku-4-5-vertex",
    "claude-sonnet-4-5-vertex",
    "claude-opus-4-7-vertex",
]


def _is_upstream_token_count_unsupported(error: str) -> bool:
    lowered = error.lower()
    return "not supported for token counting" in lowered


@pytest.mark.covers("llm.messages.vertex.count_tokens.nonstream.works")
def test_count_tokens_vertex_ai(compat_result):
    """Probe `/v1/messages/count_tokens` for each Vertex AI tier and
    assert the response shape."""
    base_url, api_key = require_proxy(compat_result)

    failures = []
    for model in VERTEX_AI_MODELS:
        result = probe_count_tokens(
            base_url=base_url, api_key=api_key, model=model
        )
        shape_error = assert_count_tokens_shape(result)
        if shape_error is not None:
            if _is_upstream_token_count_unsupported(shape_error):
                compat_result.add(
                    {
                        "status": "not_applicable",
                        "reason": f"[{model}] Vertex does not support count_tokens: {shape_error}",
                    }
                )
                continue
            error = f"[{model}] count_tokens probe failed: {shape_error}"
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        compat_result.add({"status": "pass"})

    if failures:
        pytest.fail("; ".join(failures), pytrace=False)
