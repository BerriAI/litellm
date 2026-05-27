"""LIT-1756 regression test.

Asserts that ``litellm.aresponses`` populates
``response._hidden_params["litellm_overhead_time_ms"]`` - the field that
``ProxyBaseLLMRequestProcessing.get_custom_headers`` reads to emit the
``x-litellm-overhead-duration-ms`` HTTP response header on ``/v1/responses``.

Before the fix, the responses API handler called ``AsyncHTTPHandler.post(...)``
without passing ``logging_obj=``, so the ``@track_llm_api_timing`` decorator
could not record ``llm_api_duration_ms``, and ``litellm_overhead_time_ms``
stayed ``None``.
"""
import asyncio
import os
import time
from unittest.mock import patch

import httpx
import pytest

import litellm


def _fake_response_body():
    return {
        "id": "resp_test_001",
        "object": "response",
        "created_at": int(time.time()),
        "model": "gpt-4o",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "m1",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": "Hello", "annotations": []}
                ],
            }
        ],
        "usage": {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10},
    }


@pytest.mark.asyncio
async def test_aresponses_populates_litellm_overhead_time_ms():
    """Driving ``litellm.aresponses`` with a stubbed httpx send must produce
    a response whose ``_hidden_params`` contains a numeric
    ``litellm_overhead_time_ms``.

    This is the field that the proxy reads to emit
    ``x-litellm-overhead-duration-ms``.  Regresses LIT-1756.
    """
    os.environ.setdefault("OPENAI_API_KEY", "sk-fake")

    async def fake_send(self, request, **_kwargs):
        # Simulate a small but measurable upstream latency so that
        # `_response_ms - llm_api_duration_ms` produces a non-trivial overhead.
        await asyncio.sleep(0.05)
        return httpx.Response(
            200, json=_fake_response_body(), request=request
        )

    with patch.object(httpx.AsyncClient, "send", fake_send):
        response = await litellm.aresponses(model="openai/gpt-4o", input="hi")

    hidden = getattr(response, "_hidden_params", None) or {}
    assert (
        "litellm_overhead_time_ms" in hidden
    ), f"litellm_overhead_time_ms missing; hidden keys={sorted(hidden.keys())}"
    overhead = hidden["litellm_overhead_time_ms"]
    assert isinstance(
        overhead, (int, float)
    ) and overhead >= 0, f"litellm_overhead_time_ms not a non-negative number: {overhead!r}"
