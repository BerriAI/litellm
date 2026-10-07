from __future__ import annotations

import asyncio
from typing import Final
from unittest.mock import AsyncMock, Mock

import pytest

import litellm
from litellm import Router
from litellm.llms.custom_httpx.http_handler import HTTPHandler


@pytest.mark.parametrize(
    "num_retries, expected_call_count",
    [(0, 1), (1, 2), (2, 3), (3, 4)],
)
def test_router_timeout_with_retries_anthropic_model(
    num_retries: int,
    expected_call_count: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "num_retries", num_retries)
    monkeypatch.setattr(litellm, "request_timeout", 0.000001)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    model: Final = "claude-haiku-4-5-20251001"
    router: Final = Router(
        model_list=[
            {
                "model_name": "claude-3-haiku",
                "litellm_params": {
                    "model": f"anthropic/{model}",
                    "api_key": "test-key",
                },
            }
        ],
    )
    client: Final = HTTPHandler()
    client_post: Final = Mock(side_effect=TimeoutError("Request timed out."))
    monkeypatch.setattr(client, "post", client_post)

    with pytest.raises(litellm.Timeout):
        router.completion(
            model="claude-3-haiku",
            messages=[{"role": "user", "content": "Router timeout retry test"}],
            client=client,
        )

    assert client_post.call_count == expected_call_count


@pytest.mark.parametrize("stream", [True, False])
def test_unit_test_streaming_timeout(stream: bool) -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "timeout-test-model",
                "litellm_params": {
                    "model": "openai/timeout-test-model",
                    "api_key": "test-key",
                },
            }
        ]
    )
    kwargs: Final = {"stream": stream}
    data: Final = {"timeout": 0.01, "stream_timeout": 0.0000001}

    assert router._get_stream_timeout(kwargs=kwargs, data=data) == 0.0000001
    assert router._get_non_stream_timeout(kwargs=kwargs, data=data) == 0.01
    assert router._get_timeout(kwargs=kwargs, data=data) == (0.0000001 if stream else 0.01)
