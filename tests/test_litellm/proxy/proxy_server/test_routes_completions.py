"""Behavior pins for ``proxy_server.py`` text-completions routes.

Pins (PR2):
    - POST /v1/completions
    - POST /completions
    - POST /engines/{model:path}/completions
    - POST /openai/deployments/{model:path}/completions
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy import common_request_processing, proxy_server

from .conftest import normalize  # type: ignore[import-not-found]

HAPPY_RESPONSE = {
    "id": "cmpl-test",
    "object": "text_completion",
    "created": 0,
    "model": "gpt-3.5-turbo-instruct",
    "choices": [
        {
            "index": 0,
            "text": "Hello from mock",
            "finish_reason": "stop",
            "logprobs": None,
        }
    ],
    "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
}


@pytest.fixture
def patched_completion(monkeypatch):
    monkeypatch.setattr(proxy_server, "llm_router", MagicMock())
    monkeypatch.setattr(
        proxy_server, "proxy_logging_obj", MagicMock(post_call_failure_hook=AsyncMock())
    )

    async def _fake_process(self, *args, **kwargs):
        return dict(HAPPY_RESPONSE)

    monkeypatch.setattr(
        common_request_processing.ProxyBaseLLMRequestProcessing,
        "base_process_llm_request",
        _fake_process,
    )
    yield


@pytest.fixture
def completion_pipeline_raises(monkeypatch):
    monkeypatch.setattr(proxy_server, "llm_router", MagicMock())
    monkeypatch.setattr(
        proxy_server, "proxy_logging_obj", MagicMock(post_call_failure_hook=AsyncMock())
    )

    async def _raise(self, *args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(
        common_request_processing.ProxyBaseLLMRequestProcessing,
        "base_process_llm_request",
        _raise,
    )
    yield


_COMPLETION_PATHS = [
    "/v1/completions",
    "/completions",
    "/engines/gpt-3.5-turbo-instruct/completions",
    "/openai/deployments/gpt-3.5-turbo-instruct/completions",
]


@pytest.mark.parametrize("path", _COMPLETION_PATHS)
def test_completion_happy_path(client, auth_as, patched_completion, path):
    """Pins all four ``POST .../completions`` aliases (happy path).

    Covers ``POST /v1/completions``, ``POST /completions``,
    ``POST /engines/{model:path}/completions``, and
    ``POST /openai/deployments/{model:path}/completions``.
    """
    payload = {
        "model": "gpt-3.5-turbo-instruct",
        "prompt": "Once upon",
        "max_tokens": 5,
    }
    with auth_as():
        response = client.post(path, json=payload)
    assert response.status_code == 200
    assert normalize(response.json()) == {
        "id": "<VOLATILE>",
        "object": "text_completion",
        "created": "<VOLATILE>",
        "model": "gpt-3.5-turbo-instruct",
        "choices": [
            {
                "index": 0,
                "text": "Hello from mock",
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    }


@pytest.mark.parametrize("path", _COMPLETION_PATHS)
def test_completion_pipeline_error(client, auth_as, completion_pipeline_raises, path):
    """Pins all four ``POST .../completions`` aliases (error path).

    Covers ``POST /v1/completions``, ``POST /completions``,
    ``POST /engines/{model:path}/completions``, and
    ``POST /openai/deployments/{model:path}/completions``.
    """
    payload = {"model": "gpt-3.5-turbo-instruct", "prompt": "boom"}
    with auth_as():
        response = client.post(path, json=payload)
    assert response.status_code == 500
    assert response.headers.get("content-type", "").startswith("application/json")
