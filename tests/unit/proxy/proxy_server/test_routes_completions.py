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
from fastapi import HTTPException

import litellm
from litellm.proxy import common_request_processing, proxy_server
from litellm.types.router import RouterErrors

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
def completion_pipeline_raises(request, monkeypatch):
    failure = getattr(request, "param", ValueError("boom"))
    monkeypatch.setattr(proxy_server, "llm_router", MagicMock())
    monkeypatch.setattr(
        proxy_server, "proxy_logging_obj", MagicMock(post_call_failure_hook=AsyncMock())
    )

    async def _raise(self, *args, **kwargs):
        raise failure

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


_STREAM_CUT_BEFORE_THE_FIRST_CHUNK = litellm.APIConnectionError(
    message=(
        "Azure_textException APIConnectionError - Response payload is not completed\n"
        "Traceback (most recent call last):\n"
        '  File "/usr/local/lib/python3.13/site-packages/litellm/litellm_core_utils/streaming_handler.py", '
        "line 2363, in __anext__\n"
        "httpx.ReadError: Response payload is not completed"
    ),
    llm_provider="azure_text",
    model="gpt-3.5-turbo-instruct",
)


@pytest.mark.parametrize("completion_pipeline_raises", [_STREAM_CUT_BEFORE_THE_FIRST_CHUNK], indirect=True)
def test_a_stream_cut_before_the_first_chunk_answers_without_the_servers_traceback(
    client, auth_as, completion_pipeline_raises
):
    payload = {"model": "gpt-3.5-turbo-instruct", "prompt": "hi", "stream": True}
    with auth_as():
        response = client.post("/v1/completions", json=payload)
    assert response.status_code == 500
    assert response.json()["error"]["message"] == (
        "litellm.APIConnectionError: Azure_textException APIConnectionError - Response payload is not completed"
    )


@pytest.mark.parametrize(
    ("failure", "expected_status", "expected_type"),
    [
        pytest.param(ValueError("boom"), 500, "internal_server_error", id="unexpected_failure"),
        pytest.param(
            HTTPException(status_code=403, detail="Blocked by the content policy"),
            403,
            "permission_error",
            id="hook_policy_block",
        ),
        pytest.param(HTTPException(status_code=401, detail="key expired"), 401, "authentication_error", id="hook_401"),
        pytest.param(
            ValueError(f"{RouterErrors.no_deployments_with_tag_routing.value}. Passed model=gpt-4o and tags=['x']"),
            401,
            "auth_error",
            id="tag_routing_denial",
        ),
    ],
)
def test_completion_failure_body_types_the_status_it_answers_with(
    client, auth_as, monkeypatch, failure, expected_status, expected_type
):
    monkeypatch.setattr(proxy_server, "llm_router", MagicMock())
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", MagicMock(post_call_failure_hook=AsyncMock()))

    async def _raise(self, *args, **kwargs):
        raise failure

    monkeypatch.setattr(common_request_processing.ProxyBaseLLMRequestProcessing, "base_process_llm_request", _raise)
    with auth_as():
        response = client.post("/v1/completions", json={"model": "gpt-4o", "prompt": "hi"})
    assert response.status_code == expected_status
    error = response.json()["error"]
    assert (error["type"], error["param"], error["code"]) == (expected_type, None, str(expected_status))
