"""Behavior pins for ``proxy_server.py`` embeddings routes.

Pins (PR2):
    - POST /v1/embeddings
    - POST /embeddings
    - POST /engines/{model:path}/embeddings
    - POST /openai/deployments/{model:path}/embeddings
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy import common_request_processing, proxy_server

from .conftest import normalize  # type: ignore[import-not-found]

HAPPY_RESPONSE = {
    "object": "list",
    "model": "text-embedding-ada-002",
    "data": [{"embedding": [0.0, 0.1, 0.2], "index": 0, "object": "embedding"}],
    "usage": {"prompt_tokens": 1, "total_tokens": 1},
}


@pytest.fixture
def patched_embedding(monkeypatch):
    router = MagicMock()
    router.model_names = ["text-embedding-ada-002"]
    router.get_deployment_by_model_group_name = MagicMock(return_value=None)
    monkeypatch.setattr(proxy_server, "llm_router", router)
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
def embedding_pipeline_raises(monkeypatch):
    router = MagicMock()
    router.model_names = []
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(
        proxy_server, "proxy_logging_obj", MagicMock(post_call_failure_hook=AsyncMock())
    )

    from litellm.proxy._types import ProxyException

    async def _raise(self, *args, **kwargs):
        raise ValueError("boom")

    async def _handler(self, *, e, user_api_key_dict, proxy_logging_obj, version=None):
        return ProxyException(
            message="boom", type="bad_request_error", param="model", code=400
        )

    monkeypatch.setattr(
        common_request_processing.ProxyBaseLLMRequestProcessing,
        "base_process_llm_request",
        _raise,
    )
    monkeypatch.setattr(
        common_request_processing.ProxyBaseLLMRequestProcessing,
        "_handle_llm_api_exception",
        _handler,
    )
    yield


_EMBED_PATHS = [
    "/v1/embeddings",
    "/embeddings",
    "/engines/text-embedding-ada-002/embeddings",
    "/openai/deployments/text-embedding-ada-002/embeddings",
]


@pytest.mark.parametrize("path", _EMBED_PATHS)
def test_embeddings_happy_path(client, auth_as, patched_embedding, path):
    """Pins all four ``POST .../embeddings`` aliases (happy path).

    Covers ``POST /v1/embeddings``, ``POST /embeddings``,
    ``POST /engines/{model:path}/embeddings``, and
    ``POST /openai/deployments/{model:path}/embeddings``.
    """
    payload = {"model": "text-embedding-ada-002", "input": "hello"}
    with auth_as():
        response = client.post(path, json=payload)
    assert response.status_code == 200
    assert normalize(response.json()) == {
        "object": "list",
        "model": "text-embedding-ada-002",
        "data": [{"embedding": [0.0, 0.1, 0.2], "index": 0, "object": "embedding"}],
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    }


@pytest.mark.parametrize("path", _EMBED_PATHS)
def test_embeddings_pipeline_error(client, auth_as, embedding_pipeline_raises, path):
    """Pins all four ``POST .../embeddings`` aliases (error path).

    Covers ``POST /v1/embeddings``, ``POST /embeddings``,
    ``POST /engines/{model:path}/embeddings``, and
    ``POST /openai/deployments/{model:path}/embeddings``.
    """
    payload = {"model": "text-embedding-ada-002", "input": "boom"}
    with auth_as():
        response = client.post(path, json=payload)
    assert response.status_code == 400
    assert response.content  # non-empty error body
