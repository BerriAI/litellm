import asyncio
import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../../.."))
import litellm
from litellm.llms.xinference.rerank.transformation import (
    DEFAULT_XINFERENCE_API_BASE,
    XinferenceRerankConfig,
)


def test_xinference_rerank_defaults_and_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XINFERENCE_API_BASE", raising=False)
    monkeypatch.delenv("XINFERENCE_API_KEY", raising=False)
    config = XinferenceRerankConfig()

    assert config.get_complete_url(api_base=None, model="bge-reranker") == f"{DEFAULT_XINFERENCE_API_BASE}/rerank"
    monkeypatch.setenv("XINFERENCE_API_BASE", "http://env-xinference.test/v1")
    assert config.get_complete_url(api_base=None, model="bge-reranker") == "http://env-xinference.test/v1/rerank"

    no_auth_headers = config.validate_environment(headers={}, model="bge-reranker")
    assert no_auth_headers["Authorization"] == "Bearer stub-xinference-key"

    monkeypatch.setenv("XINFERENCE_API_KEY", "env-key")
    env_auth_headers = config.validate_environment(headers={}, model="bge-reranker")
    assert env_auth_headers["Authorization"] == "Bearer env-key"

    caller_auth_headers = config.validate_environment(
        headers={"Authorization": "Bearer caller-token"},
        model="bge-reranker",
    )
    assert caller_auth_headers["Authorization"] == "Bearer caller-token"


@pytest.mark.parametrize("sync_mode", [True, False])
@patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post")
@patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
def test_xinference_rerank_uses_base_handler(
    mock_sync_post: MagicMock,
    mock_async_post: MagicMock,
    sync_mode: bool,
) -> None:
    response_data = {
        "results": [
            {"index": 1, "relevance_score": 0.92, "document": "Xinference supports rerank."},
            {"index": 0, "relevance_score": 0.24, "document": "An unrelated document."},
        ]
    }

    api_base = "http://xinference.example.test/v1"
    request_headers = {"Authorization": "Bearer caller-token", "x-request-id": "req-123"}

    if sync_mode:
        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.text = json.dumps(response_data)
        mock_sync_post.return_value = mock_response

        response = litellm.rerank(
            model="xinference/bge-reranker-large",
            query="Does Xinference support rerank?",
            documents=["An unrelated document.", "Xinference supports rerank."],
            top_n=2,
            api_base=api_base,
            headers=request_headers,
        )

        mock_sync_post.assert_called_once()
        call_kwargs = mock_sync_post.call_args.kwargs
    else:
        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.text = json.dumps(response_data)
        mock_async_post.return_value = mock_response

        response = asyncio.run(
            litellm.arerank(
                model="xinference/bge-reranker-large",
                query="Does Xinference support rerank?",
                documents=["An unrelated document.", "Xinference supports rerank."],
                top_n=2,
                api_base=api_base,
                headers=request_headers,
            )
        )

        mock_async_post.assert_called_once()
        call_kwargs = mock_async_post.call_args.kwargs

    assert call_kwargs["url"] == "http://xinference.example.test/v1/rerank"
    assert call_kwargs["headers"]["Authorization"] == "Bearer caller-token"
    assert call_kwargs["headers"]["x-request-id"] == "req-123"

    request_body = json.loads(call_kwargs["data"])
    assert request_body == {
        "model": "bge-reranker-large",
        "query": "Does Xinference support rerank?",
        "documents": ["An unrelated document.", "Xinference supports rerank."],
        "top_n": 2,
    }

    assert response.results == [
        {
            "index": 1,
            "relevance_score": 0.92,
            "document": {"text": "Xinference supports rerank."},
        },
        {
            "index": 0,
            "relevance_score": 0.24,
            "document": {"text": "An unrelated document."},
        },
    ]
