"""
Integration tests for DeepInfra rerank functionality.
Tests the full rerank flow following the repository patterns.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm


@pytest.mark.parametrize("sync_mode", [True, False])
@patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post")
@patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
def test_deepinfra_rerank_with_queries_param(
    mock_sync_post, mock_async_post, sync_mode
):
    """Test DeepInfra rerank with multiple queries parameter."""
    mock_response_data = {
        "scores": [0.8, 0.6, 0.2],
        "input_tokens": 35,
        "request_id": "deepinfra-multi-query-123",
        "inference_status": {"status": "success", "runtime_ms": 200},
    }

    def return_val():
        return mock_response_data

    if sync_mode:
        mock_response = MagicMock()
        mock_response.json = return_val
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.text = json.dumps(mock_response_data)
        mock_sync_post.return_value = mock_response

        response = litellm.rerank(
            model="deepinfra/Qwen/Qwen3-Reranker-4B",
            query="hello",
            documents=["hello", "world", "test"],
            queries=["hello", "hi there"],  # DeepInfra specific param
            custom_llm_provider="deepinfra",
            api_key="test_key",
            api_base="https://api.deepinfra.com",
        )

        mock_sync_post.assert_called_once()
        # Verify that queries parameter was passed in request
        call_data = json.loads(mock_sync_post.call_args.kwargs["data"])
        assert "queries" in call_data
        assert call_data["queries"] == ["hello", "hi there"]
    else:
        mock_response = AsyncMock()
        mock_response.json = return_val
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.text = json.dumps(mock_response_data)
        mock_async_post.return_value = mock_response

        response = asyncio.run(
            litellm.arerank(
                model="deepinfra/Qwen/Qwen3-Reranker-4B",
                query="hello",
                documents=["hello", "world", "test"],
                queries=["hello", "hi there"],
                custom_llm_provider="deepinfra",
                api_key="test_key",
                api_base="https://api.deepinfra.com",
            )
        )

        mock_async_post.assert_called_once()
        call_data = json.loads(mock_async_post.call_args.kwargs["data"])
        assert "queries" in call_data
        assert call_data["queries"] == ["hello", "hi there"]

    assert response.results is not None
    assert len(response.results) == 3


@patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
def test_deepinfra_rerank_with_env_vars(mock_post, monkeypatch):
    """Test DeepInfra rerank with environment variable configuration."""
    monkeypatch.setenv("DEEPINFRA_API_KEY", "env_test_key")
    monkeypatch.setenv("DEEPINFRA_API_BASE", "https://custom-deepinfra.com")

    mock_response_data = {
        "scores": [0.88, 0.22],
        "input_tokens": 28,
        "request_id": "env-test-123",
    }

    def return_val():
        return mock_response_data

    mock_response = MagicMock()
    mock_response.json = return_val
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.text = json.dumps(mock_response_data)
    mock_post.return_value = mock_response

    response = litellm.rerank(
        model="deepinfra/Qwen/Qwen3-Reranker-0.6B",
        query="hello",
        documents=["hello", "world"],
        custom_llm_provider="deepinfra",
    )

    mock_post.assert_called_once()

    # Verify headers contain env API key
    headers = mock_post.call_args.kwargs.get("headers", {})
    assert "Bearer env_test_key" in headers.get("Authorization", "")

    assert response.results is not None


@patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
def test_deepinfra_rerank_defaults_api_base_when_missing(mock_post, monkeypatch):
    """With no api_base anywhere, the call still goes out against DeepInfra's own base."""
    monkeypatch.delenv("DEEPINFRA_API_BASE", raising=False)

    mock_response = MagicMock()
    mock_response.json = lambda: {"scores": [0.9, 0.1], "input_tokens": 20}
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_post.return_value = mock_response

    response = litellm.rerank(
        model="deepinfra/Qwen/Qwen3-Reranker-0.6B",
        query="hello",
        documents=["hello", "world"],
        custom_llm_provider="deepinfra",
        api_key="test_key",
        # api_base is intentionally missing
    )

    assert "api.deepinfra.com" in mock_post.call_args.kwargs["url"]
    assert [result["relevance_score"] for result in response.results] == [0.9, 0.1]


def test_deepinfra_rerank_models():
    """Test that DeepInfra Qwen rerank models are recognized."""
    # These should not raise errors during model validation
    models = [
        "deepinfra/Qwen/Qwen3-Reranker-0.6B",
        "deepinfra/Qwen/Qwen3-Reranker-4B",
        "deepinfra/Qwen/Qwen3-Reranker-8B",
    ]

    for model in models:
        resolved_model, provider, _, api_base = litellm.get_llm_provider(model=model)
        assert provider == "deepinfra"
        assert resolved_model == model.removeprefix("deepinfra/")
        assert api_base == "https://api.deepinfra.com/v1/openai"
