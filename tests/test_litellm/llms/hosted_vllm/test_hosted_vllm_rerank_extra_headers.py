"""
Test that extra_headers are correctly forwarded for hosted_vllm rerank calls.

Regression test for the bug where `extra_headers` (e.g. x-model-name used by
gateway routing) were silently dropped when using the hosted_vllm rerank provider.
"""

import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

# Minimal vLLM-style rerank response
vllm_rerank_response = {
    "id": "rerank-test-id",
    "results": [
        {"index": 0, "relevance_score": 0.97, "document": {"text": "doc1"}},
        {"index": 1, "relevance_score": 0.75, "document": {"text": "doc2"}},
    ],
    "usage": {"total_tokens": 10},
}

test_query = "machine learning"
test_documents = ["Machine learning is AI.", "The weather is sunny."]


def test_hosted_vllm_rerank_extra_headers_forwarded():
    """
    extra_headers passed to litellm.rerank() must be included in the outbound
    HTTP request when using the hosted_vllm provider.

    Reproduces the bug where gateway routing headers (x-model-name) were dropped,
    causing health-check 404s and router cooldown.
    """
    client = HTTPHandler()
    extra_headers = {"x-model-name": "my-inferset//models/my-model"}

    with patch.object(client, "post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(vllm_rerank_response)
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_response.raise_for_status = lambda: None
        mock_post.return_value = mock_response

        response = litellm.rerank(
            model="hosted_vllm/my-model",
            query=test_query,
            documents=test_documents,
            top_n=2,
            api_base="http://localhost:8031/v1",
            client=client,
            extra_headers=extra_headers,
        )

        assert isinstance(response, litellm.RerankResponse)
        assert mock_post.called, "HTTP client post should be called"

        call_kwargs = mock_post.call_args.kwargs
        sent_headers = call_kwargs.get("headers", {})

        header_found = any(
            k.lower() == "x-model-name" for k in sent_headers.keys()
        )
        assert header_found, (
            "extra_headers['x-model-name'] must be forwarded to the provider. "
            f"Actual headers sent: {list(sent_headers.keys())}"
        )
        print(f"✓ extra_headers forwarded correctly: {sent_headers}")


@pytest.mark.asyncio
async def test_hosted_vllm_rerank_extra_headers_forwarded_async():
    """Async variant of the extra_headers forwarding test."""
    client = AsyncHTTPHandler()
    extra_headers = {"x-model-name": "my-inferset//models/my-model"}

    with patch.object(client, "post", new_callable=AsyncMock) as mock_post:
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(vllm_rerank_response)
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_response.raise_for_status = lambda: None
        mock_post.return_value = mock_response

        response = await litellm.arerank(
            model="hosted_vllm/my-model",
            query=test_query,
            documents=test_documents,
            top_n=2,
            api_base="http://localhost:8031/v1",
            client=client,
            extra_headers=extra_headers,
        )

        assert isinstance(response, litellm.RerankResponse)
        assert mock_post.called, "HTTP client post should be called"

        call_kwargs = mock_post.call_args.kwargs
        sent_headers = call_kwargs.get("headers", {})

        header_found = any(
            k.lower() == "x-model-name" for k in sent_headers.keys()
        )
        assert header_found, (
            "extra_headers['x-model-name'] must be forwarded to the provider (async). "
            f"Actual headers sent: {list(sent_headers.keys())}"
        )
        print(f"✓ extra_headers forwarded correctly (async): {sent_headers}")


def test_hosted_vllm_rerank_headers_and_extra_headers_merged():
    """
    Both `headers` (from proxy forwarding) and `extra_headers` (explicit) must be merged
    and sent to the hosted_vllm provider.
    """
    client = HTTPHandler()
    proxy_headers = {"x-forwarded-for": "10.0.0.1"}
    extra_headers = {"x-model-name": "my-inferset//models/my-model"}

    with patch.object(client, "post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(vllm_rerank_response)
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_response.raise_for_status = lambda: None
        mock_post.return_value = mock_response

        response = litellm.rerank(
            model="hosted_vllm/my-model",
            query=test_query,
            documents=test_documents,
            top_n=2,
            api_base="http://localhost:8031/v1",
            client=client,
            headers=proxy_headers,
            extra_headers=extra_headers,
        )

        assert isinstance(response, litellm.RerankResponse)
        call_kwargs = mock_post.call_args.kwargs
        sent_headers = call_kwargs.get("headers", {})
        sent_lower = {k.lower(): v for k, v in sent_headers.items()}

        assert "x-model-name" in sent_lower, (
            f"extra_headers not found. Sent: {list(sent_headers.keys())}"
        )
        assert "x-forwarded-for" in sent_lower, (
            f"proxy headers not found. Sent: {list(sent_headers.keys())}"
        )
        print(f"✓ Both header sources merged: {list(sent_headers.keys())}")
