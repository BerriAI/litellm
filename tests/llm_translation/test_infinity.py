import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path


import litellm

import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path
from test_rerank import assert_response_shape
import litellm


@pytest.mark.asyncio()
async def test_infinity_rerank():
    mock_response = AsyncMock()

    def return_val():
        return {
            "id": "cmpl-mockid",
            "results": [{"index": 0, "relevance_score": 0.95}],
            "usage": {"prompt_tokens": 100, "total_tokens": 150},
        }

    mock_response.json = return_val
    mock_response.headers = {"key": "value"}
    mock_response.status_code = 200

    expected_payload = {
        "model": "rerank-model",
        "query": "hello",
        "top_n": 3,
        "documents": ["hello", "world"],
    }

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        response = await litellm.arerank(
            model="infinity/rerank-model",
            query="hello",
            documents=["hello", "world"],
            top_n=3,
            api_base="https://api.infinity.ai",
        )

        print("async re rank response: ", response)

        # Assert
        mock_post.assert_called_once()
        print("call args", mock_post.call_args)
        args_to_api = mock_post.call_args.kwargs["data"]
        _url = mock_post.call_args.kwargs["url"]
        print("Arguments passed to API=", args_to_api)
        print("url = ", _url)
        assert _url == "https://api.infinity.ai/rerank"

        request_data = json.loads(args_to_api)
        assert request_data["query"] == expected_payload["query"]
        assert request_data["documents"] == expected_payload["documents"]
        assert request_data["top_n"] == expected_payload["top_n"]
        assert request_data["model"] == expected_payload["model"]

        assert response.id is not None
        assert response.results is not None
        assert response.meta["tokens"]["input_tokens"] == 100
        assert (
            response.meta["tokens"]["output_tokens"] == 50
        )  # total_tokens - prompt_tokens

        assert_response_shape(response, custom_llm_provider="infinity")


@pytest.mark.asyncio()
async def test_infinity_rerank_with_env(monkeypatch):
    # Set up mock response
    mock_response = AsyncMock()

    def return_val():
        return {
            "id": "cmpl-mockid",
            "results": [{"index": 0, "relevance_score": 0.95}],
            "usage": {"prompt_tokens": 100, "total_tokens": 150},
        }

    mock_response.json = return_val
    mock_response.headers = {"key": "value"}
    mock_response.status_code = 200

    # Set environment variable
    monkeypatch.setenv("INFINITY_API_BASE", "https://env.infinity.ai")

    expected_payload = {
        "model": "rerank-model",
        "query": "hello",
        "top_n": 3,
        "documents": ["hello", "world"],
    }

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        response = await litellm.arerank(
            model="infinity/rerank-model",
            query="hello",
            documents=["hello", "world"],
            top_n=3,
        )

        print("async re rank response: ", response)

        # Assert
        mock_post.assert_called_once()
        print("call args", mock_post.call_args)
        args_to_api = mock_post.call_args.kwargs["data"]
        _url = mock_post.call_args.kwargs["url"]
        print("Arguments passed to API=", args_to_api)
        print("url = ", _url)
        assert _url == "https://env.infinity.ai/rerank"

        request_data = json.loads(args_to_api)
        assert request_data["query"] == expected_payload["query"]
        assert request_data["documents"] == expected_payload["documents"]
        assert request_data["top_n"] == expected_payload["top_n"]
        assert request_data["model"] == expected_payload["model"]

        assert response.id is not None
        assert response.results is not None
        assert response.meta["tokens"]["input_tokens"] == 100
        assert (
            response.meta["tokens"]["output_tokens"] == 50
        )  # total_tokens - prompt_tokens

        assert_response_shape(response, custom_llm_provider="infinity")
