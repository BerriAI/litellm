import asyncio
import httpx
import json
import pytest
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import litellm
from litellm.exceptions import BadRequestError, InternalServerError
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.utils import CustomStreamWrapper


def mock_chat_response() -> Dict[str, Any]:
    return {
        "id": "chatcmpl_3f78f09a-489c-4b8d-a587-f162c7497891",
        "object": "chat.completion",
        "created": 1726285449,
        "model": "dbrx-instruct-071224",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! I'm an AI assistant. I'm doing well. How can I help?",
                    "function_call": None,
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 230,
            "completion_tokens": 38,
            "total_tokens": 268,
            "completion_tokens_details": None,
        },
        "system_fingerprint": None,
    }


def mock_chat_streaming_response_chunks() -> List[str]:
    return [
        json.dumps(
            {
                "id": "chatcmpl_8a7075d1-956e-4960-b3a6-892cd4649ff3",
                "object": "chat.completion.chunk",
                "created": 1726469651,
                "model": "dbrx-instruct-071224",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "Hello"},
                        "finish_reason": None,
                        "logprobs": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 230,
                    "completion_tokens": 1,
                    "total_tokens": 231,
                },
            }
        ),
        json.dumps(
            {
                "id": "chatcmpl_8a7075d1-956e-4960-b3a6-892cd4649ff3",
                "object": "chat.completion.chunk",
                "created": 1726469651,
                "model": "dbrx-instruct-071224",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": " world"},
                        "finish_reason": None,
                        "logprobs": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 230,
                    "completion_tokens": 1,
                    "total_tokens": 231,
                },
            }
        ),
        json.dumps(
            {
                "id": "chatcmpl_8a7075d1-956e-4960-b3a6-892cd4649ff3",
                "object": "chat.completion.chunk",
                "created": 1726469651,
                "model": "dbrx-instruct-071224",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "!"},
                        "finish_reason": "stop",
                        "logprobs": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 230,
                    "completion_tokens": 1,
                    "total_tokens": 231,
                },
            }
        ),
    ]


def mock_chat_streaming_response_chunks_bytes() -> List[bytes]:
    string_chunks = mock_chat_streaming_response_chunks()
    bytes_chunks = [chunk.encode("utf-8") + b"\n" for chunk in string_chunks]
    # Simulate the end of the stream
    bytes_chunks.append(b"")
    return bytes_chunks


def mock_http_handler_chat_streaming_response() -> MagicMock:
    mock_stream_chunks = mock_chat_streaming_response_chunks()

    def mock_iter_lines():
        for chunk in mock_stream_chunks:
            for line in chunk.splitlines():
                yield line

    mock_response = MagicMock()
    mock_response.iter_lines.side_effect = mock_iter_lines
    mock_response.status_code = 200

    return mock_response


def mock_http_handler_chat_async_streaming_response() -> MagicMock:
    mock_stream_chunks = mock_chat_streaming_response_chunks()

    async def mock_iter_lines():
        for chunk in mock_stream_chunks:
            for line in chunk.splitlines():
                yield line

    mock_response = MagicMock()
    mock_response.aiter_lines.return_value = mock_iter_lines()
    mock_response.status_code = 200

    return mock_response


def mock_databricks_client_chat_streaming_response() -> MagicMock:
    mock_stream_chunks = mock_chat_streaming_response_chunks_bytes()

    def mock_read_from_stream(size=-1):
        if mock_stream_chunks:
            return mock_stream_chunks.pop(0)
        return b""

    mock_response = MagicMock()
    streaming_response_mock = MagicMock()
    streaming_response_iterator_mock = MagicMock()
    # Mock the __getitem__("content") method to return the streaming response
    mock_response.__getitem__.return_value = streaming_response_mock
    # Mock the streaming response __enter__ method to return the streaming response iterator
    streaming_response_mock.__enter__.return_value = streaming_response_iterator_mock

    streaming_response_iterator_mock.read1.side_effect = mock_read_from_stream
    streaming_response_iterator_mock.closed = False

    return mock_response


def mock_embedding_response() -> Dict[str, Any]:
    return {
        "object": "list",
        "model": "bge-large-en-v1.5",
        "data": [
            {
                "index": 0,
                "object": "embedding",
                "embedding": [
                    0.06768798828125,
                    -0.01291656494140625,
                    -0.0501708984375,
                    0.0245361328125,
                    -0.030364990234375,
                ],
            }
        ],
        "usage": {
            "prompt_tokens": 8,
            "total_tokens": 8,
            "completion_tokens": 0,
            "completion_tokens_details": None,
        },
    }


@pytest.mark.parametrize("set_base", [True, False])
def test_throws_if_only_one_of_api_base_or_api_key_set(monkeypatch, set_base):
    if set_base:
        monkeypatch.setenv(
            "DATABRICKS_API_BASE",
            "https://my.workspace.cloud.databricks.com/serving-endpoints",
        )
        monkeypatch.delenv(
            "DATABRICKS_API_KEY",
        )
        err_msg = "A call is being made to LLM Provider but no key is set"
    else:
        monkeypatch.setenv("DATABRICKS_API_KEY", "dapimykey")
        monkeypatch.delenv("DATABRICKS_API_BASE")
        err_msg = "A call is being made to LLM Provider but no api base is set"

    with pytest.raises(BadRequestError) as exc:
        litellm.completion(
            model="databricks/dbrx-instruct-071224",
            messages={"role": "user", "content": "How are you?"},
        )
    assert err_msg in str(exc)

    with pytest.raises(BadRequestError) as exc:
        litellm.embedding(
            model="databricks/bge-12312",
            input=["Hello", "World"],
        )
    assert err_msg in str(exc)


def test_completions_with_sync_http_handler(monkeypatch):
    base_url = "https://my.workspace.cloud.databricks.com/serving-endpoints"
    api_key = "dapimykey"
    monkeypatch.setenv("DATABRICKS_API_BASE", base_url)
    monkeypatch.setenv("DATABRICKS_API_KEY", api_key)

    sync_handler = HTTPHandler()
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_chat_response()

    expected_response_json = {
        **mock_chat_response(),
        **{
            "model": "databricks/dbrx-instruct-071224",
        },
    }

    messages = [{"role": "user", "content": "How are you?"}]

    with patch.object(HTTPHandler, "post", return_value=mock_response) as mock_post:
        response = litellm.completion(
            model="databricks/dbrx-instruct-071224",
            messages=messages,
            client=sync_handler,
            temperature=0.5,
            extraparam="testpassingextraparam",
        )
        assert response.to_dict() == expected_response_json

        mock_post.assert_called_once_with(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(
                {
                    "model": "dbrx-instruct-071224",
                    "messages": messages,
                    "temperature": 0.5,
                    "extraparam": "testpassingextraparam",
                    "stream": False,
                }
            ),
        )


def test_completions_with_async_http_handler(monkeypatch):
    base_url = "https://my.workspace.cloud.databricks.com/serving-endpoints"
    api_key = "dapimykey"
    monkeypatch.setenv("DATABRICKS_API_BASE", base_url)
    monkeypatch.setenv("DATABRICKS_API_KEY", api_key)

    async_handler = AsyncHTTPHandler()
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_chat_response()

    expected_response_json = {
        **mock_chat_response(),
        **{
            "model": "databricks/dbrx-instruct-071224",
        },
    }

    messages = [{"role": "user", "content": "How are you?"}]

    with patch.object(
        AsyncHTTPHandler, "post", return_value=mock_response
    ) as mock_post:
        response = asyncio.run(
            litellm.acompletion(
                model="databricks/dbrx-instruct-071224",
                messages=messages,
                client=async_handler,
                temperature=0.5,
                extraparam="testpassingextraparam",
            )
        )
        assert response.to_dict() == expected_response_json

        mock_post.assert_called_once_with(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(
                {
                    "model": "dbrx-instruct-071224",
                    "messages": messages,
                    "temperature": 0.5,
                    "extraparam": "testpassingextraparam",
                    "stream": False,
                }
            ),
        )


def test_completions_streaming_with_sync_http_handler(monkeypatch):
    base_url = "https://my.workspace.cloud.databricks.com/serving-endpoints"
    api_key = "dapimykey"
    monkeypatch.setenv("DATABRICKS_API_BASE", base_url)
    monkeypatch.setenv("DATABRICKS_API_KEY", api_key)

    sync_handler = HTTPHandler()

    messages = [{"role": "user", "content": "How are you?"}]
    mock_response = mock_http_handler_chat_streaming_response()

    with patch.object(HTTPHandler, "post", return_value=mock_response) as mock_post:
        response_stream: CustomStreamWrapper = litellm.completion(
            model="databricks/dbrx-instruct-071224",
            messages=messages,
            client=sync_handler,
            temperature=0.5,
            extraparam="testpassingextraparam",
            stream=True,
        )
        response = list(response_stream)
        assert "dbrx-instruct-071224" in str(response)
        assert "chatcmpl" in str(response)
        assert len(response) == 4

        mock_post.assert_called_once_with(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(
                {
                    "model": "dbrx-instruct-071224",
                    "messages": messages,
                    "temperature": 0.5,
                    "stream": True,
                    "extraparam": "testpassingextraparam",
                }
            ),
            stream=True,
        )


def test_completions_streaming_with_async_http_handler(monkeypatch):
    base_url = "https://my.workspace.cloud.databricks.com/serving-endpoints"
    api_key = "dapimykey"
    monkeypatch.setenv("DATABRICKS_API_BASE", base_url)
    monkeypatch.setenv("DATABRICKS_API_KEY", api_key)

    async_handler = AsyncHTTPHandler()

    messages = [{"role": "user", "content": "How are you?"}]
    mock_response = mock_http_handler_chat_async_streaming_response()

    with patch.object(
        AsyncHTTPHandler, "post", return_value=mock_response
    ) as mock_post:
        response_stream: CustomStreamWrapper = asyncio.run(
            litellm.acompletion(
                model="databricks/dbrx-instruct-071224",
                messages=messages,
                client=async_handler,
                temperature=0.5,
                extraparam="testpassingextraparam",
                stream=True,
            )
        )

        # Use async list gathering for the response
        async def gather_responses():
            return [item async for item in response_stream]

        response = asyncio.run(gather_responses())
        assert "dbrx-instruct-071224" in str(response)
        assert "chatcmpl" in str(response)
        assert len(response) == 4

        mock_post.assert_called_once_with(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(
                {
                    "model": "dbrx-instruct-071224",
                    "messages": messages,
                    "temperature": 0.5,
                    "stream": True,
                    "extraparam": "testpassingextraparam",
                }
            ),
            stream=True,
        )


def test_embeddings_with_sync_http_handler(monkeypatch):
    base_url = "https://my.workspace.cloud.databricks.com/serving-endpoints"
    api_key = "dapimykey"
    monkeypatch.setenv("DATABRICKS_API_BASE", base_url)
    monkeypatch.setenv("DATABRICKS_API_KEY", api_key)

    sync_handler = HTTPHandler()
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_embedding_response()

    inputs = ["Hello", "World"]

    with patch.object(HTTPHandler, "post", return_value=mock_response) as mock_post:
        response = litellm.embedding(
            model="databricks/bge-large-en-v1.5",
            input=inputs,
            client=sync_handler,
            extraparam="testpassingextraparam",
        )
        assert response.to_dict() == mock_embedding_response()

        mock_post.assert_called_once_with(
            f"{base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(
                {
                    "model": "bge-large-en-v1.5",
                    "input": inputs,
                    "extraparam": "testpassingextraparam",
                }
            ),
        )


def test_embeddings_with_async_http_handler(monkeypatch):
    base_url = "https://my.workspace.cloud.databricks.com/serving-endpoints"
    api_key = "dapimykey"
    monkeypatch.setenv("DATABRICKS_API_BASE", base_url)
    monkeypatch.setenv("DATABRICKS_API_KEY", api_key)

    async_handler = AsyncHTTPHandler()
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_embedding_response()

    inputs = ["Hello", "World"]

    with patch.object(
        AsyncHTTPHandler, "post", return_value=mock_response
    ) as mock_post:
        response = asyncio.run(
            litellm.aembedding(
                model="databricks/bge-large-en-v1.5",
                input=inputs,
                client=async_handler,
                extraparam="testpassingextraparam",
            )
        )
        assert response.to_dict() == mock_embedding_response()

        mock_post.assert_called_once_with(
            f"{base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(
                {
                    "model": "bge-large-en-v1.5",
                    "input": inputs,
                    "extraparam": "testpassingextraparam",
                }
            ),
        )
