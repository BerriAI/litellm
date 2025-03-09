import asyncio
import httpx
import json
import pytest
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch, ANY

import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm.exceptions import BadRequestError
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.utils import CustomStreamWrapper
from base_llm_unit_tests import BaseLLMChatTest

try:
    import databricks.sdk

    databricks_sdk_installed = True
except ImportError:
    databricks_sdk_installed = False


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
            "completion_tokens_details": None,
            "total_tokens": 268,
            "prompt_tokens_details": None,
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
            "prompt_tokens_details": None,
        },
    }


@pytest.mark.parametrize("set_base", [True, False])
def test_throws_if_api_base_or_api_key_not_set_without_databricks_sdk(
    monkeypatch, set_base
):
    # Simulate that the databricks SDK is not installed
    monkeypatch.setitem(sys.modules, "databricks.sdk", None)

    err_msg = "the Databricks base URL and API key are not set"

    if set_base:
        monkeypatch.setenv(
            "DATABRICKS_API_BASE",
            "https://my.workspace.cloud.databricks.com/serving-endpoints",
        )
        monkeypatch.delenv(
            "DATABRICKS_API_KEY",
        )
    else:
        monkeypatch.setenv("DATABRICKS_API_KEY", "dapimykey")
        monkeypatch.delenv(
            "DATABRICKS_API_BASE",
        )

    with pytest.raises(BadRequestError) as exc:
        litellm.completion(
            model="databricks/dbrx-instruct-071224",
            messages=[{"role": "user", "content": "How are you?"}],
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
            url=f"{base_url}/chat/completions",
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
            timeout=ANY,
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
            data=ANY,
            stream=True,
            timeout=ANY,
        )

        actual_data = json.loads(
            mock_post.call_args.kwargs["data"]
        )  # Deserialize the actual data
        expected_data = {
            "model": "dbrx-instruct-071224",
            "messages": messages,
            "temperature": 0.5,
            "stream": True,
            "extraparam": "testpassingextraparam",
        }
        assert actual_data == expected_data, f"Unexpected JSON data: {actual_data}"


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
            data=ANY,
            stream=True,
        )

        actual_data = json.loads(
            mock_post.call_args.kwargs["data"]
        )  # Deserialize the actual data
        expected_data = {
            "model": "dbrx-instruct-071224",
            "messages": messages,
            "temperature": 0.5,
            "stream": True,
            "extraparam": "testpassingextraparam",
        }
        assert actual_data == expected_data, f"Unexpected JSON data: {actual_data}"


@pytest.mark.skipif(not databricks_sdk_installed, reason="Databricks SDK not installed")
def test_completions_uses_databricks_sdk_if_api_key_and_base_not_specified(monkeypatch):
    monkeypatch.delenv("DATABRICKS_API_BASE")
    monkeypatch.delenv("DATABRICKS_API_KEY")
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.config import Config

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

    base_url = "https://my.workspace.cloud.databricks.com"
    api_key = "dapimykey"
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    messages = [{"role": "user", "content": "How are you?"}]

    mock_workspace_client: WorkspaceClient = MagicMock()
    mock_config: Config = MagicMock()
    # Simulate the behavior of the config property and its methods
    mock_config.authenticate.side_effect = lambda: headers
    mock_config.host = base_url  # Assign directly as if it's a property
    mock_workspace_client.config = mock_config

    with patch(
        "databricks.sdk.WorkspaceClient", return_value=mock_workspace_client
    ), patch.object(HTTPHandler, "post", return_value=mock_response) as mock_post:
        response = litellm.completion(
            model="databricks/dbrx-instruct-071224",
            messages=messages,
            client=sync_handler,
            temperature=0.5,
            extraparam="testpassingextraparam",
        )
        assert response.to_dict() == expected_response_json

        mock_post.assert_called_once_with(
            f"{base_url}/serving-endpoints/chat/completions",
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


@pytest.mark.skipif(not databricks_sdk_installed, reason="Databricks SDK not installed")
def test_embeddings_uses_databricks_sdk_if_api_key_and_base_not_specified(monkeypatch):
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.config import Config

    base_url = "https://my.workspace.cloud.databricks.com/serving-endpoints"
    api_key = "dapimykey"
    monkeypatch.setenv("DATABRICKS_API_BASE", base_url)
    monkeypatch.setenv("DATABRICKS_API_KEY", api_key)

    sync_handler = HTTPHandler()
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_embedding_response()

    base_url = "https://my.workspace.cloud.databricks.com"
    api_key = "dapimykey"
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    inputs = ["Hello", "World"]

    mock_workspace_client: WorkspaceClient = MagicMock()
    mock_config: Config = MagicMock()
    # Simulate the behavior of the config property and its methods
    mock_config.authenticate.side_effect = lambda: headers
    mock_config.host = base_url  # Assign directly as if it's a property
    mock_workspace_client.config = mock_config

    with patch(
        "databricks.sdk.WorkspaceClient", return_value=mock_workspace_client
    ), patch.object(HTTPHandler, "post", return_value=mock_response) as mock_post:
        response = litellm.embedding(
            model="databricks/bge-large-en-v1.5",
            input=inputs,
            client=sync_handler,
            extraparam="testpassingextraparam",
        )
        assert response.to_dict() == mock_embedding_response()

        mock_post.assert_called_once_with(
            f"{base_url}/serving-endpoints/embeddings",
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


class TestDatabricksCompletion(BaseLLMChatTest):
    def get_base_completion_call_args(self) -> dict:
        return {"model": "databricks/databricks-dbrx-instruct"}

    def test_pdf_handling(self, pdf_messages):
        pytest.skip("Databricks does not support PDF handling")

    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly. Relevant issue: https://github.com/BerriAI/litellm/issues/6833"""
        pytest.skip("Databricks is openai compatible")


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_databricks_embeddings(sync_mode):
    import openai

    try:
        litellm.set_verbose = True
        litellm.drop_params = True

        if sync_mode:
            response = litellm.embedding(
                model="databricks/databricks-bge-large-en",
                input=["good morning from litellm"],
                instruction="Represent this sentence for searching relevant passages:",
            )
        else:
            response = await litellm.aembedding(
                model="databricks/databricks-bge-large-en",
                input=["good morning from litellm"],
                instruction="Represent this sentence for searching relevant passages:",
            )

        print(f"response: {response}")

        openai.types.CreateEmbeddingResponse.model_validate(
            response.model_dump(), strict=True
        )
        # stubbed endpoint is setup to return this
        # assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
