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
from base_llm_unit_tests import BaseLLMChatTest, BaseAnthropicChatTest

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


def mock_chat_response_claude_prompt_caching() -> Dict[str, Any]:
    return {
        "id": "msg_01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "object": "chat.completion",
        "created": 1761118943,
        "model": "claude-3-7-sonnet", # Mock model name for testing
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The text you've provided consists entirely of the phrase \"example text\" repeated many times without any variation or additional content. There is no specific information, narrative, argument, or structured content to explain. This appears to be placeholder or filler text that would typically be replaced with actual content in a final document.",
                    "refusal": None,
                    "function_call": None,
                    "tool_calls": None,
                    "annotations": None,
                    "audio": None,
                },
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": 1556,
            "completion_tokens": 65,
            "total_tokens": 1621,
            "completion_tokens_details": None,
            "prompt_tokens_details": None,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 1552,
        },
        "service_tier": None,
        "system_fingerprint": None,
    }

def mock_chat_response_claude_prompt_caching_repeat() -> Dict[str, Any]:
    return {
        "id": "msg_01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "object": "chat.completion",
        "created": 1761118943,
        "model": "claude-3-7-sonnet", # Mock model name for testing
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The text you've provided consists entirely of the phrase \"example text\" repeated many times without any variation or additional content. There is no specific information, narrative, argument, or structured content to explain. This appears to be placeholder or filler text that would typically be replaced with actual content in a final document.",
                    "refusal": None,
                    "function_call": None,
                    "tool_calls": None,
                    "annotations": None,
                    "audio": None,
                },
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": 1556,
            "completion_tokens": 65,
            "total_tokens": 1621,
            "completion_tokens_details": None,
            "prompt_tokens_details": None,
            "cache_read_input_tokens": 1552,
            "cache_creation_input_tokens": 0,
        },
        "service_tier": None,
        "system_fingerprint": None,
    }


def mock_chat_response_nonclaude_prompt_caching() -> Dict[str, Any]:
    return {
        "id": "msg_01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "object": "chat.completion",
        "created": 1761119150,
        "model": "gpt-oss-20b",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "reasoning",
                            "summary": [
                                {
                                    "type": "summary_text",
                                    "text": "The user just posted a block of text repeated: \"example textexample\" many times. It is unclear what they want. The instruction says: \"You are a helpful assistant that explains the content of the given text.\" So I need to explain the content.\n\nThe content is basically a repeated phrase 'example textexample' many times, possibly a demonstration of repeated words or filler text. Perhaps they test that the assistant enumerates or condenses. Should I explain that it is a repeated phrase used maybe as placeholder text? It looks like a placeholder or filler. Could say that it's essentially nonsense.\n\nExplain that the text consists of the word \"example\" concatenated with \"text\" repeated many times. It's not meaningful content. Might indicate filler text for page layout.\n\nAlternatively, explain why repeated 'example textexample' (without whitespace in some places?) is repeated. This could be a test. The user probably expects a response like: \"It says 'example textexample' several times.\" So I should summarize: The text is a repeated phrase used as filler.\n\nGiven the instruction, let's explain the content. Mention that it's repetitive placeholder, no meaningful content, just repeated phrase. Also note that \"example text\" repeated words. No specific meaning beyond being placeholder.\n\nSo respond: This is basically a placeholder used in design documents: the phrase \"example text\" repeated to fill a space, no distinct meaning beyond placeholder usage. 'text' might be part of the 'example text' phrase or 'textexample' it's concatenated. These might serve to fill text boxes, test fonts, etc.\n\nAlso mention the pattern: Could be used for testing text rendering, typographic layouts, measuring dimensions.\n\nAnswer accordingly."
                                }
                            ]
                        },
                        {
                            "type": "text",
                            "text": "The passage you pasted is essentially a block of **placeholder text**.  \nIt repeats the phrase \"example textexample\" (or \"example text\" in some places) over and over again.  There isn't any hidden message, concept, or argument buried in it – the purpose is purely to fill space, imitate real content, or test something like typography, layout, or rendering.\n\nIn design and copy‑editing, such repeated strings are often used to:\n\n* **Fill a page or template** so the designer can see how multiple lines of content will look.\n* **Test the appearance of fonts, line‑height, paragraph spacing, and other typographic settings.**\n* **Serve as a stand"
                        }
                    ],
                    "refusal": None,
                    "function_call": None,
                    "tool_calls": None,
                    "annotations": None,
                    "audio": None,
                },
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": 1638,
            "completion_tokens": 500,
            "total_tokens": 2138,
            "completion_tokens_details": None,
            "prompt_tokens_details": None,
        },
        "service_tier": None,
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

    err_msg = ["the Databricks base URL and API key are not set", "Missing API Key"]

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
    assert any(msg in str(exc) for msg in err_msg)

    with pytest.raises(BadRequestError) as exc:
        litellm.embedding(
            model="databricks/bge-12312",
            input=["Hello", "World"],
        )
    assert any(msg in str(exc) for msg in err_msg)


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

        assert (
            mock_post.call_args.kwargs["headers"]["Content-Type"] == "application/json"
        )
        assert (
            mock_post.call_args.kwargs["headers"]["Authorization"]
            == f"Bearer {api_key}"
        )
        assert mock_post.call_args.kwargs["url"] == f"{base_url}/chat/completions"
        assert mock_post.call_args.kwargs["stream"] == False

        actual_data = json.loads(
            mock_post.call_args.kwargs["data"]
        )  # Deserialize the actual data
        expected_data = {
            "model": "dbrx-instruct-071224",
            "messages": messages,
            "temperature": 0.5,
            "extraparam": "testpassingextraparam",
        }
        assert actual_data == expected_data, f"Unexpected JSON data: {actual_data}"


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

        assert (
            mock_post.call_args.kwargs["headers"]["Content-Type"] == "application/json"
        )
        assert (
            mock_post.call_args.kwargs["headers"]["Authorization"]
            == f"Bearer {api_key}"
        )
        assert mock_post.call_args.kwargs["url"] == f"{base_url}/chat/completions"
        assert mock_post.call_args.kwargs["stream"] == False

        actual_data = json.loads(
            mock_post.call_args.kwargs["data"]
        )  # Deserialize the actual data
        expected_data = {
            "model": "dbrx-instruct-071224",
            "messages": messages,
            "temperature": 0.5,
            "extraparam": "testpassingextraparam",
        }
        assert actual_data == expected_data, f"Unexpected JSON data: {actual_data}"


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

        assert (
            mock_post.call_args.kwargs["headers"]["Content-Type"] == "application/json"
        )
        assert (
            mock_post.call_args.kwargs["headers"]["Authorization"]
            == f"Bearer {api_key}"
        )
        assert mock_post.call_args.kwargs["url"] == f"{base_url}/chat/completions"
        assert mock_post.call_args.kwargs["stream"] == True

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

        assert (
            mock_post.call_args.kwargs["headers"]["Content-Type"] == "application/json"
        )
        assert (
            mock_post.call_args.kwargs["headers"]["Authorization"]
            == f"Bearer {api_key}"
        )
        assert mock_post.call_args.kwargs["url"] == f"{base_url}/chat/completions"
        assert mock_post.call_args.kwargs["stream"] == True

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

        assert (
            mock_post.call_args.kwargs["headers"]["Content-Type"] == "application/json"
        )
        assert (
            mock_post.call_args.kwargs["headers"]["Authorization"]
            == f"Bearer {api_key}"
        )
        assert (
            mock_post.call_args.kwargs["url"]
            == f"{base_url}/serving-endpoints/chat/completions"
        )
        assert mock_post.call_args.kwargs["stream"] == False
        assert mock_post.call_args.kwargs["data"] == json.dumps(
            {
                "model": "dbrx-instruct-071224",
                "messages": messages,
                "temperature": 0.5,
                "extraparam": "testpassingextraparam",
                "stream": False,
            }
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


@pytest.mark.skip(reason="Databricks rate limit errors")
class TestDatabricksCompletion(BaseLLMChatTest, BaseAnthropicChatTest):
    def get_base_completion_call_args(self) -> dict:
        return {"model": "databricks/databricks-claude-3-7-sonnet"}

    def get_base_completion_call_args_with_thinking(self) -> dict:
        return {
            "model": "databricks/databricks-claude-3-7-sonnet",
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        }

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


def test_completion_with_prompt_caching_claude_model(monkeypatch):
    base_url = "https://my.workspace.cloud.databricks.com/serving-endpoints"
    api_key = "dapimykey"
    monkeypatch.setenv("DATABRICKS_API_BASE", base_url)
    monkeypatch.setenv("DATABRICKS_API_KEY", api_key)

    sync_handler = HTTPHandler()
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_chat_response_claude_prompt_caching()

    mock_text = 'example text' * 512
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant that explains the content of the given text."
                }
            ]
        },
        {
            "role": "user", 
            "content": [
                {
                    "type": "text", 
                    "text": mock_text,
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        }
    ]

    with patch.object(HTTPHandler, "post", return_value=mock_response) as mock_post:
        response = litellm.completion(
            model="databricks/databricks-claude-3-7-sonnet",
            messages=messages,
            client=sync_handler,
            temperature=0.5
        )
        assert (
            mock_post.call_args.kwargs["headers"]["Content-Type"] == "application/json"
        )
        assert (
            mock_post.call_args.kwargs["headers"]["Authorization"]
            == f"Bearer {api_key}"
        )
        assert mock_post.call_args.kwargs["url"] == f"{base_url}/chat/completions"
        assert mock_post.call_args.kwargs["stream"] == False

        # TODO: add test for entire expected output schema in the future
        # Check the response object returned from litellm.completion()
        assert 'claude-3-7-sonnet' in response['model']
        assert response['usage']['cache_read_input_tokens'] == 0
        assert response['usage']['cache_creation_input_tokens'] == 1552
        assert response['usage']['prompt_tokens'] == 1556
        assert response['usage']['completion_tokens'] == 65
        assert response['usage']['total_tokens'] == 1621


def test_completion_with_prompt_caching_claude_model_repeat(monkeypatch):
    base_url = "https://my.workspace.cloud.databricks.com/serving-endpoints"
    api_key = "dapimykey"
    monkeypatch.setenv("DATABRICKS_API_BASE", base_url)
    monkeypatch.setenv("DATABRICKS_API_KEY", api_key)

    sync_handler = HTTPHandler()
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_chat_response_claude_prompt_caching_repeat()

    mock_text = 'example text' * 512
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant that explains the content of the given text."
                }
            ]
        },
        {
            "role": "user", 
            "content": [
                {
                    "type": "text", 
                    "text": mock_text,
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        }
    ]

    with patch.object(HTTPHandler, "post", return_value=mock_response) as mock_post:
        response = litellm.completion(
            model="databricks/databricks-claude-3-7-sonnet",
            messages=messages,
            client=sync_handler,
            temperature=0.5,
            extraparam="testpassingextraparam",
        )
        assert (
            mock_post.call_args.kwargs["headers"]["Content-Type"] == "application/json"
        )
        assert (
            mock_post.call_args.kwargs["headers"]["Authorization"]
            == f"Bearer {api_key}"
        )
        assert mock_post.call_args.kwargs["url"] == f"{base_url}/chat/completions"
        assert mock_post.call_args.kwargs["stream"] == False

        
        # TODO: add test for entire expected output schema in the future
        # Check the response object returned from litellm.completion()
        assert 'claude-3-7-sonnet' in response['model']
        assert response['usage']['cache_read_input_tokens'] == 1552
        assert response['usage']['cache_creation_input_tokens'] == 0
        assert response['usage']['prompt_tokens'] == 1556
        assert response['usage']['completion_tokens'] == 65
        assert response['usage']['total_tokens'] == 1621


def test_completion_with_prompt_caching_nonclaude_model(monkeypatch):
    base_url = "https://my.workspace.cloud.databricks.com/serving-endpoints"
    api_key = "dapimykey"
    monkeypatch.setenv("DATABRICKS_API_BASE", base_url)
    monkeypatch.setenv("DATABRICKS_API_KEY", api_key)

    sync_handler = HTTPHandler()
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_chat_response_nonclaude_prompt_caching()

    mock_text = 'example text' * 512
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant that explains the content of the given text."
                }
            ]
        },
        {
            "role": "user", 
            "content": [
                {
                    "type": "text", 
                    "text": mock_text,
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        }
    ]

    with patch.object(HTTPHandler, "post", return_value=mock_response) as mock_post:
        response = litellm.completion(
            model="databricks/databricks-gpt-oss-20b",
            messages=messages,
            client=sync_handler,
            temperature=0.5,
            extraparam="testpassingextraparam",
        )
        assert (
            mock_post.call_args.kwargs["headers"]["Content-Type"] == "application/json"
        )
        assert (
            mock_post.call_args.kwargs["headers"]["Authorization"]
            == f"Bearer {api_key}"
        )
        assert mock_post.call_args.kwargs["url"] == f"{base_url}/chat/completions"
        assert mock_post.call_args.kwargs["stream"] == False

        # TODO: add test for entire expected output schema in the future
        # Check the response object returned from litellm.completion()
        assert 'gpt-oss-20b' in response['model']
        assert ('cache_read_input_tokens' not in response['usage']) or response['usage']['cache_read_input_tokens'] in [0, None]
        assert ('cache_creation_input_tokens' not in response['usage']) or response['usage']['cache_creation_input_tokens'] in [0, None]
        assert response['usage']['prompt_tokens'] == 1638
        assert response['usage']['completion_tokens'] == 500
        assert response['usage']['total_tokens'] == 2138
        