"""
Test cases for OpenAI-like embedding handler
"""

import json
from datetime import datetime, timezone
from typing import Final
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest
import respx

import litellm

from litellm.llms.openai_like.embedding.handler import OpenAILikeEmbeddingHandler
from litellm.types.utils import EmbeddingResponse


class TestOpenAILikeEmbeddingHandler:
    """Test OpenAI-like embedding handler functionality"""

    def test_encoding_format_none_filtered_out(self):
        """
        Test that encoding_format=None is filtered out from the request payload.

        According to OpenAI API spec, encoding_format should be omitted if not specified,
        not sent as None or empty string. This prevents errors with providers like VLLM
        that reject empty encoding_format values.
        """
        handler = OpenAILikeEmbeddingHandler()

        # Mock the HTTP client
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
            "model": "test-model",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response

        # Mock logging object
        mock_logging = MagicMock()

        # Call embedding with encoding_format=None
        optional_params = {"encoding_format": None}

        with patch.object(
            handler,
            "_validate_environment",
            return_value=("http://test.com/v1/embeddings", {}),
        ):
            response = handler.embedding(
                model="test-model",
                input=["test input"],
                timeout=60.0,
                logging_obj=mock_logging,
                api_key="test-key",
                api_base="http://test.com",
                optional_params=optional_params,
                client=mock_client,
            )

        # Verify the request was made
        assert mock_client.post.called

        # Get the data that was sent in the request
        call_args = mock_client.post.call_args
        sent_data = json.loads(call_args[1]["data"])

        # Assert that encoding_format is NOT in the sent data
        assert "encoding_format" not in sent_data, (
            "encoding_format=None should be filtered out from the request payload"
        )

        # Assert that model and input are still present
        assert sent_data["model"] == "test-model"
        assert sent_data["input"] == ["test input"]

    def test_encoding_format_empty_string_filtered_out(self):
        """
        Test that encoding_format="" (empty string) is filtered out from the request payload.

        This is the specific case mentioned in the issue where VLLM rejects empty string
        encoding_format values with error: "unknown variant ``, expected float or base64"
        """
        handler = OpenAILikeEmbeddingHandler()

        # Mock the HTTP client
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
            "model": "test-model",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response

        # Mock logging object
        mock_logging = MagicMock()

        # Call embedding with encoding_format="" (empty string)
        optional_params = {"encoding_format": ""}

        with patch.object(
            handler,
            "_validate_environment",
            return_value=("http://test.com/v1/embeddings", {}),
        ):
            response = handler.embedding(
                model="test-model",
                input=["test input"],
                timeout=60.0,
                logging_obj=mock_logging,
                api_key="test-key",
                api_base="http://test.com",
                optional_params=optional_params,
                client=mock_client,
            )

        # Verify the request was made
        assert mock_client.post.called

        # Get the data that was sent in the request
        call_args = mock_client.post.call_args
        sent_data = json.loads(call_args[1]["data"])

        # Assert that encoding_format is NOT in the sent data
        assert "encoding_format" not in sent_data, (
            "encoding_format='' (empty string) should be filtered out from the request payload"
        )

    def test_encoding_format_float_preserved(self):
        """
        Test that encoding_format="float" is preserved in the request payload.
        """
        handler = OpenAILikeEmbeddingHandler()

        # Mock the HTTP client
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
            "model": "test-model",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response

        # Mock logging object
        mock_logging = MagicMock()

        # Call embedding with encoding_format="float"
        optional_params = {"encoding_format": "float"}

        with patch.object(
            handler,
            "_validate_environment",
            return_value=("http://test.com/v1/embeddings", {}),
        ):
            response = handler.embedding(
                model="test-model",
                input=["test input"],
                timeout=60.0,
                logging_obj=mock_logging,
                api_key="test-key",
                api_base="http://test.com",
                optional_params=optional_params,
                client=mock_client,
            )

        # Verify the request was made
        assert mock_client.post.called

        # Get the data that was sent in the request
        call_args = mock_client.post.call_args
        sent_data = json.loads(call_args[1]["data"])

        # Assert that encoding_format IS in the sent data with correct value
        assert "encoding_format" in sent_data, "encoding_format='float' should be preserved in the request payload"
        assert sent_data["encoding_format"] == "float"

    def test_encoding_format_base64_preserved(self):
        """
        Test that encoding_format="base64" is preserved in the request payload.
        """
        handler = OpenAILikeEmbeddingHandler()

        # Mock the HTTP client
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
            "model": "test-model",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response

        # Mock logging object
        mock_logging = MagicMock()

        # Call embedding with encoding_format="base64"
        optional_params = {"encoding_format": "base64"}

        with patch.object(
            handler,
            "_validate_environment",
            return_value=("http://test.com/v1/embeddings", {}),
        ):
            response = handler.embedding(
                model="test-model",
                input=["test input"],
                timeout=60.0,
                logging_obj=mock_logging,
                api_key="test-key",
                api_base="http://test.com",
                optional_params=optional_params,
                client=mock_client,
            )

        # Verify the request was made
        assert mock_client.post.called

        # Get the data that was sent in the request
        call_args = mock_client.post.call_args
        sent_data = json.loads(call_args[1]["data"])

        # Assert that encoding_format IS in the sent data with correct value
        assert "encoding_format" in sent_data, "encoding_format='base64' should be preserved in the request payload"
        assert sent_data["encoding_format"] == "base64"

    def test_other_optional_params_preserved(self):
        """
        Test that other optional parameters are preserved when encoding_format is filtered.
        """
        handler = OpenAILikeEmbeddingHandler()

        # Mock the HTTP client
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
            "model": "test-model",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response

        # Mock logging object
        mock_logging = MagicMock()

        # Call embedding with encoding_format=None and other params
        optional_params = {
            "encoding_format": None,
            "dimensions": 512,
            "user": "test-user",
        }

        with patch.object(
            handler,
            "_validate_environment",
            return_value=("http://test.com/v1/embeddings", {}),
        ):
            response = handler.embedding(
                model="test-model",
                input=["test input"],
                timeout=60.0,
                logging_obj=mock_logging,
                api_key="test-key",
                api_base="http://test.com",
                optional_params=optional_params,
                client=mock_client,
            )

        # Verify the request was made
        assert mock_client.post.called

        # Get the data that was sent in the request
        call_args = mock_client.post.call_args
        sent_data = json.loads(call_args[1]["data"])

        # Assert that encoding_format is NOT in the sent data
        assert "encoding_format" not in sent_data

        # Assert that other parameters ARE preserved
        assert sent_data["dimensions"] == 512
        assert sent_data["user"] == "test-user"
        assert sent_data["model"] == "test-model"
        assert sent_data["input"] == ["test input"]

    def test_no_optional_params(self):
        """
        Test that the handler works correctly when no optional params are provided.
        """
        handler = OpenAILikeEmbeddingHandler()

        # Mock the HTTP client
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
            "model": "test-model",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response

        # Mock logging object
        mock_logging = MagicMock()

        # Call embedding with empty optional_params
        optional_params = {}

        with patch.object(
            handler,
            "_validate_environment",
            return_value=("http://test.com/v1/embeddings", {}),
        ):
            response = handler.embedding(
                model="test-model",
                input=["test input"],
                timeout=60.0,
                logging_obj=mock_logging,
                api_key="test-key",
                api_base="http://test.com",
                optional_params=optional_params,
                client=mock_client,
            )

        # Verify the request was made
        assert mock_client.post.called

        # Get the data that was sent in the request
        call_args = mock_client.post.call_args
        sent_data = json.loads(call_args[1]["data"])

        # Assert that only model and input are in the sent data
        assert sent_data["model"] == "test-model"
        assert sent_data["input"] == ["test input"]
        assert "encoding_format" not in sent_data


_OPENAI_LIKE_EMBEDDING_BODY: Final = {
    "object": "list",
    "data": [{"object": "embedding", "index": 0, "embedding": [0.5, 0.25]}],
    "model": "jina-embeddings-v3",
    "usage": {"prompt_tokens": 2, "total_tokens": 2},
}


def test_llamafile_embedding_posts_to_the_env_api_base(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLAMAFILE_API_BASE", "http://llamafile.test/v1")
    route: Final = respx_mock.post("http://llamafile.test/v1/embeddings").respond(json=_OPENAI_LIKE_EMBEDDING_BODY)

    response: Final = litellm.embedding(model="llamafile/jina-embeddings-v3", input=["Hello world"])

    assert route.call_count == 1
    assert json.loads(route.calls.last.request.content) == {"model": "jina-embeddings-v3", "input": ["Hello world"]}
    assert response.data[0]["embedding"] == [0.5, 0.25]


@pytest.mark.parametrize("sync_mode", [True, False])
async def test_lm_studio_embedding_posts_to_the_env_api_base(
    sync_mode: bool, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("LM_STUDIO_API_BASE", "http://lm-studio.test")
    route: Final = respx_mock.post("http://lm-studio.test/embeddings").respond(json=_OPENAI_LIKE_EMBEDDING_BODY)
    request: Final = {"model": "lm_studio/jina-embeddings-v3", "input": ["Hello world"]}

    response: Final = litellm.embedding(**request) if sync_mode else await litellm.aembedding(**request)

    assert route.call_count == 1
    assert json.loads(route.calls.last.request.content) == {"model": "jina-embeddings-v3", "input": ["Hello world"]}
    assert response.data[0]["embedding"] == [0.5, 0.25]


@pytest.mark.parametrize("provider", ("openai_like", "llamafile", "lm_studio"))
@pytest.mark.parametrize("is_async", (False, True))
@pytest.mark.parametrize("header_mode", ("headers", "extra_headers", "both", "empty", "custom_auth"))
async def test_embedding_headers_stay_on_the_http_request(
    provider: str, is_async: bool, header_mode: str, respx_mock: respx.MockRouter
) -> None:
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    route: Final = respx_mock.post("https://embedding.test/v1/embeddings").respond(json=_OPENAI_LIKE_EMBEDDING_BODY)
    headers: Final = (
        {"authorization": "Basic caller-token", "content-type": "application/json; charset=utf-8"}
        if header_mode == "custom_auth"
        else {"X-Main": "main-value"}
        if header_mode in ("headers", "both")
        else {}
    )
    extra_headers: Final = {"X-Extra": "extra-value"} if header_mode in ("extra_headers", "both") else {}
    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(respx_mock.handler))
    try:
        response: Final = (
            await litellm.aembedding(
                model=f"{provider}/test-model",
                input=["hello"],
                api_base="https://embedding.test/v1",
                api_key="test-key",
                headers=headers,
                extra_headers=extra_headers,
                client=client,
            )
            if is_async
            else litellm.embedding(
                model=f"{provider}/test-model",
                input=["hello"],
                api_base="https://embedding.test/v1",
                api_key="test-key",
                headers=headers,
                extra_headers=extra_headers,
            )
        )
    finally:
        await client.close()
    request: Final = route.calls.last.request
    assert route.call_count == 1
    assert request.headers.get_list("content-type") == [headers.get("content-type", "application/json")]
    assert request.headers.get_list("authorization") == [headers.get("authorization", "Bearer test-key")]
    for name, value in {**headers, **extra_headers}.items():
        assert request.headers[name] == value
    assert json.loads(request.content) == {"model": "test-model", "input": ["hello"]}
    assert response.data == _OPENAI_LIKE_EMBEDDING_BODY["data"]


def test_direct_embedding_headers_preserve_caller_arguments(respx_mock: respx.MockRouter) -> None:
    from litellm.litellm_core_utils.litellm_logging import Logging

    route: Final = respx_mock.post("https://embedding.test/v1/embeddings").respond(json=_OPENAI_LIKE_EMBEDDING_BODY)
    headers: Final = {"X-Main": "main-value"}
    params: Final = {"extra_headers": {"X-Extra": "extra-value"}, "user": "test-user"}
    logging: Final = Logging(
        model="test-model",
        messages=["hello"],
        stream=False,
        call_type="embedding",
        start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        litellm_call_id="test-call",
        function_id="test-function",
    )
    response: Final = OpenAILikeEmbeddingHandler().embedding(
        model="test-model",
        input=["hello"],
        timeout=10.0,
        logging_obj=logging,
        api_key="test-key",
        api_base="https://embedding.test/v1",
        optional_params=params,
        headers=headers,
        model_response=EmbeddingResponse(),
    )
    request: Final = route.calls.last.request
    assert route.call_count == 1
    assert request.headers["content-type"] == "application/json"
    assert request.headers["x-main"] == headers["X-Main"]
    assert request.headers["x-extra"] == "extra-value"
    assert json.loads(request.content) == {"model": "test-model", "input": ["hello"], "user": "test-user"}
    assert response.data == _OPENAI_LIKE_EMBEDDING_BODY["data"]
    assert headers == {"X-Main": "main-value"}
    assert params == {"extra_headers": {"X-Extra": "extra-value"}, "user": "test-user"}
