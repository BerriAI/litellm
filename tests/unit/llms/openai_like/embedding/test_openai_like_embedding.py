"""
Test cases for OpenAI-like embedding handler
"""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest

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
        assert (
            "encoding_format" not in sent_data
        ), "encoding_format=None should be filtered out from the request payload"

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
        assert (
            "encoding_format" not in sent_data
        ), "encoding_format='' (empty string) should be filtered out from the request payload"

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
        assert (
            "encoding_format" in sent_data
        ), "encoding_format='float' should be preserved in the request payload"
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
        assert (
            "encoding_format" in sent_data
        ), "encoding_format='base64' should be preserved in the request payload"
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

    def test_extra_headers_sent_as_http_headers_not_body(self):
        """
        Test that extra headers passed via headers argument are sent as HTTP headers
        and NOT included in the JSON payload body.
        """
        handler = OpenAILikeEmbeddingHandler()

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

        mock_logging = MagicMock()

        handler.embedding(
            model="test-model",
            input=["test input"],
            timeout=60.0,
            logging_obj=mock_logging,
            api_key="test-key",
            api_base="http://test.com",
            optional_params={},
            client=mock_client,
            headers={"X-Custom-Header": "test-value"},
        )

        assert mock_client.post.called
        call_args = mock_client.post.call_args
        sent_headers = call_args[1]["headers"]
        sent_data = json.loads(call_args[1]["data"])

        # Header must be present in HTTP headers
        assert sent_headers.get("X-Custom-Header") == "test-value"
        assert sent_headers.get("Authorization") == "Bearer test-key"
        assert sent_headers.get("Content-Type") == "application/json"

        # extra_headers must NOT be in JSON payload body
        assert "extra_headers" not in sent_data
        assert sent_data["model"] == "test-model"
        assert sent_data["input"] == ["test input"]

    def test_extra_headers_in_optional_params_extracted_to_http_headers(self):
        """
        Test that extra_headers inside optional_params are extracted to HTTP headers
        and stripped from the request body.
        """
        handler = OpenAILikeEmbeddingHandler()

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

        mock_logging = MagicMock()
        optional_params = {"extra_headers": {"X-Tenant-ID": "tenant-99"}}

        handler.embedding(
            model="test-model",
            input=["test input"],
            timeout=60.0,
            logging_obj=mock_logging,
            api_key="test-key",
            api_base="http://test.com",
            optional_params=optional_params,
            client=mock_client,
        )

        assert mock_client.post.called
        call_args = mock_client.post.call_args
        sent_headers = call_args[1]["headers"]
        sent_data = json.loads(call_args[1]["data"])

        assert sent_headers.get("X-Tenant-ID") == "tenant-99"
        assert "extra_headers" not in sent_data
        assert "extra_headers" not in optional_params

    def test_litellm_embedding_openai_like_extra_headers(self):
        """
        Integration test through litellm.embedding:
        Ensure extra_headers passed to litellm.embedding(custom_llm_provider="openai_like", ...)
        are forwarded as HTTP request headers and not sent in the JSON body.
        """
        import litellm

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

        litellm.embedding(
            model="test-model",
            input=["hello world"],
            custom_llm_provider="openai_like",
            api_base="http://test.com/v1",
            api_key="sk-test",
            extra_headers={"X-User-Id": "user-42"},
            client=mock_client,
        )

        assert mock_client.post.called
        call_args = mock_client.post.call_args
        sent_headers = call_args[1]["headers"]
        sent_data = json.loads(call_args[1]["data"])

        assert sent_headers.get("X-User-Id") == "user-42"
        assert "extra_headers" not in sent_data
        assert sent_data["model"] == "test-model"
        assert sent_data["input"] == ["hello world"]

    @pytest.mark.asyncio
    async def test_litellm_aembedding_openai_like_extra_headers(self):
        """
        Integration test through litellm.aembedding:
        Ensure extra_headers passed to litellm.aembedding(custom_llm_provider="openai_like", ...)
        are forwarded as HTTP request headers and not sent in the JSON body.
        """
        from unittest.mock import AsyncMock

        import litellm
        from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

        mock_client = AsyncMock(spec=AsyncHTTPHandler)
        mock_response = Mock()
        mock_response.json.return_value = {
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
            "model": "test-model",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response

        await litellm.aembedding(
            model="test-model",
            input=["hello world"],
            custom_llm_provider="openai_like",
            api_base="http://test.com/v1",
            api_key="sk-test",
            extra_headers={"X-User-Id": "user-42"},
            client=mock_client,
        )

        assert mock_client.post.called
        call_args = mock_client.post.call_args
        sent_headers = call_args[1]["headers"]
        sent_data = json.loads(call_args[1]["data"])

        assert sent_headers.get("X-User-Id") == "user-42"
        assert "extra_headers" not in sent_data
        assert sent_data["model"] == "test-model"
        assert sent_data["input"] == ["hello world"]

    def test_case_insensitive_content_type_no_duplicate(self):
        """
        Test that when a caller passes lowercase 'content-type', a duplicate
        'Content-Type' header is not added.
        """
        handler = OpenAILikeEmbeddingHandler()

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

        mock_logging = MagicMock()

        handler.embedding(
            model="test-model",
            input=["test input"],
            timeout=60.0,
            logging_obj=mock_logging,
            api_key="test-key",
            api_base="http://test.com",
            optional_params={},
            client=mock_client,
            headers={"content-type": "application/json; charset=utf-8"},
        )

        assert mock_client.post.called
        call_args = mock_client.post.call_args
        sent_headers = call_args[1]["headers"]

        assert sent_headers.get("content-type") == "application/json; charset=utf-8"
        assert "Content-Type" not in sent_headers

    def test_caller_headers_not_mutated(self):
        """
        Test that caller-provided headers are not mutated in-place when extra_headers
        or defaults (Content-Type, Authorization) are applied.
        """
        handler = OpenAILikeEmbeddingHandler()

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

        mock_logging = MagicMock()
        caller_headers = {"X-Custom": "original-val"}
        optional_params = {"extra_headers": {"X-Extra": "added-val"}}

        handler.embedding(
            model="test-model",
            input=["test input"],
            timeout=60.0,
            logging_obj=mock_logging,
            api_key="test-key",
            api_base="http://test.com",
            optional_params=optional_params,
            client=mock_client,
            headers=caller_headers,
        )

        # caller_headers must not have been mutated
        assert caller_headers == {"X-Custom": "original-val"}

    def test_validate_environment_does_not_mutate_headers(self):
        """
        Test that _validate_environment returns a fresh dict with defaults
        without mutating the caller's input headers dictionary.
        """
        handler = OpenAILikeEmbeddingHandler()
        original_headers = {"X-My-Header": "value"}

        api_base, resolved_headers = handler._validate_environment(
            api_key="sk-test-key",
            api_base="http://test.com",
            endpoint_type="embeddings",
            headers=original_headers,
            custom_endpoint=None,
        )

        assert original_headers == {"X-My-Header": "value"}
        assert resolved_headers["X-My-Header"] == "value"
        assert resolved_headers["Content-Type"] == "application/json"
        assert resolved_headers["Authorization"] == "Bearer sk-test-key"

    def test_case_insensitive_authorization_no_duplicate(self):
        """
        Test that when a caller passes lowercase 'authorization', a duplicate
        'Authorization' header is not added even when api_key is set.
        """
        handler = OpenAILikeEmbeddingHandler()

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

        mock_logging = MagicMock()

        handler.embedding(
            model="test-model",
            input=["test input"],
            timeout=60.0,
            logging_obj=mock_logging,
            api_key="test-key",
            api_base="http://test.com",
            optional_params={},
            client=mock_client,
            headers={"authorization": "Bearer custom-token"},
        )

        assert mock_client.post.called
        call_args = mock_client.post.call_args
        sent_headers = call_args[1]["headers"]

        assert sent_headers.get("authorization") == "Bearer custom-token"
        assert "Authorization" not in sent_headers

    def test_validate_environment_case_insensitive_authorization(self):
        """
        Test that _validate_environment detects lowercase 'authorization'
        case-insensitively and does not add a duplicate 'Authorization' header.
        """
        handler = OpenAILikeEmbeddingHandler()
        original_headers = {"authorization": "Bearer custom-token"}

        api_base, resolved_headers = handler._validate_environment(
            api_key="sk-test-key",
            api_base="http://test.com",
            endpoint_type="embeddings",
            headers=original_headers,
            custom_endpoint=None,
        )

        assert original_headers == {"authorization": "Bearer custom-token"}
        assert resolved_headers.get("authorization") == "Bearer custom-token"
        assert "Authorization" not in resolved_headers


