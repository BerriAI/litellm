"""
Test transformation logic for hosted_vllm embeddings.

This test verifies that the transformation layer correctly handles parameters,
especially ensuring that encoding_format is not included when not provided.
"""

import json
from typing import Final
from unittest.mock import Mock, patch

import httpx
import pytest
import respx
from pydantic import TypeAdapter

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.hosted_vllm.embedding.transformation import (
    HostedVLLMEmbeddingConfig,
)
from litellm.types.utils import EmbeddingResponse

_API_BASE: Final = "https://vllm.example.com/v1"
_API_URL: Final = f"{_API_BASE}/embeddings"
_MODEL: Final = "hosted_vllm/nomic-ai/nomic-embed-text-v1.5"
_UPSTREAM_MODEL: Final = "nomic-ai/nomic-embed-text-v1.5"
_JSON_OBJECT: Final = TypeAdapter(dict[str, object])


def _request_json(request: httpx.Request) -> dict[str, object]:
    return _JSON_OBJECT.validate_json(request.content)


async def _embed(sync_mode: bool, router: respx.MockRouter, embedding_input: str | list[str]) -> EmbeddingResponse:
    if sync_mode:
        return litellm.embedding(model=_MODEL, input=embedding_input, api_base=_API_BASE)
    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(router.async_handler))
    return await litellm.aembedding(model=_MODEL, input=embedding_input, api_base=_API_BASE, client=client)


class TestHostedVLLMEmbeddingTransformation:
    """Test suite for hosted_vllm embedding transformation logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = HostedVLLMEmbeddingConfig()
        self.model = "hosted_vllm/BAAI/bge-small-en-v1.5"

    def test_transform_embedding_request_basic(self):
        """Test basic embedding request transformation."""
        input_data = ["hello world"]
        result = self.config.transform_embedding_request(
            model=self.model,
            input=input_data,
            optional_params={},
            headers={},
        )

        expected_result = {
            "model": "BAAI/bge-small-en-v1.5",  # prefix should be stripped
            "input": input_data,
        }
        assert result == expected_result

    def test_transform_embedding_request_string_input(self):
        """Test that string input is converted to list."""
        input_data = "hello world"
        result = self.config.transform_embedding_request(
            model=self.model,
            input=input_data,
            optional_params={},
            headers={},
        )

        assert result["input"] == ["hello world"]
        assert result["model"] == "BAAI/bge-small-en-v1.5"

    def test_transform_embedding_request_with_dimensions(self):
        """Test embedding request with dimensions parameter."""
        input_data = ["hello world"]
        optional_params = {"dimensions": 384}

        result = self.config.transform_embedding_request(
            model=self.model,
            input=input_data,
            optional_params=optional_params,
            headers={},
        )

        assert result["model"] == "BAAI/bge-small-en-v1.5"
        assert result["input"] == input_data
        assert result["dimensions"] == 384

    def test_encoding_format_not_included_when_not_provided(self):
        """
        Test that encoding_format is NOT included in the request when not provided.

        This is critical because vLLM rejects requests with encoding_format=None or
        encoding_format="" with error: "unknown variant ``, expected float or base64"
        """
        input_data = ["hello world"]

        # Test with no encoding_format in optional_params
        result = self.config.transform_embedding_request(
            model=self.model,
            input=input_data,
            optional_params={},
            headers={},
        )

        assert "encoding_format" not in result, "encoding_format should not be in request when not provided"

    def test_encoding_format_not_included_when_none(self):
        """
        Test that encoding_format is NOT included when explicitly set to None.
        """
        input_data = ["hello world"]
        optional_params = {"encoding_format": None}

        result = self.config.transform_embedding_request(
            model=self.model,
            input=input_data,
            optional_params=optional_params,
            headers={},
        )

        # encoding_format=None should be passed through, but filtered later
        # by the HTTP handler
        assert result.get("encoding_format") is None

    def test_encoding_format_included_when_float(self):
        """Test that encoding_format is included when set to 'float'."""
        input_data = ["hello world"]
        optional_params = {"encoding_format": "float"}

        result = self.config.transform_embedding_request(
            model=self.model,
            input=input_data,
            optional_params=optional_params,
            headers={},
        )

        assert result["encoding_format"] == "float"

    def test_encoding_format_included_when_base64(self):
        """Test that encoding_format is included when set to 'base64'."""
        input_data = ["hello world"]
        optional_params = {"encoding_format": "base64"}

        result = self.config.transform_embedding_request(
            model=self.model,
            input=input_data,
            optional_params=optional_params,
            headers={},
        )

        assert result["encoding_format"] == "base64"

    def test_get_supported_openai_params(self):
        """Test that supported OpenAI parameters are correctly listed."""
        supported = self.config.get_supported_openai_params(self.model)

        assert "timeout" in supported
        assert "dimensions" in supported
        assert "encoding_format" in supported
        assert "user" in supported

    def test_map_openai_params(self):
        """Test mapping of OpenAI parameters."""
        non_default_params = {
            "dimensions": 512,
            "encoding_format": "float",
            "user": "test-user",
        }

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert result["dimensions"] == 512
        assert result["encoding_format"] == "float"
        assert result["user"] == "test-user"

    def test_map_openai_params_filters_unsupported(self):
        """Test that unsupported parameters are not mapped."""
        non_default_params = {
            "dimensions": 512,
            "unsupported_param": "value",
        }

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert result["dimensions"] == 512
        assert "unsupported_param" not in result

    def test_get_complete_url(self):
        """Test URL construction for embeddings endpoint."""
        api_base = "https://test-vllm.example.com/v1"

        url = self.config.get_complete_url(
            api_base=api_base,
            api_key="test-key",
            model=self.model,
            optional_params={},
            litellm_params={},
        )

        assert url == "https://test-vllm.example.com/v1/embeddings"

    def test_get_complete_url_adds_embeddings_suffix(self):
        """Test that /embeddings is added if not present."""
        api_base = "https://test-vllm.example.com"

        url = self.config.get_complete_url(
            api_base=api_base,
            api_key="test-key",
            model=self.model,
            optional_params={},
            litellm_params={},
        )

        assert url == "https://test-vllm.example.com/embeddings"

    def test_validate_environment_with_api_key(self):
        """Test environment validation with API key."""
        headers = {}

        result = self.config.validate_environment(
            headers=headers,
            model=self.model,
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="test-api-key",
        )

        assert "Authorization" in result
        assert result["Authorization"] == "Bearer test-api-key"
        assert result["Content-Type"] == "application/json"

    def test_encoding_format_not_sent_in_actual_request(self):
        """
        E2E test that encoding_format is not sent when not provided.

        This test mocks the HTTP client to verify the actual request payload.
        Patches HTTPHandler.post at the class level so the mock is used when
        base_llm_http_handler calls sync_httpx_client.post() with the passed client.
        """
        from litellm.llms.custom_httpx.http_handler import HTTPHandler

        client = HTTPHandler()

        with patch.object(HTTPHandler, "post") as mock_post:
            # Mock response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_response.json.return_value = {
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "index": 0,
                        "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                    }
                ],
                "model": "BAAI/bge-small-en-v1.5",
                "usage": {
                    "prompt_tokens": 5,
                    "total_tokens": 5,
                },
            }
            mock_response.text = json.dumps(mock_response.json.return_value)
            mock_post.return_value = mock_response

            litellm.embedding(
                model=self.model,
                input=["Hello world"],
                api_base="https://test-vllm.example.com/v1",
                client=client,
                caching=False,
            )

            # Verify the request was made
            mock_post.assert_called_once()

            # Get the data that was sent
            call_kwargs = mock_post.call_args[1]
            sent_data = json.loads(call_kwargs["data"])

            # Assert that encoding_format is NOT in the sent data
            assert "encoding_format" not in sent_data, "encoding_format should not be in request when not provided"
            assert sent_data["model"] == "BAAI/bge-small-en-v1.5"
            assert sent_data["input"] == ["Hello world"]

    @pytest.mark.parametrize(
        "provider_params",
        [
            {"extra_body": {"truncate": "END", "input_type": "query"}},
            {"truncate": "END", "input_type": "query"},
        ],
    )
    def test_provider_params_are_sent_at_the_top_level_of_the_request(self, provider_params: dict[str, object]) -> None:
        from litellm.llms.custom_httpx.http_handler import HTTPHandler

        client = HTTPHandler()

        with patch.object(HTTPHandler, "post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_response.json.return_value = {
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                "model": "nvidia/nv-embedqa-e5-v5",
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            }
            mock_response.text = json.dumps(mock_response.json.return_value)
            mock_post.return_value = mock_response

            litellm.embedding(
                model="hosted_vllm/nvidia/nv-embedqa-e5-v5",
                input=["Hello world"],
                api_base="https://integrate.api.nvidia.com/v1",
                api_key="fake-key",
                client=client,
                caching=False,
                **provider_params,
            )

            sent_data = json.loads(mock_post.call_args.kwargs["data"])

        assert sent_data["truncate"] == "END"
        assert sent_data["input_type"] == "query"
        assert "extra_body" not in sent_data
        assert sent_data["model"] == "nvidia/nv-embedqa-e5-v5"
        assert sent_data["input"] == ["Hello world"]


@pytest.mark.parametrize("sync_mode", [True, False])
async def test_hosted_vllm_embedding_sends_model_and_parses_usage(
    respx_mock: respx.MockRouter, sync_mode: bool
) -> None:
    route: Final = respx_mock.post(_API_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
                "model": _UPSTREAM_MODEL,
                "usage": {"prompt_tokens": 4, "total_tokens": 4},
            },
        )
    )

    response: Final = await _embed(sync_mode, respx_mock, "hello vLLM")

    request: Final = route.calls[0].request
    assert str(request.url) == _API_URL
    assert _request_json(request) == {"model": _UPSTREAM_MODEL, "input": ["hello vLLM"]}
    assert response.data[0]["index"] == 0
    assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]
    assert response.usage.total_tokens == 4


@pytest.mark.parametrize("sync_mode", [True, False])
async def test_hosted_vllm_embedding_preserves_order_for_multiple_inputs(
    respx_mock: respx.MockRouter, sync_mode: bool
) -> None:
    route: Final = respx_mock.post(_API_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
                    {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
                    {"object": "embedding", "index": 2, "embedding": [0.5, 0.6]},
                ],
                "model": _UPSTREAM_MODEL,
                "usage": {"prompt_tokens": 9, "total_tokens": 9},
            },
        )
    )
    inputs: Final = ["first sentence", "second sentence", "third sentence"]

    response: Final = await _embed(sync_mode, respx_mock, inputs)

    assert _request_json(route.calls[0].request) == {"model": _UPSTREAM_MODEL, "input": inputs}
    assert tuple(item["index"] for item in response.data) == (0, 1, 2)
    assert tuple(tuple(item["embedding"]) for item in response.data) == ((0.1, 0.2), (0.3, 0.4), (0.5, 0.6))


def test_hosted_vllm_embedding_sends_api_key_as_bearer_token(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(_API_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.7, 0.8]}],
                "model": _UPSTREAM_MODEL,
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )
    )

    response: Final = litellm.embedding(
        model=_MODEL,
        input="authenticated input",
        api_base=_API_BASE,
        api_key="hosted-vllm-test-key",
    )

    request: Final = route.calls[0].request
    assert request.headers["Authorization"] == "Bearer hosted-vllm-test-key"
    assert _request_json(request) == {"model": _UPSTREAM_MODEL, "input": ["authenticated input"]}
    assert response.data[0]["embedding"] == [0.7, 0.8]


def test_hosted_vllm_embedding_repeats_identical_input_and_result(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(_API_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.25, 0.75]}],
                "model": _UPSTREAM_MODEL,
                "usage": {"prompt_tokens": 8, "total_tokens": 8},
            },
        )
    )
    input_text: Final = "repeated embedding input"

    responses: Final = tuple(
        litellm.embedding(model=_MODEL, input=input_text, api_base=_API_BASE) for _ in range(2)
    )

    assert tuple(_request_json(call.request) for call in route.calls) == (
        {"model": _UPSTREAM_MODEL, "input": [input_text]},
        {"model": _UPSTREAM_MODEL, "input": [input_text]},
    )
    assert tuple(tuple(response.data[0]["embedding"]) for response in responses) == ((0.25, 0.75), (0.25, 0.75))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
