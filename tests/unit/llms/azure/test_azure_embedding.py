import os
import sys
from typing import Final
from unittest.mock import MagicMock, patch

import pytest
import respx
from httpx import Response

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)

import litellm
from litellm.llms.azure.azure import AzureChatCompletion
from litellm.types.utils import EmbeddingResponse, Usage


def _make_embedding_response() -> EmbeddingResponse:
    return EmbeddingResponse(
        model="text-embedding-3-large",
        usage=Usage(prompt_tokens=3, completion_tokens=0, total_tokens=3),
        data=[{"embedding": [0.1, 0.2, 0.3], "index": 0, "object": "embedding"}],
    )


def _make_logging_obj() -> MagicMock:
    return MagicMock()


class TestAzureV1AsyncEmbedding:
    def test_aembedding_receives_api_version(self):
        """Regression: api_version must be forwarded to aembedding() when aembedding=True.
        Without the fix, it was silently dropped, causing AsyncAzureOpenAI to be used
        instead of AsyncOpenAI for Azure AI Foundry (v1) endpoints. Fixes #24848."""
        handler = AzureChatCompletion()

        with patch.object(handler, "aembedding") as mock_aembedding:
            handler.embedding(
                model="text-embedding-3-large",
                input=["hello world"],
                api_base="https://my-endpoint.openai.azure.com",
                api_version="v1",
                timeout=60.0,
                logging_obj=_make_logging_obj(),
                model_response=_make_embedding_response(),
                optional_params={},
                api_key="fake-key",
                aembedding=True,
                litellm_params={},
            )

        mock_aembedding.assert_called_once()
        _, kwargs = mock_aembedding.call_args
        assert kwargs.get("api_version") == "v1"

    def test_get_azure_openai_client_returns_async_openai_for_v1(self):
        from openai import AsyncAzureOpenAI, AsyncOpenAI

        handler = AzureChatCompletion()
        client = handler.get_azure_openai_client(
            api_key="fake-key",
            api_base="https://my-endpoint.openai.azure.com",
            api_version="v1",
            _is_async=True,
            litellm_params={},
        )

        assert isinstance(client, AsyncOpenAI)
        assert not isinstance(client, AsyncAzureOpenAI)

    def test_get_azure_openai_client_uses_v1_base_url(self):
        handler = AzureChatCompletion()
        client = handler.get_azure_openai_client(
            api_key="fake-key",
            api_base="https://my-endpoint.openai.azure.com",
            api_version="v1",
            _is_async=True,
            litellm_params={},
        )

        assert client is not None
        assert "/openai/v1/" in str(client.base_url)

    @pytest.mark.parametrize("api_version", ["v1", "latest", "preview"])
    def test_all_v1_variants_use_openai_client(self, api_version: str):
        from openai import AsyncOpenAI

        handler = AzureChatCompletion()
        client = handler.get_azure_openai_client(
            api_key="fake-key",
            api_base="https://my-endpoint.openai.azure.com",
            api_version=api_version,
            _is_async=True,
            litellm_params={},
        )

        assert isinstance(client, AsyncOpenAI)


@pytest.mark.respx(assert_all_called=True)
def test_azure_embedding_max_retries_zero_sends_one_request(respx_mock: respx.MockRouter) -> None:
    url: Final = "https://example-resource.openai.azure.com/openai/deployments/text-embedding-ada-002/embeddings"
    route: Final = respx_mock.post(url__startswith=url).mock(
        return_value=Response(500, json={"error": {"message": "temporary failure"}})
    )

    litellm.in_memory_llm_clients_cache.flush_cache()
    with pytest.raises(litellm.APIError):
        litellm.embedding(
            model="azure/text-embedding-ada-002",
            input=["hello"],
            api_base="https://example-resource.openai.azure.com",
            api_key="azure-test-key",
            api_version="2024-02-01",
            max_retries=0,
        )

    assert route.call_count == 1
