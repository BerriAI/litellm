import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)

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
