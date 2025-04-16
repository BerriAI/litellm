import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


from base_embedding_unit_tests import BaseLLMEmbeddingTest
from test_rerank import assert_response_shape
import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from unittest.mock import patch, MagicMock


class TestVoyageAI(BaseLLMEmbeddingTest):
    def get_custom_llm_provider(self) -> litellm.LlmProviders:
        return litellm.LlmProviders.VOYAGE

    def get_base_embedding_call_args(self) -> dict:
        return {
            "model": "voyage/voyage-3-lite",
        }


def test_voyage_ai_embedding_extra_params():
    try:

        client = HTTPHandler()
        litellm.set_verbose = True

        with patch.object(client, "post") as mock_client:
            response = litellm.embedding(
                model="voyage/voyage-3-lite",
                input=["a"],
                dimensions=512,
                input_type="document",
                client=client,
            )

            mock_client.assert_called_once()
            json_data = json.loads(mock_client.call_args.kwargs["data"])

            print("request data to voyage ai", json.dumps(json_data, indent=4))

            # Assert the request parameters
            assert json_data["input"] == ["a"]
            assert json_data["model"] == "voyage-3-lite"
            assert json_data["output_dimension"] == 512
            assert json_data["input_type"] == "document"

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_voyage_ai_embedding_prompt_token_mapping():
    try:

        client = HTTPHandler()
        litellm.set_verbose = True

        with patch.object(client, "post", return_value=MagicMock(status_code=200, json=lambda: {"usage": {"total_tokens": 120}})) as mock_client:
            response = litellm.embedding(
                model="voyage/voyage-3-lite",
                input=["a"],
                dimensions=512,
                input_type="document",
                client=client,
            )

            mock_client.assert_called_once()
            # Assert the response
            assert response.usage.prompt_tokens == 120
            assert response.usage.total_tokens == 120

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


### Rerank Tests
@pytest.mark.asyncio()
async def test_voyage_ai_rerank():
    mock_response = AsyncMock()

    def return_val():
        return {
            "id": "cmpl-mockid",
            "results": [{"index": 0, "relevance_score": 0.95}],
            "usage": {"total_tokens": 150},
        }

    mock_response.json = return_val
    mock_response.headers = {"key": "value"}
    mock_response.status_code = 200

    expected_payload = {
        "model": "rerank-model",
        "query": "hello",
        "top_k": 1,
        "documents": ["hello", "world", "foo", "bar"],
    }

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        response = await litellm.arerank(
            model="voyage/rerank-model",
            query="hello",
            documents=["hello", "world", "foo", "bar"],
            top_k=1,
        )

        print("async re rank response: ", response)

        # Assert
        mock_post.assert_called_once()
        args_to_api = mock_post.call_args.kwargs["data"]
        _url = mock_post.call_args.kwargs["url"]
        assert _url == "https://api.voyageai.com/v1/rerank"

        request_data = json.loads(args_to_api)
        assert request_data["query"] == expected_payload["query"]
        assert request_data["documents"] == expected_payload["documents"]
        assert request_data["top_k"] == expected_payload["top_k"]
        assert request_data["model"] == expected_payload["model"]

        assert response.id is not None
        assert response.results is not None
        assert response.meta["tokens"]["total_tokens"] == 150

        assert_response_shape(response, custom_llm_provider="voyage")
            