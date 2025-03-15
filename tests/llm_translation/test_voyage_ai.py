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