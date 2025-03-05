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


class TestNebius(BaseLLMEmbeddingTest):
    def get_custom_llm_provider(self) -> litellm.LlmProviders:
        return litellm.LlmProviders.NEBIUS

    def get_base_embedding_call_args(self) -> dict:
        return {
            "model": "nebius/BAAI/bge-en-icl",
        }


def test_nebius_embedding_extra_params():
    try:

        client = HTTPHandler()
        litellm.set_verbose = True

        with patch.object(client, "post") as mock_client:
            response = litellm.embedding(
                model="nebius/BAAI/bge-en-icl",
                input=["a"],
                dimensions=512,
                input_type="document",
                client=client,
            )

            mock_client.assert_called_once()
            json_data = json.loads(mock_client.call_args.kwargs["data"])

            print("Request data to Nebius AI Studio", json.dumps(json_data, indent=4))

            # Assert the request parameters
            assert json_data["input"] == ["a"]
            assert json_data["model"] == "BAAI/bge-en-icl"
            assert json_data["output_dimension"] == 512
            assert json_data["input_type"] == "document"

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
