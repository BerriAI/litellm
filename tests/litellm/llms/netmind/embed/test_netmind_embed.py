from unittest.mock import MagicMock, patch
import pytest
import sys
import os
from openai.types.create_embedding_response import CreateEmbeddingResponse
from openai.types import Embedding
from openai._models import construct_type

sys.path.insert(0, os.path.abspath("../../../../../"))

import litellm
from litellm import embedding

user_message = "How are you doing today?"
messages = [{"content": user_message, "role": "user"}]


def mock_response():
    res = {
        'data': [
            {'embedding': [0.0] * 4096, 'index': 0}
        ],
        'model': 'nvidia/NV-Embed-v2', 'id': 'c1e128c95b484fd09ff33b8f5ab2f9f1'
    }
    response = MagicMock()
    response.parse.return_value = construct_type(type_=CreateEmbeddingResponse, value=res)
    return response


@patch("litellm.llms.openai.openai.OpenAI")
def test_completion_netmind_chat(mock_openai):
    litellm.set_verbose = True
    model_name = "netmind/nvidia/NV-Embed-v2"

    mock_client = mock_openai.return_value
    mock_completion = MagicMock()
    mock_completion.create.return_value = mock_response()
    mock_client.embeddings.with_raw_response = mock_completion

    try:
        response = embedding(
            model=model_name,
            messages=messages,
            max_tokens=10,
        )
        assert isinstance(response, litellm.EmbeddingResponse)
        assert len(response.data[0]['embedding']) == 4096
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
