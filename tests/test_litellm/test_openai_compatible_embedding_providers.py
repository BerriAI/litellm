from unittest.mock import MagicMock, patch

import pytest

import litellm


@pytest.mark.parametrize(
    "provider",
    ["byteplus", "digitalocean", "siliconflow", "deepinfra", "nscale", "novita"],
)
def test_openai_compatible_provider_embedding_dispatch(provider):
    mock_response = MagicMock()
    mock_response.parse.return_value = MagicMock(
        model_dump=lambda: {
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
            "model": "test-embedding",
            "object": "list",
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }
    )
    mock_response.headers = {}

    with patch("litellm.llms.openai.openai.OpenAIChatCompletion._get_openai_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.embeddings.with_raw_response.create.return_value = mock_response

        response = litellm.embedding(
            model=f"{provider}/test-embedding",
            input="Hello world",
            api_key="test-key",
            api_base="http://test.example/v1",
        )

    assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]
    assert mock_client.embeddings.with_raw_response.create.called
