from unittest.mock import patch

import litellm


def mock_embedding_response(*args, **kwargs):
    """Mock response mimicking litellm.embedding output."""

    class MockResponse:
        def __init__(self):
            self.data = [{"embedding": [0.1, 0.2, 0.3]}]  # Example embedding vector
            self.usage = litellm.Usage()  # Mock Usage object
            self.model = kwargs.get("model", "sambanova/E5-Mistral-7B-Instruct")
            self.object = "embedding"

        def __getitem__(self, key):
            return getattr(self, key)

    return MockResponse()


def test_sambanova_embeddings():
    """Mocked test for SambaNova embeddings using MagicMock."""
    with patch("litellm.embedding", side_effect=mock_embedding_response) as mock_embed:
        response = litellm.embedding(
            model="sambanova/E5-Mistral-7B-Instruct",
            input=["good morning from litellm"],
        )

        # Assertions to verify that the mock was called correctly
        mock_embed.assert_called_once_with(
            model="sambanova/E5-Mistral-7B-Instruct",
            input=["good morning from litellm"],
        )

        # Assertions to check the structure of the mocked response
        assert isinstance(response.data, list)
        assert "embedding" in response.data[0]
        assert isinstance(response.data[0]["embedding"], list)
        assert response.model == "sambanova/E5-Mistral-7B-Instruct"
        assert response.object == "embedding"
