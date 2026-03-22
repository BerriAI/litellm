"""
Test that the generic OpenAI embedding handler filters None/empty values
from optional_params before constructing the request payload.

This mirrors the filtering already present in the openai_like handler
(litellm/llms/openai_like/embedding/handler.py L108-110) and prevents
sending encoding_format=null to strict OpenAI-compatible servers
(llama.cpp, vLLM, TEI) that reject null JSON values.

Refs:
  - https://docs.litellm.ai/blog/vllm-embeddings-incident
  - https://github.com/BerriAI/litellm/issues/19174
"""

from unittest.mock import MagicMock, patch


from litellm.llms.openai.openai import OpenAIChatCompletion


def _make_handler_and_call(optional_params: dict) -> dict:
    """
    Helper: call OpenAIChatCompletion.embedding() with the given optional_params,
    intercepting the `data` dict before it reaches the OpenAI SDK.

    Returns the `data` dict that would have been sent.
    """
    handler = OpenAIChatCompletion()

    captured_data = {}

    def fake_make_sync(openai_client, data, timeout, logging_obj):
        captured_data.update(data)
        # Return a mock that looks like an EmbeddingResponse
        mock_resp = MagicMock()
        mock_resp.model_dump.return_value = {
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
            "model": "test-model",
            "usage": {"prompt_tokens": 3, "total_tokens": 3},
        }
        return {}, mock_resp

    mock_logging = MagicMock()

    with patch.object(handler, "make_sync_openai_embedding_request", side_effect=fake_make_sync):
        with patch.object(handler, "_get_openai_client", return_value=MagicMock()):
            handler.embedding(
                model="test-model",
                input=["test input"],
                timeout=60.0,
                logging_obj=mock_logging,
                model_response=MagicMock(),
                optional_params=optional_params,
                api_key="test-key",
                api_base="http://localhost:8099",
            )

    return captured_data


class TestOpenAIEmbeddingEncodingFormatFiltering:
    """Verify that None/empty optional_params are filtered in the openai handler."""

    def test_encoding_format_none_filtered_out(self):
        """encoding_format=None must NOT appear in the request payload."""
        data = _make_handler_and_call({"encoding_format": None})
        assert "encoding_format" not in data, (
            "encoding_format=None should be filtered out"
        )
        assert data["model"] == "test-model"
        assert data["input"] == ["test input"]

    def test_encoding_format_empty_string_filtered_out(self):
        """encoding_format='' must NOT appear in the request payload."""
        data = _make_handler_and_call({"encoding_format": ""})
        assert "encoding_format" not in data, (
            "encoding_format='' should be filtered out"
        )

    def test_encoding_format_float_preserved(self):
        """encoding_format='float' must be preserved in the request payload."""
        data = _make_handler_and_call({"encoding_format": "float"})
        assert data["encoding_format"] == "float"

    def test_encoding_format_base64_preserved(self):
        """encoding_format='base64' must be preserved in the request payload."""
        data = _make_handler_and_call({"encoding_format": "base64"})
        assert data["encoding_format"] == "base64"

    def test_other_optional_params_preserved(self):
        """Other params (dimensions, user) must survive when encoding_format=None is dropped."""
        data = _make_handler_and_call({
            "encoding_format": None,
            "dimensions": 4096,
            "user": "test-user",
        })
        assert "encoding_format" not in data
        assert data["dimensions"] == 4096
        assert data["user"] == "test-user"
        assert data["model"] == "test-model"

    def test_no_optional_params(self):
        """Empty optional_params should work without errors."""
        data = _make_handler_and_call({})
        assert data["model"] == "test-model"
        assert data["input"] == ["test input"]
        assert "encoding_format" not in data
