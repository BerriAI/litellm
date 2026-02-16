import importlib
import json
import os
import sys
from unittest.mock import patch, MagicMock, AsyncMock

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

import litellm
import pytest


MOCK_EMBEDDING_RESPONSE = [[0.1, 0.2, 0.3, 0.4, 0.5]]


@pytest.fixture
def reload_huggingface_modules():
    """
    Reload modules to ensure fresh references after conftest reloads litellm.
    This ensures the HTTPHandler class being patched is the same one used by
    the embedding handler during parallel test execution.
    """
    import litellm.llms.custom_httpx.http_handler as http_handler_module
    import litellm.llms.huggingface.embedding.handler as hf_embedding_handler_module

    importlib.reload(http_handler_module)
    importlib.reload(hf_embedding_handler_module)
    yield


@pytest.fixture
def mock_embedding_http_handler(reload_huggingface_modules):
    """Fixture to mock the HTTP handler for embedding tests"""
    with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post") as mock_post:
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_EMBEDDING_RESPONSE
        mock_post.return_value = mock_response
        yield mock_post


@pytest.fixture
def mock_embedding_async_http_handler(reload_huggingface_modules):
    """Fixture to mock the async HTTP handler for embedding tests"""
    with patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post", new_callable=AsyncMock) as mock_post:
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = MOCK_EMBEDDING_RESPONSE
        mock_post.return_value = mock_response
        yield mock_post

class TestHuggingFaceEmbedding:
    @pytest.fixture(autouse=True)
    def setup(self, mock_embedding_http_handler, mock_embedding_async_http_handler):
        self.mock_get_task_patcher = patch("litellm.llms.huggingface.embedding.handler.get_hf_task_embedding_for_model")
        self.mock_get_task = self.mock_get_task_patcher.start()

        def mock_get_task_side_effect(model, task_type, api_base):
            if task_type is not None:
                return task_type
            return "sentence-similarity"

        self.mock_get_task.side_effect = mock_get_task_side_effect

        self.model = "huggingface/BAAI/bge-m3"
        self.mock_http = mock_embedding_http_handler
        self.mock_async_http = mock_embedding_async_http_handler
        litellm.set_verbose = False

        yield

        self.mock_get_task_patcher.stop()

    def test_input_type_preserved_in_optional_params(self):
        input_text = ["hello world"]

        response = litellm.embedding(
            model=self.model,
            input=input_text,
            input_type="embed",
        )

        self.mock_http.assert_called_once()
        post_call_args = self.mock_http.call_args
        request_data = json.loads(post_call_args[1]["data"])

        # When input_type="embed", it should use simple format {"inputs": [...]}
        # NOT the sentence-similarity format which would require 2+ sentences
        assert "inputs" in request_data
        assert request_data["inputs"] == input_text

        # Should NOT have sentence-similarity format
        assert "source_sentence" not in str(request_data)
        assert "sentences" not in str(request_data)

    def test_embedding_with_sentence_similarity_task(self):
        """Test embedding when task type is sentence-similarity (requires 2+ sentences)"""

        similarity_response = {
            "similarities": [[0, 0.9], [1, 0.8]]
        }

        self.mock_http.return_value.json.return_value = similarity_response

        # Test with 2+ sentences (required for sentence-similarity)
        input_text = ["This is the source sentence", "This is sentence one", "This is sentence two"]

        response = litellm.embedding(
            model=self.model,
            input=input_text,
            # Use the model's natural task type (sentence-similarity)
        )

        self.mock_http.assert_called_once()
        post_call_args = self.mock_http.call_args
        request_data = json.loads(post_call_args[1]["data"])

        assert "inputs" in request_data
        assert "source_sentence" in request_data["inputs"]
        assert "sentences" in request_data["inputs"]
        assert request_data["inputs"]["source_sentence"] == input_text[0]
        assert request_data["inputs"]["sentences"] == input_text[1:]