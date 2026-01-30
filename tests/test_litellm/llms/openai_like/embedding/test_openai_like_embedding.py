"""
Test cases for OpenAI-like embedding handler
"""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest

from litellm.llms.openai_like.embedding.handler import OpenAILikeEmbeddingHandler
from litellm.types.utils import EmbeddingResponse


class TestOpenAILikeEmbeddingHandler:
    """Test OpenAI-like embedding handler functionality"""

    def test_encoding_format_none_filtered_out(self):
        """
        Test that encoding_format=None is filtered out from the request payload.
        
        According to OpenAI API spec, encoding_format should be omitted if not specified,
        not sent as None or empty string. This prevents errors with providers like VLLM
        that reject empty encoding_format values.
        """
        handler = OpenAILikeEmbeddingHandler()
        
        # Mock the HTTP client
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [0.1, 0.2, 0.3],
                    "index": 0
                }
            ],
            "model": "test-model",
            "usage": {
                "prompt_tokens": 5,
                "total_tokens": 5
            }
        }
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response
        
        # Mock logging object
        mock_logging = MagicMock()
        
        # Call embedding with encoding_format=None
        optional_params = {"encoding_format": None}
        
        with patch.object(handler, '_validate_environment', return_value=("http://test.com/v1/embeddings", {})):
            response = handler.embedding(
                model="test-model",
                input=["test input"],
                timeout=60.0,
                logging_obj=mock_logging,
                api_key="test-key",
                api_base="http://test.com",
                optional_params=optional_params,
                client=mock_client
            )
        
        # Verify the request was made
        assert mock_client.post.called
        
        # Get the data that was sent in the request
        call_args = mock_client.post.call_args
        sent_data = json.loads(call_args[1]['data'])
        
        # Assert that encoding_format is NOT in the sent data
        assert "encoding_format" not in sent_data, (
            "encoding_format=None should be filtered out from the request payload"
        )
        
        # Assert that model and input are still present
        assert sent_data["model"] == "test-model"
        assert sent_data["input"] == ["test input"]

    def test_encoding_format_empty_string_filtered_out(self):
        """
        Test that encoding_format="" (empty string) is filtered out from the request payload.
        
        This is the specific case mentioned in the issue where VLLM rejects empty string
        encoding_format values with error: "unknown variant ``, expected float or base64"
        """
        handler = OpenAILikeEmbeddingHandler()
        
        # Mock the HTTP client
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [0.1, 0.2, 0.3],
                    "index": 0
                }
            ],
            "model": "test-model",
            "usage": {
                "prompt_tokens": 5,
                "total_tokens": 5
            }
        }
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response
        
        # Mock logging object
        mock_logging = MagicMock()
        
        # Call embedding with encoding_format="" (empty string)
        optional_params = {"encoding_format": ""}
        
        with patch.object(handler, '_validate_environment', return_value=("http://test.com/v1/embeddings", {})):
            response = handler.embedding(
                model="test-model",
                input=["test input"],
                timeout=60.0,
                logging_obj=mock_logging,
                api_key="test-key",
                api_base="http://test.com",
                optional_params=optional_params,
                client=mock_client
            )
        
        # Verify the request was made
        assert mock_client.post.called
        
        # Get the data that was sent in the request
        call_args = mock_client.post.call_args
        sent_data = json.loads(call_args[1]['data'])
        
        # Assert that encoding_format is NOT in the sent data
        assert "encoding_format" not in sent_data, (
            "encoding_format='' (empty string) should be filtered out from the request payload"
        )

    def test_encoding_format_float_preserved(self):
        """
        Test that encoding_format="float" is preserved in the request payload.
        """
        handler = OpenAILikeEmbeddingHandler()
        
        # Mock the HTTP client
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [0.1, 0.2, 0.3],
                    "index": 0
                }
            ],
            "model": "test-model",
            "usage": {
                "prompt_tokens": 5,
                "total_tokens": 5
            }
        }
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response
        
        # Mock logging object
        mock_logging = MagicMock()
        
        # Call embedding with encoding_format="float"
        optional_params = {"encoding_format": "float"}
        
        with patch.object(handler, '_validate_environment', return_value=("http://test.com/v1/embeddings", {})):
            response = handler.embedding(
                model="test-model",
                input=["test input"],
                timeout=60.0,
                logging_obj=mock_logging,
                api_key="test-key",
                api_base="http://test.com",
                optional_params=optional_params,
                client=mock_client
            )
        
        # Verify the request was made
        assert mock_client.post.called
        
        # Get the data that was sent in the request
        call_args = mock_client.post.call_args
        sent_data = json.loads(call_args[1]['data'])
        
        # Assert that encoding_format IS in the sent data with correct value
        assert "encoding_format" in sent_data, (
            "encoding_format='float' should be preserved in the request payload"
        )
        assert sent_data["encoding_format"] == "float"

    def test_encoding_format_base64_preserved(self):
        """
        Test that encoding_format="base64" is preserved in the request payload.
        """
        handler = OpenAILikeEmbeddingHandler()
        
        # Mock the HTTP client
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [0.1, 0.2, 0.3],
                    "index": 0
                }
            ],
            "model": "test-model",
            "usage": {
                "prompt_tokens": 5,
                "total_tokens": 5
            }
        }
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response
        
        # Mock logging object
        mock_logging = MagicMock()
        
        # Call embedding with encoding_format="base64"
        optional_params = {"encoding_format": "base64"}
        
        with patch.object(handler, '_validate_environment', return_value=("http://test.com/v1/embeddings", {})):
            response = handler.embedding(
                model="test-model",
                input=["test input"],
                timeout=60.0,
                logging_obj=mock_logging,
                api_key="test-key",
                api_base="http://test.com",
                optional_params=optional_params,
                client=mock_client
            )
        
        # Verify the request was made
        assert mock_client.post.called
        
        # Get the data that was sent in the request
        call_args = mock_client.post.call_args
        sent_data = json.loads(call_args[1]['data'])
        
        # Assert that encoding_format IS in the sent data with correct value
        assert "encoding_format" in sent_data, (
            "encoding_format='base64' should be preserved in the request payload"
        )
        assert sent_data["encoding_format"] == "base64"

    def test_other_optional_params_preserved(self):
        """
        Test that other optional parameters are preserved when encoding_format is filtered.
        """
        handler = OpenAILikeEmbeddingHandler()
        
        # Mock the HTTP client
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [0.1, 0.2, 0.3],
                    "index": 0
                }
            ],
            "model": "test-model",
            "usage": {
                "prompt_tokens": 5,
                "total_tokens": 5
            }
        }
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response
        
        # Mock logging object
        mock_logging = MagicMock()
        
        # Call embedding with encoding_format=None and other params
        optional_params = {
            "encoding_format": None,
            "dimensions": 512,
            "user": "test-user"
        }
        
        with patch.object(handler, '_validate_environment', return_value=("http://test.com/v1/embeddings", {})):
            response = handler.embedding(
                model="test-model",
                input=["test input"],
                timeout=60.0,
                logging_obj=mock_logging,
                api_key="test-key",
                api_base="http://test.com",
                optional_params=optional_params,
                client=mock_client
            )
        
        # Verify the request was made
        assert mock_client.post.called
        
        # Get the data that was sent in the request
        call_args = mock_client.post.call_args
        sent_data = json.loads(call_args[1]['data'])
        
        # Assert that encoding_format is NOT in the sent data
        assert "encoding_format" not in sent_data
        
        # Assert that other parameters ARE preserved
        assert sent_data["dimensions"] == 512
        assert sent_data["user"] == "test-user"
        assert sent_data["model"] == "test-model"
        assert sent_data["input"] == ["test input"]

    def test_no_optional_params(self):
        """
        Test that the handler works correctly when no optional params are provided.
        """
        handler = OpenAILikeEmbeddingHandler()
        
        # Mock the HTTP client
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [0.1, 0.2, 0.3],
                    "index": 0
                }
            ],
            "model": "test-model",
            "usage": {
                "prompt_tokens": 5,
                "total_tokens": 5
            }
        }
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response
        
        # Mock logging object
        mock_logging = MagicMock()
        
        # Call embedding with empty optional_params
        optional_params = {}
        
        with patch.object(handler, '_validate_environment', return_value=("http://test.com/v1/embeddings", {})):
            response = handler.embedding(
                model="test-model",
                input=["test input"],
                timeout=60.0,
                logging_obj=mock_logging,
                api_key="test-key",
                api_base="http://test.com",
                optional_params=optional_params,
                client=mock_client
            )
        
        # Verify the request was made
        assert mock_client.post.called
        
        # Get the data that was sent in the request
        call_args = mock_client.post.call_args
        sent_data = json.loads(call_args[1]['data'])
        
        # Assert that only model and input are in the sent data
        assert sent_data["model"] == "test-model"
        assert sent_data["input"] == ["test input"]
        assert "encoding_format" not in sent_data
