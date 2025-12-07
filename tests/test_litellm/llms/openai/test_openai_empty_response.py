"""
Test for issue #17209: Clearer error when LLM endpoint returns empty response
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.llms.openai.openai import OpenAIChatCompletion
from litellm.llms.openai.common_utils import OpenAIError

class TestEmptyResponseHandling:
    """Test that empty/invalid responses from LLM endpoints produce clear error messages"""

    def test_sync_empty_string_response_raises_clear_error(self):
        """
        Test that when an OpenAI-compatible endpoint returns an empty string,
        we get a clear error instead of "'str' object has no attribute 'model_dump'"
        """
        openai_chat = OpenAIChatCompletion()

        # Mock the raw response to return an empty string from parse()
        mock_raw_response = MagicMock()
        mock_raw_response.headers = {}
        mock_raw_response.parse.return_value = ""  # Empty string response

        mock_client = MagicMock()
        mock_client.chat.completions.with_raw_response.create.return_value = (
            mock_raw_response
        )

        with pytest.raises(OpenAIError) as exc_info:
            openai_chat.make_sync_openai_chat_completion_request(
                openai_client=mock_client,
                data={"messages": [{"role": "user", "content": "test"}]},
                timeout=30,
                logging_obj=MagicMock(),
            )

        assert "Empty or invalid response from LLM endpoint" in str(exc_info.value)
        assert "Check the reverse proxy or model server configuration" in str(
            exc_info.value
        )

    def test_sync_none_response_raises_clear_error(self):
        """Test that None response also produces a clear error"""
        openai_chat = OpenAIChatCompletion()

        mock_raw_response = MagicMock()
        mock_raw_response.headers = {}
        mock_raw_response.parse.return_value = None

        mock_client = MagicMock()
        mock_client.chat.completions.with_raw_response.create.return_value = (
            mock_raw_response
        )

        with pytest.raises(OpenAIError) as exc_info:
            openai_chat.make_sync_openai_chat_completion_request(
                openai_client=mock_client,
                data={"messages": [{"role": "user", "content": "test"}]},
                timeout=30,
                logging_obj=MagicMock(),
            )

        assert "Empty or invalid response from LLM endpoint" in str(exc_info.value)

    def test_valid_response_passes_through(self):
        """Test that a valid response with model_dump passes through correctly"""
        openai_chat = OpenAIChatCompletion()

        # Create a mock response that has model_dump (like a real Pydantic model)
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {"choices": []}

        mock_raw_response = MagicMock()
        mock_raw_response.headers = {"x-request-id": "123"}
        mock_raw_response.parse.return_value = mock_response

        mock_client = MagicMock()
        mock_client.chat.completions.with_raw_response.create.return_value = (
            mock_raw_response
        )

        headers, response = openai_chat.make_sync_openai_chat_completion_request(
            openai_client=mock_client,
            data={"messages": [{"role": "user", "content": "test"}]},
            timeout=30,
            logging_obj=MagicMock(),
        )

        assert response == mock_response
        assert headers == {"x-request-id": "123"}

    def test_sync_streaming_response_passes_through_without_model_dump(self):
        """
        Test that streaming responses (which don't have model_dump) pass through
        correctly without raising an error. This validates the fix for VLLM streaming.
        """
        openai_chat = OpenAIChatCompletion()

        # Create a mock response WITHOUT model_dump (like an AsyncStream/Iterator)
        mock_stream = MagicMock(spec=[])  # spec=[] means no attributes

        mock_raw_response = MagicMock()
        mock_raw_response.headers = {"x-request-id": "123"}
        mock_raw_response.parse.return_value = mock_stream

        mock_client = MagicMock()
        mock_client.chat.completions.with_raw_response.create.return_value = (
            mock_raw_response
        )

        # Key: data has stream=True - this should bypass the model_dump check
        headers, response = openai_chat.make_sync_openai_chat_completion_request(
            openai_client=mock_client,
            data={"messages": [{"role": "user", "content": "test"}], "stream": True},
            timeout=30,
            logging_obj=MagicMock(),
        )

        assert response == mock_stream
        assert headers == {"x-request-id": "123"}
