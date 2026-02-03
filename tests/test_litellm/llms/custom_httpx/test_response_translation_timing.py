"""
Tests for response translation timing tracking.

Verifies that response_translation_time_ms is correctly stored in model_call_details
when transform_response is called during completion.
"""

import os
import sys
import time
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.types.utils import ModelResponse


class TestResponseTranslationTiming:
    """Test suite for response translation timing tracking."""

    def test_response_translation_time_is_tracked(self):
        """
        Test that response_translation_time_ms is stored in model_call_details
        when transform_response completes successfully.
        """
        handler = BaseLLMHTTPHandler()
        
        # Create a mock config that simulates work during transform_response
        mock_config = Mock(spec=BaseConfig)
        
        # Simulate a transform_response that takes measurable time
        def transform_response_with_delay(*args, **kwargs):
            time.sleep(0.005)  # 5ms delay to ensure measurable time
            return ModelResponse()
        
        mock_config.transform_response = transform_response_with_delay
        mock_config.transform_request = Mock(return_value={"model": "gpt-4", "messages": []})
        mock_config.get_complete_url = Mock(return_value="https://api.openai.com/v1/chat/completions")
        mock_config.sign_request = Mock(return_value=({}, None))
        mock_config.validate_environment = Mock(return_value={})
        mock_config.should_fake_stream = Mock(return_value=False)
        mock_config.max_retry_on_unprocessable_entity_error = 1
        mock_config.should_retry_llm_api_inside_llm_translation_on_http_error = Mock(return_value=False)
        mock_config.get_error_class = Mock(side_effect=Exception("No error class"))
        
        # Mock HTTP client and response
        from litellm.llms.custom_httpx.http_handler import HTTPHandler
        mock_client = Mock(spec=HTTPHandler)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "choices": [{"message": {"role": "assistant", "content": "Hello"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_client.post = Mock(return_value=mock_response)
        
        # Create logging object to capture timing
        mock_logging_obj = Mock()
        mock_logging_obj.model_call_details = {}
        mock_logging_obj.update_environment_variables = Mock()
        mock_logging_obj.pre_call = Mock()
        mock_logging_obj.stream = False
        
        # Execute completion
        handler.completion(
            model="gpt-4",
            messages=[],
            api_base="https://api.openai.com/v1/chat/completions",
            custom_llm_provider="openai",
            model_response=ModelResponse(),
            encoding=None,
            logging_obj=mock_logging_obj,
            optional_params={},
            timeout=30.0,
            litellm_params={},
            acompletion=False,
            stream=False,
            fake_stream=False,
            headers={},
            client=mock_client,
            provider_config=mock_config,
        )
        
        # Verify timing was tracked
        assert "response_translation_time_ms" in mock_logging_obj.model_call_details
        translation_time = mock_logging_obj.model_call_details["response_translation_time_ms"]
        
        # Verify it's a numeric value
        assert isinstance(translation_time, (int, float))
        
        # Verify it's positive (should be at least 5ms due to our delay)
        assert translation_time > 0
        assert translation_time >= 4.0  # Allow some margin for timing variance

    def test_response_translation_time_is_zero_for_instant_transform(self):
        """
        Test that response_translation_time_ms is still tracked even when
        transform_response is very fast (near-zero time).
        """
        handler = BaseLLMHTTPHandler()
        
        mock_config = Mock(spec=BaseConfig)
        # Very fast transform_response (no delay)
        mock_config.transform_response = Mock(return_value=ModelResponse())
        mock_config.transform_request = Mock(return_value={"model": "gpt-4", "messages": []})
        mock_config.get_complete_url = Mock(return_value="https://api.openai.com/v1/chat/completions")
        mock_config.sign_request = Mock(return_value=({}, None))
        mock_config.validate_environment = Mock(return_value={})
        mock_config.should_fake_stream = Mock(return_value=False)
        mock_config.max_retry_on_unprocessable_entity_error = 1
        mock_config.should_retry_llm_api_inside_llm_translation_on_http_error = Mock(return_value=False)
        mock_config.get_error_class = Mock(side_effect=Exception("No error class"))
        
        from litellm.llms.custom_httpx.http_handler import HTTPHandler
        mock_client = Mock(spec=HTTPHandler)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "choices": [{"message": {"role": "assistant", "content": "Hello"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_client.post = Mock(return_value=mock_response)
        
        mock_logging_obj = Mock()
        mock_logging_obj.model_call_details = {}
        mock_logging_obj.update_environment_variables = Mock()
        mock_logging_obj.pre_call = Mock()
        mock_logging_obj.stream = False
        
        handler.completion(
            model="gpt-4",
            messages=[],
            api_base="https://api.openai.com/v1/chat/completions",
            custom_llm_provider="openai",
            model_response=ModelResponse(),
            encoding=None,
            logging_obj=mock_logging_obj,
            optional_params={},
            timeout=30.0,
            litellm_params={},
            acompletion=False,
            stream=False,
            fake_stream=False,
            headers={},
            client=mock_client,
            provider_config=mock_config,
        )
        
        # Verify timing is still tracked even for fast operations
        assert "response_translation_time_ms" in mock_logging_obj.model_call_details
        translation_time = mock_logging_obj.model_call_details["response_translation_time_ms"]
        assert isinstance(translation_time, (int, float))
        # Should be >= 0 (can be very small but should be tracked)
        assert translation_time >= 0

    def test_response_translation_time_not_set_on_transform_error(self):
        """
        Test that response_translation_time_ms is not set if transform_response
        raises an exception before completion.
        """
        handler = BaseLLMHTTPHandler()
        
        mock_config = Mock(spec=BaseConfig)
        # transform_response that raises an error
        mock_config.transform_response = Mock(side_effect=ValueError("Transform error"))
        mock_config.transform_request = Mock(return_value={"model": "gpt-4", "messages": []})
        mock_config.get_complete_url = Mock(return_value="https://api.openai.com/v1/chat/completions")
        mock_config.sign_request = Mock(return_value=({}, None))
        mock_config.validate_environment = Mock(return_value={})
        mock_config.should_fake_stream = Mock(return_value=False)
        mock_config.max_retry_on_unprocessable_entity_error = 1
        mock_config.should_retry_llm_api_inside_llm_translation_on_http_error = Mock(return_value=False)
        mock_config.get_error_class = Mock(side_effect=Exception("No error class"))
        
        from litellm.llms.custom_httpx.http_handler import HTTPHandler
        mock_client = Mock(spec=HTTPHandler)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "choices": [{"message": {"role": "assistant", "content": "Hello"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_client.post = Mock(return_value=mock_response)
        
        mock_logging_obj = Mock()
        mock_logging_obj.model_call_details = {}
        mock_logging_obj.update_environment_variables = Mock()
        mock_logging_obj.pre_call = Mock()
        mock_logging_obj.stream = False
        
        # Call should raise error
        with pytest.raises(ValueError, match="Transform error"):
            handler.completion(
                model="gpt-4",
                messages=[],
                api_base="https://api.openai.com/v1/chat/completions",
                custom_llm_provider="openai",
                model_response=ModelResponse(),
                encoding=None,
                logging_obj=mock_logging_obj,
                optional_params={},
                timeout=30.0,
                litellm_params={},
                acompletion=False,
                stream=False,
                fake_stream=False,
                headers={},
                client=mock_client,
                provider_config=mock_config,
            )
        
        # Verify timing was not set due to error
        assert "response_translation_time_ms" not in mock_logging_obj.model_call_details
