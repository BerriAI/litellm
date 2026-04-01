import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)

import json
from unittest.mock import MagicMock, patch

import pytest

from litellm.llms.azure_ai.anthropic.handler import AzureAnthropicChatCompletion
from litellm.types.utils import ModelResponse


class TestAzureAnthropicChatCompletion:
    def test_inherits_from_anthropic_chat_completion(self):
        """Test that AzureAnthropicChatCompletion inherits from AnthropicChatCompletion"""
        handler = AzureAnthropicChatCompletion()
        assert isinstance(handler, AzureAnthropicChatCompletion)
        # Check that it has methods from parent class
        assert hasattr(handler, "acompletion_function")
        assert hasattr(handler, "acompletion_stream_function")

    @patch("litellm.utils.ProviderConfigManager")
    @patch("litellm.llms.azure_ai.anthropic.handler.AzureAnthropicConfig")
    def test_completion_uses_azure_anthropic_config(self, mock_azure_config, mock_provider_manager):
        """Test that completion method uses AzureAnthropicConfig"""
        handler = AzureAnthropicChatCompletion()
        mock_config = MagicMock()
        mock_config.transform_request.return_value = {"model": "claude-sonnet-4-5", "messages": []}
        mock_config.transform_response.return_value = ModelResponse()
        mock_config_instance = MagicMock()
        mock_config_instance.validate_environment.return_value = {"x-api-key": "test-api-key", "anthropic-version": "2023-06-01"}
        mock_config_instance.transform_request.return_value = {"model": "claude-sonnet-4-5", "messages": []}
        mock_azure_config.return_value = mock_config_instance
        mock_provider_manager.get_provider_chat_config.return_value = mock_config

        model = "claude-sonnet-4-5"
        messages = [{"role": "user", "content": "Hello"}]
        api_base = "https://test.services.ai.azure.com/anthropic/v1/messages"
        custom_llm_provider = "azure_anthropic"
        custom_prompt_dict = {}
        model_response = ModelResponse()
        print_verbose = MagicMock()
        encoding = MagicMock()
        api_key = "test-api-key"
        logging_obj = MagicMock()
        optional_params = {}
        timeout = 60.0
        litellm_params = {"api_key": "test-api-key"}
        headers = {}

        with patch.object(
            handler, "acompletion_function", return_value=ModelResponse()
        ) as mock_acompletion:
            handler.completion(
                model=model,
                messages=messages,
                api_base=api_base,
                custom_llm_provider=custom_llm_provider,
                custom_prompt_dict=custom_prompt_dict,
                model_response=model_response,
                print_verbose=print_verbose,
                encoding=encoding,
                api_key=api_key,
                logging_obj=logging_obj,
                optional_params=optional_params,
                timeout=timeout,
                litellm_params=litellm_params,
                headers=headers,
                acompletion=True,
            )

            # Verify AzureAnthropicConfig was used
            mock_azure_config.assert_called_once()
            mock_config_instance.validate_environment.assert_called_once()

    @patch("litellm.llms.anthropic.chat.handler.make_sync_call")
    @patch("litellm.utils.ProviderConfigManager")
    @patch("litellm.llms.azure_ai.anthropic.handler.AzureAnthropicConfig")
    def test_completion_streaming(self, mock_azure_config, mock_provider_manager, mock_make_sync_call):
        # Note: decorators are applied in reverse order
        """Test completion with streaming"""
        handler = AzureAnthropicChatCompletion()
        mock_config = MagicMock()
        mock_config.transform_request.return_value = {
            "model": "claude-sonnet-4-5",
            "messages": [],
            "stream": True,
        }
        mock_config_instance = MagicMock()
        mock_config_instance.validate_environment.return_value = {"x-api-key": "test-api-key", "anthropic-version": "2023-06-01"}
        mock_config_instance.transform_request.return_value = {
            "model": "claude-sonnet-4-5",
            "messages": [],
            "stream": True,
        }
        mock_azure_config.return_value = mock_config_instance
        mock_provider_manager.get_provider_chat_config.return_value = mock_config

        # Mock streaming response
        mock_stream = MagicMock()
        mock_headers = MagicMock()
        mock_make_sync_call.return_value = (mock_stream, mock_headers)

        model = "claude-sonnet-4-5"
        messages = [{"role": "user", "content": "Hello"}]
        api_base = "https://test.services.ai.azure.com/anthropic/v1/messages"
        custom_llm_provider = "azure_anthropic"
        custom_prompt_dict = {}
        model_response = ModelResponse()
        print_verbose = MagicMock()
        encoding = MagicMock()
        api_key = "test-api-key"
        logging_obj = MagicMock()
        optional_params = {"stream": True}
        timeout = 60.0
        litellm_params = {"api_key": "test-api-key"}
        headers = {}

        result = handler.completion(
            model=model,
            messages=messages,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            custom_prompt_dict=custom_prompt_dict,
            model_response=model_response,
            print_verbose=print_verbose,
            encoding=encoding,
            api_key=api_key,
            logging_obj=logging_obj,
            optional_params=optional_params,
            timeout=timeout,
            litellm_params=litellm_params,
            headers=headers,
            acompletion=False,
        )

        # Verify streaming was handled
        mock_make_sync_call.assert_called_once()
        assert result is not None

    @patch("litellm.llms.custom_httpx.http_handler._get_httpx_client")
    @patch("litellm.utils.ProviderConfigManager")
    @patch("litellm.llms.azure_ai.anthropic.handler.AzureAnthropicConfig")
    def test_completion_non_streaming(self, mock_azure_config, mock_provider_manager, mock_get_client):
        # Note: decorators are applied in reverse order
        """Test completion without streaming"""
        handler = AzureAnthropicChatCompletion()
        mock_config = MagicMock()
        mock_config.transform_request.return_value = {
            "model": "claude-sonnet-4-5",
            "messages": [],
        }
        mock_response = ModelResponse()
        mock_config.transform_response.return_value = mock_response
        mock_config_instance = MagicMock()
        mock_config_instance.validate_environment.return_value = {"x-api-key": "test-api-key", "anthropic-version": "2023-06-01"}
        mock_config_instance.transform_request.return_value = {
            "model": "claude-sonnet-4-5",
            "messages": [],
        }
        mock_azure_config.return_value = mock_config_instance
        mock_provider_manager.get_provider_chat_config.return_value = mock_config

        model = "claude-sonnet-4-5"
        messages = [{"role": "user", "content": "Hello"}]
        api_base = "https://test.services.ai.azure.com/anthropic/v1/messages"
        custom_llm_provider = "azure_anthropic"
        custom_prompt_dict = {}
        model_response = ModelResponse()
        print_verbose = MagicMock()
        encoding = MagicMock()
        api_key = "test-api-key"
        logging_obj = MagicMock()
        optional_params = {}
        timeout = 60.0
        litellm_params = {"api_key": "test-api-key"}
        headers = {}

        # Mock HTTP client
        mock_client = MagicMock()
        mock_response_obj = MagicMock()
        mock_response_obj.status_code = 200
        mock_response_obj.text = json.dumps({
            "id": "test-id",
            "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        })
        mock_response_obj.json.return_value = {
            "id": "test-id",
            "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        mock_client.post.return_value = mock_response_obj
        mock_get_client.return_value = mock_client

        result = handler.completion(
            model=model,
            messages=messages,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            custom_prompt_dict=custom_prompt_dict,
            model_response=model_response,
            print_verbose=print_verbose,
            encoding=encoding,
            api_key=api_key,
            logging_obj=logging_obj,
            optional_params=optional_params,
            timeout=timeout,
            litellm_params=litellm_params,
            headers=headers,
            client=None,  # Let it create the client
            acompletion=False,
        )

        # Verify non-streaming was handled
        mock_client.post.assert_called_once()
        assert result is not None

