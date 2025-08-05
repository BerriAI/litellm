"""
Tests for Wandb Hub chat completion transformation
Maps to: litellm/llms/wandb_hub/chat/transformation.py
"""
import os
from unittest import mock
from unittest.mock import MagicMock, patch
import pytest

from litellm.llms.wandb_hub.chat.transformation import WandbHubChatConfig
from litellm.types.utils import ModelResponse


class TestWandbHubChatConfig:
    """Test WandbHubChatConfig transformation logic"""
    
    def test_wandb_hub_config_initialization(self):
        """Test WandbHubChatConfig initializes correctly"""
        config = WandbHubChatConfig()
        assert hasattr(config, '_get_openai_compatible_provider_info')
        assert hasattr(config, '_validate_project_id')
        assert hasattr(config, 'transform_request')
        assert hasattr(config, 'transform_response')

    def test_wandb_hub_get_openai_compatible_provider_info(self):
        """Test Wandb Hub provider info retrieval"""
        config = WandbHubChatConfig()
        
        # Test with default values (no env vars set)
        with mock.patch.dict(os.environ, {}, clear=True):
            api_base, api_key = config._get_openai_compatible_provider_info(None, None)
            assert api_base == "https://api.inference.wandb.ai/v1"
            assert api_key == ""
        
        # Test with environment variables
        with mock.patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}):
            api_base, api_key = config._get_openai_compatible_provider_info(None, None)
            assert api_base == "https://api.inference.wandb.ai/v1"
            assert api_key == "test-key"
        
        # Test with explicit parameters (should override env vars)
        with mock.patch.dict(os.environ, {"WANDB_API_KEY": "env-key"}):
            api_base, api_key = config._get_openai_compatible_provider_info(
                "https://custom.wandb.ai/v1", "param-key"
            )
            assert api_base == "https://custom.wandb.ai/v1"
            assert api_key == "param-key"

    def test_validate_project_id_success(self):
        """Test successful project_id validation"""
        config = WandbHubChatConfig()
        
        # Test with project_id
        optional_params = {"project_id": "my-project"}
        project_id = config._validate_project_id(optional_params)
        assert project_id == "my-project"
        
        # Test with project
        optional_params = {"project": "my-project-2"}
        project_id = config._validate_project_id(optional_params)
        assert project_id == "my-project-2"
        
        # Test with both (project_id takes precedence)
        optional_params = {"project_id": "project-id", "project": "project"}
        project_id = config._validate_project_id(optional_params)
        assert project_id == "project-id"

    def test_validate_project_id_failure(self):
        """Test project_id validation failure"""
        config = WandbHubChatConfig()
        
        # Test with no project_id or project
        optional_params = {}
        with pytest.raises(ValueError, match="project_id is required for Wandb Hub"):
            config._validate_project_id(optional_params)
        
        # Test with empty values
        optional_params = {"project_id": "", "project": ""}
        with pytest.raises(ValueError, match="project_id is required for Wandb Hub"):
            config._validate_project_id(optional_params)

    def test_transform_request(self):
        """Test request transformation"""
        config = WandbHubChatConfig()
        
        model = "moonshotai/Kimi-K2-Instruct"
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {"project_id": "test-project", "max_tokens": 100}
        litellm_params = {}
        headers = {}
        
        # Mock the parent transform_request
        with patch.object(config.__class__.__bases__[0], 'transform_request') as mock_parent:
            mock_parent.return_value = {"model": model, "messages": messages, "max_tokens": 100}
            
            result = config.transform_request(
                model=model,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                headers=headers
            )
            
            # Check that project_id was removed from optional_params copy passed to parent
            called_args = mock_parent.call_args
            called_optional_params = called_args.kwargs['optional_params']
            assert "project_id" not in called_optional_params
            assert "project" not in called_optional_params
            assert called_optional_params["max_tokens"] == 100
            
            # Check that project header was added
            called_headers = called_args.kwargs['headers']
            assert called_headers["project"] == "test-project"

    def test_transform_response(self):
        """Test response transformation"""
        config = WandbHubChatConfig()
        
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "test-id",
            "model": "moonshotai/Kimi-K2-Instruct",
            "choices": [{"message": {"content": "Hello"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1}
        }
        
        model_response = ModelResponse()
        logging_obj = MagicMock()
        request_data = {}
        messages = []
        optional_params = {}
        litellm_params = {}
        encoding = None
        
        result = config.transform_response(
            model="moonshotai/Kimi-K2-Instruct",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=request_data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding
        )
        
        # Check that wandb_hub prefix was added to model name
        assert result.model == "wandb_hub/moonshotai/Kimi-K2-Instruct"
        
        # Check that logging was called
        logging_obj.post_call.assert_called_once()

    def test_map_openai_params(self):
        """Test parameter mapping functionality"""
        config = WandbHubChatConfig()
        
        # Test max_completion_tokens conversion
        non_default_params = {"max_completion_tokens": 150, "temperature": 0.5}
        optional_params = {}
        
        # Mock the parent map_openai_params
        with patch.object(config.__class__.__bases__[0], 'map_openai_params') as mock_parent:
            mock_parent.return_value = {"temperature": 0.5}
            
            result = config.map_openai_params(
                non_default_params=non_default_params,
                optional_params=optional_params,
                model="test-model",
                drop_params=False,
                replace_max_completion_tokens_with_max_tokens=True
            )
            
            # Check that max_completion_tokens was converted to max_tokens
            assert "max_tokens" in result
            assert result["max_tokens"] == 150
            assert "max_completion_tokens" not in result
            assert result["temperature"] == 0.5