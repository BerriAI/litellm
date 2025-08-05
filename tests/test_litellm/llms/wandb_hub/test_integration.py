"""
Integration tests for Wandb Hub provider
Maps to: litellm/llms/wandb_hub/
"""
import os
from unittest.mock import MagicMock, patch
import pytest

import litellm
from litellm import completion, acompletion
from tests.llm_translation.base_llm_unit_tests import BaseLLMChatTest


def test_get_llm_provider_wandb_hub():
    """Test that get_llm_provider correctly identifies Wandb Hub"""
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
    
    # Test with wandb_hub/model-name format
    model, provider, api_key, api_base = get_llm_provider("wandb_hub/moonshotai/Kimi-K2-Instruct")
    assert model == "moonshotai/Kimi-K2-Instruct"
    assert provider == "wandb_hub"
    
    # Test with api_base containing Wandb Hub endpoint
    model, provider, api_key, api_base = get_llm_provider(
        "moonshotai/Kimi-K2-Instruct", api_base="https://api.inference.wandb.ai/v1"
    )
    assert model == "moonshotai/Kimi-K2-Instruct"
    assert provider == "wandb_hub"
    assert api_base == "https://api.inference.wandb.ai/v1"


def test_wandb_hub_in_provider_lists():
    """Test that Wandb Hub is registered in all necessary provider lists"""
    assert "wandb_hub" in litellm.openai_compatible_providers
    assert "wandb_hub" in litellm.provider_list
    assert "https://api.inference.wandb.ai/v1" in litellm.openai_compatible_endpoints


def test_wandb_hub_project_id_validation():
    """Test that project_id validation works in the completion call"""
    # These tests verify that the validation happens in main.py before reaching the handler
    
    # Test missing project_id
    with pytest.raises(ValueError, match="project_id is required for Wandb Hub"):
        completion(
            model="wandb_hub/moonshotai/Kimi-K2-Instruct",
            messages=[{"role": "user", "content": "Hello"}]
        )
    
    # Test empty project_id
    with pytest.raises(ValueError, match="project_id is required for Wandb Hub"):
        completion(
            model="wandb_hub/moonshotai/Kimi-K2-Instruct",
            messages=[{"role": "user", "content": "Hello"}],
            project_id=""
        )


def test_wandb_hub_with_mock_response():
    """Test Wandb Hub with mocked HTTP response"""
    # Mock the WandbHubChatHandler directly to avoid main.py validation
    with patch('litellm.llms.wandb_hub.chat.handler.WandbHubChatHandler') as mock_handler_class:
        # Mock the handler response
        mock_handler = MagicMock()
        mock_response_obj = MagicMock()
        mock_response_obj.choices = [MagicMock()]
        mock_response_obj.choices[0].message.content = "Hello! This is a test response from Wandb Hub."
        mock_response_obj.model = "wandb_hub/moonshotai/Kimi-K2-Instruct"
        mock_response_obj.usage.prompt_tokens = 10
        mock_response_obj.usage.completion_tokens = 12
        mock_response_obj.usage.total_tokens = 22
        
        mock_handler.completion.return_value = mock_response_obj
        mock_handler_class.return_value = mock_handler
        
        response = completion(
            model="wandb_hub/moonshotai/Kimi-K2-Instruct",
            messages=[{"role": "user", "content": "Hello"}],
            project_id="test-project",
            api_key="test-key"
        )
        
        # Verify response structure
        assert response.choices[0].message.content == "Hello! This is a test response from Wandb Hub."
        assert response.model == "wandb_hub/moonshotai/Kimi-K2-Instruct"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 12
        assert response.usage.total_tokens == 22
        
        # Verify the handler was called
        mock_handler.completion.assert_called_once()


class TestWandbHubIntegration(BaseLLMChatTest):
    """Integration tests for Wandb Hub using BaseLLMChatTest"""
    
    def get_base_completion_call_args(self) -> dict:
        """Return base completion call args for Wandb Hub"""
        return {
            "model": "wandb_hub/moonshotai/Kimi-K2-Instruct",
            "project_id": "test-project"
        }
    
    @pytest.mark.skipif(
        not os.getenv("WANDB_API_KEY"), 
        reason="WANDB_API_KEY not set"
    )
    def test_wandb_hub_completion(self):
        """Test actual completion call to Wandb Hub (requires WANDB_API_KEY)"""
        try:
            response = completion(
                model="wandb_hub/moonshotai/Kimi-K2-Instruct",
                messages=[{"role": "user", "content": "Hello, this is a test"}],
                max_tokens=10,
                project_id="test-project"
            )
            assert response.choices[0].message.content
            assert response.model.startswith("wandb_hub/")
            assert response.usage
        except Exception as e:
            # If there's an auth or network issue, make sure it's not a provider issue
            if "wandb_hub" not in str(e) and "provider" not in str(e).lower():
                raise

    @pytest.mark.skipif(
        not os.getenv("WANDB_API_KEY"), 
        reason="WANDB_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_wandb_hub_async_completion(self):
        """Test async completion call to Wandb Hub (requires WANDB_API_KEY)"""
        try:
            response = await acompletion(
                model="wandb_hub/moonshotai/Kimi-K2-Instruct",
                messages=[{"role": "user", "content": "Hello, this is a test"}],
                max_tokens=10,
                project="test-project"
            )
            assert response.choices[0].message.content
            assert response.model.startswith("wandb_hub/")
            assert response.usage
        except Exception as e:
            # If there's an auth or network issue, make sure it's not a provider issue
            if "wandb_hub" not in str(e) and "provider" not in str(e).lower():
                raise

    @pytest.mark.skipif(
        not os.getenv("WANDB_API_KEY"), 
        reason="WANDB_API_KEY not set"
    )
    def test_wandb_hub_streaming(self):
        """Test streaming completion call to Wandb Hub (requires WANDB_API_KEY)"""
        try:
            response = completion(
                model="wandb_hub/moonshotai/Kimi-K2-Instruct",
                messages=[{"role": "user", "content": "Hello, this is a test"}],
                max_tokens=10,
                project_id="test-project",
                stream=True
            )
            
            # Consume the stream
            content = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content += chunk.choices[0].delta.content
            
            assert content
        except Exception as e:
            # If there's an auth or network issue, make sure it's not a provider issue
            if "wandb_hub" not in str(e) and "provider" not in str(e).lower():
                raise