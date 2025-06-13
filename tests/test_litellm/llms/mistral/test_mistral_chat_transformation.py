import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from litellm.llms.mistral.mistral_chat_transformation import MistralConfig


@pytest.mark.asyncio
async def test_mistral_chat_transformation():
    mistral_config = MistralConfig()
    result = mistral_config._transform_messages(
        **{
            "messages": [
                {
                    "content": [
                        {"type": "text", "text": "Here is a representation of text"},
                        {
                            "type": "image_url",
                            "image_url": "https://images.pexels.com/photos/13268478/pexels-photo-13268478.jpeg",
                        },
                    ],
                    "role": "user",
                }
            ],
            "model": "mistral-medium-latest",
            "is_async": True,
        }
    )


class TestMistralReasoningSupport:
    """Test suite for Mistral Magistral reasoning functionality."""

    def test_get_supported_openai_params_magistral_model(self):
        """Test that magistral models support reasoning parameters."""
        mistral_config = MistralConfig()
        
        # Test magistral model supports reasoning parameters
        supported_params = mistral_config.get_supported_openai_params("mistral/magistral-medium-2506")
        assert "reasoning_effort" in supported_params
        assert "thinking" in supported_params
        
        # Test non-magistral model doesn't include reasoning parameters
        supported_params_normal = mistral_config.get_supported_openai_params("mistral/mistral-large-latest")
        assert "reasoning_effort" not in supported_params_normal
        assert "thinking" not in supported_params_normal

    def test_map_openai_params_reasoning_effort(self):
        """Test that reasoning_effort parameter is properly mapped for magistral models."""
        mistral_config = MistralConfig()
        
        # Test reasoning_effort mapping for magistral model
        optional_params = {}
        result = mistral_config.map_openai_params(
            non_default_params={"reasoning_effort": "low"},
            optional_params=optional_params,
            model="mistral/magistral-medium-2506",
            drop_params=False,
        )
        
        assert result.get("_add_reasoning_prompt") is True
        
        # Test reasoning_effort ignored for non-magistral model
        optional_params_normal = {}
        result_normal = mistral_config.map_openai_params(
            non_default_params={"reasoning_effort": "low"},
            optional_params=optional_params_normal,
            model="mistral/mistral-large-latest",
            drop_params=False,
        )
        
        assert "_add_reasoning_prompt" not in result_normal

    def test_map_openai_params_thinking(self):
        """Test that thinking parameter is properly mapped for magistral models."""
        mistral_config = MistralConfig()
        
        # Test thinking mapping for magistral model
        optional_params = {}
        result = mistral_config.map_openai_params(
            non_default_params={"thinking": {"budget": 1000}},
            optional_params=optional_params,
            model="mistral/magistral-small-2506",
            drop_params=False,
        )
        
        assert result.get("_add_reasoning_prompt") is True

    def test_get_mistral_reasoning_system_prompt(self):
        """Test that the reasoning system prompt is properly formatted."""
        prompt = MistralConfig._get_mistral_reasoning_system_prompt()
        
        assert "<think>" in prompt
        assert "</think>" in prompt
        assert "step-by-step" in prompt
        assert isinstance(prompt, str)
        assert len(prompt) > 50  # Ensure it's not empty

    def test_add_reasoning_system_prompt_no_existing_system_message(self):
        """Test adding reasoning system prompt when no system message exists."""
        mistral_config = MistralConfig()
        
        messages = [
            {"role": "user", "content": "What is 2+2?"}
        ]
        optional_params = {"_add_reasoning_prompt": True}
        
        result = mistral_config._add_reasoning_system_prompt_if_needed(messages, optional_params)
        
        # Should add a new system message at the beginning
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "<think>" in result[0]["content"]
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "What is 2+2?"
        
        # Should remove the internal flag
        assert "_add_reasoning_prompt" not in optional_params

    def test_add_reasoning_system_prompt_with_existing_system_message(self):
        """Test adding reasoning system prompt when system message already exists."""
        mistral_config = MistralConfig()
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"}
        ]
        optional_params = {"_add_reasoning_prompt": True}
        
        result = mistral_config._add_reasoning_system_prompt_if_needed(messages, optional_params)
        
        # Should modify existing system message
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "<think>" in result[0]["content"]
        assert "You are a helpful assistant." in result[0]["content"]
        assert result[1]["role"] == "user"
        
        # Should remove the internal flag
        assert "_add_reasoning_prompt" not in optional_params

    def test_add_reasoning_system_prompt_with_existing_list_content(self):
        """Test adding reasoning system prompt when system message has list content."""
        mistral_config = MistralConfig()
        
        messages = [
            {
                "role": "system", 
                "content": [
                    {"type": "text", "text": "You are a helpful assistant."},
                    {"type": "text", "text": "You always provide detailed explanations."}
                ]
            },
            {"role": "user", "content": "What is 2+2?"}
        ]
        optional_params = {"_add_reasoning_prompt": True}
        
        result = mistral_config._add_reasoning_system_prompt_if_needed(messages, optional_params)
        
        # Should modify existing system message preserving list format
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert isinstance(result[0]["content"], list)
        
        # First item should be the reasoning prompt
        assert result[0]["content"][0]["type"] == "text"
        assert "<think>" in result[0]["content"][0]["text"]
        
        # Original content should be preserved
        assert "You are a helpful assistant." in result[0]["content"][1]["text"]
        assert "You always provide detailed explanations." in result[0]["content"][2]["text"]
        
        assert result[1]["role"] == "user"
        
        # Should remove the internal flag
        assert "_add_reasoning_prompt" not in optional_params

    def test_add_reasoning_system_prompt_preserves_content_types(self):
        """Test that reasoning prompt preserves original content types (string vs list)."""
        mistral_config = MistralConfig()
        
        # Test with string content
        string_messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"}
        ]
        string_params = {"_add_reasoning_prompt": True}
        
        string_result = mistral_config._add_reasoning_system_prompt_if_needed(string_messages, string_params)
        assert isinstance(string_result[0]["content"], str)
        assert "<think>" in string_result[0]["content"]
        assert "You are helpful." in string_result[0]["content"]
        
        # Test with list content
        list_messages = [
            {
                "role": "system", 
                "content": [{"type": "text", "text": "You are helpful."}]
            },
            {"role": "user", "content": "Hello"}
        ]
        list_params = {"_add_reasoning_prompt": True}
        
        list_result = mistral_config._add_reasoning_system_prompt_if_needed(list_messages, list_params)
        assert isinstance(list_result[0]["content"], list)
        assert list_result[0]["content"][0]["type"] == "text"
        assert "<think>" in list_result[0]["content"][0]["text"]
        assert "You are helpful." in list_result[0]["content"][1]["text"]

    def test_add_reasoning_system_prompt_no_flag(self):
        """Test that no modification happens when _add_reasoning_prompt flag is not set."""
        mistral_config = MistralConfig()
        
        messages = [
            {"role": "user", "content": "What is 2+2?"}
        ]
        optional_params = {}
        
        result = mistral_config._add_reasoning_system_prompt_if_needed(messages, optional_params)
        
        # Should return messages unchanged
        assert result == messages
        assert len(result) == 1

    def test_transform_request_magistral_with_reasoning(self):
        """Test transform_request method for magistral model with reasoning."""
        mistral_config = MistralConfig()
        
        messages = [
            {"role": "user", "content": "What is 15 * 7?"}
        ]
        optional_params = {"_add_reasoning_prompt": True}
        
        result = mistral_config.transform_request(
            model="mistral/magistral-medium-2506",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={}
        )
        
        # Should have added system message
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "system"
        assert "<think>" in result["messages"][0]["content"]
        assert result["messages"][1]["role"] == "user"
        
        # Should remove internal flag from optional_params
        assert "_add_reasoning_prompt" not in result

    def test_transform_request_magistral_without_reasoning(self):
        """Test transform_request method for magistral model without reasoning."""
        mistral_config = MistralConfig()
        
        messages = [
            {"role": "user", "content": "What is 15 * 7?"}
        ]
        optional_params = {}
        
        result = mistral_config.transform_request(
            model="mistral/magistral-medium-2506",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={}
        )
        
        # Should not modify messages
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"

    def test_transform_request_non_magistral_with_reasoning_params(self):
        """Test that non-magistral models ignore reasoning parameters."""
        mistral_config = MistralConfig()
        
        messages = [
            {"role": "user", "content": "What is 15 * 7?"}
        ]
        optional_params = {"_add_reasoning_prompt": True}
        
        result = mistral_config.transform_request(
            model="mistral/mistral-large-latest",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={}
        )
        
        # Should not add system message for non-magistral models
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"

    def test_case_insensitive_magistral_detection(self):
        """Test that magistral model detection is case-insensitive."""
        mistral_config = MistralConfig()
        
        # Test various case combinations
        models_to_test = [
            "mistral/Magistral-medium-2506",
            "mistral/MAGISTRAL-MEDIUM-2506",
            "mistral/magistral-SMALL-2506",
            "MaGiStRaL-medium-2506"
        ]
        
        for model in models_to_test:
            supported_params = mistral_config.get_supported_openai_params(model)
            assert "reasoning_effort" in supported_params, f"Failed for model: {model}"

    def test_end_to_end_reasoning_workflow(self):
        """Test the complete workflow from parameter to system prompt injection."""
        mistral_config = MistralConfig()
        
        # Step 1: Map parameters
        optional_params = {}
        mapped_params = mistral_config.map_openai_params(
            non_default_params={"reasoning_effort": "high", "temperature": 0.7},
            optional_params=optional_params,
            model="mistral/magistral-medium-2506",
            drop_params=False,
        )
        
        assert mapped_params.get("_add_reasoning_prompt") is True
        assert mapped_params.get("temperature") == 0.7
        
        # Step 2: Transform request
        messages = [
            {"role": "user", "content": "Solve for x: 2x + 5 = 13"}
        ]
        
        result = mistral_config.transform_request(
            model="mistral/magistral-medium-2506",
            messages=messages,
            optional_params=mapped_params,
            litellm_params={},
            headers={}
        )
        
        # Verify final result
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "system"
        assert "<think>" in result["messages"][0]["content"]
        assert result["messages"][1]["role"] == "user"
        assert result["messages"][1]["content"] == "Solve for x: 2x + 5 = 13"
        assert result.get("temperature") == 0.7
        assert "_add_reasoning_prompt" not in result
