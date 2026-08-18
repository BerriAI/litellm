import pytest
import litellm
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.utils import (
    supports_reasoning,
    supports_medium_reasoning_effort,
    supports_high_reasoning_effort,
)


def test_openai_gpt_config_supported_params_includes_reasoning_effort_when_supported():
    config = OpenAIGPTConfig()
    
    litellm.register_model({
        "custom-reasoning-model": {
            "max_tokens": 4096,
            "input_cost_per_token": 0.00001,
            "output_cost_per_token": 0.00002,
            "litellm_provider": "openai",
            "mode": "chat",
            "supports_reasoning": True,
            "supports_medium_reasoning_effort": True,
            "supports_high_reasoning_effort": True,
        }
    })
    
    supported_params = config.get_supported_openai_params(model="custom-reasoning-model")
    assert "reasoning_effort" in supported_params
    assert supports_reasoning("custom-reasoning-model", custom_llm_provider="openai") is True
    assert supports_medium_reasoning_effort("custom-reasoning-model", custom_llm_provider="openai") is True
    assert supports_high_reasoning_effort("custom-reasoning-model", custom_llm_provider="openai") is True


def test_supports_medium_and_high_reasoning_effort_helpers():
    litellm.register_model({
        "plain-chat-model": {
            "max_tokens": 4096,
            "input_cost_per_token": 0.00001,
            "output_cost_per_token": 0.00002,
            "litellm_provider": "openai",
            "mode": "chat",
            "supports_reasoning": False,
        }
    })
    
    assert supports_medium_reasoning_effort("plain-chat-model", custom_llm_provider="openai") is False
    assert supports_high_reasoning_effort("plain-chat-model", custom_llm_provider="openai") is False
