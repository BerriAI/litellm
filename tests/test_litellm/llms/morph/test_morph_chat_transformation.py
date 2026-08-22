import os
from typing import Final
from unittest.mock import patch

from litellm import MorphChatConfig


def test_morph_provider_info():
    config: Final = MorphChatConfig()

    with patch.dict(os.environ, MORPH_API_KEY="test-key-from-env"):
        environment_provider_info: Final = config._get_openai_compatible_provider_info(None, None)
        assert environment_provider_info == ("https://api.morphllm.com/v1", "test-key-from-env")

    direct_provider_info: Final = config._get_openai_compatible_provider_info(None, "direct-key")
    assert direct_provider_info == ("https://api.morphllm.com/v1", "direct-key")

    custom_provider_info: Final = config._get_openai_compatible_provider_info("https://custom.morph.com", "key")
    assert custom_provider_info == ("https://custom.morph.com", "key")


def test_morph_custom_llm_provider():
    config: Final = MorphChatConfig()
    assert config.custom_llm_provider == "morph"


def test_morph_supported_params():
    config: Final = MorphChatConfig()
    apply_params: Final = tuple(config.get_supported_openai_params("morph/morph-v3-large"))
    chat_params: Final = frozenset(config.get_supported_openai_params("morph/morph-kimik3"))
    glm_params: Final = frozenset(config.get_supported_openai_params("morph/morph-glm52-744b"))
    deepseek_params: Final = frozenset(config.get_supported_openai_params("morph/morph-dsv4flash-0731"))

    assert apply_params == ("messages", "model", "stream", "temperature", "stop", "max_tokens")
    assert chat_params == frozenset(
        (
            "messages",
            "model",
            "stream",
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "seed",
            "max_tokens",
            "logit_bias",
            "tools",
            "response_format",
            "logprobs",
        )
    )
    assert "service_tier" in glm_params
    assert "logprobs" not in glm_params
    assert "logprobs" not in deepseek_params


def test_morph_maps_supported_chat_params():
    config: Final = MorphChatConfig()
    non_default_params: Final = {
        "max_completion_tokens": 1024,
        "response_format": {"type": "json_object"},
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
    }

    mapped_params: Final = config.map_openai_params(
        non_default_params=non_default_params,
        optional_params={},
        model="morph-kimik3",
        drop_params=False,
    )

    assert mapped_params["max_tokens"] == 1024
    assert mapped_params["response_format"] == non_default_params["response_format"]
    assert mapped_params["tools"] == non_default_params["tools"]
