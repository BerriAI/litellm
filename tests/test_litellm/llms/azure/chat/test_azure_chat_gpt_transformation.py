import os
import sys
from typing import Final

import pytest
from pydantic import TypeAdapter

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)

import litellm
from litellm.litellm_core_utils.prompt_templates.common_utils import TOOL_RESULT_IMAGE_BOUNDARY
from litellm.llms.azure.chat.gpt_transformation import AzureOpenAIConfig
from litellm.utils import get_optional_params

_MAPPED_PARAMS: Final = TypeAdapter(dict[str, object])
_SUPPORTED_PARAMS: Final = TypeAdapter(list[str])


class TestAzureOpenAIConfig:
    def test_is_response_format_supported_model(self):
        config = AzureOpenAIConfig()
        # New logic: Azure deployment names with suffixes and prefixes
        assert config._is_response_format_supported_model("azure/gpt-4.1-suffix")
        assert config._is_response_format_supported_model("gpt-4.1-suffix")
        assert config._is_response_format_supported_model("azure/gpt-4-1-suffix")
        assert config._is_response_format_supported_model("gpt-4-1-suffix")
        # 4o models (should always be supported)
        assert config._is_response_format_supported_model("gpt-4o")
        assert config._is_response_format_supported_model("azure/gpt-4o-custom")
        # Backwards compatibility: base names
        assert config._is_response_format_supported_model("gpt-4.1")
        assert config._is_response_format_supported_model("gpt-4-1")
        # Negative test: clearly unsupported model
        assert not config._is_response_format_supported_model("gpt-3.5-turbo")
        assert not config._is_response_format_supported_model("gpt-3-5-turbo")
        assert not config._is_response_format_supported_model("gpt-3-5-turbo-suffix")
        assert not config._is_response_format_supported_model("gpt-35-turbo-suffix")
        assert not config._is_response_format_supported_model("gpt-35-turbo")

    def test_prompt_cache_key_supported(self):
        """Test that 'prompt_cache_key' is in supported params for Azure OpenAI chat completion models.

        OpenAI's Chat Completions API supports prompt_cache_key for cache routing optimization.
        """
        config = AzureOpenAIConfig()
        supported_params = config.get_supported_openai_params("gpt-4.1-nano")
        assert "prompt_cache_key" in supported_params

        supported_params = config.get_supported_openai_params("gpt-4.1")
        assert "prompt_cache_key" in supported_params


def test_map_openai_params_with_preview_api_version():
    config = AzureOpenAIConfig()
    non_default_params = {
        "response_format": {"type": "json_object"},
    }
    optional_params = {}
    model = "azure/gpt-4-1"
    drop_params = False
    api_version = "preview"
    assert config.map_openai_params(
        non_default_params, optional_params, model, drop_params, api_version
    )


def test_transform_request_hoists_tool_message_image():
    """Azure builds its request via convert_to_azure_openai_messages without the
    OpenAIGPTConfig._transform_messages pipeline, so transform_request must hoist
    tool-message images itself; Azure rejects non-text tool content."""
    data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
    messages = [
        {"role": "user", "content": "read the screenshot"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read", "arguments": "{}"}}],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [{"type": "image_url", "image_url": {"url": data_uri}}],
        },
    ]

    request = AzureOpenAIConfig().transform_request(
        model="gpt-4o",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )

    transformed = request["messages"]
    assert [m.get("role") for m in transformed] == ["user", "assistant", "tool", "user"]
    assert isinstance(transformed[2]["content"], str)
    assert transformed[3]["content"] == [
        {"type": "text", "text": TOOL_RESULT_IMAGE_BOUNDARY},
        {"type": "image_url", "image_url": {"url": data_uri}},
    ]


@pytest.mark.parametrize(
    "model, emitted_key, absent_key",
    [
        ("gpt-5-chat", "max_completion_tokens", "max_tokens"),
        ("gpt-5-chat-latest", "max_completion_tokens", "max_tokens"),
        ("gpt-5-chat-2025-08-07", "max_completion_tokens", "max_tokens"),
        ("gpt-5", "max_completion_tokens", "max_tokens"),
        ("o3-mini", "max_completion_tokens", "max_tokens"),
        ("gpt-4o", "max_tokens", "max_completion_tokens"),
    ],
)
def test_azure_max_tokens_rename_covers_gpt_5_chat_family(model: str, emitted_key: str, absent_key: str) -> None:
    """Azure rejects `max_tokens` for the whole gpt-5 name family, gpt-5-chat* included."""
    mapped: Final = _MAPPED_PARAMS.validate_python(
        get_optional_params(model=model, custom_llm_provider="azure", max_tokens=5)
    )
    assert mapped[emitted_key] == 5
    assert absent_key not in mapped


@pytest.mark.parametrize("model", ["gpt-5-chat", "gpt-5-chat-latest"])
def test_azure_gpt_5_chat_stays_off_the_reasoning_path(model: str) -> None:
    """https://github.com/BerriAI/litellm/issues/13781: gpt-5-chat* is a regular chat model."""
    mapped: Final = _MAPPED_PARAMS.validate_python(
        get_optional_params(
            model=model,
            custom_llm_provider="azure",
            max_tokens=5,
            temperature=0.3,
            presence_penalty=0.1,
            frequency_penalty=0.2,
            stop=["stop"],
            logit_bias={"1": 1},
        )
    )
    supported: Final = _SUPPORTED_PARAMS.validate_python(
        litellm.get_supported_openai_params(model=model, custom_llm_provider="azure")
    )
    assert mapped["temperature"] == 0.3
    assert mapped["presence_penalty"] == 0.1
    assert mapped["frequency_penalty"] == 0.2
    assert mapped["stop"] == ["stop"]
    assert mapped["logit_bias"] == {"1": 1}
    assert "reasoning_effort" not in mapped
    assert "reasoning_effort" not in supported


def test_azure_gpt_5_takes_the_reasoning_path() -> None:
    """Positive control for the predicate split: gpt-5 still drops chat-only params."""
    mapped: Final = _MAPPED_PARAMS.validate_python(
        get_optional_params(
            model="gpt-5",
            custom_llm_provider="azure",
            presence_penalty=0.1,
            logit_bias={"1": 1},
            drop_params=True,
        )
    )
    supported: Final = _SUPPORTED_PARAMS.validate_python(
        litellm.get_supported_openai_params(model="gpt-5", custom_llm_provider="azure")
    )
    assert "presence_penalty" not in mapped
    assert "logit_bias" not in mapped
    assert "reasoning_effort" in supported
