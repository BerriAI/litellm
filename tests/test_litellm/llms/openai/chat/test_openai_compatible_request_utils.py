"""Tests for shared OpenAI-compatible request normalization helpers."""

import pytest

import litellm
from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.llms.openai.chat.openai_compatible_request_utils import (
    maybe_inject_json_keyword_hint_for_json_object,
    messages_contain_json_keyword,
    normalize_flat_function_tools,
    response_format_is_json_object,
)
from litellm.llms.openai.openai import OpenAIConfig
from litellm.utils import (
    requires_anthropic_request_sanitize,
    requires_json_keyword_for_json_object,
)


@pytest.fixture(autouse=True)
def use_local_model_cost_map(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(
        litellm, "model_cost", get_model_cost_map(url=litellm.model_cost_map_url)
    )
    litellm.add_known_models(model_cost_map=litellm.model_cost)


def test_requires_json_keyword_for_json_object_reads_model_map():
    assert requires_json_keyword_for_json_object("custom_openai/GLM-5.1") is True
    assert requires_json_keyword_for_json_object("gpt-4o") is False


def test_requires_anthropic_request_sanitize_reads_model_map():
    assert requires_anthropic_request_sanitize("custom_openai/GLM-5.1") is True
    assert requires_anthropic_request_sanitize("claude-opus-4-7") is False


def test_response_format_is_json_object():
    assert response_format_is_json_object({"response_format": {"type": "json_object"}})
    assert not response_format_is_json_object({"response_format": {"type": "text"}})
    assert not response_format_is_json_object({})


def test_messages_contain_json_keyword_nested():
    assert not messages_contain_json_keyword(
        [{"role": "user", "content": [{"type": "text", "text": "no match"}]}]
    )
    assert messages_contain_json_keyword(
        [{"role": "user", "content": [{"type": "text", "text": "Return JSON"}]}]
    )


def test_normalize_flat_function_tools_coerces_non_dict_parameters():
    tools = [
        {
            "type": "function",
            "name": "search",
            "description": "Search",
            "parameters": "invalid",
        }
    ]
    normalized = normalize_flat_function_tools(tools)
    assert normalized[0]["function"]["parameters"] == {"type": "object"}


def test_normalize_flat_function_tools_preserves_existing_function_wrapper():
    tools = [
        {
            "type": "function",
            "function": {"name": "search", "parameters": {"type": "object"}},
        }
    ]
    assert normalize_flat_function_tools(tools) == tools


def test_normalize_flat_function_tools_skips_function_without_name():
    tools = [{"type": "function", "description": "missing name"}]
    assert normalize_flat_function_tools(tools) == tools


def test_json_hint_appends_to_existing_system_message():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Reply with one word."},
    ]
    result = maybe_inject_json_keyword_hint_for_json_object(
        model="custom_openai/GLM-5.1",
        messages=messages,
        optional_params={"response_format": {"type": "json_object"}},
    )
    assert len(result) == 2
    assert result[0]["role"] == "system"
    assert "helpful" in str(result[0]["content"]).lower()
    assert "json" in str(result[0]["content"]).lower()


def test_json_hint_appends_to_system_message_list_content():
    messages = [
        {
            "role": "developer",
            "content": [{"type": "text", "text": "You are helpful."}],
        },
        {"role": "user", "content": "Reply with one word."},
    ]
    result = maybe_inject_json_keyword_hint_for_json_object(
        model="custom_openai/GLM-5.1",
        messages=messages,
        optional_params={"response_format": {"type": "json_object"}},
    )
    assert len(result) == 2
    content = result[0]["content"]
    assert isinstance(content, list)
    assert content[-1]["text"] == "Return valid JSON only."


def test_json_hint_sets_content_when_system_message_content_is_none():
    messages = [
        {"role": "system", "content": None},
        {"role": "user", "content": "Reply with one word."},
    ]
    result = maybe_inject_json_keyword_hint_for_json_object(
        model="custom_openai/GLM-5.1",
        messages=messages,
        optional_params={"response_format": {"type": "json_object"}},
    )
    assert result[0]["content"] == "Return valid JSON only."


def test_json_hint_prepends_system_message_when_missing():
    messages = [{"role": "user", "content": "Reply with one word."}]
    result = maybe_inject_json_keyword_hint_for_json_object(
        model="custom_openai/GLM-5.1",
        messages=messages,
        optional_params={"response_format": {"type": "json_object"}},
    )
    assert result[0]["role"] == "system"
    assert "json" in str(result[0]["content"]).lower()


def test_json_hint_skips_when_prompt_already_contains_json():
    messages = [{"role": "user", "content": "Return JSON only."}]
    result = maybe_inject_json_keyword_hint_for_json_object(
        model="custom_openai/GLM-5.1",
        messages=messages,
        optional_params={"response_format": {"type": "json_object"}},
    )
    assert result == messages


def test_openai_config_delegates_to_shared_normalize_flat_function_tools():
    config = OpenAIConfig()
    tools = [{"type": "shell", "environment": {"type": "local"}}]
    assert config._normalize_flat_function_tools(
        tools
    ) == normalize_flat_function_tools(tools)


def test_openai_config_json_helper_shims():
    config = OpenAIConfig()
    optional_params = {"response_format": {"type": "json_object"}}
    assert config._response_format_is_json_object(optional_params) is True
    assert config._messages_contain_json_keyword([{"role": "user", "content": "json"}])
    result = config._maybe_inject_json_keyword_hint(
        model="custom_openai/GLM-5.1",
        messages=[{"role": "user", "content": "hi"}],
        optional_params=optional_params,
    )
    assert result[0]["role"] == "system"


def test_openai_gpt_config_json_helper_shims():
    config = OpenAIGPTConfig()
    optional_params = {"response_format": {"type": "json_object"}}
    assert config._response_format_is_json_object(optional_params) is True
    assert config._messages_contain_json_keyword([{"role": "user", "content": "json"}])
    result = config._maybe_inject_json_keyword_hint(
        model="custom_openai/GLM-5.1",
        messages=[{"role": "user", "content": "hi"}],
        optional_params=optional_params,
    )
    assert result[0]["role"] == "system"


@pytest.mark.asyncio
async def test_openai_gpt_config_async_transform_request_when_base_class():
    config = OpenAIGPTConfig.__new__(OpenAIGPTConfig)
    previous = OpenAIGPTConfig._is_base_class
    OpenAIGPTConfig._is_base_class = True
    try:
        result = await config.async_transform_request(
            model="custom_openai/GLM-5.1",
            messages=[{"role": "user", "content": "Reply with one word."}],
            optional_params={"response_format": {"type": "json_object"}},
            litellm_params={"custom_llm_provider": "custom_openai"},
            headers={},
        )
    finally:
        OpenAIGPTConfig._is_base_class = previous

    assert result["messages"][0]["role"] == "system"
    assert "json" in str(result["messages"][0]["content"]).lower()
