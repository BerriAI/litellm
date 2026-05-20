"""Tests for shared OpenAI-compatible request normalization helpers."""

import pytest

import litellm
from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map
from litellm.llms.openai.chat.openai_compatible_request_utils import (
    maybe_inject_json_keyword_hint_for_json_object,
    normalize_flat_function_tools,
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


def test_json_hint_prepends_system_message_when_missing():
    messages = [{"role": "user", "content": "Reply with one word."}]
    result = maybe_inject_json_keyword_hint_for_json_object(
        model="custom_openai/GLM-5.1",
        messages=messages,
        optional_params={"response_format": {"type": "json_object"}},
    )
    assert result[0]["role"] == "system"
    assert "json" in str(result[0]["content"]).lower()


def test_openai_config_delegates_to_shared_normalize_flat_function_tools():
    config = OpenAIConfig()
    tools = [{"type": "shell", "environment": {"type": "local"}}]
    assert config._normalize_flat_function_tools(
        tools
    ) == normalize_flat_function_tools(tools)
