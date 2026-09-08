
import pytest

import litellm


from litellm.llms.fireworks_ai.completion.transformation import (
    FireworksAITextCompletionConfig,
)


@pytest.fixture(autouse=True)
def force_local_model_cost(monkeypatch):
    """Force local model cost map usage for all tests in this file."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map

    litellm.model_cost = get_model_cost_map(url=litellm.model_cost_map_url)


_REASONING_MODEL = "fireworks_ai/accounts/fireworks/models/glm-5p1"
_NON_REASONING_MODEL = "fireworks_ai/accounts/fireworks/models/llama-v3-70b-instruct"


def test_map_extra_body_params_strips_truncate_params():
    config = FireworksAITextCompletionConfig()
    result = config.map_extra_body_params(
        {"extra_body": {"truncate_prompt_tokens": 4096, "prompt_truncate_len": 2048}},
        _REASONING_MODEL,
    )
    assert result == {}


def test_map_extra_body_params_chat_template_kwargs_effort():
    config = FireworksAITextCompletionConfig()
    disabled = config.map_extra_body_params(
        {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        _REASONING_MODEL,
    )
    assert disabled == {"extra_body": {"reasoning_effort": "none"}}

    enabled = config.map_extra_body_params(
        {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
        _REASONING_MODEL,
    )
    assert enabled == {}

    budget = config.map_extra_body_params(
        {"extra_body": {"chat_template_kwargs": {"reasoning_budget": 512}}},
        _REASONING_MODEL,
    )
    assert budget == {"extra_body": {"reasoning_effort": 512}}

    low = config.map_extra_body_params(
        {"extra_body": {"chat_template_kwargs": {"low_effort": True}}},
        _REASONING_MODEL,
    )
    assert low == {"extra_body": {"reasoning_effort": "low"}}


def test_map_extra_body_params_chat_template_kwargs_dropped_for_non_reasoning_model():
    config = FireworksAITextCompletionConfig()
    result = config.map_extra_body_params(
        {"extra_body": {"chat_template_kwargs": {"reasoning_budget": 512}}},
        _NON_REASONING_MODEL,
    )
    assert result == {}


def test_map_extra_body_params_chat_template_kwargs_extra_body_thinking_wins():
    config = FireworksAITextCompletionConfig()
    thinking = {"type": "enabled", "budget_tokens": 4096}
    result = config.map_extra_body_params(
        {"extra_body": {"thinking": thinking, "chat_template_kwargs": {"enable_thinking": False}}},
        _REASONING_MODEL,
    )
    assert result == {"extra_body": {"thinking": thinking}}


def test_map_extra_body_params_top_level_reasoning_effort_moves_into_extra_body():
    config = FireworksAITextCompletionConfig()
    result = config.map_extra_body_params(
        {
            "reasoning_effort": "high",
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        _REASONING_MODEL,
    )
    assert result == {"extra_body": {"reasoning_effort": "high"}}


def test_map_extra_body_params_top_level_thinking_moves_into_extra_body():
    config = FireworksAITextCompletionConfig()
    thinking = {"type": "enabled", "budget_tokens": 1024}
    result = config.map_extra_body_params(
        {"thinking": thinking, "max_tokens": 300},
        _REASONING_MODEL,
    )
    assert result == {"max_tokens": 300, "extra_body": {"thinking": thinking}}
    assert "reasoning_effort" not in {
        k for k in result if k != "extra_body"
    }


def test_map_extra_body_params_top_level_response_format_moves_into_extra_body():
    config = FireworksAITextCompletionConfig()
    native = {"type": "json_object"}
    result = config.map_extra_body_params(
        {
            "response_format": native,
            "extra_body": {"response_format": {"type": "json_schema"}},
        },
        _REASONING_MODEL,
    )
    assert result == {"extra_body": {"response_format": native}}


def test_map_extra_body_params_guided_params():
    config = FireworksAITextCompletionConfig()
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    guided_json = config.map_extra_body_params(
        {"extra_body": {"guided_json": schema}}, _REASONING_MODEL
    )
    assert guided_json == {
        "extra_body": {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema},
            }
        }
    }

    guided_choice = config.map_extra_body_params(
        {"extra_body": {"guided_choice": ["yes", "no"]}}, _REASONING_MODEL
    )
    assert guided_choice == {
        "extra_body": {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "choice",
                    "schema": {"type": "string", "enum": ["yes", "no"]},
                },
            }
        }
    }


def test_map_extra_body_params_guided_native_response_format_wins():
    config = FireworksAITextCompletionConfig()
    native = {"type": "json_object"}
    result = config.map_extra_body_params(
        {
            "response_format": native,
            "extra_body": {"guided_json": {"type": "object"}},
        },
        _REASONING_MODEL,
    )
    assert result == {"extra_body": {"response_format": native}}


def test_map_extra_body_params_strips_unsupported_and_preserves_passthrough():
    config = FireworksAITextCompletionConfig()
    result = config.map_extra_body_params(
        {
            "extra_body": {
                "min_tokens": 10,
                "top_k": 40,
                "best_of": 2,
                "include_reasoning": True,
                "nvext": {"verbosity": 1},
            }
        },
        _REASONING_MODEL,
    )
    assert result == {"extra_body": {"min_tokens": 10, "top_k": 40}}


def test_transform_text_completion_request_keeps_sdk_rejected_keys_in_extra_body():
    config = FireworksAITextCompletionConfig()
    data = config.transform_text_completion_request(
        model="glm-5p1",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={
            "max_tokens": 10,
            "reasoning_effort": "low",
            "extra_body": {
                "truncate_prompt_tokens": 4096,
                "chat_template_kwargs": {"low_effort": True},
                "best_of": 2,
                "top_k": 40,
            },
        },
        headers={},
    )
    assert data["model"] == "accounts/fireworks/models/glm-5p1"
    assert data["prompt"] == "hi"
    assert data["max_tokens"] == 10
    assert "reasoning_effort" not in data
    assert data["extra_body"]["reasoning_effort"] == "low"
    assert data["extra_body"]["top_k"] == 40
    assert "truncate_prompt_tokens" not in data["extra_body"]
    assert "prompt_truncate_len" not in data["extra_body"]
    assert "chat_template_kwargs" not in data["extra_body"]
    assert "best_of" not in data["extra_body"]
    assert "response_format" not in data
