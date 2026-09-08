from types import MappingProxyType

from litellm.litellm_core_utils.thinking_param_translation import (
    ThinkingParamsState,
    apply_thinking_param_translation,
    translate_thinking_params,
)
from litellm.utils import get_optional_params


def _extra_body_model_info(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "supports_reasoning": True,
        "thinking_param": "thinking.type",
        "thinking_values": ["enabled", "disabled"],
        "reasoning_effort_values": ["low", "high", "max"],
        "thinking_send_via": "extra_body",
    }
    return {**base, **overrides}


def test_translate_thinking_type_and_effort_to_extra_body():
    result = apply_thinking_param_translation(
        model_info=_extra_body_model_info(),
        thinking={"type": "enabled", "budget_tokens": 1024},
        reasoning_effort="high",
        existing_extra_body=None,
    )
    assert result.thinking is None
    assert result.reasoning_effort is None
    assert dict(result.extra_body) == {
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "reasoning_effort": "high",
    }


def test_translate_enable_thinking_bool():
    result = apply_thinking_param_translation(
        model_info=_extra_body_model_info(
            thinking_param="enable_thinking",
            thinking_values=["true", "false"],
        ),
        thinking={"type": "enabled"},
        reasoning_effort=None,
        existing_extra_body=None,
    )
    assert result.thinking is None
    assert dict(result.extra_body) == {"enable_thinking": True}


def test_translate_chat_template_kwargs():
    result = apply_thinking_param_translation(
        model_info=_extra_body_model_info(
            thinking_param="chat_template_kwargs",
            thinking_values=[],
            reasoning_effort_values=["low", "medium", "high"],
        ),
        thinking={"type": "disabled"},
        reasoning_effort="medium",
        existing_extra_body=None,
    )
    assert dict(result.extra_body) == {
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_effort": "medium",
    }


def test_translate_clamps_effort_aliases():
    result = apply_thinking_param_translation(
        model_info=_extra_body_model_info(reasoning_effort_values=["low", "high", "max"]),
        thinking=None,
        reasoning_effort="xhigh",
        existing_extra_body=None,
    )
    assert result.reasoning_effort is None
    assert result.extra_body["reasoning_effort"] == "max"


def test_translate_provider_mapped_keeps_thinking_moves_effort():
    result = translate_thinking_params(
        model_info=_extra_body_model_info(thinking_send_via="provider_mapped"),
        state=ThinkingParamsState(
            thinking={"type": "enabled"},
            reasoning_effort="high",
            extra_body=MappingProxyType({}),
        ),
    )
    assert result.thinking == {"type": "enabled"}
    assert result.reasoning_effort is None
    assert dict(result.extra_body) == {"reasoning_effort": "high"}


def test_translate_noop_without_model_info():
    state = ThinkingParamsState(
        thinking={"type": "enabled"},
        reasoning_effort="high",
        extra_body=MappingProxyType({}),
    )
    assert translate_thinking_params(model_info=None, state=state) is state


def test_translate_noop_when_send_via_na():
    result = apply_thinking_param_translation(
        model_info=_extra_body_model_info(thinking_send_via="n/a"),
        thinking={"type": "enabled"},
        reasoning_effort="high",
        existing_extra_body=None,
    )
    assert result.thinking == {"type": "enabled"}
    assert result.reasoning_effort == "high"
    assert dict(result.extra_body) == {}


def test_get_optional_params_openai_drop_translates_via_model_info():
    optional_params = get_optional_params(
        model="deepseek-v4-flash",
        custom_llm_provider="openai",
        drop_params=True,
        thinking={"type": "enabled"},
        reasoning_effort="high",
        model_info=_extra_body_model_info(),
    )
    assert optional_params.get("thinking") is None
    assert optional_params.get("reasoning_effort") is None
    assert optional_params["extra_body"]["thinking"] == {"type": "enabled"}
    assert optional_params["extra_body"]["reasoning_effort"] == "high"


def test_get_optional_params_openai_drop_without_model_info_drops_params():
    optional_params = get_optional_params(
        model="gpt-4o",
        custom_llm_provider="openai",
        drop_params=True,
        thinking={"type": "enabled"},
        reasoning_effort="high",
    )
    extra_body = optional_params.get("extra_body") or {}
    assert "thinking" not in extra_body
    assert "reasoning_effort" not in extra_body
    assert optional_params.get("thinking") is None
    assert optional_params.get("reasoning_effort") is None
