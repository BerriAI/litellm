import importlib
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


def test_translate_chat_template_kwargs_preserves_existing_nested_keys():
    result = apply_thinking_param_translation(
        model_info=_extra_body_model_info(
            thinking_param="chat_template_kwargs",
            thinking_values=[],
        ),
        thinking={"type": "enabled"},
        reasoning_effort=None,
        existing_extra_body={"chat_template_kwargs": {"reasoning_budget": 512}},
    )
    assert result.extra_body["chat_template_kwargs"]["reasoning_budget"] == 512
    assert result.extra_body["chat_template_kwargs"]["enable_thinking"] is True


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


def test_get_optional_params_does_not_reintroduce_dropped_thinking():
    optional_params = get_optional_params(
        model="deepseek-v4-flash",
        custom_llm_provider="openai",
        drop_params=True,
        thinking={"type": "enabled"},
        additional_drop_params=["thinking"],
        model_info=_extra_body_model_info(
            thinking_param="chat_template_kwargs",
            thinking_values=[],
        ),
    )
    extra_body = optional_params.get("extra_body") or {}
    assert optional_params.get("thinking") is None
    assert "enable_thinking" not in extra_body
    assert "chat_template_kwargs" not in extra_body


def test_batch_completion_vllm_passes_model_info(monkeypatch):
    batch_completion_mod = importlib.import_module("litellm.batch_completion.main")

    captured: dict[str, object] = {}
    looked_up: dict[str, object] = _extra_body_model_info(thinking_param="enable_thinking")

    def fake_get_optional_params(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {}

    def fake_batch_completions(**kwargs: object) -> list[str]:
        return ["ok"]

    def fake_get_model_info(**kwargs: object) -> dict[str, object]:
        return looked_up

    monkeypatch.setattr(batch_completion_mod, "get_optional_params", fake_get_optional_params)
    monkeypatch.setattr(batch_completion_mod.vllm_handler, "batch_completions", fake_batch_completions)
    monkeypatch.setattr(batch_completion_mod, "get_model_info", fake_get_model_info)

    batch_completion_mod.batch_completion(
        model="vllm/some-model",
        messages=[[{"role": "user", "content": "hi"}]],
    )
    assert captured.get("model_info") == looked_up
