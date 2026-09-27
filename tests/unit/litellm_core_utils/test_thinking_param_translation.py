import copy
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import httpx
import openai
import pytest

import litellm
from litellm.litellm_core_utils.thinking_param_translation import (
    ThinkingParamsState,
    translate_thinking_params,
)
from litellm.utils import get_optional_params

_EXTRA_BODY_MODEL_INFO: Final = MappingProxyType(
    {
        "supports_reasoning": True,
        "thinking_param": "thinking.type",
        "thinking_values": ["enabled", "disabled"],
        "reasoning_effort_values": ["low", "high", "max"],
        "thinking_send_via": "extra_body",
    }
)

_CHAT_COMPLETION_RESPONSE: Final = MappingProxyType(
    {
        "id": "chatcmpl-thinking",
        "object": "chat.completion",
        "created": 0,
        "model": "deepseek-v4-flash",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
)


def _model_info(**overrides: object) -> Mapping[str, object]:
    return MappingProxyType({**_EXTRA_BODY_MODEL_INFO, **overrides})


def _state(
    *,
    thinking: object = None,
    reasoning_effort: object = None,
    extra_body: Mapping[str, object] = MappingProxyType({}),
) -> ThinkingParamsState:
    return ThinkingParamsState(thinking=thinking, reasoning_effort=reasoning_effort, extra_body=extra_body)


def _as_plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _as_plain(item) for key, item in value.items()}
    return value


def _recording_client(bodies: list[object]) -> openai.OpenAI:
    def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=dict(_CHAT_COMPLETION_RESPONSE))

    return openai.OpenAI(api_key="test-key", http_client=httpx.Client(transport=httpx.MockTransport(respond)))


@pytest.mark.parametrize(
    ("model_info", "thinking", "reasoning_effort", "expected_extra_body"),
    [
        pytest.param(
            _model_info(),
            {"type": "enabled", "budget_tokens": 1024, "clear_thinking": False},
            "high",
            {
                "thinking": {"type": "enabled", "budget_tokens": 1024, "clear_thinking": False},
                "reasoning_effort": "high",
            },
            id="thinking_type_keeps_caller_thinking_keys",
        ),
        pytest.param(
            _model_info(),
            {"type": "auto"},
            None,
            {"thinking": {"type": "enabled"}},
            id="thinking_type_auto_falls_back_to_enabled",
        ),
        pytest.param(
            _model_info(),
            False,
            None,
            {"thinking": {"type": "disabled"}},
            id="thinking_type_from_bool",
        ),
        pytest.param(
            _model_info(),
            "enabled",
            None,
            {"thinking": {"type": "enabled"}},
            id="thinking_type_from_string",
        ),
        pytest.param(
            _model_info(thinking_param="thinking", thinking_values=[]),
            {"type": "enabled", "budget_tokens": 2048},
            None,
            {"thinking": {"type": "enabled", "budget_tokens": 2048}},
            id="thinking_dict_passthrough",
        ),
        pytest.param(
            _model_info(thinking_param="thinking", thinking_values=[]),
            True,
            None,
            {"thinking": {"type": "enabled"}},
            id="thinking_from_bool",
        ),
        pytest.param(
            _model_info(thinking_param="enable_thinking"),
            {"type": "enabled"},
            None,
            {"enable_thinking": True},
            id="enable_thinking_from_type",
        ),
        pytest.param(
            _model_info(thinking_param="enable_thinking"),
            {"enabled": True},
            None,
            {"enable_thinking": True},
            id="enable_thinking_from_enabled_flag",
        ),
        pytest.param(
            _model_info(thinking_param="enable_thinking"),
            "true",
            None,
            {"enable_thinking": True},
            id="enable_thinking_from_string",
        ),
        pytest.param(
            _model_info(thinking_param="enable_thinking"),
            False,
            None,
            {"enable_thinking": False},
            id="enable_thinking_from_bool",
        ),
        pytest.param(
            _model_info(thinking_param="enable_thinking"),
            1,
            None,
            {"enable_thinking": False},
            id="enable_thinking_unrecognized_value_is_disabled",
        ),
        pytest.param(
            _model_info(thinking_param="chat_template_kwargs", reasoning_effort_values=["low", "medium", "high"]),
            {"type": "disabled"},
            "medium",
            {"chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "medium"},
            id="chat_template_kwargs",
        ),
        pytest.param(
            _model_info(reasoning_effort_values=["low", "high", "max"]),
            None,
            "xhigh",
            {"reasoning_effort": "max"},
            id="effort_clamped_to_alias",
        ),
        pytest.param(
            _model_info(reasoning_effort_values=["low"]),
            None,
            "minimal",
            {"reasoning_effort": "low"},
            id="effort_clamped_down",
        ),
        pytest.param(
            _model_info(reasoning_effort_values=[]),
            None,
            "high",
            {"reasoning_effort": "high"},
            id="effort_passthrough_without_allowed_values",
        ),
        pytest.param(
            _model_info(reasoning_effort_values=None),
            None,
            "high",
            {"reasoning_effort": "high"},
            id="effort_passthrough_when_allowed_values_missing",
        ),
    ],
)
def test_translate_moves_params_into_extra_body(
    model_info: Mapping[str, object],
    thinking: object,
    reasoning_effort: object,
    expected_extra_body: dict[str, object],
):
    result = translate_thinking_params(
        model_info=model_info, state=_state(thinking=thinking, reasoning_effort=reasoning_effort)
    )

    assert (result.thinking, result.reasoning_effort) == (None, None)
    assert _as_plain(result.extra_body) == expected_extra_body


@pytest.mark.parametrize(
    ("model_info", "thinking", "reasoning_effort"),
    [
        pytest.param(None, {"type": "enabled"}, "high", id="no_model_info"),
        pytest.param(_model_info(thinking_send_via="n/a"), {"type": "enabled"}, "high", id="send_via_not_applicable"),
        pytest.param(_model_info(supports_reasoning=False), {"type": "enabled"}, "high", id="reasoning_unsupported"),
        pytest.param(_model_info(), None, None, id="nothing_requested"),
        pytest.param(_model_info(thinking_param="unknown"), {"type": "enabled"}, None, id="unknown_thinking_param"),
        pytest.param(_model_info(thinking_param=None), {"type": "enabled"}, None, id="thinking_param_missing"),
        pytest.param(_model_info(), {"type": "adaptive"}, None, id="thinking_type_not_allowed"),
        pytest.param(_model_info(), {"budget_tokens": 1024}, None, id="thinking_type_unreadable"),
        pytest.param(_model_info(), 1, None, id="thinking_type_unsupported_value"),
        pytest.param(
            _model_info(thinking_param="thinking", thinking_values=[]), "adaptive", None, id="thinking_value_unmapped"
        ),
        pytest.param(_model_info(reasoning_effort_values=["low"]), None, "ultra", id="effort_without_fallback"),
        pytest.param(_model_info(), None, 5, id="effort_not_a_string"),
    ],
)
def test_translate_returns_state_unchanged_when_nothing_applies(
    model_info: Mapping[str, object] | None,
    thinking: object,
    reasoning_effort: object,
):
    state = _state(thinking=thinking, reasoning_effort=reasoning_effort)

    assert translate_thinking_params(model_info=model_info, state=state) is state


def test_translate_provider_mapped_keeps_thinking_and_moves_effort():
    thinking = {"type": "enabled"}

    result = translate_thinking_params(
        model_info=_model_info(thinking_send_via="provider_mapped", supports_reasoning=False),
        state=_state(thinking=thinking, reasoning_effort="high"),
    )

    assert result.thinking is thinking
    assert result.reasoning_effort is None
    assert _as_plain(result.extra_body) == {"reasoning_effort": "high"}


def test_translate_keeps_caller_extra_body_values_over_translated_ones():
    result = translate_thinking_params(
        model_info=_model_info(thinking_param="chat_template_kwargs", thinking_values=[]),
        state=_state(
            thinking={"type": "enabled"},
            reasoning_effort="high",
            extra_body={"chat_template_kwargs": {"enable_thinking": False, "reasoning_budget": 512}, "top_k": 20},
        ),
    )

    assert (result.thinking, result.reasoning_effort) == (None, None)
    assert _as_plain(result.extra_body) == {
        "chat_template_kwargs": {"enable_thinking": False, "reasoning_budget": 512},
        "reasoning_effort": "high",
        "top_k": 20,
    }


def test_get_optional_params_moves_thinking_into_extra_body():
    optional_params = get_optional_params(
        model="deepseek-v4-flash",
        custom_llm_provider="openai",
        drop_params=True,
        thinking={"type": "enabled"},
        reasoning_effort="high",
        model_info=_model_info(),
    )

    assert "thinking" not in optional_params
    assert "reasoning_effort" not in optional_params
    assert optional_params["extra_body"] == {"thinking": {"type": "enabled"}, "reasoning_effort": "high"}


def test_get_optional_params_without_model_info_drops_thinking():
    optional_params = get_optional_params(
        model="gpt-4o",
        custom_llm_provider="openai",
        drop_params=True,
        thinking={"type": "enabled"},
        reasoning_effort="high",
    )

    assert "thinking" not in optional_params
    assert "thinking" not in (optional_params.get("extra_body") or {})


def test_get_optional_params_does_not_reintroduce_dropped_thinking():
    optional_params = get_optional_params(
        model="deepseek-v4-flash",
        custom_llm_provider="openai",
        drop_params=True,
        thinking={"type": "enabled"},
        additional_drop_params=["thinking"],
        model_info=_model_info(thinking_param="chat_template_kwargs", thinking_values=[]),
    )

    assert "thinking" not in optional_params
    assert optional_params.get("extra_body") in (None, {})


def test_get_optional_params_leaves_caller_extra_body_untouched_and_serializable():
    caller_extra_body = {"chat_template_kwargs": {"reasoning_budget": 512}}

    optional_params = get_optional_params(
        model="deepseek-v4-flash",
        custom_llm_provider="openai",
        thinking={"type": "enabled"},
        extra_body=caller_extra_body,
        model_info=_model_info(thinking_param="chat_template_kwargs", thinking_values=[]),
    )

    expected_extra_body = {"chat_template_kwargs": {"reasoning_budget": 512, "enable_thinking": True}}
    assert optional_params["extra_body"] == expected_extra_body
    assert json.loads(json.dumps(optional_params["extra_body"])) == expected_extra_body
    assert copy.deepcopy(optional_params)["extra_body"] == expected_extra_body
    assert caller_extra_body == {"chat_template_kwargs": {"reasoning_budget": 512}}


def test_completion_sends_translated_thinking_on_the_wire():
    bodies: list[object] = []

    litellm.completion(
        model="openai/deepseek-v4-flash",
        messages=[{"role": "user", "content": "hi"}],
        thinking={"type": "enabled", "budget_tokens": 1024},
        reasoning_effort="high",
        model_info=dict(_model_info()),
        client=_recording_client(bodies),
        num_retries=0,
    )

    assert bodies == [
        {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "reasoning_effort": "high",
        }
    ]


def test_batch_completion_translates_every_request_like_completion():
    bodies: list[object] = []

    litellm.batch_completion(
        model="openai/deepseek-v4-flash",
        messages=[[{"role": "user", "content": "one"}], [{"role": "user", "content": "two"}]],
        thinking={"type": "enabled"},
        model_info=dict(_model_info(thinking_param="enable_thinking")),
        client=_recording_client(bodies),
        num_retries=0,
        max_workers=1,
    )

    assert sorted(bodies, key=lambda body: json.dumps(body, sort_keys=True)) == [
        {"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "one"}], "enable_thinking": True},
        {"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "two"}], "enable_thinking": True},
    ]
