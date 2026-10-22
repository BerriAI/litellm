from collections.abc import Mapping
from functools import reduce
from types import MappingProxyType
from typing import Final

import pytest
from pydantic import BaseModel, ConfigDict

import litellm
from litellm.completion_extras.litellm_interactions_transformation import (
    transform_chat_to_interactions_request,
)
from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.prompt_templates.factory import encode_tool_call_id_with_signature
from litellm.types.utils import ChatCompletionMessageToolCall, Function, Message


class _MovieReview(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    rating: float


def test_basic_multiturn_and_system_instruction_omits_store_by_default() -> None:
    prepared: Final = transform_chat_to_interactions_request(
        messages=[
            {"role": "system", "content": "Be concise."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hi", "cache_control": {"type": "ephemeral"}},
                ],
            },
            {"role": "assistant", "content": "Hello."},
            {"role": "user", "content": "2+2?"},
        ],
        model="gemini-3-flash-preview",
        optional_params={"temperature": 0.4, "max_completion_tokens": 128},
        litellm_params={},
    )

    assert prepared.input_steps == [
        {"type": "user_input", "content": [{"type": "text", "text": "Hi"}]},
        {"type": "model_output", "content": [{"type": "text", "text": "Hello."}]},
        {"type": "user_input", "content": [{"type": "text", "text": "2+2?"}]},
    ]
    assert prepared.optional_params == {
        "system_instruction": "Be concise.",
        "generation_config": {
            "temperature": 0.4,
            "max_output_tokens": 128,
        },
    }


def test_explicit_store_and_previous_interaction_id_pass_through() -> None:
    prepared: Final = transform_chat_to_interactions_request(
        messages=[{"role": "user", "content": "Continue"}],
        model="gemini-3-flash-preview",
        optional_params={"store": False, "previous_interaction_id": "int_prev_42"},
        litellm_params={},
    )

    assert prepared.optional_params == {
        "store": False,
        "previous_interaction_id": "int_prev_42",
    }


def test_stateless_tool_loop_replays_signature_and_resolves_tool_name() -> None:
    encoded_id: Final = encode_tool_call_id_with_signature("call_weather_1", "sig_base64_xyz")
    assistant_msg: Final = Message(
        role="assistant",
        content=None,
        tool_calls=[
            ChatCompletionMessageToolCall(
                id=encoded_id,
                type="function",
                function=Function(name="get_weather", arguments='{"city": "Tokyo"}'),
            )
        ],
    )

    prepared: Final = transform_chat_to_interactions_request(
        messages=[
            {"role": "user", "content": "Weather in Tokyo?"},
            assistant_msg,
            {"role": "tool", "tool_call_id": encoded_id, "content": '{"temp_c": 21}'},
        ],
        model="gemini-3-flash-preview",
        optional_params={
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Fetch weather",
                        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        },
        litellm_params={},
    )

    assert prepared.input_steps == [
        {"type": "user_input", "content": [{"type": "text", "text": "Weather in Tokyo?"}]},
        {
            "type": "function_call",
            "id": "call_weather_1",
            "name": "get_weather",
            "arguments": {"city": "Tokyo"},
            "signature": "sig_base64_xyz",
        },
        {
            "type": "function_result",
            "call_id": "call_weather_1",
            "name": "get_weather",
            "result": [{"type": "text", "text": '{"temp_c": 21}'}],
        },
    ]
    assert prepared.optional_params["tools"] == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Fetch weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ]
    assert prepared.optional_params["generation_config"] == {
        "tool_choice": {"allowed_tools": {"mode": "any", "tools": ["get_weather"]}},
    }


def test_structured_output_pydantic_model_and_thinking_config() -> None:
    prepared: Final = transform_chat_to_interactions_request(
        messages=[{"role": "user", "content": "Review Arrival"}],
        model="gemini-3-flash-preview",
        optional_params={
            "response_format": _MovieReview,
            "reasoning_effort": "none",
            "web_search_options": {},
        },
        litellm_params={},
    )

    assert prepared.optional_params == {
        "tools": [{"type": "google_search"}],
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": _MovieReview.model_json_schema(),
        },
        "generation_config": {
            "thinking_level": "minimal",
            "thinking_summaries": "none",
        },
    }


@pytest.mark.parametrize(
    ("model", "expected_level"),
    (
        ("gemini/future-unknown-model", "low"),
        ("gemini-3.1-pro-preview", "low"),
        ("gemini-2.5-flash", "low"),
        ("gemini-3-flash-preview", "minimal"),
    ),
)
def test_disabled_thinking_defaults_to_low_for_unknown_or_non_gemini3_flash_models(
    model: str,
    expected_level: str,
) -> None:
    prepared: Final = transform_chat_to_interactions_request(
        messages=[{"role": "user", "content": "hi"}],
        model=model,
        optional_params={"reasoning_effort": "none"},
        litellm_params={},
    )

    assert prepared.optional_params["generation_config"] == {
        "thinking_level": expected_level,
        "thinking_summaries": "none",
    }


def test_ai_studio_fetches_remote_image_while_vertex_preserves_uri() -> None:
    calls: Final[list[tuple[str, str | None]]] = []  # mutable-ok: test spy accumulator

    def fake_fetcher(url: str, fmt: str | None) -> tuple[str, str]:
        calls.append((url, fmt))
        return "image/png", "ZmFrZV9wbmdfYnl0ZXM="

    messages: Final = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}],
        }
    ]

    gemini_prepared: Final = transform_chat_to_interactions_request(
        messages=messages,
        model="gemini-3-flash-preview",
        optional_params={},
        litellm_params={},
        custom_llm_provider="gemini",
        media_fetcher=fake_fetcher,
    )
    vertex_prepared: Final = transform_chat_to_interactions_request(
        messages=messages,
        model="gemini-3-flash-preview",
        optional_params={},
        litellm_params={},
        custom_llm_provider="vertex_ai",
        media_fetcher=fake_fetcher,
    )

    assert calls == [("https://example.com/cat.png", None)]
    assert gemini_prepared.input_steps == [
        {
            "type": "user_input",
            "content": [{"type": "image", "mime_type": "image/png", "data": "ZmFrZV9wbmdfYnl0ZXM="}],
        }
    ]
    assert vertex_prepared.input_steps == [
        {
            "type": "user_input",
            "content": [{"type": "image", "uri": "https://example.com/cat.png"}],
        }
    ]


@pytest.mark.parametrize(
    "bad_param",
    (
        {"cached_content": "cachedContents/abc"},
        {"video_metadata": {"start_offset": "1s"}},
        {"logprobs": True},
        {"n": 2},
        {"frequency_penalty": 0.5},
    ),
)
def test_unsupported_params_raise_by_default_and_drop_when_requested(
    bad_param: Mapping[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)

    with pytest.raises(UnsupportedParamsError):
        transform_chat_to_interactions_request(
            messages=[{"role": "user", "content": "hi"}],
            model="gemini-3-flash-preview",
            optional_params=bad_param,
            litellm_params={},
        )

    dropped: Final = transform_chat_to_interactions_request(
        messages=[{"role": "user", "content": "hi"}],
        model="gemini-3-flash-preview",
        optional_params=bad_param,
        litellm_params={"drop_params": True},
    )
    assert dropped.optional_params == {}


def _wrap_layer(inner: object, _: int) -> object:
    return MappingProxyType({"layer": (inner,)})


def test_iterative_normalization_rejects_excessive_nesting_depth() -> None:
    seed: Final[object] = _MovieReview(title="base", rating=9.0)
    deeply_nested: Final = reduce(_wrap_layer, range(70), seed)

    with pytest.raises(ValueError, match="maximum JSON nesting depth"):
        transform_chat_to_interactions_request(
            messages=[{"role": "user", "content": "hi"}],
            model="gemini-3-flash-preview",
            optional_params={"extra_body": deeply_nested},
            litellm_params={},
        )
