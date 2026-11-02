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
        {"type": "thought", "signature": "sig_base64_xyz"},
        {
            "type": "function_call",
            "id": "call_weather_1",
            "name": "get_weather",
            "arguments": {"city": "Tokyo"},
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
        ("gemini/gemini-2.5-pro", "low"),
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


@pytest.mark.parametrize(
    ("model", "litellm_params", "expected_level"),
    (
        ("gemini-2.5-pro", {}, "low"),
        ("gemini-3.1-pro-preview", {}, "low"),
        (
            "gemini/gemini-3.1-pro-preview",
            {"model_info": {"supports_minimal_reasoning_effort": True}},
            "minimal",
        ),
        (
            "gemini/gemini-2.5-pro",
            {"model_info": {"reasoning_effort_levels": ["minimal", "low", "medium", "high"]}},
            "minimal",
        ),
    ),
)
def test_minimal_thinking_level_honors_model_capability_flags_and_clamps_unsupported_models(
    model: str,
    litellm_params: Mapping[str, object],
    expected_level: str,
) -> None:
    disabled_prepared: Final = transform_chat_to_interactions_request(
        messages=[{"role": "user", "content": "hi"}],
        model=model,
        optional_params={"reasoning_effort": "none"},
        litellm_params=litellm_params,
    )
    minimal_prepared: Final = transform_chat_to_interactions_request(
        messages=[{"role": "user", "content": "hi"}],
        model=model,
        optional_params={"reasoning_effort": "minimal"},
        litellm_params=litellm_params,
    )

    assert disabled_prepared.optional_params["generation_config"] == {
        "thinking_level": expected_level,
        "thinking_summaries": "none",
    }
    assert minimal_prepared.optional_params["generation_config"] == {
        "thinking_level": expected_level,
        "thinking_summaries": "auto",
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


def test_unsigned_assistant_preamble_and_tool_calls_order_model_output_before_function_call() -> None:
    assistant_msg: Final = {
        "role": "assistant",
        "content": "Let me inspect both files first.",
        "tool_calls": [
            {
                "id": "call_ls",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"cmd": "ls"}'},
            },
            {
                "id": "call_pwd",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"cmd": "pwd"}'},
            },
        ],
    }

    gemini4_prepared: Final = transform_chat_to_interactions_request(
        messages=[
            {"role": "user", "content": "Check workspace"},
            assistant_msg,
            {"role": "tool", "tool_call_id": "call_ls", "content": "a.py\n"},
            {"role": "tool", "tool_call_id": "call_pwd", "content": "/repo\n"},
        ],
        model="gemini-4-argon",
        optional_params={},
        litellm_params={},
    )
    gemini25_prepared: Final = transform_chat_to_interactions_request(
        messages=[
            {"role": "user", "content": "Check workspace"},
            assistant_msg,
            {"role": "tool", "tool_call_id": "call_ls", "content": "a.py\n"},
            {"role": "tool", "tool_call_id": "call_pwd", "content": "/repo\n"},
        ],
        model="gemini-2.5-flash",
        optional_params={},
        litellm_params={},
    )

    assert gemini4_prepared.input_steps == [
        {"type": "user_input", "content": [{"type": "text", "text": "Check workspace"}]},
        {"type": "thought", "signature": "c2tpcF90aG91Z2h0X3NpZ25hdHVyZV92YWxpZGF0b3I="},
        {"type": "model_output", "content": [{"type": "text", "text": "Let me inspect both files first."}]},
        {"type": "function_call", "id": "call_ls", "name": "bash", "arguments": {"cmd": "ls"}},
        {"type": "function_call", "id": "call_pwd", "name": "bash", "arguments": {"cmd": "pwd"}},
        {
            "type": "function_result",
            "call_id": "call_ls",
            "name": "bash",
            "result": [{"type": "text", "text": "a.py\n"}],
        },
        {
            "type": "function_result",
            "call_id": "call_pwd",
            "name": "bash",
            "result": [{"type": "text", "text": "/repo\n"}],
        },
    ]
    assert gemini25_prepared.input_steps == [
        {"type": "user_input", "content": [{"type": "text", "text": "Check workspace"}]},
        {"type": "model_output", "content": [{"type": "text", "text": "Let me inspect both files first."}]},
        {"type": "function_call", "id": "call_ls", "name": "bash", "arguments": {"cmd": "ls"}},
        {"type": "function_call", "id": "call_pwd", "name": "bash", "arguments": {"cmd": "pwd"}},
        {
            "type": "function_result",
            "call_id": "call_ls",
            "name": "bash",
            "result": [{"type": "text", "text": "a.py\n"}],
        },
        {
            "type": "function_result",
            "call_id": "call_pwd",
            "name": "bash",
            "result": [{"type": "text", "text": "/repo\n"}],
        },
    ]


def test_unsigned_thinking_summary_and_provider_thought_signatures_merge_onto_thought_step() -> None:
    encoded_id: Final = encode_tool_call_id_with_signature("call_grep", "sig_from_tool_id")
    merged_prepared: Final = transform_chat_to_interactions_request(
        messages=[
            {"role": "user", "content": "Search for main"},
            {
                "role": "assistant",
                "content": "Searching now.",
                "thinking_blocks": [{"type": "thinking", "thinking": "need ripgrep"}],
                "tool_calls": [
                    {
                        "id": encoded_id,
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"cmd": "rg main"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": encoded_id, "content": "main.py:1\n"},
        ],
        model="gemini-4-argon",
        optional_params={},
        litellm_params={},
    )
    psf_prepared: Final = transform_chat_to_interactions_request(
        messages=[
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": "Hello there.",
                "provider_specific_fields": {"thought_signatures": ["sig_from_msg_psf"]},
            },
            {"role": "user", "content": "Next?"},
        ],
        model="gemini-4-argon",
        optional_params={},
        litellm_params={},
    )

    assert merged_prepared.input_steps == [
        {"type": "user_input", "content": [{"type": "text", "text": "Search for main"}]},
        {
            "type": "thought",
            "signature": "sig_from_tool_id",
            "summary": [{"type": "text", "text": "need ripgrep"}],
        },
        {"type": "model_output", "content": [{"type": "text", "text": "Searching now."}]},
        {"type": "function_call", "id": "call_grep", "name": "bash", "arguments": {"cmd": "rg main"}},
        {
            "type": "function_result",
            "call_id": "call_grep",
            "name": "bash",
            "result": [{"type": "text", "text": "main.py:1\n"}],
        },
    ]
    assert psf_prepared.input_steps == [
        {"type": "user_input", "content": [{"type": "text", "text": "Hi"}]},
        {"type": "thought", "signature": "sig_from_msg_psf"},
        {"type": "model_output", "content": [{"type": "text", "text": "Hello there."}]},
        {"type": "user_input", "content": [{"type": "text", "text": "Next?"}]},
    ]


def test_drop_params_omits_unsupported_sampling_params_and_normalizes_builtin_tools_and_provider_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {"gemini/gemini-3.1-flash-lite-preview": {"supports_sampling_params": False}},
    )

    with pytest.raises(UnsupportedParamsError):
        transform_chat_to_interactions_request(
            messages=[{"role": "user", "content": "hi"}],
            model="gemini/gemini-3.1-flash-lite-preview",
            optional_params={"temperature": 0.7, "top_p": 0.9},
            litellm_params={},
        )

    dropped: Final = transform_chat_to_interactions_request(
        messages=[{"role": "user", "content": "hi"}],
        model="gemini/gemini-3.1-flash-lite-preview",
        optional_params={
            "temperature": 0.7,
            "top_p": 0.9,
            "seed": 42,
            "tools": [
                {"type": "google_search_retrieval"},
                {"type": "urlContext"},
                {"type": "codeExecution"},
                {"type": "google_maps"},
            ],
        },
        litellm_params={"drop_params": True},
    )
    assert dropped.optional_params == {
        "generation_config": {"seed": 42},
        "tools": [
            {"type": "google_search"},
            {"type": "url_context"},
            {"type": "code_execution"},
            {"type": "google_maps"},
        ],
    }

    with pytest.raises(UnsupportedParamsError) as vertex_exc:
        transform_chat_to_interactions_request(
            messages=[{"role": "user", "content": "hi"}],
            model="gemini-3-flash-preview",
            optional_params={"cached_content": "cachedContents/123"},
            litellm_params={},
            custom_llm_provider="vertex_ai",
        )
    assert getattr(vertex_exc.value, "llm_provider", None) == "vertex_ai"


def test_vertex_mixed_preamble_and_tool_call_replay_defaults_store_false_while_preserving_turn1_and_gemini() -> None:
    encoded_id: Final = encode_tool_call_id_with_signature("call_ls", "sig_vtx_1")
    mixed_messages: Final = [
        {"role": "user", "content": "Inspect repo"},
        {
            "role": "assistant",
            "content": "Checking directory contents.",
            "thinking_blocks": [{"type": "thinking", "thinking": "", "signature": "sig_vtx_1"}],
            "tool_calls": [
                {
                    "id": encoded_id,
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"cmd": "ls"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": encoded_id, "content": "README.md\n"},
    ]
    tool_only_messages: Final = [
        {"role": "user", "content": "Inspect repo"},
        {
            "role": "assistant",
            "content": None,
            "thinking_blocks": [{"type": "thinking", "thinking": "", "signature": "sig_vtx_1"}],
            "tool_calls": [
                {
                    "id": encoded_id,
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"cmd": "ls"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": encoded_id, "content": "README.md\n"},
    ]

    vertex_turn1: Final = transform_chat_to_interactions_request(
        messages=[{"role": "user", "content": "Inspect repo"}],
        model="gemini-4-argon",
        optional_params={},
        litellm_params={},
        custom_llm_provider="vertex_ai",
    )
    vertex_tool_only: Final = transform_chat_to_interactions_request(
        messages=tool_only_messages,
        model="gemini-4-argon",
        optional_params={},
        litellm_params={},
        custom_llm_provider="vertex_ai",
    )
    vertex_mixed: Final = transform_chat_to_interactions_request(
        messages=mixed_messages,
        model="gemini-4-argon",
        optional_params={},
        litellm_params={},
        custom_llm_provider="vertex_ai",
    )
    gemini_mixed: Final = transform_chat_to_interactions_request(
        messages=mixed_messages,
        model="gemini-4-argon",
        optional_params={},
        litellm_params={},
        custom_llm_provider="gemini",
    )
    vertex_explicit_store_true: Final = transform_chat_to_interactions_request(
        messages=mixed_messages,
        model="gemini-4-argon",
        optional_params={"store": True},
        litellm_params={},
        custom_llm_provider="vertex_ai",
    )

    assert "store" not in vertex_turn1.optional_params
    assert "store" not in vertex_tool_only.optional_params
    assert vertex_mixed.optional_params == {"store": False}
    assert [step["type"] for step in vertex_mixed.input_steps if isinstance(step, dict)] == [
        "user_input",
        "thought",
        "model_output",
        "function_call",
        "function_result",
    ]
    assert "store" not in gemini_mixed.optional_params
    assert vertex_explicit_store_true.optional_params == {"store": True}


@pytest.mark.parametrize(
    ("raw_choice", "expected_choice"),
    (
        (
            {
                "type": "allowed_tools",
                "allowed_tools": {
                    "mode": "auto",
                    "tools": [
                        {"type": "function", "function": {"name": "get_weather"}},
                        {"type": "function", "name": "get_time"},
                    ],
                },
            },
            {"allowed_tools": {"mode": "auto", "tools": ["get_weather", "get_time"]}},
        ),
        (
            {
                "type": "allowed_tools",
                "allowed_tools": {
                    "mode": "required",
                    "tools": [{"type": "custom", "custom": {"name": "code_runner"}}],
                },
            },
            {"allowed_tools": {"mode": "any", "tools": ["code_runner"]}},
        ),
    ),
)
def test_openai_allowed_tools_tool_choice_translates_mode_and_names_for_interactions(
    raw_choice: Mapping[str, object],
    expected_choice: Mapping[str, object],
) -> None:
    prepared: Final = transform_chat_to_interactions_request(
        messages=[{"role": "user", "content": "hi"}],
        model="gemini-3-flash-preview",
        optional_params={"tool_choice": raw_choice},
        litellm_params={},
    )
    assert prepared.optional_params == {"generation_config": {"tool_choice": expected_choice}}


@pytest.mark.parametrize(
    "bad_choice",
    (
        {"allowed_tools": "not_a_dict"},
        {"allowed_tools": ["get_weather"]},
        {"allowed_tools": {"mode": "any"}, "extra": 1},
        {"allowed_tools": {"mode": "invalid_mode", "tools": ["get_weather"]}},
        {"allowed_tools": {"mode": "any", "tools": [123]}},
        {
            "type": "allowed_tools",
            "allowed_tools": {"mode": "required", "tools": [{"type": "mcp", "server_label": "s"}]},
        },
        {
            "type": "allowed_tools",
            "allowed_tools": {
                "mode": "required",
                "tools": [{"type": "function", "name": "a", "function": {"name": "b"}}],
            },
        },
    ),
)
def test_malformed_allowed_tools_tool_choice_is_rejected_unless_drop_params_true(
    bad_choice: Mapping[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)

    with pytest.raises(UnsupportedParamsError):
        transform_chat_to_interactions_request(
            messages=[{"role": "user", "content": "hi"}],
            model="gemini-3-flash-preview",
            optional_params={"tool_choice": bad_choice},
            litellm_params={},
        )

    dropped: Final = transform_chat_to_interactions_request(
        messages=[{"role": "user", "content": "hi"}],
        model="gemini-3-flash-preview",
        optional_params={"tool_choice": bad_choice},
        litellm_params={"drop_params": True},
    )
    assert dropped.optional_params == {}
