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
from litellm.constants import MAX_IMAGE_URL_DOWNLOAD_SIZE_MB
from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.prompt_templates.factory import encode_tool_call_id_with_signature
from litellm.types.utils import ChatCompletionMessageToolCall, Function, Message


class _MovieReview(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    rating: float


def test_basic_multiturn_and_system_instruction_leave_store_to_googles_default() -> None:
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


def test_callers_store_choice_is_forwarded() -> None:
    prepared: Final = transform_chat_to_interactions_request(
        messages=[{"role": "user", "content": "Continue"}],
        model="gemini-3-flash-preview",
        optional_params={"store": True},
        litellm_params={},
    )

    assert prepared.optional_params == {"store": True}


def test_callers_previous_interaction_id_is_rejected_so_chat_cannot_read_a_stored_interaction() -> None:
    with pytest.raises(litellm.BadRequestError, match="previous_interaction_id"):
        transform_chat_to_interactions_request(
            messages=[{"role": "user", "content": "What did I say earlier?"}],
            model="gemini-4-neon",
            optional_params={"previous_interaction_id": "v1_other_keys_turn"},
            litellm_params={"drop_params": True},
        )


@pytest.mark.parametrize("environment", ("env_other_keys_sandbox", {"environment_id": "env_other_keys_sandbox"}))
def test_callers_environment_is_rejected_so_chat_cannot_reuse_a_sandbox_it_does_not_own(environment: object) -> None:
    with pytest.raises(litellm.BadRequestError, match="environment"):
        transform_chat_to_interactions_request(
            messages=[{"role": "user", "content": "List the files"}],
            model="gemini-4-neon",
            optional_params={"environment": environment},
            litellm_params={"drop_params": True},
        )


def test_stateless_tool_loop_replays_signature_and_resolves_tool_name() -> None:
    encoded_id: Final = encode_tool_call_id_with_signature("call_weather_1", "c2lnX2Jhc2U2NF94eXo=")
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
        {"type": "thought", "signature": "c2lnX2Jhc2U2NF94eXo="},
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
    encoded_id: Final = encode_tool_call_id_with_signature("call_grep", "c2lnX2Zyb21fdG9vbF9pZA==")
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
            "signature": "c2lnX2Zyb21fdG9vbF9pZA==",
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


def test_store_is_only_sent_when_the_caller_sets_it() -> None:
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

    assert vertex_turn1.optional_params == {}
    assert vertex_tool_only.optional_params == {}
    assert vertex_mixed.optional_params == {}
    assert [step["type"] for step in vertex_mixed.input_steps if isinstance(step, dict)] == [
        "user_input",
        "thought",
        "model_output",
        "function_call",
        "function_result",
    ]
    assert gemini_mixed.optional_params == {}
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


def test_remote_image_is_sent_as_bare_base64_with_its_detected_mime_type(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "litellm.litellm_core_utils.prompt_templates.factory.convert_url_to_base64",
        lambda url, max_bytes=None: "data:image/webp;base64,ZmFrZQ==",
    )

    prepared = transform_chat_to_interactions_request(
        model="gemini-4-neon",
        messages=[
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/cat"}}]}
        ],
        optional_params={},
        litellm_params={},
        custom_llm_provider="gemini",
    )

    assert prepared.input_steps[0]["content"] == [{"type": "image", "mime_type": "image/webp", "data": "ZmFrZQ=="}]


def test_remote_image_download_is_capped_at_the_configured_image_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    download_caps: Final[list[int | None]] = []  # mutable-ok: download spy

    def fake_download(url: str, max_bytes: int | None = None) -> str:
        download_caps.append(max_bytes)
        return "data:image/webp;base64,ZmFrZQ=="

    monkeypatch.setattr("litellm.litellm_core_utils.prompt_templates.factory.convert_url_to_base64", fake_download)

    transform_chat_to_interactions_request(
        model="gemini-4-neon",
        messages=[
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/cat"}}]}
        ],
        optional_params={},
        litellm_params={},
        custom_llm_provider="gemini",
    )

    assert download_caps == [int(MAX_IMAGE_URL_DOWNLOAD_SIZE_MB * 1024 * 1024)]


@pytest.mark.parametrize("unsupported", ({"background": True}, {"safety_settings": [{"category": "c"}]}))
def test_params_the_chat_bridge_cannot_honor_are_rejected_even_with_drop_params(
    unsupported: Mapping[str, object],
) -> None:
    with pytest.raises(litellm.BadRequestError):
        transform_chat_to_interactions_request(
            model="gemini-4-neon",
            messages=[{"role": "user", "content": "hi"}],
            optional_params=dict(unsupported),
            litellm_params={"drop_params": True},
            custom_llm_provider="gemini",
        )


@pytest.mark.parametrize(
    "tool",
    (
        {"type": "file_search", "file_search_store_names": ["fileSearchStores/other-team"]},
        {"file_search": {"file_search_store_names": ["fileSearchStores/other-team"]}},
    ),
)
def test_file_search_tool_is_rejected_so_chat_cannot_search_a_store_the_key_was_not_granted(
    tool: dict[str, object],
) -> None:
    with pytest.raises(litellm.BadRequestError, match="file_search"):
        transform_chat_to_interactions_request(
            messages=[{"role": "user", "content": "What is in the store?"}],
            model="gemini-4-neon",
            optional_params={"tools": [tool]},
            litellm_params={"drop_params": True},
        )


def test_function_tool_carrying_a_builtin_key_stays_a_function_so_a_denied_builtin_is_not_enabled() -> None:
    prepared: Final = transform_chat_to_interactions_request(
        messages=[{"role": "user", "content": "Run it"}],
        model="gemini-4-neon",
        optional_params={
            "tools": [{"type": "function", "function": {"name": "allowed_fn", "parameters": {}}, "code_execution": {}}]
        },
        litellm_params={},
    )

    assert [tool["type"] for tool in prepared.optional_params["tools"]] == ["function"]


@pytest.mark.parametrize(("token", "sent"), (("tok-1", "tok-1"), ("", None)), ids=("token", "empty"))
def test_empty_continuation_token_is_dropped_so_google_does_not_reject_the_request(
    token: str, sent: str | None
) -> None:
    prepared: Final = transform_chat_to_interactions_request(
        messages=[{"role": "user", "content": "Continue"}],
        model="gemini-4-neon",
        optional_params={"continuation_token": token},
        litellm_params={},
    )

    assert prepared.optional_params.get("continuation_token") == sent


def test_forced_bridge_is_rejected_while_global_safety_settings_are_configured_so_they_are_not_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "vertex_ai_safety_settings", [{"category": "HARM_CATEGORY_HARASSMENT"}])

    with pytest.raises(litellm.BadRequestError, match="safety_settings"):
        transform_chat_to_interactions_request(
            messages=[{"role": "user", "content": "hi"}],
            model="gemini-4-neon",
            optional_params={},
            litellm_params={"drop_params": True},
        )


def test_global_safety_settings_reach_interactions_in_its_own_shape_ahead_of_the_request_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        litellm,
        "vertex_ai_safety_settings",
        [{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}],
    )

    prepared: Final = transform_chat_to_interactions_request(
        messages=[{"role": "user", "content": "hi"}],
        model="gemini-4-neon",
        optional_params={
            "safety_settings": [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_LOW_AND_ABOVE"}]
        },
        litellm_params={},
        custom_llm_provider="vertex_ai",
    )

    assert prepared.optional_params == {"safety_settings": [{"type": "hate_speech", "threshold": "block_none"}]}


@pytest.mark.parametrize(
    ("file_part", "expected"),
    (
        (
            {"file_id": "https://www.youtube.com/watch?v=abc", "format": "video/mp4"},
            {"type": "video", "uri": "https://www.youtube.com/watch?v=abc"},
        ),
        (
            {"file_data": "data:audio/mp3;base64,QUJD"},
            {"type": "audio", "mime_type": "audio/mp3", "data": "QUJD"},
        ),
        (
            {"file_data": "data:application/pdf;base64,QUJD"},
            {"type": "document", "mime_type": "application/pdf", "data": "QUJD"},
        ),
    ),
    ids=("video_uri", "inline_audio", "inline_pdf"),
)
def test_file_part_is_sent_as_the_media_kind_its_mime_type_names(
    file_part: Mapping[str, str], expected: Mapping[str, str]
) -> None:
    prepared: Final = transform_chat_to_interactions_request(
        messages=[{"role": "user", "content": [{"type": "file", "file": file_part}]}],
        model="gemini-4-neon",
        optional_params={},
        litellm_params={},
    )

    assert prepared.input_steps == [{"type": "user_input", "content": [expected]}]


@pytest.mark.parametrize("blank", ("", None, [{"type": "text", "text": ""}]))
def test_blank_final_user_turn_after_an_assistant_reply_still_ends_the_request_on_a_user_turn(blank: object) -> None:
    prepared = transform_chat_to_interactions_request(
        model="gemini-4-neon",
        messages=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": blank},
        ],
        optional_params={},
        litellm_params={},
        custom_llm_provider="gemini",
    )

    assert prepared.input_steps == [
        {"type": "user_input", "content": [{"type": "text", "text": "hello"}]},
        {"type": "model_output", "content": [{"type": "text", "text": "hi there"}]},
        {"type": "user_input", "content": [{"type": "text", "text": " "}]},
    ]


@pytest.mark.parametrize(
    ("maps_options", "expected_tool"),
    [
        ({}, {"type": "google_maps"}),
        (
            {"enableWidget": "ENABLE_WIDGET", "retrievalConfig": {"latitude": 37.42, "longitude": -122.08}},
            {"type": "google_maps", "latitude": 37.42, "longitude": -122.08, "enable_widget": True},
        ),
        ({"latitude": 48, "longitude": 2.35}, {"type": "google_maps", "latitude": 48, "longitude": 2.35}),
    ],
)
def test_google_maps_tool_keeps_the_caller_location_on_interactions(
    maps_options: dict[str, object], expected_tool: dict[str, object]
) -> None:
    sent: Final = transform_chat_to_interactions_request(
        messages=[{"role": "user", "content": "coffee near me"}],
        model="gemini/gemini-3.1-flash-lite-preview",
        optional_params={"tools": [{"googleMaps": maps_options}]},
        litellm_params={},
    )
    assert sent.optional_params["tools"] == [expected_tool]


@pytest.mark.parametrize("role", ["user", "assistant"])
def test_native_image_part_naming_a_stored_file_is_refused_so_it_cannot_skip_the_file_owner_check(role: str) -> None:
    with pytest.raises(UnsupportedParamsError) as refused:
        transform_chat_to_interactions_request(
            messages=[
                {
                    "role": role,
                    "content": [{"type": "image", "uri": "https://generativelanguage.googleapis.com/v1beta/files/abc"}],
                },
                {"role": "user", "content": "transcribe it"},
            ],
            model="gemini/gemini-3.1-flash-lite-preview",
            optional_params={},
            litellm_params={},
        )
    assert refused.value.status_code == 400


def test_message_with_the_native_model_role_is_not_forwarded_as_a_model_turn() -> None:
    file_part: Final = {
        "type": "image_url",
        "image_url": {"url": "https://generativelanguage.googleapis.com/v1beta/files/abc"},
    }
    sent: Final = transform_chat_to_interactions_request(
        messages=[{"role": "model", "content": [file_part]}, {"role": "user", "content": "transcribe it"}],
        model="gemini/gemini-3.1-flash-lite-preview",
        optional_params={},
        litellm_params={},
    )
    assert sent.input_steps == [{"type": "user_input", "content": [{"type": "text", "text": "transcribe it"}]}]
