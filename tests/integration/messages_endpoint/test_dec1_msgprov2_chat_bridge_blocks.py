"""Content blocks across the Messages to chat completions bridge, in both directions.

When a deployment has no native Messages config (a `hosted_vllm/` model here, which also stands for every
chat-only OpenAI-compatible provider such as Baseten, Fireworks, Groq or Together), `POST /v1/messages` is
served by `LiteLLMAnthropicMessagesAdapter` (`litellm/llms/anthropic/messages/adapters/transformation.py`).

Request direction: a user `image` block becomes an OpenAI `image_url` part with a `data:` URL (or the remote
URL), a `tool_result` becomes a `role: tool` message carrying `tool_call_id` and placed before the user text of
the same turn, list content in a tool result is collapsed to a string when it is a single text part, assistant
text blocks are joined into one string, `tool_use` becomes `tool_calls` with JSON string arguments, tool names
longer than 64 characters are shortened to `<55 chars>_<8 hex of sha256>`, and `tool_choice: none` becomes
`"none"`. Answer direction: `reasoning_content` becomes a leading `thinking` block, text then tool calls keep
that order, the original long tool name is restored, finish `content_filter` becomes `stop_reason: refusal` with
`stop_details`, and an unknown finish becomes `end_turn`. Usage splits `prompt_tokens` into `input_tokens`,
`cache_read_input_tokens` and `cache_creation_input_tokens`.

The chat bridge target is PYTHON_ONLY in `litellm/rust_bridge/catalog.py`; this file is the contract a Rust
bridge must keep.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Final

import pytest
from integration._support.client import Gateway, object_value
from integration._support.wire import Reply, Request
from integration.messages_endpoint.dec1_messages_support import JSON_OBJECT, Rig
from integration.messages_endpoint.dec1_msgprov2_support import (
    HOSTED_VLLM_BACKEND,
    HOSTED_VLLM_KEY,
    chat_message_reply,
)
from integration.messages_endpoint.dec1_msgprov_support import (
    PNG_BASE64,
    TOOL,
    chat_echo,
    path_of,
    provider_deployment,
    unique_text,
)
from pydantic import JsonValue

_ECHO: Final = chat_echo(HOSTED_VLLM_BACKEND)


def _vllm(
    gateway: Gateway, respond: Callable[[Request], Reply] = _ECHO, **params: JsonValue
) -> AbstractContextManager[Rig]:
    return provider_deployment(
        gateway,
        respond,
        model=f"hosted_vllm/{HOSTED_VLLM_BACKEND}",
        base_suffix="/v1",
        api_key=HOSTED_VLLM_KEY,
        **params,
    )


def _answer(rig: Rig, **fields: JsonValue) -> dict[str, JsonValue]:
    response: Final = rig.post(rig.body(**fields))
    assert response.status_code == 200, response.text
    return JSON_OBJECT.validate_json(response.content)


def _messages(rig: Rig) -> list[dict[str, JsonValue]]:
    raw: Final = rig.one_body()["messages"]
    assert isinstance(raw, list)
    return [object_value(message) for message in raw]


# ---------------------------------------------------------------- Request direction


def test_the_bridge_posts_chat_completions_under_the_api_base(gateway: Gateway) -> None:
    """A `hosted_vllm/` deployment with `api_base` ending in `/v1` receives `POST /v1/chat/completions` with
    `Authorization: Bearer synthetic-vllm-key`, and nothing at `/v1/messages`."""
    with _vllm(gateway) as rig:
        _answer(rig)
        call: Final = rig.one_call()
        assert path_of(call) == "/v1/chat/completions"
        assert call.headers["authorization"] == f"Bearer {HOSTED_VLLM_KEY}"


def test_a_base64_image_becomes_an_image_url_data_part(gateway: Gateway) -> None:
    """A user turn of `[image(base64 png), text]` reaches the provider as one user message whose content is
    `[{"type": "image_url", "image_url": {"url": "data:image/png;base64,<data>"}}, {"type": "text", ...}]`, in
    the same order (`anthropic_image_source_to_openai_url`)."""
    prompt: Final = unique_text()
    content: Final[list[JsonValue]] = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_BASE64}},
        {"type": "text", "text": prompt},
    ]
    with _vllm(gateway) as rig:
        _answer(rig, messages=[{"role": "user", "content": content}])
        assert _messages(rig) == [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG_BASE64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]


def test_a_url_image_becomes_an_image_url_part_with_the_same_url(gateway: Gateway) -> None:
    """An image with `source: {"type": "url", "url": "https://images.invalid/cat.png"}` is sent as
    `image_url.url` equal to that URL. UNVERIFIED: that the hosted vLLM chat transform does not download and
    inline the remote image; the `.invalid` host guarantees a download attempt would fail loudly."""
    url: Final = "https://images.invalid/cat.png"
    content: Final[list[JsonValue]] = [
        {"type": "image", "source": {"type": "url", "url": url}},
        {"type": "text", "text": unique_text()},
    ]
    with _vllm(gateway) as rig:
        _answer(rig, messages=[{"role": "user", "content": content}])
        part: Final = object_value(_messages(rig)[0]["content"][0])  # pyright: ignore[reportIndexIssue]  # nested JSON
        assert part == {"type": "image_url", "image_url": {"url": url}}


def test_a_tool_result_becomes_a_tool_message_before_the_text_of_its_turn(gateway: Gateway) -> None:
    """A user turn `[tool_result(toolu_a, "14C"), text("and tomorrow?")]` after an assistant `tool_use` becomes
    `{"role": "tool", "tool_call_id": "toolu_a", "content": "14C"}` followed by a separate user message with the
    text part: OpenAI requires tool messages to follow the assistant tool call directly."""
    question: Final = unique_text()
    history: Final[list[JsonValue]] = [
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_a", "name": "get_weather", "input": {"city": "Oslo"}}],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_a", "content": "14C"},
                {"type": "text", "text": "and tomorrow?"},
            ],
        },
    ]
    with _vllm(gateway) as rig:
        _answer(rig, messages=history, tools=[TOOL])
        sent: Final = _messages(rig)
        assert sent[2] == {"role": "tool", "tool_call_id": "toolu_a", "content": "14C"}
        assert sent[3] == {"role": "user", "content": [{"type": "text", "text": "and tomorrow?"}]}
        assert len(sent) == 4


def test_two_tool_results_keep_their_order_and_ids(gateway: Gateway) -> None:
    """An assistant turn with two `tool_use` blocks is answered by two `tool_result` blocks. They become two
    `role: tool` messages in the caller's order, each with its own `tool_call_id`, after one assistant message
    carrying both `tool_calls`."""
    history: Final[list[JsonValue]] = [
        {"role": "user", "content": unique_text()},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Oslo"}},
                {"type": "tool_use", "id": "toolu_2", "name": "get_weather", "input": {"city": "Rome"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "2C"},
                {"type": "tool_result", "tool_use_id": "toolu_2", "content": "21C"},
            ],
        },
    ]
    with _vllm(gateway) as rig:
        _answer(rig, messages=history, tools=[TOOL])
        sent: Final = _messages(rig)
        calls: Final = sent[1]["tool_calls"]
        assert isinstance(calls, list)
        assert [object_value(call)["id"] for call in calls] == ["toolu_1", "toolu_2"]
        assert [object_value(call)["function"] for call in calls] == [
            {"name": "get_weather", "arguments": json.dumps({"city": "Oslo"})},
            {"name": "get_weather", "arguments": json.dumps({"city": "Rome"})},
        ]
        assert sent[2:] == [
            {"role": "tool", "tool_call_id": "toolu_1", "content": "2C"},
            {"role": "tool", "tool_call_id": "toolu_2", "content": "21C"},
        ]


def test_a_single_text_part_tool_result_is_collapsed_to_a_string(gateway: Gateway) -> None:
    """`tool_result.content: [{"type": "text", "text": "dry"}]` is sent as `content: "dry"`, the plain string
    form every OpenAI-compatible server accepts (`_tool_result_content`)."""
    history: Final[list[JsonValue]] = [
        {"role": "user", "content": unique_text()},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_s", "name": "get_weather", "input": {}}]},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_s", "content": [{"type": "text", "text": "dry"}]}
            ],
        },
    ]
    with _vllm(gateway) as rig:
        _answer(rig, messages=history, tools=[TOOL])
        assert _messages(rig)[2] == {"role": "tool", "tool_call_id": "toolu_s", "content": "dry"}


def test_a_multi_part_tool_result_stays_a_list_of_parts(gateway: Gateway) -> None:
    """A tool result with two text parts and an image becomes a tool message whose `content` is the list of the two
    text parts in order; only a single text part collapses to a string. Pinned from CI run 44176682961: the image
    part is dropped, since an OpenAI `tool` message carries text parts only; a port that starts forwarding it as
    `image_url` changes what a chat provider sees."""
    parts: Final[list[JsonValue]] = [
        {"type": "text", "text": "first"},
        {"type": "text", "text": "second"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_BASE64}},
    ]
    history: Final[list[JsonValue]] = [
        {"role": "user", "content": unique_text()},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_m", "name": "get_weather", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_m", "content": parts}]},
    ]
    with _vllm(gateway) as rig:
        _answer(rig, messages=history, tools=[TOOL])
        assert _messages(rig)[2]["content"] == [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]


def test_a_tool_result_error_flag_is_not_sent_to_a_chat_provider(gateway: Gateway) -> None:
    """`is_error: true` on a tool result has no OpenAI field, so the tool message is exactly
    `{"role": "tool", "tool_call_id": ..., "content": ...}` without `is_error` (`keep_tool_errors` is off for
    generic chat providers). A port that copies the flag would get a 400 from strict servers."""
    history: Final[list[JsonValue]] = [
        {"role": "user", "content": unique_text()},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_e", "name": "get_weather", "input": {}}]},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_e", "content": "timeout", "is_error": True}],
        },
    ]
    with _vllm(gateway) as rig:
        _answer(rig, messages=history, tools=[TOOL])
        assert _messages(rig)[2] == {"role": "tool", "tool_call_id": "toolu_e", "content": "timeout"}


def test_assistant_text_blocks_are_joined_into_one_string(gateway: Gateway) -> None:
    """An assistant turn with two text blocks `"Hello "` and `"there."` and no cache_control is replayed as one
    assistant message with `content: "Hello there."`."""
    history: Final[list[JsonValue]] = [
        {"role": "user", "content": unique_text()},
        {"role": "assistant", "content": [{"type": "text", "text": "Hello "}, {"type": "text", "text": "there."}]},
        {"role": "user", "content": "continue"},
    ]
    with _vllm(gateway) as rig:
        _answer(rig, messages=history)
        assert _messages(rig)[1] == {"role": "assistant", "content": "Hello there."}


def test_tool_choice_none_is_sent_as_none(gateway: Gateway) -> None:
    """Anthropic `tool_choice: {"type": "none"}` becomes OpenAI `tool_choice: "none"` and the tools are still
    sent as functions."""
    with _vllm(gateway) as rig:
        _answer(rig, tools=[TOOL], tool_choice={"type": "none"})
        body: Final = rig.one_body()
        assert body["tool_choice"] == "none"
        assert object_value(object_value(body["tools"][0])["function"])["name"] == "get_weather"  # pyright: ignore[reportIndexIssue]  # nested JSON


def test_a_tool_name_over_64_characters_is_shortened_and_restored(gateway: Gateway) -> None:
    """A tool named with 70 characters is sent to the provider as `<first 55 chars>_<first 8 hex of the sha256
    of the full name>` (64 characters, OpenAI's limit). When the provider calls that shortened name, the caller's
    `tool_use` block carries the original 70 character name again (`create_tool_name_mapping`)."""
    long_name: Final = "lookup_" + "x" * 63
    short_name: Final = f"{long_name[:55]}_{hashlib.sha256(long_name.encode()).hexdigest()[:8]}"
    tool: Final[dict[str, JsonValue]] = {**TOOL, "name": long_name}
    call: Final[dict[str, JsonValue]] = {
        "id": "call_long",
        "type": "function",
        "function": {"name": short_name, "arguments": "{}"},
    }
    reply: Final = chat_message_reply(
        HOSTED_VLLM_BACKEND, {"role": "assistant", "content": None, "tool_calls": [call]}, finish_reason="tool_calls"
    )
    with _vllm(gateway, lambda _r: reply) as rig:
        payload: Final = _answer(rig, tools=[tool])
        sent_tool: Final = object_value(object_value(rig.one_body()["tools"][0])["function"])  # pyright: ignore[reportIndexIssue]  # nested JSON
        assert (len(short_name), sent_tool["name"]) == (64, short_name)
        assert object_value(payload["content"][0])["name"] == long_name  # pyright: ignore[reportIndexIssue]  # nested JSON


def test_sampling_fields_without_an_anthropic_twin_are_copied_through(gateway: Gateway) -> None:
    """`temperature: 0.2` and `top_p: 0.9` are not in the adapter's translatable set, so they are copied into the
    chat request unchanged, next to `max_tokens: 64`."""
    with _vllm(gateway) as rig:
        _answer(rig, temperature=0.2, top_p=0.9)
        body: Final = rig.one_body()
        assert (body["temperature"], body["top_p"], body["max_tokens"]) == (0.2, 0.9, 64)


# ---------------------------------------------------------------- Answer direction


def test_reasoning_content_becomes_a_leading_thinking_block(gateway: Gateway) -> None:
    """A chat answer with `reasoning_content: "think first"` and `content: "done"` becomes
    `content: [{"type": "thinking", "thinking": "think first", "signature": ""}, {"type": "text", "text": "done"}]`:
    thinking always comes before text, and a provider that signs nothing gives an empty string signature
    (`signature=""`, litellm/llms/anthropic/messages/adapters/transformation.py:1559 on main; pinned from CI run
    44176682961)."""
    reply: Final = chat_message_reply(
        HOSTED_VLLM_BACKEND, {"role": "assistant", "content": "done", "reasoning_content": "think first"}
    )
    with _vllm(gateway, lambda _r: reply) as rig:
        assert _answer(rig)["content"] == [
            {"type": "thinking", "thinking": "think first", "signature": ""},
            {"type": "text", "text": "done"},
        ]


def test_text_and_tool_calls_come_back_as_text_then_tool_use(gateway: Gateway) -> None:
    """A chat answer with `content: "let me check"` and two `tool_calls` becomes a text block followed by two
    `tool_use` blocks in the provider's order, with `stop_reason: tool_use`."""
    calls: Final[list[JsonValue]] = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": json.dumps({"city": "Oslo"})},
        },
        {
            "id": "call_2",
            "type": "function",
            "function": {"name": "get_weather", "arguments": json.dumps({"city": "Rome"})},
        },
    ]
    reply: Final = chat_message_reply(
        HOSTED_VLLM_BACKEND,
        {"role": "assistant", "content": "let me check", "tool_calls": calls},
        finish_reason="tool_calls",
    )
    with _vllm(gateway, lambda _r: reply) as rig:
        payload: Final = _answer(rig, tools=[TOOL])
        assert payload["stop_reason"] == "tool_use"
        assert payload["content"] == [
            {"type": "text", "text": "let me check"},
            {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "Oslo"}},
            {"type": "tool_use", "id": "call_2", "name": "get_weather", "input": {"city": "Rome"}},
        ]


def test_a_content_filter_finish_becomes_a_refusal(gateway: Gateway) -> None:
    """Finish `content_filter` becomes `stop_reason: "refusal"` with
    `stop_details: {"type": "refusal", "category": null, "explanation": null}`
    (`_translate_openai_finish_reason_to_anthropic`, `refusal_stop_details`). A port that maps it to `end_turn`
    hides moderation from Anthropic clients."""
    reply: Final = chat_message_reply(
        HOSTED_VLLM_BACKEND, {"role": "assistant", "content": ""}, finish_reason="content_filter"
    )
    with _vllm(gateway, lambda _r: reply) as rig:
        payload: Final = _answer(rig)
        assert payload["stop_reason"] == "refusal"
        assert payload["stop_details"] == {"type": "refusal", "category": None, "explanation": None}


def test_an_openai_refusal_message_becomes_text_and_a_refusal_with_its_explanation(gateway: Gateway) -> None:
    """A chat answer with `content: null`, `refusal: "I can't help with that."` and finish `stop` becomes a text
    block with that sentence, `stop_reason: "refusal"`, and `stop_details.explanation` equal to the sentence
    (`refusal_reply_text`). UNVERIFIED: that the chat transform keeps `refusal` on the message for hosted vLLM."""
    sentence: Final = "I can't help with that."
    reply: Final = chat_message_reply(HOSTED_VLLM_BACKEND, {"role": "assistant", "content": None, "refusal": sentence})
    with _vllm(gateway, lambda _r: reply) as rig:
        payload: Final = _answer(rig)
        assert payload["content"] == [{"type": "text", "text": sentence}]
        assert payload["stop_reason"] == "refusal"
        assert object_value(payload["stop_details"])["explanation"] == sentence


def test_an_unknown_finish_reason_becomes_end_turn(gateway: Gateway) -> None:
    """A provider finish of `eos` (vLLM style) has no Anthropic twin and is answered as `stop_reason: end_turn`,
    with `stop_sequence: null`. UNVERIFIED: litellm may first normalise `eos` to `stop`; either way the caller
    sees `end_turn`."""
    reply: Final = chat_message_reply(HOSTED_VLLM_BACKEND, {"role": "assistant", "content": "ok"}, finish_reason="eos")
    with _vllm(gateway, lambda _r: reply) as rig:
        payload: Final = _answer(rig)
        assert (payload["stop_reason"], payload["stop_sequence"]) == ("end_turn", None)


def test_the_bridged_message_has_the_anthropic_envelope(gateway: Gateway) -> None:
    """The caller gets `type: "message"`, `role: "assistant"`, an `id`, a `model`, and `usage` with
    `input_tokens: 11` and `output_tokens: 7` from `prompt_tokens` and `completion_tokens`; no `choices`,
    `object` or `created` fields leak from the chat shape."""
    with _vllm(gateway) as rig:
        payload: Final = _answer(rig)
        assert (payload["type"], payload["role"]) == ("message", "assistant")
        assert isinstance(payload["id"], str) and payload["id"]
        usage: Final = object_value(payload["usage"])
        assert (usage["input_tokens"], usage["output_tokens"]) == (11, 7)
        assert not {"choices", "object", "created"} & set(payload)


def test_cache_write_tokens_are_reported_as_cache_creation_and_removed_from_input(gateway: Gateway) -> None:
    """Chat usage `prompt_tokens: 30` with `prompt_tokens_details: {"cached_tokens": 10, "cache_write_tokens": 5}`
    becomes `input_tokens: 15`, `cache_read_input_tokens: 10` and `cache_creation_input_tokens: 5`, so no
    token is billed twice (`translate_openai_usage_to_anthropic_usage_delta`). UNVERIFIED: that litellm's
    `Usage` keeps `cache_write_tokens` inside `prompt_tokens_details` for hosted vLLM."""
    usage: Final[dict[str, JsonValue]] = {
        "prompt_tokens": 30,
        "completion_tokens": 4,
        "total_tokens": 34,
        "prompt_tokens_details": {"cached_tokens": 10, "cache_write_tokens": 5},
    }
    reply: Final = chat_message_reply(HOSTED_VLLM_BACKEND, {"role": "assistant", "content": "ok"}, usage=usage)
    with _vllm(gateway, lambda _r: reply) as rig:
        answered: Final = object_value(_answer(rig)["usage"])
        assert (
            answered["input_tokens"],
            answered["cache_read_input_tokens"],
            answered["cache_creation_input_tokens"],
            answered["output_tokens"],
        ) == (15, 10, 5, 4)


def test_no_cache_fields_are_reported_without_cached_tokens(gateway: Gateway) -> None:
    """Plain usage with no cached tokens gives `usage` with `input_tokens` and `output_tokens` and neither
    `cache_read_input_tokens` nor `cache_creation_input_tokens` (the adapter only adds them when positive)."""
    with _vllm(gateway) as rig:
        usage: Final = object_value(_answer(rig)["usage"])
        assert "cache_read_input_tokens" not in usage and "cache_creation_input_tokens" not in usage, usage


@pytest.mark.parametrize("arguments", [""], ids=["empty"])
def test_unparseable_tool_arguments_still_give_a_tool_use_block(gateway: Gateway, arguments: str) -> None:
    """A provider tool call with empty `arguments` still becomes one `tool_use` block named `get_weather` with a
    JSON object `input`, and the call is answered with HTTP 200 rather than 500 (`parse_tool_call_arguments`).
    Malformed text is the next test."""
    call: Final[dict[str, JsonValue]] = {
        "id": "call_bad",
        "type": "function",
        "function": {"name": "get_weather", "arguments": arguments},
    }
    reply: Final = chat_message_reply(
        HOSTED_VLLM_BACKEND, {"role": "assistant", "content": None, "tool_calls": [call]}, finish_reason="tool_calls"
    )
    with _vllm(gateway, lambda _r: reply) as rig:
        block: Final = object_value(_answer(rig, tools=[TOOL])["content"][0])  # pyright: ignore[reportIndexIssue]  # nested JSON
        assert (block["type"], block["name"]) == ("tool_use", "get_weather")
        assert isinstance(block["input"], dict)


def test_malformed_tool_arguments_fail_the_call_with_an_anthropic_api_error(gateway: Gateway) -> None:
    """A provider tool call whose `arguments` is `{not json`. Pinned from CI run 44176682961: the adapter's
    `parse_tool_call_arguments` (litellm/llms/anthropic/messages/adapters/transformation.py:1594-1598 on main)
    raises, so the caller gets HTTP 500 with an Anthropic `api_error` whose message starts `Failed to parse tool
    call arguments for tool 'get_weather' (Anthropic pass-through adapter)`, not a `tool_use` block."""
    call: Final[dict[str, JsonValue]] = {
        "id": "call_bad",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{not json"},
    }
    reply: Final = chat_message_reply(
        HOSTED_VLLM_BACKEND, {"role": "assistant", "content": None, "tool_calls": [call]}, finish_reason="tool_calls"
    )
    with _vllm(gateway, lambda _r: reply) as rig:
        response: Final = rig.post(rig.body(tools=[TOOL]))
        assert response.status_code == 500, response.text
        error: Final = object_value(JSON_OBJECT.validate_json(response.content)["error"])
        assert error["type"] == "api_error"
        assert str(error["message"]).startswith(
            "Failed to parse tool call arguments for tool 'get_weather' (Anthropic pass-through adapter)"
        )
