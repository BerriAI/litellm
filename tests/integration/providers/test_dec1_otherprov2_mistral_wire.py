"""Contract: Mistral chat tool calls, JSON mode, message history rules, prompt cache key and stream content blocks.

A `mistral/<model>` deployment posts to `<api_base>/v1/chat/completions` (`/v1` is added when missing) with
`Authorization: Bearer <key>` (`MistralConfig`, `litellm/llms/mistral/chat/transformation.py`). Mistral's API is
stricter than OpenAI's about tool choice values and message history, so the proxy reshapes requests:

- `tool_choice` is only sent when tools are present, `required` becomes `any`, an object naming one function is
  kept, and anything else (for example `allowed_tools`) is refused with 400 unless `drop_params` is set;
- tool schemas lose `$id` and `$schema`;
- `name` is dropped from every message except a tool message with a non-blank name; assistant tool calls are
  rebuilt as `{id, type: function, function}`; a tool message with null content is sent with `""`; an assistant
  message with nothing in it is left out of the history;
- `response_format` (JSON mode and JSON schema) is forwarded as sent;
- `prompt_cache_key` is only taken by `*.mistral.ai` hosts: to any other base it is refused with 400, withheld
  under `drop_params`, and sent when the caller lists it in `allowed_openai_params`;
- stream deltas whose `content` is a list of text blocks reach the caller as plain text.

Basic URL shapes, `max_completion_tokens`/`seed`/`required` mapping and the strict-flag strip are in
`test_dec1_openai_compatible_chat_wire.py`; Magistral reasoning is in `test_dec1_otherprov_reasoning_and_fireworks_wire.py`;
Mistral OCR and embeddings are in `test_dec1_media2_ocr_wire.py` and `test_dec1_embedrerank_embed_selfhosted_wire.py`.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value
from integration._support.wire import Reply
from integration.providers.dec1_otherprov2_support import (
    TOOL_CALL_ID,
    chunk,
    delta_choice,
    message_roles,
    raw_stream,
    tool_call,
    tool_call_reply,
    upstream_error,
    usage_with,
)
from integration.providers.dec1_otherprov_support import (
    TOOL,
    Call,
    chat,
    deployment,
    error_message,
    first_choice,
    joined_delta,
    marker,
    peer,
    served,
    stream_chunks,
)
from pydantic import JsonValue

_MISTRAL: Final = "mistral/mistral-large-latest"
_CACHE_KEY_REASON: Final = "since only the Mistral API (*.mistral.ai) takes it"


def test_mistral_tool_choice_without_tools_is_not_sent(gateway: Gateway) -> None:
    """The caller sends `tool_choice: "auto"` and no tools. `map_openai_params` only maps `tool_choice` when the
    request also carries a non-empty `tools` list (`mistral/chat/transformation.py:280-291`), so Mistral receives
    neither key and the call succeeds."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MISTRAL, api_base_suffix="/v1")
        served(chat(gateway, model, tool_choice="auto"))
        sent: Final = provider.only().body
        assert "tool_choice" not in sent and "tools" not in sent, sent


def test_mistral_tool_choice_naming_one_function_is_kept(gateway: Gateway) -> None:
    """`tool_choice: {"type": "function", "function": {"name": "get_weather"}}` with that tool reaches Mistral
    as the same object (`map_tool_choice`, `mistral/chat/transformation.py:219-231`)."""
    choice: Final[dict[str, JsonValue]] = {"type": "function", "function": {"name": "get_weather"}}
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MISTRAL, api_base_suffix="/v1")
        served(chat(gateway, model, tools=[TOOL], tool_choice=choice))
        assert provider.only().body["tool_choice"] == choice


def test_mistral_refuses_a_tool_choice_it_cannot_express(gateway: Gateway) -> None:
    """`tool_choice: {"type": "allowed_tools", ...}` has no Mistral form, so the call is answered 400 with
    `the Mistral API takes tool_choice only as 'auto', 'none', 'any', 'required'` in the message
    (`_tool_choice_to_send`, `mistral/chat/transformation.py:233-252`, `:101-104`) and Mistral is not called."""
    choice: Final[dict[str, JsonValue]] = {
        "type": "allowed_tools",
        "allowed_tools": {"mode": "auto", "tools": [{"type": "function", "function": {"name": "get_weather"}}]},
    }
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MISTRAL, api_base_suffix="/v1")
        response: Final = chat(gateway, model, tools=[TOOL], tool_choice=choice)
        assert response.status_code == 400, response.text
        assert "the Mistral API takes tool_choice only as 'auto', 'none', 'any', 'required'" in error_message(response)
        assert provider.posts() == ()


def test_mistral_drops_an_inexpressible_tool_choice_under_drop_params(gateway: Gateway) -> None:
    """The same `allowed_tools` choice on a deployment with `drop_params: true` is dropped
    (`mistral/chat/transformation.py:238-243`): Mistral receives the tool and no `tool_choice` key."""
    choice: Final[dict[str, JsonValue]] = {"type": "allowed_tools", "allowed_tools": {"mode": "auto", "tools": []}}
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MISTRAL, api_base_suffix="/v1", drop_params=True)
        served(chat(gateway, model, tools=[TOOL], tool_choice=choice))
        sent: Final = provider.only().body
        assert "tool_choice" not in sent and len(list_value(sent["tools"])) == 1, sent


def test_mistral_tool_schema_loses_id_and_schema_keys(gateway: Gateway) -> None:
    """A tool whose parameters carry `$id` and `$schema` (which break Mistral's grammar validation). Mistral
    receives the parameters without either key and with `properties` intact (`_clean_tool_schema_for_mistral`,
    `mistral/chat/transformation.py:517-547`, `remove_json_schema_refs`, `litellm/utils.py:5271-5294`)."""
    tool: Final[dict[str, JsonValue]] = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "$id": "https://schemas.example.com/weather",
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MISTRAL, api_base_suffix="/v1")
        served(chat(gateway, model, tools=[tool]))
        parameters: Final = object_value(
            object_value(object_value(list_value(provider.only().body["tools"])[0])["function"])["parameters"]
        )
        assert parameters == {"type": "object", "properties": {"city": {"type": "string"}}}, parameters


@pytest.mark.parametrize(
    "response_format",
    [
        {"type": "json_object"},
        {"type": "json_schema", "json_schema": {"name": "city", "schema": {"type": "object"}, "strict": True}},
    ],
    ids=["json-object", "json-schema"],
)
def test_mistral_forwards_response_format_as_sent(gateway: Gateway, response_format: dict[str, JsonValue]) -> None:
    """Mistral takes JSON mode and JSON schema natively, so `response_format` reaches Mistral exactly as the
    caller sent it (`mistral/chat/transformation.py:306-307`), with no tool-call emulation."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MISTRAL, api_base_suffix="/v1")
        served(chat(gateway, model, response_format=response_format))
        sent: Final = provider.only().body
        assert sent["response_format"] == response_format and "tools" not in sent, sent


def test_mistral_keeps_name_only_on_tool_messages(gateway: Gateway) -> None:
    """A history with a user message named `alice`, an assistant tool call, and a tool message named
    `get_weather`. Mistral receives the user message without `name` and the tool message with `name:
    "get_weather"` (`_handle_name_in_message`, `mistral/chat/transformation.py:783-799`)."""
    text: Final = marker()
    messages: Final[list[JsonValue]] = [
        {"role": "user", "content": text, "name": "alice"},
        {"role": "assistant", "content": None, "tool_calls": [tool_call()]},
        {"role": "tool", "tool_call_id": TOOL_CALL_ID, "name": "get_weather", "content": "sunny"},
    ]
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MISTRAL, api_base_suffix="/v1")
        served(chat(gateway, model, messages=messages, tools=[TOOL]))
        sent: Final = list_value(provider.only().body["messages"])
        assert "name" not in object_value(sent[0]), sent
        assert object_value(sent[2])["name"] == "get_weather", sent


def test_mistral_rebuilds_replayed_tool_calls_in_its_own_shape(gateway: Gateway) -> None:
    """An assistant tool call replayed with an extra `index: 0` field (as streamed answers carry). Mistral
    receives that call as exactly `{"id", "type": "function", "function"}` (`_handle_tool_call_message`,
    `mistral/chat/transformation.py:801-817`), so the extra field is gone."""
    replayed: Final[dict[str, JsonValue]] = {"index": 0, **tool_call()}
    messages: Final[list[JsonValue]] = [
        {"role": "user", "content": marker()},
        {"role": "assistant", "content": None, "tool_calls": [replayed]},
        {"role": "tool", "tool_call_id": TOOL_CALL_ID, "content": "sunny"},
    ]
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MISTRAL, api_base_suffix="/v1")
        served(chat(gateway, model, messages=messages, tools=[TOOL]))
        sent_call: Final = object_value(
            list_value(object_value(list_value(provider.only().body["messages"])[1])["tool_calls"])[0]
        )
        assert sent_call == tool_call(), sent_call


def test_mistral_sends_a_null_tool_result_as_an_empty_string(gateway: Gateway) -> None:
    """A tool message answering the replayed call has `content: null`. Mistral rejects null tool content, so the
    message is sent with `content: ""` (`_with_tool_reply_content`, `mistral/chat/transformation.py:849-855`)."""
    messages: Final[list[JsonValue]] = [
        {"role": "user", "content": marker()},
        {"role": "assistant", "content": None, "tool_calls": [tool_call()]},
        {"role": "tool", "tool_call_id": TOOL_CALL_ID, "content": None},
    ]
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MISTRAL, api_base_suffix="/v1")
        served(chat(gateway, model, messages=messages, tools=[TOOL]))
        tool_message: Final = object_value(list_value(provider.only().body["messages"])[2])
        assert tool_message["content"] == "", tool_message


def test_mistral_leaves_an_empty_assistant_message_out_of_the_history(gateway: Gateway) -> None:
    """A history user, assistant with `content: ""` and nothing else, user. Mistral does not take empty
    assistant content (`_is_empty_assistant_message`, `mistral/chat/transformation.py:857-874`, `_sent_flags`
    `:703-722`), so it receives only the two user messages."""
    messages: Final[list[JsonValue]] = [
        {"role": "user", "content": marker()},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": marker()},
    ]
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MISTRAL, api_base_suffix="/v1")
        served(chat(gateway, model, messages=messages))
        assert message_roles(provider.only().body) == ("user", "user")


def test_mistral_tool_call_answer_reaches_the_caller(gateway: Gateway) -> None:
    """Mistral answers one tool call with finish reason `tool_calls`. The caller gets the same id and function
    name and `finish_reason: "tool_calls"`; `content` is null (Mistral's empty string content becomes null,
    `mistral/chat/transformation.py:876-894`)."""
    with peer(tool_call_reply) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MISTRAL, api_base_suffix="/v1")
        choice: Final = first_choice(served(chat(gateway, model, tools=[TOOL])))
        message: Final = object_value(choice["message"])
        found: Final = object_value(list_value(message["tool_calls"])[0])
        assert (choice["finish_reason"], found["id"]) == ("tool_calls", TOOL_CALL_ID), choice
        assert object_value(found["function"])["name"] == "get_weather", choice


def test_mistral_refuses_prompt_cache_key_for_a_non_mistral_host(gateway: Gateway) -> None:
    """The peer is not a `*.mistral.ai` host, so `prompt_cache_key: "dec1-cache"` is answered 400 with
    `since only the Mistral API (*.mistral.ai) takes it` in the message (`_prompt_cache_key_withheld`,
    `mistral/chat/transformation.py:339-375`) and Mistral is not called. Self-hosted Mistral-compatible servers
    reject the unknown field, which is why the proxy guards it."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MISTRAL, api_base_suffix="/v1")
        response: Final = chat(gateway, model, prompt_cache_key="dec1-cache")
        assert response.status_code == 400, response.text
        assert _CACHE_KEY_REASON in error_message(response)
        assert provider.posts() == ()


def test_mistral_withholds_prompt_cache_key_from_a_non_mistral_host_under_drop_params(gateway: Gateway) -> None:
    """The same request on a deployment with `drop_params: true` succeeds and Mistral receives no
    `prompt_cache_key` (`mistral/chat/transformation.py:363-364`, `:957-958`)."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MISTRAL, api_base_suffix="/v1", drop_params=True)
        served(chat(gateway, model, prompt_cache_key="dec1-cache"))
        assert "prompt_cache_key" not in provider.only().body


def test_mistral_sends_prompt_cache_key_when_the_caller_allows_it(gateway: Gateway) -> None:
    """The caller sends `prompt_cache_key: "dec1-cache"` with `allowed_openai_params: ["prompt_cache_key"]`; the
    guard is skipped (`mistral/chat/transformation.py:357-358`) and Mistral receives the key as sent."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MISTRAL, api_base_suffix="/v1")
        served(chat(gateway, model, prompt_cache_key="dec1-cache", allowed_openai_params=["prompt_cache_key"]))
        assert provider.only().body["prompt_cache_key"] == "dec1-cache"


def test_mistral_stream_text_block_lists_reach_the_caller_as_plain_text(gateway: Gateway) -> None:
    """Mistral streams deltas whose `content` is `[{"type": "text", "text": "dec1 "}]` then `[{"type": "text",
    "text": "blocks"}]`. `MistralChatResponseIterator` normalizes list content to a string
    (`mistral/chat/transformation.py:1034-1055`), so the caller's joined `delta.content` is `dec1 blocks`."""

    def respond(call: Call) -> Reply:
        return raw_stream(
            chunk(
                call.body, choices=delta_choice({"role": "assistant", "content": [{"type": "text", "text": "dec1 "}]})
            ),
            chunk(call.body, choices=delta_choice({"content": [{"type": "text", "text": "blocks"}]})),
            chunk(call.body, choices=delta_choice({}, finish_reason="stop"), usage=usage_with()),
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MISTRAL, api_base_suffix="/v1")
        assert joined_delta(stream_chunks(gateway, model), "content") == "dec1 blocks"


@pytest.mark.parametrize("status", [401, 429])
def test_mistral_error_status_reaches_the_caller(gateway: Gateway, status: int) -> None:
    """Mistral answers 401 or 429 with message `dec1 mistral failure`. Mistral is mapped like OpenAI
    (`exception_mapping_utils.py:2733-2749`), so the caller gets the same status and message after one call."""
    with peer(lambda call: upstream_error(status, "dec1 mistral failure")) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MISTRAL, api_base_suffix="/v1")
        response: Final = chat(gateway, model)
        assert response.status_code == status, response.text
        assert "dec1 mistral failure" in error_message(response)
        assert len(provider.posts()) == 1
