"""How `/v1/responses` input items reach a chat provider through the Responses to chat bridge.

The bridge (`litellm/responses/litellm_completion_transformation/transformation.py`) turns every Responses input
shape into chat messages before the provider transform runs: a string `input` is one user message, `instructions`
is a leading system message, a message item keeps its role and its content (string, or a list whose
`input_text`/`output_text` parts become `text` parts, `input_image` an `image_url` part with `detail` defaulting to
`auto`, `input_file` a `file` part), a `function_call` item is an assistant message with one `tool_calls` entry and
`content: null`, consecutive `function_call` items share one assistant message, a `function_call_output` is a
`tool` message, and a `reasoning` item is replayed as `reasoning_content` on the next assistant message, never as
visible text. These rules are what a Rust port of the bridge has to reproduce, and they are provider agnostic:
the tests read them on the `use_chat_completions_api` OpenAI carrier, whose wire body is the bridge output
unchanged, and contrast them with DeepSeek (which collapses text lists to a string) and Anthropic (which needs the
tool turn shape).
"""

from __future__ import annotations

import json
from typing import Final

from integration._support.client import Gateway, list_value, object_value
from integration.providers.dec1_respapi_support import (
    CHAT_PATH,
    MESSAGES_PATH,
    anthropic,
    chat_bridge,
    chat_messages,
    deepseek,
    marker,
    peer,
    responses,
    served,
)
from pydantic import JsonValue

WEATHER_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "name": "get_weather",
    "description": "Look up the weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}


def test_a_string_input_is_one_user_message_at_the_chat_completions_path(gateway: Gateway) -> None:
    """A bridged `openai/gpt-4o-mini` deployment (`use_chat_completions_api: true`) gets `input: "<marker>"`.

    The peer must receive exactly one `POST /v1/chat/completions` (never `/v1/responses`) whose `messages` is
    `[{"role": "user", "content": "<marker>"}]` and whose `model` is `gpt-4o-mini`.
    """
    prompt: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        served(responses(gateway, model, input=prompt))
        call: Final = provider.only()
        assert call.path == CHAT_PATH
        assert call.body["model"] == "gpt-4o-mini"
        assert chat_messages(call) == [{"role": "user", "content": prompt}]


def test_instructions_become_a_leading_system_message(gateway: Gateway) -> None:
    """`instructions: "Answer in one word."` with a string input.

    The chat body must start with `{"role": "system", "content": "Answer in one word."}` followed by the user
    message (`transform_instructions_to_system_message`, `transformation.py:2960`).
    """
    prompt: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        served(responses(gateway, model, input=prompt, instructions="Answer in one word."))
        assert chat_messages(provider.only()) == [
            {"role": "system", "content": "Answer in one word."},
            {"role": "user", "content": prompt},
        ]


def test_an_empty_instructions_string_adds_no_system_message(gateway: Gateway) -> None:
    """`instructions: ""` is falsy, so `transform_responses_api_input_to_messages` adds no system message.

    The chat body must hold only the user message. A port that tests `is not None` instead of truthiness would
    send an empty system message, which Anthropic refuses.
    """
    prompt: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        served(responses(gateway, model, input=prompt, instructions=""))
        assert chat_messages(provider.only()) == [{"role": "user", "content": prompt}]


def test_a_message_list_keeps_every_role_and_its_order(gateway: Gateway) -> None:
    """The input is a list of three easy messages: user, assistant, user, each with string content.

    The chat body must carry the same three messages with the same roles and contents in the same order.
    """
    first: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        served(
            responses(
                gateway,
                model,
                input=[
                    {"role": "user", "content": first},
                    {"role": "assistant", "content": "noted"},
                    {"role": "user", "content": "and now?"},
                ],
            )
        )
        assert chat_messages(provider.only()) == [
            {"role": "user", "content": first},
            {"role": "assistant", "content": "noted"},
            {"role": "user", "content": "and now?"},
        ]


def test_a_message_item_without_a_role_is_a_user_message(gateway: Gateway) -> None:
    """A `{"type": "message", "content": "<marker>"}` item with no `role`.

    `_input_item_role` defaults the role to `user` (`transformation.py:600`), so the chat body must hold
    `{"role": "user", "content": "<marker>"}`.
    """
    prompt: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        served(responses(gateway, model, input=[{"type": "message", "content": prompt}]))
        assert chat_messages(provider.only()) == [{"role": "user", "content": prompt}]


def test_a_developer_message_keeps_the_developer_role_on_an_openai_chat_wire(gateway: Gateway) -> None:
    """A `{"role": "developer", "content": "be terse"}` item before the user turn on the OpenAI chat carrier.

    The role is forwarded verbatim (`_input_item_role` does no validation), so the chat body must hold
    `{"role": "developer", "content": "be terse"}` first. UNVERIFIED: assumes the gpt-4o-mini chat transform
    does not rewrite `developer` to `system`; see `litellm/llms/openai/chat/gpt_transformation.py`.
    """
    prompt: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        served(
            responses(
                gateway,
                model,
                input=[{"role": "developer", "content": "be terse"}, {"role": "user", "content": prompt}],
            )
        )
        assert chat_messages(provider.only())[0] == {"role": "developer", "content": "be terse"}


def test_input_text_parts_become_text_parts(gateway: Gateway) -> None:
    """A user message whose content is two `input_text` parts.

    The bridge strips the `input_` prefix (`_get_chat_completion_request_content_type`), so the chat body must
    hold `[{"type": "text", "text": "<marker>"}, {"type": "text", "text": "second part"}]` as the content.
    """
    prompt: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        served(
            responses(
                gateway,
                model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_text", "text": "second part"},
                        ],
                    }
                ],
            )
        )
        assert chat_messages(provider.only())[0]["content"] == [
            {"type": "text", "text": prompt},
            {"type": "text", "text": "second part"},
        ]


def test_deepseek_receives_input_text_parts_collapsed_into_one_string(gateway: Gateway) -> None:
    """The same two `input_text` parts on a DeepSeek deployment.

    The bridge builds two text parts, then DeepSeek's chat transform joins the texts with no separator
    (`convert_content_list_to_str`), so the peer must see `content == "<marker>second part"`. The contrast with
    the OpenAI carrier shows the bridge itself never collapses content.
    """
    prompt: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deepseek(scenario, provider)
        served(
            responses(
                gateway,
                model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_text", "text": "second part"},
                        ],
                    }
                ],
            )
        )
        call: Final = provider.only()
        assert call.path == CHAT_PATH
        assert chat_messages(call) == [{"role": "user", "content": f"{prompt}second part"}]


def test_output_text_parts_on_an_assistant_turn_become_text_parts(gateway: Gateway) -> None:
    """A replayed assistant output message `{"type": "message", "role": "assistant", "content": [{"type":
    "output_text", "text": "earlier answer", "annotations": []}]}` sits between two user turns.

    The chat body's second message must be an assistant message whose content holds one `text` part with
    `earlier answer`; `output_text` maps to `text` and `annotations` is not forwarded.
    """
    prompt: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        served(
            responses(
                gateway,
                model,
                input=[
                    {"role": "user", "content": prompt},
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "earlier answer", "annotations": []}],
                    },
                    {"role": "user", "content": "go on"},
                ],
            )
        )
        assistant: Final = chat_messages(provider.only())[1]
        assert assistant["role"] == "assistant"
        assert assistant["content"] == [{"type": "text", "text": "earlier answer"}]


def test_an_input_image_url_becomes_an_image_url_part_with_auto_detail(gateway: Gateway) -> None:
    """A user message with `{"type": "input_image", "image_url": "https://example.invalid/cat.png"}` and no
    `detail`.

    `_responses_image_part` (`transformation.py:353`) builds `{"type": "image_url", "image_url": {"url": ...,
    "detail": "auto"}}`, which must be the second content part on the wire.
    """
    prompt: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        served(
            responses(
                gateway,
                model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": "https://example.invalid/cat.png"},
                        ],
                    }
                ],
            )
        )
        parts: Final = list_value(chat_messages(provider.only())[0]["content"])
        assert parts[1] == {
            "type": "image_url",
            "image_url": {"url": "https://example.invalid/cat.png", "detail": "auto"},
        }


def test_an_input_image_keeps_the_detail_the_caller_asked_for(gateway: Gateway) -> None:
    """An `input_image` with a base64 data URL and `detail: "low"`.

    The wire part must be `{"type": "image_url", "image_url": {"url": <the data URL>, "detail": "low"}}`.
    """
    data_url: Final = "data:image/png;base64,iVBORw0KGgo="
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        served(
            responses(
                gateway,
                model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": marker()},
                            {"type": "input_image", "image_url": data_url, "detail": "low"},
                        ],
                    }
                ],
            )
        )
        parts: Final = list_value(chat_messages(provider.only())[0]["content"])
        assert parts[1] == {"type": "image_url", "image_url": {"url": data_url, "detail": "low"}}


def test_an_input_file_id_becomes_a_file_part(gateway: Gateway) -> None:
    """A user message with `{"type": "input_file", "file_id": "file-dec1respapi"}`.

    `_responses_file_part` (`transformation.py:384`) builds `{"type": "file", "file": {"file_id":
    "file-dec1respapi"}}`, which must be the second content part on the chat wire. UNVERIFIED: assumes the
    managed files hook leaves an unmanaged `file-` id untouched.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        served(
            responses(
                gateway,
                model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": marker()},
                            {"type": "input_file", "file_id": "file-dec1respapi"},
                        ],
                    }
                ],
            )
        )
        parts: Final = list_value(chat_messages(provider.only())[0]["content"])
        assert parts[1] == {"type": "file", "file": {"file_id": "file-dec1respapi"}}


def test_an_input_file_with_inline_data_keeps_its_filename(gateway: Gateway) -> None:
    """An `input_file` with `filename: "notes.pdf"` and `file_data: "data:application/pdf;base64,JVBERi0="`.

    The wire part must be `{"type": "file", "file": {"file_data": ..., "filename": "notes.pdf"}}`: both keys are
    copied when set (`_responses_file_part`).
    """
    file_data: Final = "data:application/pdf;base64,JVBERi0="
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        served(
            responses(
                gateway,
                model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": marker()},
                            {"type": "input_file", "filename": "notes.pdf", "file_data": file_data},
                        ],
                    }
                ],
            )
        )
        part: Final = object_value(list_value(chat_messages(provider.only())[0]["content"])[1])
        assert part["type"] == "file"
        assert object_value(part["file"]) == {"file_data": file_data, "filename": "notes.pdf"}


def test_a_function_call_item_becomes_an_assistant_tool_call_with_null_content(gateway: Gateway) -> None:
    """The caller replays a `function_call` item (`call_id: call_dec1a`, `name: get_weather`, `arguments:
    {"city":"Oslo"}`) and its `function_call_output` (`sunny`).

    The chat body must hold, after the user turn, `{"role": "assistant", "content": null, "tool_calls": [{"id":
    "call_dec1a", "type": "function", "function": {"name": "get_weather", "arguments": "{\\"city\\":\\"Oslo\\"}"}}]}`
    and then `{"role": "tool", "tool_call_id": "call_dec1a", "content": "sunny"}`.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        served(
            responses(
                gateway,
                model,
                tools=[WEATHER_TOOL],
                input=[
                    {"role": "user", "content": marker()},
                    {
                        "type": "function_call",
                        "call_id": "call_dec1a",
                        "name": "get_weather",
                        "arguments": '{"city":"Oslo"}',
                    },
                    {"type": "function_call_output", "call_id": "call_dec1a", "output": "sunny"},
                ],
            )
        )
        messages: Final = chat_messages(provider.only())
        assistant: Final = messages[1]
        assert assistant["role"] == "assistant" and assistant.get("content") is None, assistant
        calls: Final = list_value(assistant["tool_calls"])
        assert len(calls) == 1
        call: Final = object_value(calls[0])
        assert (call["id"], call["type"]) == ("call_dec1a", "function")
        assert object_value(call["function"])["name"] == "get_weather"
        assert json.loads(str(object_value(call["function"])["arguments"])) == {"city": "Oslo"}
        tool: Final = messages[2]
        assert (tool["role"], tool["tool_call_id"], tool["content"]) == ("tool", "call_dec1a", "sunny")


def test_two_consecutive_function_calls_share_one_assistant_message(gateway: Gateway) -> None:
    """Two `function_call` items in a row (`call_dec1a` Oslo, `call_dec1b` Rome), then both outputs.

    The bridge merges consecutive calls into one assistant message (`transformation.py:1745`), because
    Anthropic refuses back to back assistant turns. The chat body must hold one assistant message with two
    `tool_calls` (`call_dec1a`, `call_dec1b` in that order) followed by two `tool` messages in the same order.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        served(
            responses(
                gateway,
                model,
                tools=[WEATHER_TOOL],
                input=[
                    {"role": "user", "content": marker()},
                    {
                        "type": "function_call",
                        "call_id": "call_dec1a",
                        "name": "get_weather",
                        "arguments": '{"city":"Oslo"}',
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_dec1b",
                        "name": "get_weather",
                        "arguments": '{"city":"Rome"}',
                    },
                    {"type": "function_call_output", "call_id": "call_dec1a", "output": "sunny"},
                    {"type": "function_call_output", "call_id": "call_dec1b", "output": "rain"},
                ],
            )
        )
        messages: Final = chat_messages(provider.only())
        assert [message["role"] for message in messages] == ["user", "assistant", "tool", "tool"]
        assert [object_value(call)["id"] for call in list_value(messages[1]["tool_calls"])] == [
            "call_dec1a",
            "call_dec1b",
        ]
        assert [message["tool_call_id"] for message in messages[2:]] == ["call_dec1a", "call_dec1b"]


def test_a_function_call_output_with_an_empty_call_id_is_dropped(gateway: Gateway) -> None:
    """A `function_call_output` whose `call_id` is `""` follows the user turn.

    `_transform_responses_api_tool_call_output_to_chat_completion_message` returns no message for an empty call
    id, so the chat body must hold only the user message and no `tool` message.
    """
    prompt: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        served(
            responses(
                gateway,
                model,
                input=[
                    {"role": "user", "content": prompt},
                    {"type": "function_call_output", "call_id": "", "output": "orphan"},
                ],
            )
        )
        assert chat_messages(provider.only()) == [{"role": "user", "content": prompt}]


def test_a_structured_function_call_output_reaches_the_tool_message_as_text_parts(gateway: Gateway) -> None:
    """A `function_call_output` whose `output` is a list `[{"type": "input_text", "text": "sunny"}]`.

    `tool_output_to_chat_content` (`transformation.py:472`) normalises the list into chat text blocks, so the
    `tool` message content must be `[{"type": "text", "text": "sunny"}]`. UNVERIFIED: the exact normalised
    block shape; read `tool_output_to_chat_content`.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        served(
            responses(
                gateway,
                model,
                tools=[WEATHER_TOOL],
                input=[
                    {"role": "user", "content": marker()},
                    {"type": "function_call", "call_id": "call_dec1s", "name": "get_weather", "arguments": "{}"},
                    {
                        "type": "function_call_output",
                        "call_id": "call_dec1s",
                        "output": [{"type": "input_text", "text": "sunny"}],
                    },
                ],
            )
        )
        tool: Final = chat_messages(provider.only())[2]
        assert tool["role"] == "tool"
        assert tool["content"] == [{"type": "text", "text": "sunny"}]


def test_a_replayed_function_call_reaches_anthropic_as_tool_use_and_tool_result(gateway: Gateway) -> None:
    """The same `function_call` (`toolu_dec1a`) plus `function_call_output` replay on an Anthropic deployment.

    Anthropic must receive `POST /v1/messages` with an assistant turn holding a `tool_use` block (`id:
    toolu_dec1a`, `name: get_weather`, `input: {"city": "Oslo"}`) followed by a user turn whose `tool_result`
    block has `tool_use_id: toolu_dec1a`.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = anthropic(scenario, provider)
        served(
            responses(
                gateway,
                model,
                tools=[WEATHER_TOOL],
                input=[
                    {"role": "user", "content": marker()},
                    {
                        "type": "function_call",
                        "call_id": "toolu_dec1a",
                        "name": "get_weather",
                        "arguments": '{"city":"Oslo"}',
                    },
                    {"type": "function_call_output", "call_id": "toolu_dec1a", "output": "sunny"},
                ],
            )
        )
        call: Final = provider.only()
        assert call.path == MESSAGES_PATH
        messages: Final = chat_messages(call)
        tool_use: Final = [
            object_value(block)
            for block in list_value(messages[1]["content"])
            if object_value(block).get("type") == "tool_use"
        ]
        assert [(block["id"], block["name"], block["input"]) for block in tool_use] == [
            ("toolu_dec1a", "get_weather", {"city": "Oslo"})
        ]
        results: Final = [
            object_value(block)
            for block in list_value(messages[2]["content"])
            if object_value(block).get("type") == "tool_result"
        ]
        assert [block["tool_use_id"] for block in results] == ["toolu_dec1a"]


def test_a_reasoning_item_is_replayed_as_reasoning_content_on_the_next_assistant_turn(gateway: Gateway) -> None:
    """On DeepSeek, the input replays `{"type": "reasoning", "id": "rs_dec1", "summary": [{"type":
    "summary_text", "text": "think first"}]}` and then the assistant message `earlier answer`.

    The reasoning item becomes a reasoning only assistant message that `_merge_reasoning_only_assistant_messages`
    folds into the following assistant turn, so the wire must hold one assistant message with `content` that
    reads `earlier answer` and `reasoning_content: "think first"`, and no message whose content is `think first`.
    DeepSeek rejects multi turn requests without `reasoning_content`.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deepseek(scenario, provider)
        served(
            responses(
                gateway,
                model,
                input=[
                    {"role": "user", "content": marker()},
                    {
                        "type": "reasoning",
                        "id": "rs_dec1",
                        "summary": [{"type": "summary_text", "text": "think first"}],
                    },
                    {"role": "assistant", "content": "earlier answer"},
                    {"role": "user", "content": "and then?"},
                ],
            )
        )
        messages: Final = chat_messages(provider.only())
        assistants: Final = [message for message in messages if message["role"] == "assistant"]
        assert len(assistants) == 1, messages
        assert assistants[0]["content"] == "earlier answer"
        assert assistants[0]["reasoning_content"] == "think first"
        assert all(message.get("content") != "think first" for message in messages)


def test_a_reasoning_item_with_only_encrypted_content_is_not_replayed_as_text(gateway: Gateway) -> None:
    """A `reasoning` item that carries only `encrypted_content: "opaque-dec1"` and an empty `summary`.

    With no readable text and no decodable thinking blocks the bridge emits no message for it
    (`transformation.py:2558`), so no wire message may contain `opaque-dec1`. UNVERIFIED: assumes
    `opaque-dec1` does not decode as a bridge minted thinking block payload.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deepseek(scenario, provider)
        served(
            responses(
                gateway,
                model,
                input=[
                    {"role": "user", "content": marker()},
                    {"type": "reasoning", "id": "rs_dec1enc", "summary": [], "encrypted_content": "opaque-dec1"},
                    {"role": "assistant", "content": "earlier answer"},
                    {"role": "user", "content": "and then?"},
                ],
            )
        )
        assert "opaque-dec1" not in json.dumps(provider.only().body)


def test_an_item_reference_with_no_content_is_left_out(gateway: Gateway) -> None:
    """An `{"type": "item_reference", "id": "msg_dec1ref"}` item between two user turns.

    It has no `content`, so the generic branch returns no message (`transformation.py:2563`). The chat body must
    hold exactly the two user messages.
    """
    first: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        served(
            responses(
                gateway,
                model,
                input=[
                    {"role": "user", "content": first},
                    {"type": "item_reference", "id": "msg_dec1ref"},
                    {"role": "user", "content": "second"},
                ],
            )
        )
        assert [message["content"] for message in chat_messages(provider.only())] == [first, "second"]
