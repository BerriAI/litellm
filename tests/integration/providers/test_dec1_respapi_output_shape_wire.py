"""The Responses object a caller gets back when the answer came from a chat provider through the bridge.

`transform_chat_completion_response_to_responses_api_response` (`litellm/responses/litellm_completion_transformation/
transformation.py:3602`) rebuilds a Responses object from a chat completion: `object: response`, a `status` mapped
from the chat `finish_reason` (`stop`, `tool_calls` are `completed`; `length`, `content_filter`, `refusal` are
`incomplete` with `incomplete_details.reason` of `max_output_tokens` or `content_filter`), an `output` list that
holds reasoning items first (`rs_` ids, from `reasoning_content` or thinking blocks), then message items (`msg_`
ids, `output_text` parts with `annotations`), then `function_call` items whose `id` is `fc_<call id>` and whose
`call_id` is the provider's tool call id. Usage is renamed (`prompt_tokens` to `input_tokens`, cached and
reasoning details moved into `input_tokens_details` and `output_tokens_details`), and request fields
(`instructions`, `metadata`, `temperature`, `top_p`, `max_output_tokens`, `tools`, `user`, `parallel_tool_calls`)
are echoed. The proxy then rewrites `id` into a managed `resp_` id. Clients like the OpenAI Agents SDK and Codex
read every one of these fields, so each is pinned.
"""

from __future__ import annotations

import json
from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value, string_value
from integration._support.wire import Reply
from integration.providers.dec1_respapi_support import (
    COMPLETION_TOKENS,
    PROMPT_TOKENS,
    REPLY_TEXT,
    Call,
    anthropic,
    anthropic_message,
    chat_bridge,
    chat_json,
    deepseek,
    json_reply,
    output_items,
    output_text,
    peer,
    responses,
    served,
    tool_call,
    usage,
)
from pydantic import JsonValue

WEATHER_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "name": "get_weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
}


def _answering(**chat: JsonValue):  # noqa: ANN202  # local factory of a peer handler
    def respond(call: Call) -> Reply:
        return json_reply(chat_json(call.body, **chat))  # pyright: ignore[reportArgumentType]

    return respond


def test_a_text_answer_is_a_completed_response_with_one_message_item(gateway: Gateway) -> None:
    """The chat provider answers `REPLY_TEXT` with `finish_reason: stop`.

    The caller gets `object: response`, `status: completed`, `incomplete_details: null`, and exactly one
    `message` item with `role: assistant`, `status: completed`, an id starting `msg_`, and one `output_text`
    part holding `REPLY_TEXT` with `annotations: []`.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        body: Final = served(responses(gateway, model))
        assert (body["object"], body["status"], body.get("incomplete_details")) == ("response", "completed", None)
        messages: Final = output_items(body, "message")
        assert len(messages) == 1
        assert (messages[0]["role"], messages[0]["status"]) == ("assistant", "completed")
        assert string_value(messages[0]["id"]).startswith("msg_")
        assert messages[0]["content"] == [{"type": "output_text", "text": REPLY_TEXT, "annotations": []}]


def test_the_response_id_is_a_managed_resp_id_not_the_chat_completion_id(gateway: Gateway) -> None:
    """The chat provider answers with `id: chatcmpl-dec1fixed`.

    The bridge copies the chat id into the Responses object and the proxy rewrites it into a managed id, so the
    caller's `id` starts with `resp_` and never equals or contains `chatcmpl-dec1fixed` in clear text.
    `previous_response_id`, GET and DELETE all depend on that id carrying the deployment.
    """
    with peer(_answering(identity="chatcmpl-dec1fixed")) as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        identity: Final = string_value(served(responses(gateway, model))["id"])
        assert identity.startswith("resp_"), identity
        assert "chatcmpl-dec1fixed" not in identity


def test_created_at_is_the_chat_completion_created_time(gateway: Gateway) -> None:
    """The chat answer has `created: 1`. The Responses object's `created_at` is copied from it
    (`transformation.py:3628`), so it must be `1`. UNVERIFIED: assumes no later hook restamps `created_at`."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        assert served(responses(gateway, model))["created_at"] == 1


def test_usage_is_renamed_to_input_and_output_tokens(gateway: Gateway) -> None:
    """The chat usage is 11 prompt and 7 completion tokens. The caller's usage must be `input_tokens: 11`,
    `output_tokens: 7`, `total_tokens: 18`, with no `prompt_tokens` or `completion_tokens` keys."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        body_usage: Final = object_value(served(responses(gateway, model))["usage"])
        assert (body_usage["input_tokens"], body_usage["output_tokens"], body_usage["total_tokens"]) == (
            PROMPT_TOKENS,
            COMPLETION_TOKENS,
            PROMPT_TOKENS + COMPLETION_TOKENS,
        )
        assert "prompt_tokens" not in body_usage and "completion_tokens" not in body_usage


def test_cached_prompt_tokens_move_to_input_tokens_details(gateway: Gateway) -> None:
    """The chat usage carries `prompt_tokens_details: {"cached_tokens": 4}`.

    `transform_chat_completion_usage_to_responses_usage` copies it to `usage.input_tokens_details.cached_tokens
    == 4` (`transformation.py:4114`).
    """
    with (
        peer(_answering(reported_usage=usage(prompt_tokens_details={"cached_tokens": 4}))) as provider,
        gateway.scenario() as scenario,
    ):
        model: Final = chat_bridge(scenario, provider)
        details: Final = object_value(object_value(served(responses(gateway, model))["usage"])["input_tokens_details"])
        assert details["cached_tokens"] == 4


def test_reasoning_tokens_move_to_output_tokens_details(gateway: Gateway) -> None:
    """The chat usage carries `completion_tokens_details: {"reasoning_tokens": 3}`. The caller's
    `usage.output_tokens_details.reasoning_tokens` must be `3`."""
    with (
        peer(_answering(reported_usage=usage(completion_tokens_details={"reasoning_tokens": 3}))) as provider,
        gateway.scenario() as scenario,
    ):
        model: Final = chat_bridge(scenario, provider)
        details: Final = object_value(object_value(served(responses(gateway, model))["usage"])["output_tokens_details"])
        assert details["reasoning_tokens"] == 3


@pytest.mark.parametrize(
    ("finish_reason", "reason"),
    [("length", "max_output_tokens"), ("content_filter", "content_filter")],
)
def test_a_truncating_finish_reason_is_an_incomplete_response_with_its_reason(
    gateway: Gateway, finish_reason: str, reason: str
) -> None:
    """The chat answer ends with `finish_reason` `length` or `content_filter`.

    `_map_chat_completion_finish_reason_to_responses_status` maps both to `incomplete` and
    `_INCOMPLETE_REASON_BY_FINISH_REASON` (`transformation.py:217`) names the reason, so the body must hold
    `status: incomplete` and `incomplete_details == {"reason": <reason>}`; the message item is `incomplete` too.
    """
    with peer(_answering(finish_reason=finish_reason)) as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        body: Final = served(responses(gateway, model))
        assert body["status"] == "incomplete"
        assert body["incomplete_details"] == {"reason": reason}
        assert output_items(body, "message")[0]["status"] == "incomplete"


def test_a_tool_calls_finish_is_completed_with_a_function_call_item(gateway: Gateway) -> None:
    """The chat answer has one tool call `call_dec1out` to `get_weather` with `{"city":"Oslo"}` and
    `finish_reason: tool_calls`.

    The body must be `completed` and hold one `function_call` item with `call_id: call_dec1out`, `id:
    fc_call_dec1out` (`openai_shaped_tool_call_item_id`), `name: get_weather`, `status: completed`, and no
    message item, because the chat content was null.
    """
    respond: Final = _answering(
        content=None,
        finish_reason="tool_calls",
        message={"tool_calls": [tool_call("call_dec1out", "get_weather", '{"city":"Oslo"}')]},
    )
    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        body: Final = served(responses(gateway, model, tools=[WEATHER_TOOL]))
        assert body["status"] == "completed"
        calls: Final = output_items(body, "function_call")
        assert [(item["call_id"], item["id"], item["name"], item["status"]) for item in calls] == [
            ("call_dec1out", "fc_call_dec1out", "get_weather", "completed")
        ]
        assert output_items(body, "message") == []


def test_a_tool_call_id_that_already_starts_with_fc_keeps_its_id(gateway: Gateway) -> None:
    """The chat tool call id is `fc_dec1already`. `openai_shaped_tool_call_item_id` does not double the prefix
    (`custom_tools.py:43`), so the item's `id` and `call_id` are both `fc_dec1already`."""
    respond: Final = _answering(
        content=None,
        finish_reason="tool_calls",
        message={"tool_calls": [tool_call("fc_dec1already", "get_weather", "{}")]},
    )
    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        call: Final = output_items(served(responses(gateway, model, tools=[WEATHER_TOOL])), "function_call")[0]
        assert (call["id"], call["call_id"]) == ("fc_dec1already", "fc_dec1already")


def test_text_and_two_tool_calls_come_back_message_first_then_calls_in_order(gateway: Gateway) -> None:
    """The chat answer has content `checking` and two tool calls `call_dec1x`, `call_dec1y`.

    The output order is message items before function calls (`_transform_chat_completion_choices_to_responses_
    output`), so `output` types must read `message, function_call, function_call` with the two call ids in the
    provider's order.
    """
    respond: Final = _answering(
        content="checking",
        finish_reason="tool_calls",
        message={
            "tool_calls": [
                tool_call("call_dec1x", "get_weather", '{"city":"Oslo"}'),
                tool_call("call_dec1y", "get_weather", '{"city":"Rome"}'),
            ]
        },
    )
    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        body: Final = served(responses(gateway, model, tools=[WEATHER_TOOL]))
        kinds: Final = [object_value(item)["type"] for item in list_value(body["output"])]
        assert kinds == ["message", "function_call", "function_call"]
        assert [item["call_id"] for item in output_items(body, "function_call")] == ["call_dec1x", "call_dec1y"]


def test_reasoning_content_comes_back_as_a_reasoning_item_before_the_message(gateway: Gateway) -> None:
    """A DeepSeek chat answer has `reasoning_content: "dec1 thought"` and content `REPLY_TEXT`.

    `_extract_reasoning_output_items` puts a `reasoning` item first, with an id starting `rs_` and the
    reasoning text as its `output_text` content; the message item follows. The reasoning text must not appear
    in the message text.
    """
    with peer(_answering(message={"reasoning_content": "dec1 thought"})) as provider, gateway.scenario() as scenario:
        model: Final = deepseek(scenario, provider)
        body: Final = served(responses(gateway, model))
        output: Final = [object_value(item) for item in list_value(body["output"])]
        assert [item["type"] for item in output] == ["reasoning", "message"]
        assert string_value(output[0]["id"]).startswith("rs_")
        assert [object_value(part)["text"] for part in list_value(output[0]["content"])] == ["dec1 thought"]
        assert output_text(body) == REPLY_TEXT


def test_claude_thinking_comes_back_as_a_reasoning_item_with_encrypted_content(gateway: Gateway) -> None:
    """Anthropic answers a `thinking` block (`thinking: "dec1 claude thought"`, `signature: "sig-dec1"`) and a
    text block.

    The reasoning item must carry the thinking text in its content and a non empty `encrypted_content` (the
    thinking blocks encoded by `_encode_thinking_blocks`), so a later turn can replay the signed block.
    """

    def respond(call: Call) -> Reply:
        return json_reply(
            anthropic_message(
                call.body,
                content=[
                    {"type": "thinking", "thinking": "dec1 claude thought", "signature": "sig-dec1"},
                    {"type": "text", "text": REPLY_TEXT},
                ],
            )
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = anthropic(scenario, provider)
        body: Final = served(responses(gateway, model, reasoning={"effort": "low"}))
        reasoning: Final = output_items(body, "reasoning")
        assert len(reasoning) == 1
        assert "dec1 claude thought" in json.dumps(reasoning[0]["content"])
        assert isinstance(reasoning[0].get("encrypted_content"), str) and reasoning[0]["encrypted_content"]
        assert output_text(body) == REPLY_TEXT


def test_a_chat_refusal_comes_back_as_a_refusal_part(gateway: Gateway) -> None:
    """The chat answer has `content: null` and `refusal: "dec1 cannot help"`.

    `_message_output_content` builds a `{"type": "refusal", "refusal": "dec1 cannot help"}` part and no
    `output_text` part (`transformation.py:3953`).
    """
    with (
        peer(_answering(content=None, message={"refusal": "dec1 cannot help"})) as provider,
        gateway.scenario() as scenario,
    ):
        model: Final = chat_bridge(scenario, provider)
        message: Final = output_items(served(responses(gateway, model)), "message")[0]
        assert message["content"] == [{"type": "refusal", "refusal": "dec1 cannot help"}]


def test_url_citation_annotations_are_carried_onto_the_output_text(gateway: Gateway) -> None:
    """The chat message carries `annotations: [{"type": "url_citation", "url_citation": {"url":
    "https://example.invalid/a", "title": "A", "start_index": 0, "end_index": 4}}]`.

    `transform_chat_completion_annotations_to_response_output_annotations` flattens it to `{"type":
    "url_citation", "url": ..., "title": "A", "start_index": 0, "end_index": 4}` on the `output_text` part.
    UNVERIFIED: exact flattened key set; read `transformation.py:4029`.
    """
    annotation: Final[dict[str, JsonValue]] = {
        "type": "url_citation",
        "url_citation": {"url": "https://example.invalid/a", "title": "A", "start_index": 0, "end_index": 4},
    }
    with peer(_answering(message={"annotations": [annotation]})) as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        part: Final = object_value(
            list_value(output_items(served(responses(gateway, model)), "message")[0]["content"])[0]
        )
        annotations: Final = [object_value(item) for item in list_value(part["annotations"])]
        assert annotations[0]["type"] == "url_citation"
        assert annotations[0]["url"] == "https://example.invalid/a"


def test_request_fields_are_echoed_on_the_response(gateway: Gateway) -> None:
    """The caller sends `instructions`, `metadata: {"dec1": "yes"}`, `temperature: 0.2`, `top_p: 0.9`,
    `max_output_tokens: 64`, `user: dec1-user` and the `get_weather` tool.

    `_echoable_request_params` (`transformation.py:251`) echoes each, so the body must hold them unchanged and
    `tools[0].name == "get_weather"`.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        body: Final = served(
            responses(
                gateway,
                model,
                instructions="Be brief.",
                metadata={"dec1": "yes"},
                temperature=0.2,
                top_p=0.9,
                max_output_tokens=64,
                user="dec1-user",
                tools=[WEATHER_TOOL],
            )
        )
        assert (body["instructions"], body["metadata"], body["temperature"], body["top_p"]) == (
            "Be brief.",
            {"dec1": "yes"},
            0.2,
            0.9,
        )
        assert (body["max_output_tokens"], body["user"]) == (64, "dec1-user")
        assert object_value(list_value(body["tools"])[0])["name"] == "get_weather"


def test_unset_echo_fields_default_to_empty_values(gateway: Gateway) -> None:
    """A bare call with only `input`. The body must hold `metadata: {}`, `tools: []`, `parallel_tool_calls:
    false` and `instructions: null`, the defaults the transform fills (`transformation.py:3630`)."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        body: Final = served(responses(gateway, model))
        assert (body["metadata"], body["tools"], body["parallel_tool_calls"], body.get("instructions")) == (
            {},
            [],
            False,
            None,
        )


def test_an_anthropic_max_tokens_stop_is_an_incomplete_response(gateway: Gateway) -> None:
    """Anthropic answers with `stop_reason: max_tokens`.

    The chat transform maps it to `finish_reason: length`, so the Responses body must hold `status:
    incomplete` and `incomplete_details: {"reason": "max_output_tokens"}`.
    """

    def respond(call: Call) -> Reply:
        return json_reply(anthropic_message(call.body, stop_reason="max_tokens"))

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = anthropic(scenario, provider)
        body: Final = served(responses(gateway, model, max_output_tokens=20))
        assert body["status"] == "incomplete"
        assert body["incomplete_details"] == {"reason": "max_output_tokens"}


def test_an_anthropic_tool_use_comes_back_as_a_function_call_with_toolu_call_id(gateway: Gateway) -> None:
    """Anthropic answers one `tool_use` block (`id: toolu_dec1out`, `name: get_weather`, `input: {"city":
    "Oslo"}`).

    The caller gets one `function_call` item with `call_id: toolu_dec1out`, `id: fc_toolu_dec1out`, and
    `arguments` that parse to `{"city": "Oslo"}`; `status` is `completed`.
    """

    def respond(call: Call) -> Reply:
        return json_reply(
            anthropic_message(
                call.body,
                content=[{"type": "tool_use", "id": "toolu_dec1out", "name": "get_weather", "input": {"city": "Oslo"}}],
                stop_reason="tool_use",
            )
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = anthropic(scenario, provider)
        body: Final = served(responses(gateway, model, tools=[WEATHER_TOOL]))
        assert body["status"] == "completed"
        call: Final = output_items(body, "function_call")[0]
        assert (call["call_id"], call["id"]) == ("toolu_dec1out", "fc_toolu_dec1out")
        assert json.loads(string_value(call["arguments"])) == {"city": "Oslo"}
