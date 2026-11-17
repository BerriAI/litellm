"""Contract: how an Anthropic Messages response becomes the OpenAI chat completion the caller receives.

`AnthropicConfig.transform_parsed_response` (`litellm/llms/anthropic/chat/transformation.py`) turns the provider's
content blocks into one OpenAI choice: text blocks concatenate into `message.content`, `tool_use` blocks become
`message.tool_calls` with JSON string arguments, `thinking` and `redacted_thinking` blocks become
`message.thinking_blocks` (signatures kept) and `message.reasoning_content`, citations ride on
`message.provider_specific_fields`, and `stop_reason` is mapped by `map_finish_reason`. Usage folds Anthropic's
separate cache counters into OpenAI's `prompt_tokens` (`calculate_usage`). Structured output (`response_format`)
takes Anthropic's native `output_format` when the model supports it and the schema is closed, and otherwise a
forced `json_tool_call` tool whose input the proxy returns as the message text.

The caller-visible shapes here are what OpenAI SDK clients parse, so a rewrite that changes any of them breaks
clients that never talk to Anthropic directly. Deployments are priced at 0.001 per input token, 0.002 per output
token, 0.0001 per cache read token and 0.003 per cache creation token.
"""

from __future__ import annotations

import json
from typing import Final

import pytest
from integration._support.client import JSON_OBJECT, Gateway, list_value, object_value, string_value
from integration._support.wire import Reply
from integration.providers.dec1_anthropic_support import (
    CALL_COST,
    REPLY_TEXT,
    SIGNATURE,
    Call,
    answering,
    betas,
    chat,
    deployment,
    first_choice,
    first_message,
    json_reply,
    message_json,
    peer,
    served,
    spend_row_by_group,
    tool,
    usage_block,
)
from pydantic import JsonValue

_SCHEMA_CLOSED: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
    "required": ["city", "country"],
    "additionalProperties": False,
}
# Open means `additionalProperties: true` (or a dict, `patternProperties`, or an object with no `properties`):
# an object that lists `properties` and leaves `additionalProperties` out counts as closed
# (`_is_open_object`, litellm/litellm_core_utils/prompt_templates/json_schema_utils.py:355-359 on main).
_SCHEMA_OPEN: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": True,
}
_PLACE: Final[dict[str, JsonValue]] = {"city": "Paris", "country": "France"}


def _json_schema_format(schema: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {"type": "json_schema", "json_schema": {"name": "place", "schema": schema}}


def test_a_text_reply_becomes_the_assistant_message_content(gateway: Gateway) -> None:
    """The provider answers one text block `dec1 anthropic reply` with `stop_reason: end_turn`.

    The caller must get `object: chat.completion`, one choice at index 0 whose message has `role: assistant`,
    `content: "dec1 anthropic reply"`, no tool calls, and `finish_reason: stop`.
    """
    with peer() as provider, gateway.scenario() as scenario:
        body: Final = served(chat(gateway, deployment(scenario, provider)))
        assert body["object"] == "chat.completion"
        choice: Final = first_choice(body)
        assert choice["index"] == 0
        assert choice["finish_reason"] == "stop"
        message: Final = first_message(body)
        assert message["role"] == "assistant"
        assert message["content"] == REPLY_TEXT
        assert not message.get("tool_calls")


def test_several_text_blocks_are_concatenated_in_order(gateway: Gateway) -> None:
    """The provider answers three text blocks `alpha `, `beta `, `gamma`. The caller's `message.content` must be
    `alpha beta gamma`, the blocks joined without separators in provider order.
    """
    content: Final[list[JsonValue]] = [
        {"type": "text", "text": "alpha "},
        {"type": "text", "text": "beta "},
        {"type": "text", "text": "gamma"},
    ]
    with peer(answering(content)) as provider, gateway.scenario() as scenario:
        assert first_message(served(chat(gateway, deployment(scenario, provider))))["content"] == "alpha beta gamma"


@pytest.mark.parametrize(
    ("stop_reason", "finish_reason"),
    [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("max_tokens", "length"),
        ("model_context_window_exceeded", "length"),
        ("refusal", "content_filter"),
    ],
)
def test_the_anthropic_stop_reason_is_mapped_to_an_openai_finish_reason(
    gateway: Gateway, stop_reason: str, finish_reason: str
) -> None:
    """The provider answers a text block with each `stop_reason`. The caller's `finish_reason` must be the OpenAI
    value from `_FINISH_REASON_MAP` (`litellm/litellm_core_utils/core_helpers.py:432`): end_turn and stop_sequence to
    stop, max_tokens and model_context_window_exceeded to length, refusal to content_filter.
    """
    with (
        peer(answering([{"type": "text", "text": REPLY_TEXT}], stop_reason=stop_reason)) as provider,
        gateway.scenario() as scenario,
    ):
        assert first_choice(served(chat(gateway, deployment(scenario, provider))))["finish_reason"] == finish_reason


def test_usage_reports_input_output_and_total_tokens(gateway: Gateway) -> None:
    """The provider reports `input_tokens: 11`, `output_tokens: 7` and no cache counters.

    The caller's usage must be `prompt_tokens: 11`, `completion_tokens: 7`, `total_tokens: 18`, and
    `prompt_tokens_details.cached_tokens` must be 0.
    """
    with peer() as provider, gateway.scenario() as scenario:
        usage: Final = object_value(served(chat(gateway, deployment(scenario, provider)))["usage"])
        assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (11, 7, 18)
        assert object_value(usage["prompt_tokens_details"]).get("cached_tokens") in (0, None)


def test_cache_read_and_creation_tokens_are_added_to_prompt_tokens(gateway: Gateway) -> None:
    """The provider reports `input_tokens: 5`, `cache_read_input_tokens: 7`, `cache_creation_input_tokens: 3`,
    `output_tokens: 7`.

    Anthropic counts uncached input separately; OpenAI's `prompt_tokens` is the sum, so the caller must see
    `prompt_tokens: 15`, `total_tokens: 22`, `prompt_tokens_details.cached_tokens: 7`, and the Anthropic counters
    `cache_read_input_tokens: 7` and `cache_creation_input_tokens: 3` on usage (`calculate_usage`).
    """
    usage_in: Final = usage_block(5, 7, cache_read_input_tokens=7, cache_creation_input_tokens=3)
    with (
        peer(answering([{"type": "text", "text": REPLY_TEXT}], usage=usage_in)) as provider,
        gateway.scenario() as scenario,
    ):
        usage: Final = object_value(served(chat(gateway, deployment(scenario, provider)))["usage"])
        assert usage["prompt_tokens"] == 15
        assert usage["total_tokens"] == 22
        assert object_value(usage["prompt_tokens_details"])["cached_tokens"] == 7
        assert usage["cache_read_input_tokens"] == 7
        assert usage["cache_creation_input_tokens"] == 3


def test_the_response_cost_header_prices_plain_input_and_output(gateway: Gateway) -> None:
    """The provider reports 11 input and 7 output tokens; the deployment is priced at 0.001 and 0.002.

    The caller's `x-litellm-response-cost` header must be 11 * 0.001 + 7 * 0.002 = 0.025.
    """
    with peer() as provider, gateway.scenario() as scenario:
        response: Final = chat(gateway, deployment(scenario, provider))
        served(response)
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(CALL_COST)


def test_the_response_cost_header_prices_cache_reads_and_writes_at_their_own_rates(gateway: Gateway) -> None:
    """The provider reports 5 uncached input, 7 cache read, 3 cache creation and 7 output tokens.

    The cost must be 5 * 0.001 + 7 * 0.0001 + 3 * 0.003 + 7 * 0.002 = 0.0287: cache reads are cheaper and cache
    writes dearer than plain input. UNVERIFIED for the chat route: the same formula is pinned on /v1/messages by
    `test_x_litellm_response_cost_prices_cache_read_and_cache_creation_tokens`.
    """
    usage_in: Final = usage_block(5, 7, cache_read_input_tokens=7, cache_creation_input_tokens=3)
    with (
        peer(answering([{"type": "text", "text": REPLY_TEXT}], usage=usage_in)) as provider,
        gateway.scenario() as scenario,
    ):
        response: Final = chat(gateway, deployment(scenario, provider))
        served(response)
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(0.0287)


def test_a_successful_call_writes_a_spend_row_with_the_priced_cost(gateway: Gateway) -> None:
    """A plain call priced as above. One `LiteLLM_SpendLogs` row for the model group must land with
    `call_type: acompletion`, `custom_llm_provider: anthropic`, `prompt_tokens: 11`, `completion_tokens: 7`,
    `total_tokens: 18`, spend 0.025 and `status: success`.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider)
        served(chat(gateway, model))
        row: Final = spend_row_by_group(gateway, model)
        assert row["call_type"] == "acompletion"
        assert row["custom_llm_provider"] == "anthropic"
        assert (row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]) == (11, 7, 18)
        assert float(str(row["spend"])) == pytest.approx(CALL_COST)
        assert row["status"] == "success"


def test_a_tool_use_block_becomes_an_openai_tool_call_with_json_string_arguments(gateway: Gateway) -> None:
    """The provider answers a `tool_use` block `toolu_dec1a` calling `get_weather` with input `{"city": "Paris"}`
    and `stop_reason: tool_use`.

    The caller must get `finish_reason: tool_calls` and one tool call with `id: toolu_dec1a`, `type: function`,
    `function.name: get_weather`, and `function.arguments` a JSON string that decodes to `{"city": "Paris"}`.
    """
    block: Final[dict[str, JsonValue]] = {
        "type": "tool_use",
        "id": "toolu_dec1a",
        "name": "get_weather",
        "input": {"city": "Paris"},
    }
    with peer(answering([block], stop_reason="tool_use")) as provider, gateway.scenario() as scenario:
        body: Final = served(
            chat(gateway, deployment(scenario, provider), tools=[tool("get_weather", city={"type": "string"})])
        )
        assert first_choice(body)["finish_reason"] == "tool_calls"
        calls: Final = [object_value(c) for c in list_value(first_message(body)["tool_calls"])]
        assert len(calls) == 1
        assert (calls[0]["id"], calls[0]["type"]) == ("toolu_dec1a", "function")
        function: Final = object_value(calls[0]["function"])
        assert function["name"] == "get_weather"
        assert json.loads(string_value(function["arguments"])) == {"city": "Paris"}


def test_parallel_tool_use_blocks_become_tool_calls_in_provider_order(gateway: Gateway) -> None:
    """The provider answers a text block then two `tool_use` blocks (`toolu_dec1a` get_weather, `toolu_dec1b`
    get_time).

    The caller must get the text as `content` and two tool calls with ids `toolu_dec1a` then `toolu_dec1b`, each
    with its own name and arguments, so an agent can run both before replying.
    """
    content: Final[list[JsonValue]] = [
        {"type": "text", "text": "checking"},
        {"type": "tool_use", "id": "toolu_dec1a", "name": "get_weather", "input": {"city": "Paris"}},
        {"type": "tool_use", "id": "toolu_dec1b", "name": "get_time", "input": {"city": "Rome"}},
    ]
    tools: Final[list[JsonValue]] = [
        tool("get_weather", city={"type": "string"}),
        tool("get_time", city={"type": "string"}),
    ]
    with peer(answering(content, stop_reason="tool_use")) as provider, gateway.scenario() as scenario:
        message: Final = first_message(served(chat(gateway, deployment(scenario, provider), tools=tools)))
        assert message["content"] == "checking"
        calls: Final = [object_value(c) for c in list_value(message["tool_calls"])]
        assert [c["id"] for c in calls] == ["toolu_dec1a", "toolu_dec1b"]
        assert [object_value(c["function"])["name"] for c in calls] == ["get_weather", "get_time"]
        assert [json.loads(string_value(object_value(c["function"])["arguments"])) for c in calls] == [
            {"city": "Paris"},
            {"city": "Rome"},
        ]


def test_a_sanitized_tool_name_is_mapped_back_to_the_caller_name(gateway: Gateway) -> None:
    """The caller's tool is `weather.lookup/v1`, sent upstream as `weather_lookup_v1`. The provider answers a
    `tool_use` for `weather_lookup_v1`.

    The caller's tool call must name `weather.lookup/v1`, the name it registered, so an OpenAI-style dispatcher can
    match it (`transform_parsed_response` applies the per-request reverse map).
    """
    block: Final[dict[str, JsonValue]] = {
        "type": "tool_use",
        "id": "toolu_dec1a",
        "name": "weather_lookup_v1",
        "input": {},
    }
    with peer(answering([block], stop_reason="tool_use")) as provider, gateway.scenario() as scenario:
        message: Final = first_message(
            served(
                chat(
                    gateway, deployment(scenario, provider), tools=[tool("weather.lookup/v1", city={"type": "string"})]
                )
            )
        )
        calls: Final = [object_value(c) for c in list_value(message["tool_calls"])]
        assert object_value(calls[0]["function"])["name"] == "weather.lookup/v1"


def test_thinking_blocks_are_returned_with_signatures_and_as_reasoning_content(gateway: Gateway) -> None:
    """The provider answers a `thinking` block (text `step one`, a signature), a `redacted_thinking` block and a
    text block.

    The caller's message must carry `reasoning_content: "step one"`, `thinking_blocks` equal to the two provider
    blocks unchanged (signature and redacted data intact, so the caller can replay them), and `content` the text.
    """
    thinking: Final[dict[str, JsonValue]] = {"type": "thinking", "thinking": "step one", "signature": SIGNATURE}
    redacted: Final[dict[str, JsonValue]] = {"type": "redacted_thinking", "data": "EmwKAhgBEgy3dec1redacted"}
    content: Final[list[JsonValue]] = [thinking, redacted, {"type": "text", "text": REPLY_TEXT}]
    with peer(answering(content)) as provider, gateway.scenario() as scenario:
        message: Final = first_message(served(chat(gateway, deployment(scenario, provider))))
        assert message["reasoning_content"] == "step one"
        assert message["thinking_blocks"] == [thinking, redacted]
        assert message["content"] == REPLY_TEXT


def test_text_citations_are_returned_in_provider_specific_fields(gateway: Gateway) -> None:
    """The provider answers a text block `Paris is in France.` with one `char_location` citation.

    The caller's `message.provider_specific_fields.citations` must be a list holding one list (per cited block)
    whose one entry is the citation plus `supported_text: "Paris is in France."` (`extract_response_content`).
    """
    citation: Final[dict[str, JsonValue]] = {
        "type": "char_location",
        "cited_text": "Paris is the capital of France.",
        "document_index": 0,
        "document_title": "dec1 atlas",
        "start_char_index": 0,
        "end_char_index": 31,
    }
    content: Final[list[JsonValue]] = [{"type": "text", "text": "Paris is in France.", "citations": [citation]}]
    with peer(answering(content)) as provider, gateway.scenario() as scenario:
        message: Final = first_message(served(chat(gateway, deployment(scenario, provider))))
        fields: Final = object_value(message["provider_specific_fields"])
        assert fields["citations"] == [[{**citation, "supported_text": "Paris is in France."}]]
        assert message["content"] == "Paris is in France."


def test_a_web_search_turn_keeps_its_results_and_reports_search_requests_in_usage(gateway: Gateway) -> None:
    """The provider answers a `server_tool_use` web_search call, its `web_search_tool_result`, and a text answer,
    with `usage.server_tool_use.web_search_requests: 1`.

    The caller must get the text as `content`, no client tool call for the server tool (it already ran), the
    result under `provider_specific_fields.web_search_results`, and `usage.server_tool_use.web_search_requests: 1`.
    """
    content: Final[list[JsonValue]] = [
        {"type": "server_tool_use", "id": "srvtoolu_dec1", "name": "web_search", "input": {"query": "dec1 weather"}},
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_dec1",
            "content": [
                {"type": "web_search_result", "url": "https://dec1.invalid/w", "title": "w", "encrypted_content": "x"}
            ],
        },
        {"type": "text", "text": REPLY_TEXT},
    ]
    usage_in: Final = usage_block(server_tool_use={"web_search_requests": 1})
    with peer(answering(content, usage=usage_in)) as provider, gateway.scenario() as scenario:
        body: Final = served(
            chat(gateway, deployment(scenario, provider), web_search_options={"search_context_size": "low"})
        )
        message: Final = first_message(body)
        assert message["content"] == REPLY_TEXT
        assert not message.get("tool_calls")
        results: Final = list_value(object_value(message["provider_specific_fields"])["web_search_results"])
        assert object_value(results[0])["tool_use_id"] == "srvtoolu_dec1"
        assert object_value(object_value(body["usage"])["server_tool_use"])["web_search_requests"] == 1


def test_a_closed_json_schema_uses_native_output_format_and_returns_the_json_text(gateway: Gateway) -> None:
    """The caller asks `response_format: json_schema` with a closed schema on `claude-sonnet-4-5-20250929`
    (`supports_native_structured_output: true`). The provider answers the JSON as text.

    Upstream, the body must carry `output_format: {"type": "json_schema", "schema": ...}` with the caller's
    properties, no `json_tool_call` tool, and the `structured-outputs-2025-11-13` beta. The caller's `content` is
    the provider's JSON text and `finish_reason` is stop.
    """
    text: Final = json.dumps(_PLACE)
    with peer(answering([{"type": "text", "text": text}])) as provider, gateway.scenario() as scenario:
        body: Final = served(
            chat(gateway, deployment(scenario, provider), response_format=_json_schema_format(_SCHEMA_CLOSED))
        )
        call: Final = provider.only()
        output_format: Final = object_value(call.body["output_format"])
        assert output_format["type"] == "json_schema"
        assert object_value(output_format["schema"])["properties"] == _SCHEMA_CLOSED["properties"]
        assert "tools" not in call.body
        assert "structured-outputs-2025-11-13" in betas(call)
        assert json.loads(string_value(first_message(body)["content"])) == _PLACE
        assert first_choice(body)["finish_reason"] == "stop"


def _json_tool_respond(call: Call) -> Reply:
    if call.method != "POST":
        return json_reply({})
    block: Final[dict[str, JsonValue]] = {
        "type": "tool_use",
        "id": "toolu_dec1json",
        "name": "json_tool_call",
        "input": {"city": "Paris"},
    }
    return json_reply(message_json(call.body, content=[block], stop_reason="tool_use"))


def test_an_open_json_schema_is_sent_as_a_forced_json_tool_call(gateway: Gateway) -> None:
    """The caller asks `response_format: json_schema` with a schema whose object is open
    (`additionalProperties: true`) and sends no tools or thinking.

    An open object takes the tool path (`uses_native_structured_output_for_schema`): the upstream body must carry
    one tool named `json_tool_call` whose `input_schema.properties` are the caller's, `tool_choice: {"name":
    "json_tool_call", "type": "tool"}`, and no `output_format`.
    """
    with peer(_json_tool_respond) as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), response_format=_json_schema_format(_SCHEMA_OPEN)))
        body: Final = provider.only().body
        tools: Final = [object_value(t) for t in list_value(body["tools"])]
        assert [t["name"] for t in tools] == ["json_tool_call"]
        assert object_value(tools[0]["input_schema"])["properties"] == _SCHEMA_OPEN["properties"]
        assert body["tool_choice"] == {"name": "json_tool_call", "type": "tool"}
        assert "output_format" not in body


def test_a_json_tool_call_answer_is_returned_as_message_content_not_a_tool_call(gateway: Gateway) -> None:
    """Same open-schema request; the provider answers a `tool_use` of `json_tool_call` with input `{"city":
    "Paris"}` and `stop_reason: tool_use`.

    The caller asked for JSON, not tools, so it must get `content` that decodes to `{"city": "Paris"}`, no
    `tool_calls`, and `finish_reason: stop` (`_resolve_json_mode_non_streaming` rewrites the stop reason).
    """
    with peer(_json_tool_respond) as provider, gateway.scenario() as scenario:
        body: Final = served(
            chat(gateway, deployment(scenario, provider), response_format=_json_schema_format(_SCHEMA_OPEN))
        )
        message: Final = first_message(body)
        assert JSON_OBJECT.validate_json(string_value(message["content"])) == {"city": "Paris"}
        assert not message.get("tool_calls")
        assert first_choice(body)["finish_reason"] == "stop"


def test_response_format_json_object_without_a_schema_sends_no_tool_and_no_output_format(gateway: Gateway) -> None:
    """The caller asks `response_format: {"type": "json_object"}` with no schema.

    There is no schema to enforce, so the upstream body must carry neither `output_format` nor a `json_tool_call`
    tool (`map_response_format_to_anthropic_tool` returns None without a schema). UNVERIFIED: the native branch
    with a None schema is also expected to add nothing.
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), response_format={"type": "json_object"}))
        body: Final = provider.only().body
        assert "output_format" not in body
        assert "json_tool_call" not in json.dumps(body.get("tools", []))


def test_a_json_tool_call_next_to_a_user_tool_call_keeps_the_user_tool_and_moves_the_json_to_content(
    gateway: Gateway,
) -> None:
    """The caller sends a `get_weather` tool and an open-schema `response_format`; the provider answers both a
    `get_weather` tool_use and a `json_tool_call` tool_use.

    The caller must get exactly one tool call (`get_weather`) and the JSON payload `{"city": "Paris"}` as message
    text (`_resolve_json_mode_non_streaming` mixed case). With caller tools present the request itself takes the
    native `output_format` path; only the response handling is pinned here, which still runs because `json_mode`
    is set for every `response_format`.
    """
    content: Final[list[JsonValue]] = [
        {"type": "tool_use", "id": "toolu_dec1a", "name": "get_weather", "input": {"city": "Paris"}},
        {"type": "tool_use", "id": "toolu_dec1json", "name": "json_tool_call", "input": {"city": "Paris"}},
    ]
    with peer(answering(content, stop_reason="tool_use")) as provider, gateway.scenario() as scenario:
        message: Final = first_message(
            served(
                chat(
                    gateway,
                    deployment(scenario, provider),
                    tools=[tool("get_weather", city={"type": "string"})],
                    response_format=_json_schema_format(_SCHEMA_OPEN),
                )
            )
        )
        calls: Final = [object_value(c) for c in list_value(message["tool_calls"])]
        assert [object_value(c["function"])["name"] for c in calls] == ["get_weather"]
        assert JSON_OBJECT.validate_json(string_value(message["content"])) == {"city": "Paris"}
