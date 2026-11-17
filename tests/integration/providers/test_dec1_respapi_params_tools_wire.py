"""How `/v1/responses` request parameters and tools reach a chat provider through the bridge.

`transform_responses_api_request_to_chat_completion_request` (`litellm/responses/litellm_completion_transformation/
transformation.py:1303`) maps the Responses parameters onto chat parameters: `max_output_tokens` becomes
`max_tokens`, `temperature`, `top_p`, `user`, `service_tier` and `parallel_tool_calls` keep their names,
`text.format` becomes `response_format`, `reasoning.effort` becomes `reasoning_effort` (and `reasoning.summary`
rides along only when the target can read it), `tool_choice` is normalised to the chat shape, flat Responses
function tools are nested under `function`, a `web_search` tool becomes `web_search_options`, hosted tools with no
chat form (`image_generation`, `computer_use`, `local_shell`, `shell`) are dropped, and a `file_search` tool is
emulated with a `litellm_file_search` function tool. Responses only fields the bridge cannot translate (for
example `max_tool_calls`) are refused with 400 unless `drop_params` is on (`responses/main.py:688`), while
`store`, `truncation`, `include` and `background` never reach the chat wire. Streamed bridged calls always ask the
chat provider for `stream_options.include_usage` so the final `response.completed` can carry usage.
"""

from __future__ import annotations

import json
from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value
from integration._support.wire import Reply
from integration.providers.dec1_respapi_support import (
    REASONING_CHAT_BRIDGE,
    REPLY_TEXT,
    Call,
    anthropic,
    chat_bridge,
    chat_json,
    chat_messages,
    error_message,
    json_reply,
    output_items,
    peer,
    responses,
    served,
    stream_events,
    tool_call,
)
from pydantic import JsonValue

WEATHER_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "name": "get_weather",
    "description": "Look up the weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}
SCHEMA: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def test_max_output_tokens_becomes_max_tokens(gateway: Gateway) -> None:
    """`max_output_tokens: 48` on the OpenAI chat carrier.

    The chat body must carry `max_tokens: 48` and no `max_output_tokens` key (`transformation.py:1424`).
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        served(responses(gateway, model, max_output_tokens=48))
        body: Final = provider.only().body
        assert body["max_tokens"] == 48
        assert "max_output_tokens" not in body


def test_max_output_tokens_becomes_the_anthropic_max_tokens(gateway: Gateway) -> None:
    """`max_output_tokens: 300` on an Anthropic deployment.

    The bridge sends `max_tokens: 300` into the Anthropic chat transform, so the `/v1/messages` body must carry
    `max_tokens: 300` instead of the model default.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = anthropic(scenario, provider)
        served(responses(gateway, model, max_output_tokens=300))
        assert provider.only().body["max_tokens"] == 300


@pytest.mark.parametrize(
    ("name", "value"),
    [("temperature", 0.3), ("top_p", 0.5), ("user", "dec1-end-user"), ("service_tier", "flex")],
)
def test_sampling_and_identity_params_keep_their_chat_names(gateway: Gateway, name: str, value: JsonValue) -> None:
    """One of `temperature`, `top_p`, `user`, `service_tier` is set on the Responses call.

    The bridge copies each under the same name (`transformation.py:1416`), so the chat body must carry it with
    the value sent. UNVERIFIED for `service_tier`: assumes the gpt-4o-mini chat transform keeps it.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        served(responses(gateway, model, **{name: value}))
        assert provider.only().body[name] == value


@pytest.mark.parametrize(
    ("name", "value"),
    [("store", False), ("store", True), ("truncation", "auto"), ("include", ["reasoning.encrypted_content"])],
)
def test_responses_only_state_fields_never_reach_the_chat_wire(gateway: Gateway, name: str, value: JsonValue) -> None:
    """`store`, `truncation` or `include` is set on a bridged call.

    These are named `responses()` arguments that the bridge reads but never copies into the chat request
    (`transformation.py:1407`), so the call is served and the chat body has no such key. A port that forwards
    `store` would make OpenAI chat persist the completion.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        served(responses(gateway, model, **{name: value}))
        assert name not in provider.only().body


def test_store_false_is_echoed_on_the_bridged_response(gateway: Gateway) -> None:
    """`store: false` and `truncation: "auto"` on a bridged call.

    `_echoable_request_params` (`transformation.py:251`) echoes both, so the response body must hold `store:
    false` and `truncation: "auto"` even though neither went upstream.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        body: Final = served(responses(gateway, model, store=False, truncation="auto"))
        assert body["store"] is False
        assert body["truncation"] == "auto"


def test_a_flat_function_tool_is_nested_under_function_on_the_chat_wire(gateway: Gateway) -> None:
    """A flat Responses tool `get_weather` with `strict` unset.

    The chat body must carry `tools == [{"type": "function", "function": {"name": "get_weather", "description":
    ..., "parameters": ..., "strict": false}}]` (`_responses_tool_to_chat_form`, `transformation.py:3077`).
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        served(responses(gateway, model, tools=[WEATHER_TOOL]))
        assert provider.only().body["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Look up the weather for a city",
                    "parameters": WEATHER_TOOL["parameters"],
                    "strict": False,
                },
            }
        ]


def test_a_function_tool_without_parameters_gets_an_object_schema(gateway: Gateway) -> None:
    """A function tool `ping` with no `parameters` and no `description`.

    The bridge fills `parameters: {"type": "object"}` and `description: ""` so strict chat providers accept it.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        served(responses(gateway, model, tools=[{"type": "function", "name": "ping"}]))
        function: Final = object_value(object_value(list_value(provider.only().body["tools"])[0])["function"])
        assert function["parameters"] == {"type": "object"}
        assert function["description"] == ""


def test_parallel_tool_calls_and_tool_choice_are_dropped_when_there_are_no_tools(gateway: Gateway) -> None:
    """`tool_choice: "required"` and `parallel_tool_calls: false` with no `tools`.

    With no tools the bridge pops `tool_choice`, `tools` and `parallel_tool_calls` (`transformation.py:1433`),
    so the call is served and none of the three keys reaches the chat body.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        served(responses(gateway, model, tool_choice="required", parallel_tool_calls=False))
        body: Final = provider.only().body
        assert {"tool_choice", "tools", "parallel_tool_calls"}.isdisjoint(body), body


@pytest.mark.parametrize(
    ("sent", "expected"),
    [
        ("auto", "auto"),
        ("required", "required"),
        ("none", "none"),
        ({"type": "function", "name": "get_weather"}, {"type": "function", "function": {"name": "get_weather"}}),
        ({"type": "auto"}, "auto"),
        ({"type": "tool"}, "required"),
        ({"type": "any"}, "required"),
        ({"type": "function"}, "required"),
    ],
)
def test_tool_choice_is_normalised_to_the_chat_shape(gateway: Gateway, sent: JsonValue, expected: JsonValue) -> None:
    """`tool_choice` in one of the Responses or client dict shapes, with the `get_weather` tool.

    `_transform_tool_choice` (`transformation.py:1082`) keeps strings, nests a flat named function under
    `function`, maps `{"type": "auto"}` to `auto`, and maps `tool`, `any` and a nameless `function` to
    `required`. The chat body's `tool_choice` must equal the expected value.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        served(responses(gateway, model, tools=[WEATHER_TOOL], tool_choice=sent))
        assert provider.only().body["tool_choice"] == expected


def test_a_named_tool_choice_is_echoed_back_in_the_responses_shape(gateway: Gateway) -> None:
    """`tool_choice: {"type": "function", "name": "get_weather"}` on a bridged call.

    The response echoes the request's tool choice through `transform_tool_choice_for_responses_api_response`, so
    the body must hold `tool_choice == {"type": "function", "name": "get_weather"}`, the flat Responses form.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        body: Final = served(
            responses(gateway, model, tools=[WEATHER_TOOL], tool_choice={"type": "function", "name": "get_weather"})
        )
        assert body["tool_choice"] == {"type": "function", "name": "get_weather"}


def test_an_absent_tool_choice_is_echoed_as_auto(gateway: Gateway) -> None:
    """No `tool_choice` on a bridged call. `transform_tool_choice_for_responses_api_response(None)` returns
    `auto`, so the body must hold `tool_choice: "auto"`."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        assert served(responses(gateway, model))["tool_choice"] == "auto"


def test_a_json_schema_text_format_becomes_a_json_schema_response_format(gateway: Gateway) -> None:
    """`text: {"format": {"type": "json_schema", "name": "answer_schema", "schema": <schema>, "strict": true}}`.

    The chat body must carry `response_format == {"type": "json_schema", "json_schema": {"name":
    "answer_schema", "schema": <schema>, "strict": true}}` (`_transform_text_format_to_response_format`,
    `transformation.py:4154`).
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        served(
            responses(
                gateway,
                model,
                text={"format": {"type": "json_schema", "name": "answer_schema", "schema": SCHEMA, "strict": True}},
            )
        )
        assert provider.only().body["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "answer_schema", "schema": SCHEMA, "strict": True},
        }


def test_a_json_schema_text_format_without_name_or_strict_gets_the_defaults(gateway: Gateway) -> None:
    """`text.format` of type `json_schema` with only a `schema`.

    The bridge fills `name: "response_schema"` and `strict: false`, so the chat `response_format.json_schema`
    must be `{"name": "response_schema", "schema": <schema>, "strict": false}`.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        served(responses(gateway, model, text={"format": {"type": "json_schema", "schema": SCHEMA}}))
        assert object_value(provider.only().body["response_format"])["json_schema"] == {
            "name": "response_schema",
            "schema": SCHEMA,
            "strict": False,
        }


def test_a_json_object_text_format_becomes_a_json_object_response_format(gateway: Gateway) -> None:
    """`text: {"format": {"type": "json_object"}}`. The chat body must carry `response_format: {"type":
    "json_object"}`."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        served(responses(gateway, model, text={"format": {"type": "json_object"}}))
        assert provider.only().body["response_format"] == {"type": "json_object"}


def test_a_plain_text_format_sends_no_response_format(gateway: Gateway) -> None:
    """`text: {"format": {"type": "text"}}` maps to no `response_format` at all, so the chat body has no such
    key; the response still echoes `text.format.type == "text"`."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        body: Final = served(responses(gateway, model, text={"format": {"type": "text"}}))
        assert "response_format" not in provider.only().body
        assert object_value(object_value(body["text"])["format"])["type"] == "text"


def test_a_json_schema_text_format_reaches_anthropic_as_structured_output(gateway: Gateway) -> None:
    """The same `json_schema` text format on an Anthropic deployment.

    The Anthropic chat transform turns `response_format` into structured output for Claude Sonnet 4.5, so the
    `/v1/messages` body must carry the schema somewhere: either `output_format.schema` or a forced tool whose
    `input_schema` is the schema. UNVERIFIED which of the two applies to this model; see
    `litellm/llms/anthropic/chat/transformation.py` `_extract_json_schema_from_response_format` callers.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = anthropic(scenario, provider)
        served(responses(gateway, model, text={"format": {"type": "json_schema", "name": "s", "schema": SCHEMA}}))
        body: Final = provider.only().body
        tools: Final = [object_value(tool) for tool in list_value(body.get("tools") or [])]
        output_format: Final = body.get("output_format")
        assert (isinstance(output_format, dict) and output_format.get("schema") == SCHEMA) or any(
            tool.get("input_schema") == SCHEMA for tool in tools
        ), body


def test_reasoning_effort_reaches_a_chat_reasoning_model_as_reasoning_effort(gateway: Gateway) -> None:
    """`reasoning: {"effort": "low"}` on a bridged `openai/o3-mini` deployment.

    `_transform_reasoning_for_chat_completion` splits the object into the chat string param, so the chat body
    must carry `reasoning_effort: "low"` and no `reasoning` object.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider, model=REASONING_CHAT_BRIDGE)
        served(responses(gateway, model, reasoning={"effort": "low"}))
        body: Final = provider.only().body
        assert body["reasoning_effort"] == "low"
        assert "reasoning" not in body


def test_reasoning_summary_is_not_sent_to_a_chat_target_that_cannot_read_it(gateway: Gateway) -> None:
    """`reasoning: {"effort": "medium", "summary": "auto"}` on the bridged `o3-mini` chat deployment.

    `summary` has no chat equivalent and o3-mini on chat does not bridge back to Responses, so the bridge drops
    it (`transformation.py:1290`): the chat body must carry `reasoning_effort: "medium"` and neither
    `reasoning_summary` nor `summary`.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider, model=REASONING_CHAT_BRIDGE)
        served(responses(gateway, model, reasoning={"effort": "medium", "summary": "auto"}))
        body: Final = provider.only().body
        assert body["reasoning_effort"] == "medium"
        assert "reasoning_summary" not in body and "summary" not in body


def test_reasoning_effort_low_reaches_claude_as_a_1024_token_thinking_budget(gateway: Gateway) -> None:
    """`reasoning: {"effort": "low"}` on `anthropic/claude-sonnet-4-5-20250929` through the bridge.

    The chat `reasoning_effort: low` maps to `thinking: {"type": "enabled", "budget_tokens": 1024}`
    (`DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET`, `litellm/constants.py:371`), which the `/v1/messages` body
    must carry.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = anthropic(scenario, provider)
        served(responses(gateway, model, reasoning={"effort": "low"}))
        assert provider.only().body["thinking"] == {"type": "enabled", "budget_tokens": 1024}


def test_reasoning_is_echoed_on_the_bridged_response(gateway: Gateway) -> None:
    """`reasoning: {"effort": "low", "summary": "auto"}` on the o3-mini chat bridge. The response echoes the
    request's reasoning object, so the body's `reasoning.effort` must be `low` and `reasoning.summary` `auto`."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider, model=REASONING_CHAT_BRIDGE)
        reasoning: Final = object_value(
            served(responses(gateway, model, reasoning={"effort": "low", "summary": "auto"}))["reasoning"]
        )
        assert (reasoning["effort"], reasoning["summary"]) == ("low", "auto")


def test_a_web_search_tool_reaches_claude_as_the_hosted_web_search_tool(gateway: Gateway) -> None:
    """A Responses `{"type": "web_search", "search_context_size": "medium"}` tool on Anthropic.

    The bridge turns it into `web_search_options`, and the Anthropic transform maps that to the hosted tool
    `{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}` (`ANTHROPIC_WEB_SEARCH_TOOL_MAX_USES`
    medium is 5, `litellm/constants.py:794`), which must be among the `/v1/messages` tools.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = anthropic(scenario, provider)
        served(responses(gateway, model, tools=[{"type": "web_search", "search_context_size": "medium"}]))
        tools: Final = [object_value(tool) for tool in list_value(provider.only().body["tools"])]
        assert {"type": "web_search_20250305", "name": "web_search", "max_uses": 5} in tools, tools


def test_a_web_search_tool_is_dropped_for_a_chat_model_without_web_search_support(gateway: Gateway) -> None:
    """A `web_search` tool on the gpt-4o-mini chat bridge, whose chat params do not list `web_search_options`.

    `_should_drop_derived_web_search_options` drops the derived param instead of raising (`transformation.py:
    1165`), so the call is served and the chat body has neither `web_search_options` nor any tool. UNVERIFIED:
    assumes `get_supported_openai_params` for gpt-4o-mini omits `web_search_options`.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        served(responses(gateway, model, tools=[{"type": "web_search"}]))
        body: Final = provider.only().body
        assert "web_search_options" not in body
        assert "tools" not in body


@pytest.mark.parametrize("hosted", ["image_generation", "computer_use", "local_shell", "shell"])
def test_hosted_tools_without_a_chat_form_are_dropped_and_the_call_is_served(gateway: Gateway, hosted: str) -> None:
    """A hosted tool from `RESPONSES_TOOL_TYPES_WITHOUT_CHAT_FORM` sits beside the `get_weather` function tool.

    The bridge drops it with a warning (`transformation.py:3115`), so the call is answered 200 and the chat
    `tools` hold only `get_weather`.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        served(responses(gateway, model, tools=[{"type": hosted}, WEATHER_TOOL]))
        names: Final = [
            object_value(object_value(tool)["function"])["name"] for tool in list_value(provider.only().body["tools"])
        ]
        assert names == ["get_weather"]


def test_a_file_search_tool_is_emulated_with_a_litellm_file_search_function(gateway: Gateway) -> None:
    """A `{"type": "file_search", "vector_store_ids": ["vs_dec1"]}` tool on a bridged deployment.

    A bridged target has no native file search, so `aresponses_with_emulated_file_search` replaces the tool
    with one function tool named `litellm_file_search` whose `vector_store_id` parameter is an enum of
    `["vs_dec1"]` (`litellm/responses/file_search/emulated_handler.py:65`). The first chat call must carry it.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        served(responses(gateway, model, tools=[{"type": "file_search", "vector_store_ids": ["vs_dec1"]}]))
        tool: Final = object_value(list_value(provider.posts()[0].body["tools"])[0])
        function: Final = object_value(tool["function"])
        assert function["name"] == "litellm_file_search"
        properties: Final = object_value(object_value(function["parameters"])["properties"])
        assert object_value(properties["vector_store_id"])["enum"] == ["vs_dec1"]


def test_an_emulated_file_search_the_model_skips_still_returns_a_file_search_call_item(gateway: Gateway) -> None:
    """The model answers the emulated file search call with plain text and no tool call.

    The handler synthesises the OpenAI shape anyway (`emulated_handler.py:568`): `output[0]` is a
    `file_search_call` with `status: completed` and an id starting `fs_`, and `output[1]` is the message holding
    `REPLY_TEXT`. Only one chat call is made.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        body: Final = served(
            responses(gateway, model, tools=[{"type": "file_search", "vector_store_ids": ["vs_dec1"]}])
        )
        output: Final = [object_value(item) for item in list_value(body["output"])]
        assert output[0]["type"] == "file_search_call"
        assert output[0]["status"] == "completed"
        assert str(output[0]["id"]).startswith("fs_")
        assert output[1]["type"] == "message"
        assert len(provider.posts()) == 1


def test_an_untranslatable_responses_field_is_refused_with_400_before_the_provider(gateway: Gateway) -> None:
    """`max_tool_calls: 2` on a bridged deployment without `drop_params`.

    `max_tool_calls` is a Responses field with no chat equivalent, so `_bridge_kwargs` returns it as
    untranslatable and `_raise_untranslatable_responses_fields` (`responses/main.py:704`) answers 400 with a
    message containing `does not support parameters: ['max_tool_calls']` and `The Responses API to chat
    completions bridge cannot translate them`. The peer receives nothing.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        response: Final = responses(gateway, model, max_tool_calls=2)
        assert response.status_code == 400, response.text
        message: Final = error_message(response)
        assert "does not support parameters: ['max_tool_calls']" in message
        assert "The Responses API to chat completions bridge cannot translate them" in message
        assert provider.posts() == ()


def test_an_untranslatable_responses_field_is_dropped_under_drop_params(gateway: Gateway) -> None:
    """The same `max_tool_calls: 2` on a bridged deployment with `drop_params: true`.

    The field is dropped (`responses/main.py:700`), the call is served, and the chat body has no
    `max_tool_calls` key.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider, drop_params=True)
        served(responses(gateway, model, max_tool_calls=2))
        assert "max_tool_calls" not in provider.only().body


def test_an_untranslatable_field_listed_in_additional_drop_params_is_dropped(gateway: Gateway) -> None:
    """`max_tool_calls: 2` on a bridged deployment with `additional_drop_params: ["max_tool_calls"]` and no
    `drop_params`.

    `_explicitly_dropped_fields` removes it from the untranslatable candidates, so the call is served and the
    field never reaches the chat body.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider, additional_drop_params=["max_tool_calls"])
        served(responses(gateway, model, max_tool_calls=2))
        assert "max_tool_calls" not in provider.only().body


def test_an_untranslatable_field_named_in_allowed_openai_params_is_forwarded(gateway: Gateway) -> None:
    """`max_tool_calls: 2` with `allowed_openai_params: ["max_tool_calls"]` in the request.

    `allowed_openai_params` widens the chat field set (`_chat_completion_request_fields`), so the bridge keeps the
    kwarg and the chat body carries `max_tool_calls: 2`. UNVERIFIED: assumes the OpenAI chat transform forwards an
    allowed unknown param unchanged.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        served(responses(gateway, model, max_tool_calls=2, allowed_openai_params=["max_tool_calls"]))
        assert provider.only().body["max_tool_calls"] == 2


def test_a_streamed_bridged_call_asks_the_chat_provider_for_usage(gateway: Gateway) -> None:
    """`stream: true` on the chat bridge.

    The chat body must carry `stream: true` and `stream_options: {"include_usage": true}` (`transformation.py:
    1439`); without it the final `response.completed` would have no usage to report.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        stream_events(gateway, model)
        body: Final = provider.only().body
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}


def test_a_non_streamed_bridged_call_sends_no_stream_options(gateway: Gateway) -> None:
    """Without `stream`, the chat body has no `stream_options` and no truthy `stream`."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        served(responses(gateway, model))
        body: Final = provider.only().body
        assert "stream_options" not in body
        assert body.get("stream") in (None, False)


def test_text_verbosity_and_json_schema_reach_the_chat_wire_together(gateway: Gateway) -> None:
    """`text: {"format": {"type": "json_object"}, "verbosity": "low"}` on the explicit chat bridge of a model that
    accepts `verbosity` (`gpt-5-mini`). `_chat_verbosity` (`handler.py:40`) lifts it to a top level `verbosity`, so
    the chat body carries `verbosity: "low"` beside `response_format: {"type": "json_object"}`. UNVERIFIED that
    `gpt-5-mini` chat lists `verbosity`; the verbosity suite (`test_responses_bridge_verbosity_wire.py`) pins
    the details for `gpt-5`."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider, model="openai/gpt-5-mini")
        served(responses(gateway, model, text={"format": {"type": "json_object"}, "verbosity": "low"}))
        body: Final = provider.only().body
        assert body["verbosity"] == "low"
        assert body["response_format"] == {"type": "json_object"}


def test_a_tool_call_answer_for_a_function_tool_round_trips_its_arguments(gateway: Gateway) -> None:
    """The chat provider answers one tool call `call_dec1rt` to `get_weather` with `{"city":"Oslo"}`.

    The arguments string must come back byte for byte in the `function_call` item's `arguments`, since agents
    parse it themselves.
    """

    def respond(call: Call) -> Reply:
        return json_reply(
            chat_json(
                call.body,
                content=None,
                finish_reason="tool_calls",
                message={"tool_calls": [tool_call("call_dec1rt", "get_weather", '{"city":"Oslo"}')]},
            )
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        body: Final = served(responses(gateway, model, tools=[WEATHER_TOOL]))
        calls: Final = output_items(body, "function_call")
        assert [item["arguments"] for item in calls] == ['{"city":"Oslo"}']
        assert json.loads(str(calls[0]["arguments"])) == {"city": "Oslo"}
        assert REPLY_TEXT not in json.dumps(body)
        assert chat_messages(provider.only())[0]["role"] == "user"
