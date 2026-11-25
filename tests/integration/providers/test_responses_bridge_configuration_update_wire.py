import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import reduce
from itertools import chain
from types import MappingProxyType
from typing import Final
from urllib.parse import urlsplit

import httpx
import pytest
from integration._support.client import Gateway, Scenario, list_value, object_value
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_GPT6_MODEL: Final = "gpt-6.2-sol"
_GPT5_MODEL: Final = "gpt-5.5"
_GPT6_NONE_EFFORT_MODEL: Final = "gpt-6.1-terra"
_NON_OPENAI_MODEL: Final = "xai/responses/grok-4.20-reasoning-gv2"
_SYSTEM: Final = "You are a terse travel assistant. Answer in one short sentence."
_ANSWER: Final = '{"city": "Tokyo"}'
_CONFIGURATION_UPDATE: Final = "configuration_update"
_ADDITIONAL_TOOLS: Final = "additional_tools"
_WEATHER_ASK: Final = "weather in Tokyo"
_WEATHER_ARGUMENTS: Final = '{"city": "Tokyo"}'
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def _function_tool(name: str, description: str) -> dict[str, JsonValue]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }


def _json_schema_format(name: str, field: str) -> dict[str, JsonValue]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {field: {"type": "string"}},
                "required": [field],
                "additionalProperties": False,
            },
        },
    }


_WEATHER_TOOL: Final = _function_tool("get_weather", "Get the weather for a city.")
_TIME_TOOL: Final = _function_tool("get_local_time", "Get the local time in a city.")
_CITY_FORMAT: Final = _json_schema_format("city_answer", "city")
_COUNTRY_FORMAT: Final = _json_schema_format("country_answer", "country")


def _reasoning_item(identity: str) -> dict[str, JsonValue]:
    return {
        "type": "reasoning",
        "id": f"rs_{identity}",
        "summary": [{"type": "summary_text", "text": "Tokyo fits the question."}],
        "encrypted_content": f"enc_{identity}",
    }


def _message_item(identity: str, status: str = "completed") -> dict[str, JsonValue]:
    return {
        "type": "message",
        "id": f"msg_{identity}",
        "status": status,
        "role": "assistant",
        "phase": "final_answer",
        "content": [{"type": "output_text", "text": _ANSWER, "annotations": []}] if status == "completed" else [],
    }


def _function_call_item(identity: str, status: str = "completed") -> dict[str, JsonValue]:
    return {
        "type": "function_call",
        "id": f"fc_{identity}",
        "call_id": f"call_{identity}",
        "name": "get_weather",
        "arguments": _WEATHER_ARGUMENTS if status == "completed" else "",
        "status": status,
    }


def _response(identity: str, model: JsonValue, status: str, output: Sequence[JsonValue]) -> dict[str, JsonValue]:
    return {
        "id": f"resp_{identity}",
        "object": "response",
        "created_at": 1789788253,
        "status": status,
        "model": model,
        "output": list(output),
        "usage": {
            "input_tokens": 2048,
            "input_tokens_details": {"cached_tokens": 1024},
            "output_tokens": 16,
            "output_tokens_details": {"reasoning_tokens": 8},
            "total_tokens": 2064,
        },
    }


def _event(payload: Mapping[str, JsonValue]) -> bytes:
    return f"event: {payload['type']}\ndata: {json.dumps(payload)}\n\n".encode()


def _item_events(identity: str, index: int, item: Mapping[str, JsonValue]) -> tuple[Mapping[str, JsonValue], ...]:
    match item["type"]:
        case "message":
            return (
                {
                    "type": "response.output_item.added",
                    "output_index": index,
                    "item": _message_item(identity, "in_progress"),
                },
                {
                    "type": "response.output_text.delta",
                    "item_id": item["id"],
                    "output_index": index,
                    "content_index": 0,
                    "delta": _ANSWER,
                },
                {"type": "response.output_item.done", "output_index": index, "item": dict(item)},
            )
        case "function_call":
            return (
                {
                    "type": "response.output_item.added",
                    "output_index": index,
                    "item": _function_call_item(identity, "in_progress"),
                },
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": item["id"],
                    "output_index": index,
                    "delta": _WEATHER_ARGUMENTS,
                },
                {"type": "response.output_item.done", "output_index": index, "item": dict(item)},
            )
        case _:
            return (
                {"type": "response.output_item.added", "output_index": index, "item": dict(item)},
                {"type": "response.output_item.done", "output_index": index, "item": dict(item)},
            )


def _stream(identity: str, model: JsonValue, output: Sequence[dict[str, JsonValue]]) -> tuple[bytes, ...]:
    events: Final[tuple[Mapping[str, JsonValue], ...]] = (
        {"type": "response.created", "response": _response(identity, model, "in_progress", [])},
        *chain.from_iterable(_item_events(identity, index, item) for index, item in enumerate(output)),
        {"type": "response.completed", "response": _response(identity, model, "completed", list(output))},
    )
    return tuple(_event(payload) for payload in events)


def _asks_for_weather(body: Mapping[str, JsonValue]) -> bool:
    last: Final = object_value(list_value(body["input"])[-1])
    return last.get("role") == "user" and _WEATHER_ASK in _text(last)


def _respond(request: Request) -> Reply:
    if request.method == "GET":
        return Reply(body=b'{"object":"list","data":[]}')
    assert urlsplit(request.target).path.endswith("/responses"), request.target
    body: Final = _JSON_OBJECT.validate_json(request.body)
    identity: Final = uuid.uuid4().hex
    answer: Final = _function_call_item(identity) if _asks_for_weather(body) else _message_item(identity)
    output: Final = (_reasoning_item(identity), answer)
    if body.get("stream"):
        return Reply(content_type="text/event-stream", chunks=_stream(identity, body.get("model"), output))
    return Reply(body=json.dumps(_response(identity, body.get("model"), "completed", list(output))).encode())


def _posted_bodies(wire: Wire) -> tuple[dict[str, JsonValue], ...]:
    return tuple(_JSON_OBJECT.validate_json(request.body) for request in wire.drain() if request.method == "POST")


def _deltas(response: httpx.Response) -> list[dict[str, JsonValue]]:
    chunks: Final = [
        _JSON_OBJECT.validate_json(line[6:])
        for line in response.iter_lines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    return [
        object_value(object_value(choice)["delta"])
        for choice in chain.from_iterable(list_value(chunk["choices"]) for chunk in chunks)
    ]


def _merged_provider_fields(merged: dict[str, JsonValue], delta: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {**merged, **object_value(delta.get("provider_specific_fields") or {})}


def _tool_call(index: int, calls: Sequence[Mapping[str, JsonValue]]) -> dict[str, JsonValue]:
    parts: Final = [call for call in calls if call.get("index") == index]
    functions: Final = [object_value(part["function"]) for part in parts]
    return {
        "id": next(part["id"] for part in parts if part.get("id")),
        "type": "function",
        "function": {
            "name": next(function["name"] for function in functions if function.get("name")),
            "arguments": "".join(str(function.get("arguments") or "") for function in functions),
        },
    }


def _assembled(response: httpx.Response) -> dict[str, JsonValue]:
    deltas: Final = _deltas(response)
    empty: Final[dict[str, JsonValue]] = {}
    fields: Final = reduce(_merged_provider_fields, deltas, empty)
    reasoning_items: Final = list(chain.from_iterable(list_value(d.get("reasoning_items") or []) for d in deltas))
    calls: Final = [
        object_value(call) for call in chain.from_iterable(list_value(d.get("tool_calls") or []) for d in deltas)
    ]
    indexes: Final = sorted({int(str(call.get("index"))) for call in calls})
    tool_calls: Final[list[JsonValue]] = [_tool_call(index, calls) for index in indexes]
    return {
        "role": "assistant",
        "content": "".join(str(delta["content"]) for delta in deltas if delta.get("content")) or None,
        **({"tool_calls": tool_calls} if tool_calls else {}),
        **({"reasoning_items": reasoning_items} if reasoning_items else {}),
        **({"provider_specific_fields": fields} if fields else {}),
    }


def _stripped(message: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {key: value for key, value in message.items() if key in {"role", "content", "tool_calls"} and value}


def _question(number: int, marker: str) -> str:
    return f"Question {number}: which city should I visit next? {marker}"


@dataclass(frozen=True, slots=True)
class _Caller:
    gateway: Gateway
    model: str
    stream: bool
    keep_extra_fields: bool

    def reply(self, messages: Sequence[JsonValue], settings: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        body: Final[dict[str, JsonValue]] = {
            "model": self.model,
            "messages": list(messages),
            "stream": self.stream,
            "cache": {"no-cache": True},
            **settings,
        }
        response: Final = self.gateway.request("POST", "/v1/chat/completions", body)
        assert response.status_code == 200, response.text
        message: Final = (
            _assembled(response)
            if self.stream
            else object_value(
                object_value(list_value(_JSON_OBJECT.validate_json(response.content)["choices"])[0])["message"]
            )
        )
        return message if self.keep_extra_fields else _stripped(message)

    def ask(
        self, history: tuple[JsonValue, ...], question: str, settings: Mapping[str, JsonValue]
    ) -> tuple[JsonValue, ...]:
        user: Final[JsonValue] = {"role": "user", "content": question}
        asked: Final[tuple[JsonValue, ...]] = (*history, user)
        return (*asked, self.reply(asked, settings))

    def turn(
        self, history: tuple[JsonValue, ...], numbered: tuple[int, Mapping[str, JsonValue]], marker: str
    ) -> tuple[JsonValue, ...]:
        return self.ask(history, _question(numbered[0], marker), numbered[1])

    def answer_tool_calls(
        self, history: tuple[JsonValue, ...], settings: Mapping[str, JsonValue]
    ) -> tuple[JsonValue, ...]:
        calls: Final = list_value(object_value(history[-1]).get("tool_calls") or [])
        assert calls, history[-1]
        results: Final[tuple[JsonValue, ...]] = tuple(
            {"role": "tool", "tool_call_id": object_value(call)["id"], "content": "Rain, 18C"} for call in calls
        )
        answered: Final = (*history, *results)
        return (*answered, self.reply(answered, settings))

    def converse(self, turns: Sequence[Mapping[str, JsonValue]]) -> None:
        marker: Final = uuid.uuid4().hex
        system: Final[tuple[JsonValue, ...]] = ({"role": "system", "content": _SYSTEM},)
        reduce(lambda history, numbered: self.turn(history, numbered, marker), enumerate(turns, start=1), system)


def _openai_model(scenario: Scenario, wire: Wire, model: str, model_info: Mapping[str, JsonValue] | None = None) -> str:
    return scenario.model(model=model, model_info=model_info, api_base=wire.url, api_key="synthetic-openai-key")


def _azure_model(scenario: Scenario, wire: Wire, model: str) -> str:
    return scenario.model(
        model=f"azure/responses/{model}", api_base=wire.url, api_key="synthetic-azure-key", api_version="preview"
    )


_ROUTES: Final[Mapping[str, Callable[[Scenario, Wire, str], str]]] = MappingProxyType(
    {
        "openai-pinned": lambda scenario, wire, model: _openai_model(scenario, wire, f"openai/responses/{model}"),
        "openai-implicit": lambda scenario, wire, model: _openai_model(scenario, wire, f"openai/{model}"),
        "openai-mode-responses": lambda scenario, wire, model: _openai_model(
            scenario, wire, f"openai/{model}", model_info={"mode": "responses"}
        ),
        "azure-pinned": _azure_model,
        "as-written": lambda scenario, wire, model: _openai_model(scenario, wire, model),
    }
)


def _run(
    gateway: Gateway,
    turns: Sequence[Mapping[str, JsonValue]],
    *,
    stream: bool,
    route: str = "openai-pinned",
    model: str = _GPT6_MODEL,
    keep_extra_fields: bool = True,
) -> tuple[dict[str, JsonValue], ...]:
    with wire_server(_respond) as wire, gateway.scenario() as scenario:
        deployment: Final = _ROUTES[route](scenario, wire, model)
        _Caller(gateway, deployment, stream, keep_extra_fields).converse(turns)
        bodies: Final = _posted_bodies(wire)
    assert len(bodies) == len(turns), bodies
    return bodies


def _first_turn_body(gateway: Gateway, settings: Mapping[str, JsonValue], route: str) -> dict[str, JsonValue]:
    return _run(gateway, [settings], stream=False, route=route)[0]


def _items(body: Mapping[str, JsonValue]) -> list[dict[str, JsonValue]]:
    return [object_value(item) for item in list_value(body["input"])]


def _count(body: Mapping[str, JsonValue], kind: str) -> int:
    return sum(1 for item in _items(body) if item.get("type") == kind)


def _text(item: Mapping[str, JsonValue]) -> str:
    content: Final = item.get("content")
    if isinstance(content, str):
        return content
    return "".join(str(object_value(part).get("text")) for part in list_value(content or []))


def _question_position(body: Mapping[str, JsonValue], number: int) -> int:
    positions: Final = [
        index
        for index, item in enumerate(_items(body))
        if item.get("role") == "user" and _text(item).startswith(f"Question {number}:")
    ]
    assert len(positions) == 1, (number, body["input"])
    return positions[0]


def _before_question(body: Mapping[str, JsonValue], number: int, distance: int = 1) -> dict[str, JsonValue]:
    return _items(body)[_question_position(body, number) - distance]


def _assert_each_turn_extends_the_previous_input(bodies: Sequence[Mapping[str, JsonValue]]) -> None:
    for earlier, later in zip(bodies, bodies[1:], strict=False):
        assert _items(later)[: len(_items(earlier))] == _items(earlier), (earlier["input"], later["input"])


def _assert_top_level_stays_at_the_first_turn(bodies: Sequence[Mapping[str, JsonValue]]) -> None:
    for field in ("reasoning", "text", "tools"):
        assert [body.get(field) for body in bodies] == [bodies[0].get(field)] * len(bodies), field


def _assert_no_mid_conversation_items(bodies: Sequence[Mapping[str, JsonValue]]) -> None:
    kinds: Final = {str(item.get("type")) for body in bodies for item in _items(body)}
    assert not kinds & {_CONFIGURATION_UPDATE, _ADDITIONAL_TOOLS}, kinds


def _effort(body: Mapping[str, JsonValue]) -> JsonValue:
    return object_value(body["reasoning"]).get("effort")


def _efforts(*efforts: str, **extra: JsonValue) -> list[dict[str, JsonValue]]:
    return [{"reasoning_effort": effort, "tools": [_WEATHER_TOOL], **extra} for effort in efforts]


def _tools(*tools: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {"reasoning_effort": "low", "tools": list(tools)}


_STREAM: Final = pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])


@_STREAM
@pytest.mark.parametrize("route", ["openai-pinned", "openai-implicit", "openai-mode-responses", "azure-pinned"])
def test_regression_gpt6_effort_change_becomes_a_configuration_update_before_the_new_question(
    gateway: Gateway, route: str, stream: bool
) -> None:
    bodies: Final = _run(gateway, _efforts("low", "low", "high", "high"), stream=stream, route=route)

    _assert_top_level_stays_at_the_first_turn(bodies)
    assert _effort(bodies[0]) == "low", bodies[0]
    assert [_count(body, _CONFIGURATION_UPDATE) for body in bodies] == [0, 0, 1, 1], bodies
    assert _before_question(bodies[2], 3) == {"type": _CONFIGURATION_UPDATE, "reasoning": {"effort": "high"}}
    assert _before_question(bodies[2], 3, distance=2).get("role") == "assistant", bodies[2]["input"]
    _assert_each_turn_extends_the_previous_input(bodies)


@_STREAM
def test_regression_gpt6_reverted_effort_adds_a_second_configuration_update_at_the_later_turn(
    gateway: Gateway, stream: bool
) -> None:
    bodies: Final = _run(gateway, _efforts("low", "high", "low", "low"), stream=stream)

    _assert_top_level_stays_at_the_first_turn(bodies)
    assert [_count(body, _CONFIGURATION_UPDATE) for body in bodies] == [0, 1, 2, 2], bodies
    assert _before_question(bodies[1], 2) == {"type": _CONFIGURATION_UPDATE, "reasoning": {"effort": "high"}}
    assert _before_question(bodies[2], 3) == {"type": _CONFIGURATION_UPDATE, "reasoning": {"effort": "low"}}
    _assert_each_turn_extends_the_previous_input(bodies)


@_STREAM
def test_regression_gpt6_response_format_change_becomes_a_configuration_update(gateway: Gateway, stream: bool) -> None:
    plain: Final = {"reasoning_effort": "low"}
    structured: Final = {**plain, "response_format": _COUNTRY_FORMAT}
    bodies: Final = _run(gateway, [plain, plain, structured, structured], stream=stream)
    country_format: Final = object_value(_first_turn_body(gateway, structured, "openai-pinned")["text"])["format"]

    _assert_top_level_stays_at_the_first_turn(bodies)
    assert [_count(body, _CONFIGURATION_UPDATE) for body in bodies] == [0, 0, 1, 1], bodies
    assert _before_question(bodies[2], 3) == {"type": _CONFIGURATION_UPDATE, "text": {"format": country_format}}
    _assert_each_turn_extends_the_previous_input(bodies)


def test_guard_gpt6_switching_to_a_different_response_format_rewrites_the_top_level_text(gateway: Gateway) -> None:
    city: Final = {"reasoning_effort": "low", "response_format": _CITY_FORMAT}
    country: Final = {"reasoning_effort": "low", "response_format": _COUNTRY_FORMAT}
    bodies: Final = _run(gateway, [city, city, country, country], stream=False)
    city_text: Final = _first_turn_body(gateway, city, "openai-pinned")["text"]
    country_text: Final = _first_turn_body(gateway, country, "openai-pinned")["text"]

    assert city_text != country_text, (city_text, country_text)
    assert [body.get("text") for body in bodies] == [city_text, city_text, country_text, country_text], bodies
    assert [_count(body, _CONFIGURATION_UPDATE) for body in bodies] == [0, 0, 0, 0], bodies


@_STREAM
@pytest.mark.parametrize("route", ["openai-pinned", "azure-pinned"])
def test_regression_gpt6_added_tool_becomes_additional_tools_before_the_new_question(
    gateway: Gateway, route: str, stream: bool
) -> None:
    one_tool: Final = _tools(_WEATHER_TOOL)
    two_tools: Final = _tools(_WEATHER_TOOL, _TIME_TOOL)
    bodies: Final = _run(gateway, [one_tool, one_tool, two_tools, two_tools], stream=stream, route=route)
    time_tool: Final = _first_turn_body(gateway, _tools(_TIME_TOOL), route)["tools"]

    _assert_top_level_stays_at_the_first_turn(bodies)
    assert [_count(body, _ADDITIONAL_TOOLS) for body in bodies] == [0, 0, 1, 1], bodies
    assert _before_question(bodies[2], 3) == {"type": _ADDITIONAL_TOOLS, "role": "developer", "tools": time_tool}
    _assert_each_turn_extends_the_previous_input(bodies)


def test_guard_gpt5_effort_change_still_rewrites_the_top_level_reasoning(gateway: Gateway) -> None:
    bodies: Final = _run(gateway, _efforts("low", "low", "high", "high"), stream=False, model=_GPT5_MODEL)

    assert [_effort(body) for body in bodies] == ["low", "low", "high", "high"], bodies
    _assert_no_mid_conversation_items(bodies)


def test_guard_gpt6_caller_that_strips_extra_message_fields_gets_the_top_level_effort_rewrite(gateway: Gateway) -> None:
    bodies: Final = _run(gateway, _efforts("low", "low", "high", "high"), stream=False, keep_extra_fields=False)

    assert [_effort(body) for body in bodies] == ["low", "low", "high", "high"], bodies
    _assert_no_mid_conversation_items(bodies)


def test_guard_gpt6_truncation_auto_effort_change_sends_no_configuration_update(gateway: Gateway) -> None:
    bodies: Final = _run(gateway, _efforts("low", "low", "high", "high", truncation="auto"), stream=False)

    assert [body.get("truncation") for body in bodies] == ["auto"] * len(bodies), bodies
    assert [_effort(body) for body in bodies] == ["low", "low", "high", "high"], bodies
    _assert_no_mid_conversation_items(bodies)


@pytest.mark.parametrize("route", ["openai-pinned", "openai-implicit", "azure-pinned"])
def test_guard_gpt6_conversation_that_never_changes_settings_sends_only_messages_and_reasoning(
    gateway: Gateway, route: str
) -> None:
    bodies: Final = _run(gateway, _efforts("medium", "medium", "medium", "medium"), stream=False, route=route)

    _assert_top_level_stays_at_the_first_turn(bodies)
    assert {str(item.get("type")) for body in bodies for item in _items(body)} == {"message", "reasoning"}, bodies
    _assert_each_turn_extends_the_previous_input(bodies)


def test_guard_non_openai_bridge_provider_effort_change_still_rewrites_the_top_level_reasoning(
    gateway: Gateway,
) -> None:
    bodies: Final = _run(
        gateway, _efforts("low", "low", "high", "high"), stream=False, route="as-written", model=_NON_OPENAI_MODEL
    )

    assert [_effort(body) for body in bodies] == ["low", "low", "high", "high"], bodies
    _assert_no_mid_conversation_items(bodies)


def _run_tool_result_turn_with_new_settings(gateway: Gateway, *, stream: bool) -> tuple[dict[str, JsonValue], ...]:
    before: Final = _tools(_WEATHER_TOOL)
    after: Final = {**_tools(_WEATHER_TOOL, _TIME_TOOL), "reasoning_effort": "high"}
    marker: Final = uuid.uuid4().hex
    system: Final[tuple[JsonValue, ...]] = ({"role": "system", "content": _SYSTEM},)
    with wire_server(_respond) as wire, gateway.scenario() as scenario:
        caller: Final = _Caller(gateway, _ROUTES["openai-pinned"](scenario, wire, _GPT6_MODEL), stream, True)
        first: Final = caller.turn(system, (1, before), marker)
        called: Final = caller.ask(first, f"Question 2: what is the {_WEATHER_ASK} right now? {marker}", before)
        answered: Final = caller.answer_tool_calls(called, after)
        caller.turn(answered, (4, after), marker)
        bodies: Final = _posted_bodies(wire)
    assert len(bodies) == 4, bodies
    return bodies


def _by_type(items: Sequence[Mapping[str, JsonValue]]) -> list[Mapping[str, JsonValue]]:
    return sorted(items, key=lambda item: str(item.get("type")))


@_STREAM
def test_regression_gpt6_settings_changed_on_a_tool_result_turn_land_after_the_function_call_output(
    gateway: Gateway, stream: bool
) -> None:
    bodies: Final = _run_tool_result_turn_with_new_settings(gateway, stream=stream)
    time_tool: Final = _first_turn_body(gateway, _tools(_TIME_TOOL), "openai-pinned")["tools"]
    continued: Final = _items(bodies[2])

    _assert_top_level_stays_at_the_first_turn(bodies)
    assert _effort(bodies[0]) == "low", bodies[0]
    assert [item.get("type") for item in continued[-4:-2]] == ["function_call", "function_call_output"], continued
    assert _by_type(continued[-2:]) == _by_type(
        [
            {"type": _ADDITIONAL_TOOLS, "role": "developer", "tools": time_tool},
            {"type": _CONFIGURATION_UPDATE, "reasoning": {"effort": "high"}},
        ]
    ), continued
    assert [_count(body, _CONFIGURATION_UPDATE) for body in bodies] == [0, 0, 1, 1], bodies
    assert [_count(body, _ADDITIONAL_TOOLS) for body in bodies] == [0, 0, 1, 1], bodies
    _assert_each_turn_extends_the_previous_input(bodies)


@_STREAM
def test_regression_gpt6_effort_change_after_a_tool_removal_keeps_the_rewritten_top_level_and_sends_an_update(
    gateway: Gateway, stream: bool
) -> None:
    one_tool: Final = _tools(_WEATHER_TOOL)
    two_tools: Final = _tools(_WEATHER_TOOL, _TIME_TOOL)
    raised: Final = {**one_tool, "reasoning_effort": "high"}
    bodies: Final = _run(gateway, [one_tool, two_tools, one_tool, raised, raised], stream=stream)

    assert bodies[2]["tools"] == bodies[0]["tools"], bodies[2]["tools"]
    _assert_top_level_stays_at_the_first_turn(bodies[2:])
    assert _effort(bodies[3]) == "low", bodies[3]
    assert _before_question(bodies[3], 4) == {"type": _CONFIGURATION_UPDATE, "reasoning": {"effort": "high"}}
    assert _before_question(bodies[3], 4, distance=2).get("role") == "assistant", bodies[3]["input"]
    assert [_count(body, _CONFIGURATION_UPDATE) for body in bodies] == [0, 0, 0, 1, 1], bodies
    _assert_each_turn_extends_the_previous_input(bodies[2:])


def test_guard_gpt6_effort_dropped_to_none_with_temperature_rewrites_the_top_level_reasoning(gateway: Gateway) -> None:
    reasoning: Final = _tools(_WEATHER_TOOL)
    no_reasoning: Final = {**reasoning, "reasoning_effort": "none", "temperature": 0.3}
    bodies: Final = _run(
        gateway, [reasoning, reasoning, no_reasoning, no_reasoning], stream=False, model=_GPT6_NONE_EFFORT_MODEL
    )

    assert [_effort(body) for body in bodies] == ["low", "low", "none", "none"], bodies
    assert [body.get("temperature") for body in bodies] == [None, None, 0.3, 0.3], bodies
    _assert_no_mid_conversation_items(bodies)
