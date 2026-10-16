import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from itertools import chain
from typing import Final, Literal, TypeAlias
from urllib.parse import unquote

import pytest
from _pytest.mark.structures import ParameterSet
from integration._support import responses_vendor as rv
from integration._support.anthropic_thinking import BEDROCK_MODEL
from integration._support.client import Gateway, Scenario
from integration._support.openai_wire import chat_reply
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers.test_bedrock_auth_wire import RESPONSE as CONVERSE_RESPONSE
from integration.providers.test_chat_completions_tool_search_wire import (
    assert_lands_once_as,
    raw_answer,
    response_id,
)
from pydantic import JsonValue

_REGEX_TOOL_NAME: Final = "tool_search_tool_regex"
_BM25_TOOL_NAME: Final = "tool_search_tool_bm25"
_SEARCH_ARGUMENTS: Final = '{"pattern": "weather"}'
_AZURE_DEPLOYMENT: Final = "scripted-chat-deployment"
_AZURE_API_VERSION: Final = "2025-04-01-preview"
_GEMINI: Final = "gemini-2.5-flash"
_GEMINI_REPLY: Final[dict[str, JsonValue]] = {
    "candidates": [{"content": {"role": "model", "parts": [{"text": "gemini control"}]}, "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3, "totalTokenCount": 8},
    "modelVersion": _GEMINI,
}
_CITY_PARAMETERS: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}

Route: TypeAlias = Literal["openai", "azure", "bedrock_converse", "gemini"]


def _tool_call(call_id: str, name: str, arguments: str) -> dict[str, JsonValue]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def _search_call(marker: str, name: str = _REGEX_TOOL_NAME) -> dict[str, JsonValue]:
    return _tool_call(f"srvtoolu_{marker}", name, _SEARCH_ARGUMENTS)


def _weather_call(marker: str) -> dict[str, JsonValue]:
    return _tool_call(f"toolu_{marker}", "get_weather", '{"city": "Paris"}')


def _weather_result(marker: str) -> dict[str, JsonValue]:
    return {"role": "tool", "tool_call_id": f"toolu_{marker}", "content": f"22C marker-{marker}"}


def _history(marker: str, calls: Sequence[Mapping[str, JsonValue]], *results: JsonValue) -> list[JsonValue]:
    return [
        {"role": "user", "content": f"Weather in Paris marker-{marker}"},
        {"role": "assistant", "content": f"searching marker-{marker}", "tool_calls": [dict(call) for call in calls]},
        *results,
        {"role": "user", "content": f"And tomorrow? marker-{marker}"},
    ]


def _second_turn(marker: str) -> list[JsonValue]:
    return _history(marker, (_search_call(marker), _weather_call(marker)), _weather_result(marker))


def _body(model: str, messages: Sequence[JsonValue], stream: bool) -> dict[str, JsonValue]:
    return {
        "model": model,
        "messages": list(messages),
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Weather for a city",
                    "parameters": _CITY_PARAMETERS,
                },
            }
        ],
        "stream": stream,
        "num_retries": 0,
    }


def _called_ids(message: Mapping[str, JsonValue]) -> list[str]:
    return [str(call["id"]) for call in rv.ITEMS.validate_python(message.get("tool_calls") or [])]


def _unanswered_chat_calls(messages: Sequence[Mapping[str, JsonValue]]) -> list[str]:
    answered: Final = {str(message.get("tool_call_id")) for message in messages if message.get("role") == "tool"}
    called: Final = tuple(chain.from_iterable(_called_ids(message) for message in messages))
    return [call_id for call_id in called if call_id not in answered]


def _strict_chat_vendor(refusal: Callable[[str], str]) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.method != "POST":
            return Reply()
        body: Final = rv.JSON_OBJECT.validate_json(request.body)
        marker: Final = rv.newest_marker(request.body.decode()) or "health"
        unanswered: Final = _unanswered_chat_calls(rv.ITEMS.validate_python(body["messages"]))
        if unanswered:
            return rv.error(400, refusal(unanswered[0]), "invalid_request_error")
        return chat_reply(
            f"chatcmpl-{marker}", "scripted", f"answer marker-{marker}", stream=body.get("stream") is True
        )

    return respond


def _openai_refusal(call_id: str) -> str:
    return f"No tool output found for function call {call_id}."


def _azure_refusal(call_id: str) -> str:
    return f"An assistant message with 'tool_calls' must be followed by tool messages responding to {call_id}"


def _converse_ids(message: Mapping[str, JsonValue], kind: str) -> set[str]:
    blocks: Final = [
        rv.JSON_OBJECT.validate_python(block[kind])
        for block in rv.ITEMS.validate_python(message["content"])
        if kind in block
    ]
    return {str(block["toolUseId"]) for block in blocks}


def _missing_results(messages: Sequence[Mapping[str, JsonValue]], index: int) -> list[str]:
    following: Final = messages[index + 1] if index + 1 < len(messages) else {"content": []}
    return sorted(_converse_ids(messages[index], "toolUse") - _converse_ids(following, "toolResult"))


def _strict_converse_vendor(request: Request) -> Reply:
    assert unquote(request.target) == f"/model/{BEDROCK_MODEL}/converse", request.target
    messages: Final = rv.ITEMS.validate_python(rv.JSON_OBJECT.validate_json(request.body)["messages"])
    gaps: Final = [
        (index, _missing_results(messages, index))
        for index, message in enumerate(messages)
        if message["role"] == "assistant"
    ]
    unanswered: Final = [(index, missing) for index, missing in gaps if missing]
    if not unanswered:
        return Reply(body=CONVERSE_RESPONSE)
    index, missing = unanswered[0]
    refusal: Final = {
        "message": f"Expected toolResult blocks at messages.{index + 1}.content for the following Ids: {', '.join(missing)}"
    }
    return Reply(status=400, body=json.dumps(refusal).encode())


def _gemini_vendor(request: Request) -> Reply:
    assert request.target.split("?")[0].endswith(f"/models/{_GEMINI}:generateContent"), request.target
    return Reply(body=json.dumps(_GEMINI_REPLY).encode())


def _vendor(route: Route) -> Callable[[Request], Reply]:
    match route:
        case "openai":
            return _strict_chat_vendor(_openai_refusal)
        case "azure":
            return _strict_chat_vendor(_azure_refusal)
        case "bedrock_converse":
            return _strict_converse_vendor
        case "gemini":
            return _gemini_vendor


def _deployment(scenario: Scenario, route: Route, wire_url: str) -> str:
    match route:
        case "openai":
            return scenario.model(model="openai/gpt-5.6", api_base=f"{wire_url}/v1", api_key="scripted-openai-key")
        case "azure":
            return scenario.model(
                model=f"azure/{_AZURE_DEPLOYMENT}",
                api_base=wire_url,
                api_key="scripted-azure-key",
                api_version=_AZURE_API_VERSION,
            )
        case "bedrock_converse":
            return scenario.model(
                model=f"bedrock/converse/{BEDROCK_MODEL}",
                api_base=wire_url,
                aws_access_key_id="AKIASCRIPTEDPROVIDER",
                aws_secret_access_key="scripted-secret",
                aws_region_name="us-east-1",
                aws_bedrock_runtime_endpoint=wire_url,
            )
        case "gemini":
            return scenario.model(model=f"gemini/{_GEMINI}", api_base=wire_url, api_key="synthetic-gemini-key")


def _target(route: Route) -> str:
    match route:
        case "openai":
            return "/v1/chat/completions"
        case "azure":
            return f"/openai/deployments/{_AZURE_DEPLOYMENT}/chat/completions?api-version={_AZURE_API_VERSION}"
        case "bedrock_converse":
            return f"/model/{BEDROCK_MODEL}/converse"
        case "gemini":
            return f"/models/{_GEMINI}:generateContent"


def _only_post(wire: Wire) -> tuple[Request, dict[str, JsonValue]]:
    posted: Final = tuple(request for request in wire.drain() if request.method == "POST")
    assert len(posted) == 1, [request.target for request in posted]
    return posted[0], rv.JSON_OBJECT.validate_json(posted[0].body)


def _wire_messages(body: Mapping[str, JsonValue]) -> list[dict[str, JsonValue]]:
    return rv.ITEMS.validate_python(body.get("messages") or body.get("contents") or [])


def _chat_call_ids(messages: Sequence[Mapping[str, JsonValue]]) -> list[str]:
    (assistant,) = [message for message in messages if message.get("role") == "assistant"]
    return [str(call["id"]) for call in rv.ITEMS.validate_python(assistant.get("tool_calls") or [])]


def _chat_result_ids(messages: Sequence[Mapping[str, JsonValue]]) -> list[str]:
    return [str(message["tool_call_id"]) for message in messages if message.get("role") == "tool"]


def _assert_only_the_function_call_went_out(route: Route, body: Mapping[str, JsonValue], marker: str) -> None:
    messages: Final = _wire_messages(body)
    match route:
        case "openai" | "azure":
            assert "srvtoolu_" not in json.dumps(body) and _REGEX_TOOL_NAME not in json.dumps(body), body
            assert _chat_call_ids(messages) == [f"toolu_{marker}"], messages
            assert _chat_result_ids(messages) == [f"toolu_{marker}"], messages
        case "bedrock_converse":
            assert "srvtoolu_" not in json.dumps(body) and _REGEX_TOOL_NAME not in json.dumps(body), body
            assert [message["role"] for message in messages] == ["user", "assistant", "user"], messages
            assert _converse_ids(messages[1], "toolUse") == {f"toolu_{marker}"}, messages
            assert _converse_ids(messages[2], "toolResult") == {f"toolu_{marker}"}, messages
        case "gemini":
            assert _REGEX_TOOL_NAME in json.dumps(messages), messages


def _answer_text(route: Route, marker: str) -> str:
    match route:
        case "openai" | "azure":
            return f"answer marker-{marker}"
        case "bedrock_converse":
            return "bedrock wire control"
        case "gemini":
            return "gemini control"


def _history_cells() -> tuple[ParameterSet, ...]:
    cases: Final[tuple[tuple[Route, bool], ...]] = (
        ("openai", False),
        ("openai", True),
        ("azure", False),
        ("azure", True),
        ("bedrock_converse", False),
        ("gemini", False),
    )
    return tuple(pytest.param(route, stream, id=f"{route}-{'stream' if stream else 'sync'}") for route, stream in cases)


@pytest.mark.parametrize(("route", "stream"), _history_cells())
def test_a_claude_search_call_with_no_result_is_dropped_only_on_the_routes_that_reject_it(
    gateway: Gateway, route: Route, stream: bool
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_vendor(route)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, route, wire.url)
        answer: Final = raw_answer(gateway, _body(model, _second_turn(marker), stream))
        assert answer.status == 200, answer.text
        assert _answer_text(route, marker) in answer.text, answer.text
        request, body = _only_post(wire)
        assert unquote(request.target).endswith(_target(route)), request.target
        _assert_only_the_function_call_went_out(route, body, marker)
        assert_lands_once_as(model, (response_id(answer),), "success")


@pytest.mark.parametrize("content", [None, ""], ids=["null", "empty"])
def test_a_search_only_assistant_turn_with_no_content_is_removed(gateway: Gateway, content: JsonValue) -> None:
    marker: Final = uuid.uuid4().hex
    messages: Final[list[JsonValue]] = [
        {"role": "user", "content": f"Weather in Paris marker-{marker}"},
        {"role": "assistant", "content": content, "tool_calls": [_search_call(marker)]},
        {"role": "user", "content": f"Just tell me marker-{marker}"},
    ]
    with wire_server(_vendor("openai")) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "openai", wire.url)
        answer: Final = raw_answer(gateway, _body(model, messages, False))
        assert answer.status == 200 and f"answer marker-{marker}" in answer.text, answer.text
        _, body = _only_post(wire)
        sent: Final = _wire_messages(body)
        assert [message["role"] for message in sent] == ["user", "user"], sent
        assert "srvtoolu_" not in json.dumps(body), body


@pytest.mark.parametrize("name", [_REGEX_TOOL_NAME, _BM25_TOOL_NAME])
def test_a_search_call_the_client_answered_keeps_its_result(gateway: Gateway, name: str) -> None:
    marker: Final = uuid.uuid4().hex
    search_result: Final[dict[str, JsonValue]] = {
        "role": "tool",
        "tool_call_id": f"srvtoolu_{marker}",
        "content": f"get_weather marker-{marker}",
    }
    messages: Final = _history(
        marker, (_search_call(marker, name), _weather_call(marker)), search_result, _weather_result(marker)
    )
    with wire_server(_vendor("openai")) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "openai", wire.url)
        answer: Final = raw_answer(gateway, _body(model, messages, False))
        assert answer.status == 200 and f"answer marker-{marker}" in answer.text, answer.text
        _, body = _only_post(wire)
        sent: Final = _wire_messages(body)
        assert _chat_call_ids(sent) == [f"srvtoolu_{marker}", f"toolu_{marker}"], sent
        assert _chat_result_ids(sent) == [f"srvtoolu_{marker}", f"toolu_{marker}"], sent


def test_a_function_call_whose_id_only_looks_like_a_search_call_is_kept(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    look_alike: Final = _tool_call(f"srvtoolu_{marker}", "get_weather", '{"city": "Lyon"}')
    messages: Final = _history(marker, (look_alike, _weather_call(marker)), _weather_result(marker))
    with wire_server(_vendor("openai")) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "openai", wire.url)
        answer: Final = raw_answer(gateway, _body(model, messages, False))
        assert answer.status == 400, answer.text
        assert f"No tool output found for function call srvtoolu_{marker}" in answer.text, answer.text
        _, body = _only_post(wire)
        assert _chat_call_ids(_wire_messages(body)) == [f"srvtoolu_{marker}", f"toolu_{marker}"], body


@pytest.mark.parametrize(
    "tool_call_id",
    [5, ["srvtoolu_x"], "", "srvtoolu_" + "x" * 5000],
    ids=["int", "list", "empty", "five-kb"],
)
def test_a_malformed_answer_id_leaves_the_search_call_unanswered_and_the_proxy_serving(
    gateway: Gateway, tool_call_id: JsonValue
) -> None:
    marker: Final = uuid.uuid4().hex
    malformed: Final[dict[str, JsonValue]] = {"role": "tool", "tool_call_id": tool_call_id, "content": "x"}
    messages: Final = _history(
        marker, (_search_call(marker), _weather_call(marker)), malformed, _weather_result(marker)
    )
    with wire_server(_vendor("openai")) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "openai", wire.url)
        answer: Final = raw_answer(gateway, _body(model, messages, False))
        assert answer.status in (200, 400), answer.text
        posted: Final = tuple(request for request in wire.drain() if request.method == "POST")
        assert all("srvtoolu_" + marker not in request.body.decode() for request in posted), posted
        follow_up: Final = uuid.uuid4().hex
        served: Final = raw_answer(gateway, _body(model, _second_turn(follow_up), False))
        assert served.status == 200 and f"answer marker-{follow_up}" in served.text, served.text
