import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import httpx
import pytest
import yaml
from integration._support import responses_vendor as rv
from integration._support.client import Gateway, Scenario, eventually, gateway_from_environment
from integration._support.openai_wire import answering_model_discovery, chat_reply, responses_reply
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = pytest.mark.timeout(120)

_GPT: Final = "gpt-5.6"
_DATABRICKS: Final = "databricks-meta-llama-4-maverick"
_PROVIDER_KEY: Final = "synthetic-provider-key"
_NO_CACHE: Final[dict[str, JsonValue]] = {"no-cache": True}
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_WEATHER_SCHEMA: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "celsius": {"type": "number"}},
    "required": ["city", "celsius"],
    "additionalProperties": False,
}
_TEXT_FORMAT: Final[dict[str, JsonValue]] = {
    "type": "json_schema",
    "name": "weather",
    "schema": _WEATHER_SCHEMA,
    "strict": True,
}
_RESPONSE_FORMAT: Final[dict[str, JsonValue]] = {
    "type": "json_schema",
    "json_schema": {"name": "weather", "schema": _WEATHER_SCHEMA, "strict": True},
}
_STREAM_OPTIONS: Final[dict[str, JsonValue]] = {"stream_options": {"include_usage": True}}

Verbosity = Literal["low", "high"]


def _prompt(marker: str) -> str:
    return f"verbosity probe marker-{marker}"


def _answer(marker: str) -> str:
    return f"scripted answer marker-{marker}"


def _marker_of(request: Request) -> str:
    body: Final = request.body.decode()
    start: Final = body.rfind("marker-")
    return body[start + len("marker-") : start + len("marker-") + 32] if start >= 0 else "health"


def _vendor(request: Request) -> Reply:
    body: Final = _JSON_OBJECT.validate_json(request.body or b"{}")
    marker: Final = _marker_of(request)
    stream: Final = body.get("stream") is True
    if request.target.endswith("/responses"):
        return responses_reply(f"resp_{marker}", _GPT, _answer(marker), stream=stream)
    return chat_reply(f"chatcmpl-{marker}", str(body.get("model")), _answer(marker), stream=stream)


@contextmanager
def _upstream() -> Iterator[Wire]:
    with wire_server(answering_model_discovery(_vendor)) as wire:
        yield wire


def _bridged_gpt(scenario: Scenario, wire: Wire, **parameters: JsonValue) -> str:
    return scenario.model(
        model=f"openai/{_GPT}",
        api_base=f"{wire.url}/v1",
        api_key=_PROVIDER_KEY,
        use_chat_completions_api=True,
        **parameters,
    )


def _native_gpt(scenario: Scenario, wire: Wire) -> str:
    return scenario.model(model=f"openai/{_GPT}", api_base=f"{wire.url}/v1", api_key=_PROVIDER_KEY)


def _databricks(scenario: Scenario, wire: Wire, **parameters: JsonValue) -> str:
    return scenario.model(
        model=f"databricks/{_DATABRICKS}",
        api_base=f"{wire.url}/serving-endpoints",
        api_key=_PROVIDER_KEY,
        **parameters,
    )


def _responses(gateway: Gateway, model: str, marker: str, **fields: JsonValue) -> httpx.Response:
    return gateway.request(
        "POST", "/v1/responses", {"model": model, "input": _prompt(marker), "cache": _NO_CACHE, **fields}
    )


def _streamed_events(gateway: Gateway, model: str, marker: str, **fields: JsonValue) -> list[dict[str, JsonValue]]:
    with gateway.client.stream(
        "POST",
        "/v1/responses",
        json={"model": model, "input": _prompt(marker), "cache": _NO_CACHE, "stream": True, **fields},
        headers={"Authorization": f"Bearer {gateway.key}"},
    ) as response:
        assert response.status_code == 200, response.read()
        return [
            _JSON_OBJECT.validate_json(line.removeprefix("data: "))
            for line in response.iter_lines()
            if line.startswith("data: {")
        ]


def _output_text(response: Mapping[str, JsonValue]) -> str:
    output: Final = response["output"]
    assert isinstance(output, list), response
    return "".join(
        str(part["text"])
        for item in output
        if isinstance(item, dict) and item.get("type") == "message"
        for part in item["content"]
        if isinstance(part, dict) and part.get("type") == "output_text"
    )


def _answered(response: httpx.Response, marker: str) -> None:
    assert response.status_code == 200, response.text
    body: Final = _JSON_OBJECT.validate_json(response.text)
    assert body["status"] == "completed" and _output_text(body) == _answer(marker), body


def _streamed_answer(events: list[dict[str, JsonValue]], marker: str) -> None:
    completed: Final = [event for event in events if event["type"] == "response.completed"]
    assert len(completed) == 1, [event["type"] for event in events]
    final: Final = completed[0]["response"]
    assert isinstance(final, dict) and _output_text(final) == _answer(marker), final
    deltas: Final = "".join(str(event["delta"]) for event in events if event["type"] == "response.output_text.delta")
    assert deltas == _answer(marker), events


def _posted(wire: Wire) -> tuple[Request, ...]:
    return tuple(request for request in wire.drain() if request.method == "POST")


def _only_post(wire: Wire, target: str) -> dict[str, JsonValue]:
    posted: Final = _posted(wire)
    assert [request.target for request in posted] == [target], posted
    return _JSON_OBJECT.validate_json(posted[0].body)


def _chat_messages(marker: str) -> list[JsonValue]:
    return [{"role": "user", "content": _prompt(marker)}]


def _unsupported_param_error(response: httpx.Response) -> None:
    assert response.status_code == 400, response.text
    error: Final = _JSON_OBJECT.validate_json(response.text)["error"]
    assert isinstance(error, dict) and "verbosity" in str(error["message"]), response.text


@pytest.mark.parametrize("verbosity", ["low", "high"])
def test_regression_explicit_bridge_sends_text_verbosity_as_top_level_verbosity(
    gateway: Gateway, verbosity: Verbosity
) -> None:
    marker: Final = uuid.uuid4().hex
    with _upstream() as wire, gateway.scenario() as scenario:
        model: Final = _bridged_gpt(scenario, wire)
        _answered(_responses(gateway, model, marker, text={"verbosity": verbosity}), marker)
        sent: Final = _only_post(wire, "/v1/chat/completions")
    assert sent == {"model": _GPT, "messages": _chat_messages(marker), "verbosity": verbosity}, sent


@pytest.mark.parametrize("verbosity", ["low", "high"])
def test_regression_explicit_bridge_stream_sends_text_verbosity_as_top_level_verbosity(
    gateway: Gateway, verbosity: Verbosity
) -> None:
    marker: Final = uuid.uuid4().hex
    with _upstream() as wire, gateway.scenario() as scenario:
        model: Final = _bridged_gpt(scenario, wire)
        _streamed_answer(_streamed_events(gateway, model, marker, text={"verbosity": verbosity}), marker)
        sent: Final = _only_post(wire, "/v1/chat/completions")
    assert sent == {
        "model": _GPT,
        "messages": _chat_messages(marker),
        "stream": True,
        **_STREAM_OPTIONS,
        "verbosity": verbosity,
    }, sent


def test_regression_explicit_bridge_sends_verbosity_beside_the_json_schema_response_format(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with _upstream() as wire, gateway.scenario() as scenario:
        model: Final = _bridged_gpt(scenario, wire)
        _answered(_responses(gateway, model, marker, text={"format": _TEXT_FORMAT, "verbosity": "low"}), marker)
        sent: Final = _only_post(wire, "/v1/chat/completions")
    assert sent == {
        "model": _GPT,
        "messages": _chat_messages(marker),
        "response_format": _RESPONSE_FORMAT,
        "verbosity": "low",
    }, sent


def test_regression_automatic_bridge_refuses_unsupported_verbosity_before_the_wire(gateway: Gateway) -> None:
    refused: Final = uuid.uuid4().hex
    control: Final = uuid.uuid4().hex
    with _upstream() as wire, gateway.scenario() as scenario:
        model: Final = _databricks(scenario, wire, drop_params=False)
        _unsupported_param_error(_responses(gateway, model, refused, text={"verbosity": "low"}))
        _answered(_responses(gateway, model, control), control)
        posted: Final = _posted(wire)
    assert [(request.target, _marker_of(request)) for request in posted] == [
        ("/serving-endpoints/chat/completions", control)
    ], posted


def test_regression_automatic_bridge_stream_refuses_unsupported_verbosity_before_the_wire(gateway: Gateway) -> None:
    refused: Final = uuid.uuid4().hex
    control: Final = uuid.uuid4().hex
    with _upstream() as wire, gateway.scenario() as scenario:
        model: Final = _databricks(scenario, wire, drop_params=False)
        _unsupported_param_error(_responses(gateway, model, refused, text={"verbosity": "high"}, stream=True))
        _answered(_responses(gateway, model, control), control)
        posted: Final = _posted(wire)
    assert [(request.target, _marker_of(request)) for request in posted] == [
        ("/serving-endpoints/chat/completions", control)
    ], posted


@pytest.mark.parametrize("source", ["deployment", "request"])
def test_regression_allowed_openai_params_forwards_verbosity_through_the_automatic_bridge(
    gateway: Gateway, source: Literal["deployment", "request"]
) -> None:
    marker: Final = uuid.uuid4().hex
    allowed: Final[dict[str, JsonValue]] = {"allowed_openai_params": ["verbosity"]}
    with _upstream() as wire, gateway.scenario() as scenario:
        model: Final = _databricks(scenario, wire, drop_params=False, **(allowed if source == "deployment" else {}))
        extra: Final = allowed if source == "request" else {}
        _answered(_responses(gateway, model, marker, text={"verbosity": "low"}, **extra), marker)
        sent: Final = _only_post(wire, "/serving-endpoints/chat/completions")
    assert sent == {"model": _DATABRICKS, "messages": _chat_messages(marker), "verbosity": "low"}, sent


@pytest.mark.parametrize("source", ["deployment", "request"])
def test_guard_drop_params_drops_unsupported_verbosity_and_keeps_the_json_schema_format(
    gateway: Gateway, source: Literal["deployment", "request"]
) -> None:
    marker: Final = uuid.uuid4().hex
    with _upstream() as wire, gateway.scenario() as scenario:
        model: Final = _databricks(scenario, wire, drop_params=source == "deployment")
        extra: Final[dict[str, JsonValue]] = {"drop_params": True} if source == "request" else {}
        _answered(
            _responses(gateway, model, marker, text={"format": _TEXT_FORMAT, "verbosity": "low"}, **extra), marker
        )
        sent: Final = _only_post(wire, "/serving-endpoints/chat/completions")
    assert sent == {"model": _DATABRICKS, "messages": _chat_messages(marker), "response_format": _RESPONSE_FORMAT}, sent


def test_guard_drop_params_stream_drops_unsupported_verbosity(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with _upstream() as wire, gateway.scenario() as scenario:
        model: Final = _databricks(scenario, wire, drop_params=True)
        _streamed_answer(_streamed_events(gateway, model, marker, text={"verbosity": "high"}), marker)
        sent: Final = _only_post(wire, "/serving-endpoints/chat/completions")
    assert sent == {"model": _DATABRICKS, "messages": _chat_messages(marker), "stream": True}, sent


def test_guard_explicit_bridge_without_verbosity_sends_only_the_json_schema_format(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with _upstream() as wire, gateway.scenario() as scenario:
        model: Final = _bridged_gpt(scenario, wire)
        _answered(_responses(gateway, model, marker, text={"format": _TEXT_FORMAT}), marker)
        sent: Final = _only_post(wire, "/v1/chat/completions")
    assert sent == {"model": _GPT, "messages": _chat_messages(marker), "response_format": _RESPONSE_FORMAT}, sent


def test_guard_explicit_bridge_without_text_sends_no_verbosity(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with _upstream() as wire, gateway.scenario() as scenario:
        model: Final = _bridged_gpt(scenario, wire)
        _answered(_responses(gateway, model, marker), marker)
        sent: Final = _only_post(wire, "/v1/chat/completions")
    assert sent == {"model": _GPT, "messages": _chat_messages(marker)}, sent


@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
def test_guard_native_responses_route_forwards_text_verbosity_unchanged(gateway: Gateway, stream: bool) -> None:
    marker: Final = uuid.uuid4().hex
    text: Final[dict[str, JsonValue]] = {"format": _TEXT_FORMAT, "verbosity": "high"}
    with _upstream() as wire, gateway.scenario() as scenario:
        model: Final = _native_gpt(scenario, wire)
        if stream:
            _streamed_answer(_streamed_events(gateway, model, marker, text=text), marker)
        else:
            _answered(_responses(gateway, model, marker, text=text), marker)
        sent: Final = _only_post(wire, "/v1/responses")
    assert sent == {"model": _GPT, "input": _prompt(marker), "text": text, **({"stream": True} if stream else {})}, sent


_CLIENT_TOOL_SEARCH: Final[dict[str, JsonValue]] = {"type": "tool_search", "execution": "client"}
_CLIENT_TOOL_SEARCH_ON_THE_WIRE: Final[dict[str, JsonValue]] = {
    "type": "function",
    "function": {
        "name": "tool_search",
        "description": (
            "Search the deferred tools by keyword. The matching tools are loaded and become callable on the next turn."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query for deferred tools."},
                "limit": {"type": "integer", "description": "Maximum number of tools to return. Defaults to 8."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True, slots=True)
class _FollowUpWire:
    sent: JsonValue
    expected: dict[str, JsonValue]


def _stored_prompts_config(directory: Path) -> Path:
    stock: Final = _JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    general: Final = _JSON_OBJECT.validate_python(stock["general_settings"])
    path: Final = directory / "verbosity-stored-turns.yaml"
    path.write_text(yaml.safe_dump({**stock, "general_settings": {**general, "store_prompts_in_spend_logs": True}}))
    return path


@pytest.fixture(scope="module")
def stored_turns_proxy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[OwnedProxy, Wire]]:
    directory: Final = tmp_path_factory.mktemp("verbosity-stored-turns")
    with _upstream() as wire, gateway_from_environment() as gateway:
        with owned_proxy_process(gateway, directory, {}, config=_stored_prompts_config(directory)) as owned:
            yield owned, wire


def _stored_first_turn(proxy: Gateway, model: str, marker: str) -> str:
    first: Final = _responses(proxy, model, marker, tools=[_CLIENT_TOOL_SEARCH])
    _answered(first, marker)
    response_id: Final = str(_JSON_OBJECT.validate_json(first.text)["id"])
    eventually(
        lambda: proxy.rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model,)),
        lambda rows: any(rv.same_response(str(row["request_id"]), response_id) for row in rows),
        seconds=70,
    )
    return response_id


def _client_tool_search_follow_up(
    stored_turns_proxy: tuple[OwnedProxy, Wire], stream: bool, **fields: JsonValue
) -> _FollowUpWire:
    owned, wire = stored_turns_proxy
    proxy: Final = owned.gateway
    first: Final = uuid.uuid4().hex
    second: Final = uuid.uuid4().hex
    with proxy.scenario() as scenario:
        model: Final = _bridged_gpt(scenario, wire)
        previous: Final = _stored_first_turn(proxy, model, first)
        wire.drain()
        follow_up: Final[dict[str, JsonValue]] = {
            "previous_response_id": previous,
            "tools": [_CLIENT_TOOL_SEARCH],
            **fields,
        }
        if stream:
            _streamed_answer(_streamed_events(proxy, model, second, **follow_up), second)
        else:
            _answered(_responses(proxy, model, second, **follow_up), second)
        sent: Final = _only_post(wire, "/v1/chat/completions")
    return _FollowUpWire(
        sent,
        {
            "model": _GPT,
            "messages": [
                *_chat_messages(first),
                {"role": "assistant", "content": _answer(first)},
                *_chat_messages(second),
            ],
            "tools": [_CLIENT_TOOL_SEARCH_ON_THE_WIRE],
            **({"stream": True, **_STREAM_OPTIONS} if stream else {}),
        },
    )


@pytest.mark.timeout(240)
@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
@pytest.mark.parametrize("verbosity", ["low", "high"])
def test_regression_client_tool_search_follow_up_sends_text_verbosity_as_top_level_verbosity(
    stored_turns_proxy: tuple[OwnedProxy, Wire], stream: bool, verbosity: Verbosity
) -> None:
    observed: Final = _client_tool_search_follow_up(stored_turns_proxy, stream, text={"verbosity": verbosity})
    assert observed.sent == {**observed.expected, "verbosity": verbosity}, observed.sent


@pytest.mark.timeout(240)
@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
def test_guard_client_tool_search_follow_up_without_verbosity_sends_no_verbosity(
    stored_turns_proxy: tuple[OwnedProxy, Wire], stream: bool
) -> None:
    observed: Final = _client_tool_search_follow_up(stored_turns_proxy, stream)
    assert observed.sent == observed.expected, observed.sent
