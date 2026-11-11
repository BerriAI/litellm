import asyncio
import itertools
import json
import re
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final, Literal
from urllib.parse import urlsplit

import anthropic
import httpx
import openai
import psutil
import pytest
import yaml
from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.openai_wire import chat_reply, responses_reply
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.responses_vendor import same_response
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = pytest.mark.shares_worker_state

_API_KEY: Final = "synthetic-routing-bearer"
_MANTLE_GPT: Final = "bedrock_mantle/openai.gpt-5.6-terra"
_MANTLE_OSS: Final = "bedrock_mantle/openai.gpt-oss-120b"
_OPENAI_GPT: Final = "openai/gpt-5.4"
_PINNED: Final = "routing-chat-pinned"
_SIBLING: Final = "routing-unpinned-sibling"
_NO_RETRIES: Final[Mapping[str, JsonValue]] = MappingProxyType({"num_retries": 0})
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MARKER: Final = re.compile(r"marker-([0-9a-f]{32})")
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_SPEND_BY_MODEL_GROUP: Final = 'SELECT request_id, status, model_group FROM "LiteLLM_SpendLogs" WHERE model_group = %s'
_N_REJECTED_BY_MANTLE: Final = "'n' is not supported for values greater than 1"
_FUNCTION_TOOL: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }
)

Api = Literal["chat", "responses"]
Client = Literal["openai-sdk", "openai-sdk-async-stream", "anthropic-sdk", "anthropic-sdk-async-stream", "httpx-stream"]
Endpoint = Literal["chat", "messages", "responses"]


@dataclass(frozen=True, slots=True)
class _Attempt:
    api: Api
    marker: str
    body: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class _Answer:
    status: int
    identity: str
    call_id: str
    text: str


@dataclass(frozen=True, slots=True)
class _Call:
    model: str
    endpoint: Endpoint
    stream: bool
    marker: str


def _question(marker: str) -> str:
    return f"Question marker-{marker}"


def _answer(marker: str) -> str:
    return f"answer marker-{marker}"


def _marker_in(text: str) -> str:
    found: Final = _MARKER.search(text)
    assert found is not None, text
    return found.group(1)


def _api(target: str) -> Api:
    if target.endswith("/chat/completions"):
        return "chat"
    assert target.endswith("/responses"), target
    return "responses"


def _error(status: int, message: str, kind: str) -> Reply:
    return Reply(status=status, body=json.dumps({"error": {"message": message, "type": kind, "code": None}}).encode())


def _upstream(down: threading.Event | None = None) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.method != "POST":
            return Reply(body=b'{"object": "list", "data": []}')
        if request.headers.get("authorization") != f"Bearer {_API_KEY}":
            return _error(401, "Invalid bearer token.", "authentication_error")
        if down is not None and down.is_set():
            return _error(503, "The server is overloaded.", "server_error")
        body: Final = _JSON_OBJECT.validate_json(request.body)
        marker: Final = _marker_in(request.body.decode())
        stream: Final = body.get("stream") is True
        backend: Final = string_value(body["model"])
        if _api(request.target) == "responses":
            return responses_reply(f"resp_{marker}", backend, _answer(marker), stream=stream)
        n: Final = body.get("n")
        if isinstance(n, int) and n > 1:
            return _error(400, _N_REJECTED_BY_MANTLE, "invalid_request_error")
        return chat_reply(f"chatcmpl-{marker}", backend, _answer(marker), stream=stream)

    return respond


def _is_probe(request: Request) -> bool:
    return request.method == "POST" and _MARKER.search(request.body.decode()) is None


def _attempts(wire: Wire, *, skip_probes: bool = False) -> tuple[_Attempt, ...]:
    return tuple(
        _Attempt(_api(request.target), _marker_in(request.body.decode()), _JSON_OBJECT.validate_json(request.body))
        for request in wire.drain()
        if request.method == "POST" and not (skip_probes and _is_probe(request))
    )


def _routes(attempts: tuple[_Attempt, ...]) -> Mapping[str, tuple[Api, ...]]:
    markers: Final = frozenset(attempt.marker for attempt in attempts)
    return {marker: tuple(attempt.api for attempt in attempts if attempt.marker == marker) for marker in markers}


def _api_base(wire: Wire, backend: str) -> str:
    return wire.url + ("/openai/v1" if backend.startswith("bedrock_mantle/") else "/v1")


def _deployment(
    scenario: Scenario,
    wire: Wire,
    backend: str,
    mode: str | None,
    *,
    provider_pin: bool = True,
    api_key: str = _API_KEY,
) -> str:
    provider: Final[Mapping[str, JsonValue]] = {"custom_llm_provider": backend.split("/", 1)[0]} if provider_pin else {}
    return scenario.model(
        model=backend,
        api_base=_api_base(wire, backend),
        api_key=api_key,
        model_info=None if mode is None else {"mode": mode},
        **provider,
    )


def _wildcard(scenario: Scenario, wire: Wire, mode: str | None) -> str:
    prefix: Final = f"wild-{uuid.uuid4().hex}"
    created: Final = scenario.gateway.post(
        "/model/new",
        {
            "model_name": f"{prefix}/*",
            "litellm_params": {"model": "bedrock_mantle/*", "api_base": wire.url + "/openai/v1", "api_key": _API_KEY},
            "model_info": {} if mode is None else {"mode": mode},
        },
    )
    scenario.cleanups.callback(scenario.delete_model, string_value(object_value(created["model_info"])["id"]))
    return f"{prefix}/{_MANTLE_GPT.removeprefix('bedrock_mantle/')}"


def _raw_model_info_deployment(scenario: Scenario, wire: Wire, raw_model_info: str) -> str:
    name: Final = f"integration-{uuid.uuid4().hex}"
    params: Final = json.dumps(
        {
            "model": _MANTLE_GPT,
            "custom_llm_provider": "bedrock_mantle",
            "api_base": wire.url + "/openai/v1",
            "api_key": _API_KEY,
        }
    )
    response: Final = scenario.gateway.client.post(
        "/model/new",
        content=f'{{"model_name": "{name}", "litellm_params": {params}, "model_info": {raw_model_info}}}',
        headers={"Authorization": f"Bearer {scenario.gateway.key}", "Content-Type": "application/json"},
    )
    assert response.status_code == 200, response.text
    identity: Final = string_value(object_value(_JSON_OBJECT.validate_json(response.content)["model_info"])["id"])
    scenario.cleanups.callback(scenario.delete_model, identity)
    return name


def _proxy_url(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/")


def _messages(marker: str) -> list[JsonValue]:
    return [{"role": "user", "content": _question(marker)}]


def _identity(endpoint: Endpoint, stream: bool, text: str) -> str:
    if not stream:
        return string_value(_JSON_OBJECT.validate_json(text)["id"])
    first: Final = _JSON_OBJECT.validate_json(
        next(line for line in text.splitlines() if line.startswith("data: {")).removeprefix("data: ")
    )
    match endpoint:
        case "chat":
            return string_value(first["id"])
        case "messages":
            return string_value(object_value(first["message"])["id"])
        case "responses":
            return string_value(object_value(first["response"])["id"])


def _body(call: _Call, extra: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    match call.endpoint:
        case "chat":
            return {"model": call.model, "messages": _messages(call.marker), "stream": call.stream, **extra}
        case "messages":
            return {"model": call.model, "max_tokens": 64, "messages": _messages(call.marker), "stream": call.stream}
        case "responses":
            return {"model": call.model, "input": _question(call.marker), "stream": call.stream, **extra}


def _path(endpoint: Endpoint) -> str:
    match endpoint:
        case "chat":
            return "/v1/chat/completions"
        case "messages":
            return "/v1/messages"
        case "responses":
            return "/v1/responses"


async def _send(client: httpx.AsyncClient, key: str, call: _Call, extra: Mapping[str, JsonValue]) -> _Answer:
    async with client.stream(
        "POST",
        _path(call.endpoint),
        json=_body(call, extra),
        headers={"Authorization": f"Bearer {key}", "anthropic-version": "2023-06-01"},
    ) as response:
        text: Final = (await response.aread()).decode()
    identity: Final = _identity(call.endpoint, call.stream, text) if response.status_code == 200 else ""
    call_id: Final = response.headers["x-litellm-call-id"] if "x-litellm-call-id" in response.headers else ""
    return _Answer(response.status_code, identity, call_id, text)


async def _http(gateway: Gateway, call: _Call, extra: Mapping[str, JsonValue] = _NO_RETRIES) -> _Answer:
    async with httpx.AsyncClient(base_url=_proxy_url(gateway), timeout=60, trust_env=False) as client:
        return await _send(client, gateway.key, call, extra)


async def _burst(
    base_url: str, key: str, calls: tuple[_Call, ...], *, tolerate_drops: bool = False
) -> tuple[_Answer | None, ...]:
    async with httpx.AsyncClient(base_url=base_url, timeout=90, trust_env=False) as client:
        results: Final = await asyncio.gather(
            *(_send(client, key, call, _NO_RETRIES) for call in calls), return_exceptions=tolerate_drops
        )
    for result in results:
        assert not isinstance(result, BaseException) or isinstance(result, httpx.TransportError), repr(result)
    return tuple(result if isinstance(result, _Answer) else None for result in results)


def _openai_sdk(gateway: Gateway, model: str, marker: str) -> _Answer:
    with openai.OpenAI(base_url=f"{_proxy_url(gateway)}/v1", api_key=gateway.key, max_retries=0) as sdk:
        completion: Final = sdk.chat.completions.create(
            model=model, messages=[{"role": "user", "content": _question(marker)}]
        )
    return _Answer(200, completion.id, "", completion.choices[0].message.content or "")


async def _openai_sdk_async_stream(gateway: Gateway, model: str, marker: str) -> _Answer:
    async with openai.AsyncOpenAI(base_url=f"{_proxy_url(gateway)}/v1", api_key=gateway.key, max_retries=0) as sdk:
        stream: Final = await sdk.chat.completions.create(
            model=model, messages=[{"role": "user", "content": _question(marker)}], stream=True
        )
        chunks: Final = [chunk async for chunk in stream]
    (identity,) = {chunk.id for chunk in chunks}
    return _Answer(
        200, identity, "", "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)
    )


def _anthropic_sdk(gateway: Gateway, model: str, marker: str) -> _Answer:
    with anthropic.Anthropic(base_url=_proxy_url(gateway), api_key=gateway.key, max_retries=0) as sdk:
        message: Final = sdk.messages.create(
            model=model, max_tokens=64, messages=[{"role": "user", "content": _question(marker)}]
        )
    return _Answer(200, message.id, "", "".join(block.text for block in message.content if block.type == "text"))


async def _anthropic_sdk_async_stream(gateway: Gateway, model: str, marker: str) -> _Answer:
    async with anthropic.AsyncAnthropic(base_url=_proxy_url(gateway), api_key=gateway.key, max_retries=0) as sdk:
        async with sdk.messages.stream(
            model=model, max_tokens=64, messages=[{"role": "user", "content": _question(marker)}]
        ) as stream:
            pieces: Final = [piece async for piece in stream.text_stream]
            message: Final = await stream.get_final_message()
    return _Answer(200, message.id, "", "".join(pieces))


async def _ask(gateway: Gateway, client: Client, model: str, marker: str) -> _Answer:
    match client:
        case "openai-sdk":
            return _openai_sdk(gateway, model, marker)
        case "openai-sdk-async-stream":
            return await _openai_sdk_async_stream(gateway, model, marker)
        case "anthropic-sdk":
            return _anthropic_sdk(gateway, model, marker)
        case "anthropic-sdk-async-stream":
            return await _anthropic_sdk_async_stream(gateway, model, marker)
        case "httpx-stream":
            return await _http(gateway, _Call(model, "chat", True, marker))


def _logged_ids(answer: _Answer) -> tuple[str, ...]:
    markers: Final = frozenset(_MARKER.findall(answer.text))
    upstream: Final = (*(f"resp_{marker}" for marker in markers), *(f"chatcmpl-{marker}" for marker in markers))
    return tuple(shown for shown in (answer.identity, answer.call_id, *upstream) if shown)


def _logs_answer(request_id: JsonValue, answer: _Answer) -> bool:
    return any(same_response(string_value(request_id), shown) for shown in _logged_ids(answer))


def _logged_rows(answer: _Answer, model: str) -> list[dict[str, JsonValue]]:
    return [row for row in read_rows(_SPEND_BY_MODEL_GROUP, (model,)) if _logs_answer(row["request_id"], answer)]


def _assert_logged(answer: _Answer, model: str, status: str) -> None:
    rows: Final = eventually(lambda: _logged_rows(answer, model), lambda found: len(found) >= 1, seconds=70)
    assert len(rows) == 1, rows
    assert (rows[0]["status"], rows[0]["model_group"]) == (status, model), rows


def _assert_answered(answer: _Answer | None, marker: str) -> _Answer:
    assert answer is not None, marker
    assert answer.status == 200, answer.text
    assert _answer(marker) in answer.text, answer.text
    assert set(_MARKER.findall(answer.text)) == {marker}, answer.text
    return answer


def _waves(model: str, count: int) -> tuple[_Call, ...]:
    endpoints: Final[tuple[Endpoint, ...]] = ("chat", "messages", "responses")
    return tuple(
        _Call(model, endpoints[index % len(endpoints)], index % 2 == 0, uuid.uuid4().hex) for index in range(count)
    )


def _expected_route(call: _Call, pinned: str) -> Api:
    return "chat" if call.model == pinned and call.endpoint != "responses" else "responses"


@pytest.mark.parametrize(
    "client", ("openai-sdk", "openai-sdk-async-stream", "anthropic-sdk", "anthropic-sdk-async-stream", "httpx-stream")
)
async def test_a_chat_pin_leaves_its_unpinned_sibling_on_responses_on_every_client(
    gateway: Gateway, client: Client
) -> None:
    pinned_marker: Final = uuid.uuid4().hex
    sibling_marker: Final = uuid.uuid4().hex
    with wire_server(_upstream()) as wire, gateway.scenario() as scenario:
        pinned: Final = _deployment(scenario, wire, _MANTLE_GPT, "chat")
        sibling: Final = _deployment(scenario, wire, _MANTLE_GPT, None)
        pinned_answer: Final = _assert_answered(await _ask(gateway, client, pinned, pinned_marker), pinned_marker)
        sibling_answer: Final = _assert_answered(await _ask(gateway, client, sibling, sibling_marker), sibling_marker)
        assert _routes(_attempts(wire)) == {pinned_marker: ("chat",), sibling_marker: ("responses",)}
        _assert_logged(pinned_answer, pinned, "success")
        _assert_logged(sibling_answer, sibling, "success")


@pytest.mark.parametrize("backend", (_MANTLE_OSS, _OPENAI_GPT))
async def test_a_responses_pin_leaves_its_unpinned_sibling_on_chat(gateway: Gateway, backend: str) -> None:
    pinned_marker: Final = uuid.uuid4().hex
    sibling_marker: Final = uuid.uuid4().hex
    with wire_server(_upstream()) as wire, gateway.scenario() as scenario:
        pinned: Final = _deployment(scenario, wire, backend, "responses")
        sibling: Final = _deployment(scenario, wire, backend, None)
        pinned_answer: Final = _assert_answered(
            await _http(gateway, _Call(pinned, "chat", False, pinned_marker)), pinned_marker
        )
        sibling_answer: Final = _assert_answered(
            await _http(gateway, _Call(sibling, "chat", False, sibling_marker)), sibling_marker
        )
        assert _routes(_attempts(wire)) == {pinned_marker: ("responses",), sibling_marker: ("chat",)}
        _assert_logged(pinned_answer, pinned, "success")
        _assert_logged(sibling_answer, sibling, "success")


@pytest.mark.parametrize("stream", (False, True))
async def test_the_responses_endpoint_serves_a_chat_pinned_deployment_natively(gateway: Gateway, stream: bool) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_upstream()) as wire, gateway.scenario() as scenario:
        pinned: Final = _deployment(scenario, wire, _MANTLE_GPT, "chat")
        answer: Final = _assert_answered(await _http(gateway, _Call(pinned, "responses", stream, marker)), marker)
        assert _routes(_attempts(wire)) == {marker: ("responses",)}
        _assert_logged(answer, pinned, "success")


async def test_a_wildcard_route_keeps_the_registered_mode_beside_a_chat_pin(gateway: Gateway) -> None:
    pinned_marker: Final = uuid.uuid4().hex
    wildcard_marker: Final = uuid.uuid4().hex
    with wire_server(_upstream()) as wire, gateway.scenario() as scenario:
        pinned: Final = _deployment(scenario, wire, _MANTLE_GPT, "chat")
        wildcard: Final = _wildcard(scenario, wire, None)
        _assert_answered(await _http(gateway, _Call(pinned, "chat", False, pinned_marker)), pinned_marker)
        _assert_answered(await _http(gateway, _Call(wildcard, "chat", False, wildcard_marker)), wildcard_marker)
        assert _routes(_attempts(wire)) == {pinned_marker: ("chat",), wildcard_marker: ("responses",)}


@pytest.mark.parametrize("shape", ("wildcard", "no-provider-pin"))
async def test_a_chat_pin_without_a_provider_pin_gets_chat_completions(gateway: Gateway, shape: str) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_upstream()) as wire, gateway.scenario() as scenario:
        model: Final = (
            _wildcard(scenario, wire, "chat")
            if shape == "wildcard"
            else _deployment(scenario, wire, _MANTLE_GPT, "chat", provider_pin=False)
        )
        _assert_answered(await _http(gateway, _Call(model, "chat", False, marker)), marker)
        assert _routes(_attempts(wire)) == {marker: ("chat",)}


async def test_an_explicit_responses_prefix_wins_over_a_chat_pin(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_upstream()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/responses/gpt-5.4", api_base=wire.url + "/v1", api_key=_API_KEY, model_info={"mode": "chat"}
        )
        answer: Final = _assert_answered(await _http(gateway, _Call(model, "chat", False, marker)), marker)
        (attempt,) = _attempts(wire)
        assert (attempt.api, attempt.body["model"]) == ("responses", "gpt-5.4"), attempt
        _assert_logged(answer, model, "success")


_SUMMARY_REASONING: Final[Mapping[str, JsonValue]] = MappingProxyType({"effort": "low", "summary": "auto"})


@pytest.mark.parametrize(
    ("extra", "reasoning"),
    (
        ({"tools": [dict(_FUNCTION_TOOL)]}, None),
        ({"reasoning_effort": dict(_SUMMARY_REASONING)}, _SUMMARY_REASONING),
        ({"extra_body": {"reasoning_effort": dict(_SUMMARY_REASONING)}}, _SUMMARY_REASONING),
        ({"web_search_options": {"search_context_size": "low"}}, None),
    ),
    ids=("function-tools", "reasoning-summary", "extra-body-reasoning-summary", "web-search"),
)
async def test_a_mantle_chat_pin_sends_what_chat_cannot_serve_to_responses(
    gateway: Gateway, extra: Mapping[str, JsonValue], reasoning: Mapping[str, JsonValue] | None
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_upstream()) as wire, gateway.scenario() as scenario:
        pinned: Final = _deployment(scenario, wire, _MANTLE_GPT, "chat")
        answer: Final = _assert_answered(
            await _http(gateway, _Call(pinned, "chat", False, marker), {**_NO_RETRIES, **extra}), marker
        )
        (attempt,) = _attempts(wire)
        assert attempt.api == "responses", attempt
        assert reasoning is None or attempt.body["reasoning"] == reasoning, attempt.body
        _assert_logged(answer, pinned, "success")


def _placed(placement: str, params: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    return params if placement == "top-level" else {"extra_body": dict(params)}


async def _messages_with_thinking(
    gateway: Gateway, model: str, marker: str, thinking: Mapping[str, JsonValue]
) -> _Answer:
    async with httpx.AsyncClient(base_url=_proxy_url(gateway), timeout=60, trust_env=False) as client:
        response: Final = await client.post(
            "/v1/messages",
            json={"model": model, "max_tokens": 2048, "thinking": dict(thinking), "messages": _messages(marker)},
            headers={"Authorization": f"Bearer {gateway.key}", "anthropic-version": "2023-06-01"},
        )
    identity: Final = _identity("messages", False, response.text) if response.status_code == 200 else ""
    return _Answer(response.status_code, identity, response.headers.get("x-litellm-call-id", ""), response.text)


@pytest.mark.parametrize(("summary", "route"), ((None, "chat"), ("auto", "responses")), ids=("plain", "summary"))
async def test_a_mantle_chat_pin_moves_messages_thinking_to_responses_only_when_it_asks_for_a_summary(
    gateway: Gateway, summary: str | None, route: Api
) -> None:
    marker: Final = uuid.uuid4().hex
    thinking: Final[Mapping[str, JsonValue]] = {"type": "enabled", "budget_tokens": 1024} | (
        {} if summary is None else {"summary": summary}
    )
    with wire_server(_upstream()) as wire, gateway.scenario() as scenario:
        pinned: Final = _deployment(scenario, wire, _MANTLE_GPT, "chat")
        answer: Final = _assert_answered(await _messages_with_thinking(gateway, pinned, marker, thinking), marker)
        (attempt,) = _attempts(wire)
        assert attempt.api == route, attempt
        match route:
            case "chat":
                assert "summary" not in json.dumps(attempt.body), attempt.body
            case "responses":
                assert object_value(attempt.body["reasoning"])["summary"] == summary, attempt.body
        _assert_logged(answer, pinned, "success")


def _sent_effort(attempt: _Attempt) -> JsonValue:
    match attempt.api:
        case "chat":
            return attempt.body.get("reasoning_effort")
        case "responses":
            return object_value(attempt.body["reasoning"]).get("effort")


@pytest.mark.parametrize(
    ("backend", "mode", "effort", "route"),
    ((_OPENAI_GPT, None, "low", "responses"), (_MANTLE_GPT, "chat", "none", "chat")),
    ids=("openai-low", "mantle-chat-pin-none"),
)
@pytest.mark.parametrize("placement", ("top-level", "extra-body"))
async def test_an_effort_inside_extra_body_routes_function_tools_like_a_top_level_one(
    gateway: Gateway, backend: str, mode: str | None, effort: str, route: Api, placement: str
) -> None:
    marker: Final = uuid.uuid4().hex
    extra: Final[Mapping[str, JsonValue]] = {
        **_NO_RETRIES,
        "tools": [dict(_FUNCTION_TOOL)],
        **_placed(placement, {"reasoning_effort": effort}),
    }
    with wire_server(_upstream()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, backend, mode)
        answer: Final = _assert_answered(await _http(gateway, _Call(model, "chat", False, marker), extra), marker)
        (attempt,) = _attempts(wire)
        assert (attempt.api, _sent_effort(attempt)) == (route, effort), attempt
        _assert_logged(answer, model, "success")


_AUTO_SUMMARY_LEVELS: Final[Mapping[str, str]] = MappingProxyType(
    {"auto-summary-mantle": "auto", "auto-summary-openai": "detailed"}
)
_PLAIN_THINKING: Final[Mapping[str, JsonValue]] = MappingProxyType({"type": "enabled", "budget_tokens": 1024})


def _auto_summary_config(wire: Wire, tmp_path: Path) -> Path:
    base: Final = _JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    mantle: Final[dict[str, JsonValue]] = {
        "model": _MANTLE_GPT,
        "custom_llm_provider": "bedrock_mantle",
        "api_base": wire.url + "/openai/v1",
        "api_key": _API_KEY,
    }
    openai_codex: Final[dict[str, JsonValue]] = {
        "model": "openai/gpt-5.3-codex",
        "api_base": wire.url + "/v1",
        "api_key": _API_KEY,
    }
    config: Final = {
        **base,
        "model_list": [
            {"model_name": "auto-summary-mantle", "litellm_params": mantle},
            {"model_name": "auto-summary-openai", "litellm_params": openai_codex},
        ],
        "router_settings": {**object_value(base["router_settings"]), "num_retries": 0},
    }
    path: Final = tmp_path / "deployment-mode-routing-auto-summary.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


async def _ask_for_reasoning(gateway: Gateway, call: _Call) -> _Answer:
    if call.endpoint == "messages":
        return await _messages_with_thinking(gateway, call.model, call.marker, _PLAIN_THINKING)
    return await _http(gateway, call, {**_NO_RETRIES, "reasoning_effort": "low"})


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
async def test_the_auto_summary_asks_each_target_for_a_level_it_accepts_on_every_endpoint(
    gateway: Gateway, tmp_path: Path
) -> None:
    endpoints: Final[tuple[Endpoint, ...]] = ("chat", "messages", "responses")
    calls: Final = tuple(
        _Call(model, endpoint, False, uuid.uuid4().hex)
        for model, endpoint in itertools.product(_AUTO_SUMMARY_LEVELS, endpoints)
    )
    with wire_server(_upstream()) as wire:
        config: Final = _auto_summary_config(wire, tmp_path)
        with owned_proxy_process(gateway, tmp_path, {"LITELLM_REASONING_AUTO_SUMMARY": "true"}, config=config) as owned:
            for call in calls:
                _assert_answered(await _ask_for_reasoning(owned.gateway, call), call.marker)
        sent: Final = {
            attempt.marker: (attempt.api, object_value(attempt.body["reasoning"]).get("summary"))
            for attempt in _attempts(wire)
        }
    assert sent == {call.marker: ("responses", _AUTO_SUMMARY_LEVELS[call.model]) for call in calls}


def _health_upstream(request: Request) -> Reply:
    if request.method != "POST":
        return Reply(body=b'{"object": "list", "data": []}')
    backend: Final = string_value(_JSON_OBJECT.validate_json(request.body)["model"])
    if _api(request.target) == "responses":
        return responses_reply("resp_health", backend, "ok", stream=False)
    return chat_reply("chatcmpl-health", backend, "ok", stream=False)


@pytest.mark.parametrize(("mode", "route"), (("chat", "chat"), (None, "responses")), ids=("chat-pin", "unpinned"))
def test_a_health_check_probes_the_api_its_deployment_sends_chat_to(
    gateway: Gateway, mode: str | None, route: Api
) -> None:
    with wire_server(_health_upstream) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, _MANTLE_GPT, mode)
        response: Final = gateway.request("GET", "/health", params={"model": model})
        probed: Final = tuple(_api(request.target) for request in wire.drain() if request.method == "POST")
    assert response.status_code == 200, response.text
    assert _JSON_OBJECT.validate_json(response.content)["healthy_count"] == 1, response.text
    assert probed == (route,), probed


_INPUT_PRICE: Final = 0.001
_OUTPUT_PRICE: Final = 0.002
_SPEND_OF_MODEL_GROUP: Final = 'SELECT request_id, spend FROM "LiteLLM_SpendLogs" WHERE model_group = %s'


def _priced_chat_pin(scenario: Scenario, wire: Wire) -> str:
    return scenario.model(
        model=_MANTLE_GPT,
        custom_llm_provider="bedrock_mantle",
        api_base=_api_base(wire, _MANTLE_GPT),
        api_key=_API_KEY,
        input_cost_per_token=_INPUT_PRICE,
        output_cost_per_token=_OUTPUT_PRICE,
        model_info={"mode": "chat"},
    )


def _probe_or_marked(request: Request) -> Reply:
    if _is_probe(request):
        return _health_upstream(request)
    return _upstream()(request)


async def _answered_burst(gateway: Gateway, model: str, count: int) -> tuple[tuple[_Call, _Answer], ...]:
    calls: Final = tuple(_Call(model, "chat", False, uuid.uuid4().hex) for _ in range(count))
    answers: Final = await _burst(_proxy_url(gateway), gateway.key, calls)
    return tuple((call, _assert_answered(answer, call.marker)) for call, answer in zip(calls, answers, strict=True))


async def test_a_custom_priced_chat_pin_bills_its_price_and_leaves_its_unpinned_sibling_on_responses(
    gateway: Gateway,
) -> None:
    with wire_server(_upstream()) as wire, gateway.scenario() as scenario:
        pinned: Final = _priced_chat_pin(scenario, wire)
        sibling: Final = _deployment(scenario, wire, _MANTLE_GPT, None)
        pinned_calls: Final = await _answered_burst(gateway, pinned, 8)
        sibling_calls: Final = await _answered_burst(gateway, sibling, 8)
        assert _routes(_attempts(wire)) == {call.marker: ("chat",) for call, _ in pinned_calls} | {
            call.marker: ("responses",) for call, _ in sibling_calls
        }
        _, answer = pinned_calls[0]
        rows: Final = eventually(
            lambda: [
                row for row in read_rows(_SPEND_OF_MODEL_GROUP, (pinned,)) if _logs_answer(row["request_id"], answer)
            ],
            lambda found: len(found) == 1,
            seconds=70,
        )
        assert float(str(rows[0]["spend"])) == pytest.approx(5 * _INPUT_PRICE + 3 * _OUTPUT_PRICE), rows


async def test_a_health_probe_of_a_custom_priced_chat_pin_leaves_its_unpinned_sibling_on_responses(
    gateway: Gateway,
) -> None:
    with wire_server(_probe_or_marked) as wire, gateway.scenario() as scenario:
        pinned: Final = _priced_chat_pin(scenario, wire)
        sibling: Final = _deployment(scenario, wire, _MANTLE_GPT, None)
        async with httpx.AsyncClient(base_url=_proxy_url(gateway), timeout=60, trust_env=False) as client:
            probes: Final = await asyncio.gather(
                *(
                    client.get("/health", params={"model": pinned}, headers={"Authorization": f"Bearer {gateway.key}"})
                    for _ in range(6)
                )
            )
        probed: Final = tuple(
            _api(request.target)
            for request in wire.drain()
            if request.method == "POST" and _MARKER.search(request.body.decode()) is None
        )
        sibling_calls: Final = await _answered_burst(gateway, sibling, 8)
        assert _routes(_attempts(wire)) == {call.marker: ("responses",) for call, _ in sibling_calls}
    assert [(probe.status_code, _JSON_OBJECT.validate_json(probe.content)["healthy_count"]) for probe in probes] == [
        (200, 1)
    ] * 6, [probe.text for probe in probes]
    assert probed == ("chat",) * 6, probed


_PROBED: Final = "routing-probed-deployment"
_BACKGROUND_HEALTH_CHECKS: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"background_health_checks": True, "health_check_interval": 1}
)


def _failing_probes(probes: SimpleQueue[Api]) -> Callable[[Request], Reply]:
    respond: Final = _upstream()

    def fail_probe(request: Request) -> Reply:
        if not _is_probe(request):
            return respond(request)
        probes.put(_api(request.target))
        return _error(404, "The model `openai.gpt-5.6-terra` does not exist.", "invalid_request_error")

    return fail_probe


def _probed_deployment_config(wire: Wire, tmp_path: Path, mode: str | None, general: Mapping[str, JsonValue]) -> Path:
    base: Final = _JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    params: Final[dict[str, JsonValue]] = {
        "model": _MANTLE_GPT,
        "custom_llm_provider": "bedrock_mantle",
        "api_base": wire.url + "/openai/v1",
        "api_key": _API_KEY,
    }
    pin: Final[dict[str, JsonValue]] = {} if mode is None else {"model_info": {"mode": mode}}
    config: Final = {
        **base,
        "model_list": [{"model_name": _PROBED, "litellm_params": params, **pin}],
        "general_settings": {**object_value(base["general_settings"]), "store_model_in_db": False, **general},
        "router_settings": {
            **object_value(base["router_settings"]),
            "disable_cooldowns": False,
            "cooldown_time": 60,
            "num_retries": 0,
        },
    }
    path: Final = tmp_path / "deployment-mode-routing-failing-probe.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _drained(probes: SimpleQueue[Api]) -> tuple[Api, ...]:
    return tuple(probes.get() for _ in range(probes.qsize()))


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
@pytest.mark.parametrize(("mode", "route"), (("chat", "chat"), (None, "responses")), ids=("chat-pin", "unpinned"))
async def test_a_failing_health_probe_leaves_its_deployment_serving_the_next_requests(
    gateway: Gateway, tmp_path: Path, mode: str | None, route: Api
) -> None:
    probes: Final[SimpleQueue[Api]] = SimpleQueue()
    calls: Final = tuple(_Call(_PROBED, "chat", False, uuid.uuid4().hex) for _ in range(4))
    with wire_server(_failing_probes(probes)) as wire:
        config: Final = _probed_deployment_config(wire, tmp_path, mode, {})
        with owned_proxy_process(gateway, tmp_path, {}, config=config) as owned:
            health: Final = owned.gateway.request("GET", "/health", params={"model": _PROBED})
            answers: Final = [await _http(owned.gateway, call) for call in calls]
        served: Final = _routes(_attempts(wire, skip_probes=True))
    counts: Final = _JSON_OBJECT.validate_json(health.content)
    assert (counts["healthy_count"], counts["unhealthy_count"]) == (0, 1), health.text
    assert _drained(probes) == (route,)
    for call, answer in zip(calls, answers, strict=True):
        _assert_answered(answer, call.marker)
    assert served == {call.marker: (route,) for call in calls}


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
async def test_failing_background_health_probes_leave_a_chat_pinned_deployment_serving_requests(
    gateway: Gateway, tmp_path: Path
) -> None:
    probes: Final[SimpleQueue[Api]] = SimpleQueue()
    calls: Final = tuple(_Call(_PROBED, "chat", False, uuid.uuid4().hex) for _ in range(4))
    with wire_server(_failing_probes(probes)) as wire:
        config: Final = _probed_deployment_config(wire, tmp_path, "chat", _BACKGROUND_HEALTH_CHECKS)
        with owned_proxy_process(gateway, tmp_path, {}, config=config) as owned:
            await asyncio.to_thread(eventually, probes.qsize, lambda size: size >= 2, 60)
            answers: Final = [await _http(owned.gateway, call) for call in calls]
        served: Final = _routes(_attempts(wire, skip_probes=True))
    probed: Final = _drained(probes)
    assert len(probed) >= 2 and set(probed) == {"chat"}, probed
    for call, answer in zip(calls, answers, strict=True):
        _assert_answered(answer, call.marker)
    assert served == {call.marker: ("chat",) for call in calls}


@pytest.mark.parametrize(
    "extra_body",
    (
        {"reasoning_effort": "low", "reasoningSummary": "auto"},
        {"reasoning_effort": {"effort": "low"}, "reasoning_summary": "auto"},
    ),
    ids=("string-effort", "object-effort"),
)
@pytest.mark.parametrize(
    ("backend", "mode"), ((_OPENAI_GPT, None), (_MANTLE_GPT, "chat")), ids=("openai", "mantle-chat-pin")
)
async def test_a_summary_alias_beside_an_extra_body_effort_reaches_the_responses_api(
    gateway: Gateway, extra_body: Mapping[str, JsonValue], backend: str, mode: str | None
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_upstream()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, backend, mode)
        answer: Final = _assert_answered(
            await _http(gateway, _Call(model, "chat", False, marker), {**_NO_RETRIES, "extra_body": dict(extra_body)}),
            marker,
        )
        (attempt,) = _attempts(wire)
        assert (attempt.api, attempt.body.get("reasoning")) == ("responses", {"effort": "low", "summary": "auto"}), (
            attempt
        )
        _assert_logged(answer, model, "success")


@pytest.mark.parametrize("placement", ("top-level", "extra-body"))
async def test_a_mantle_chat_pin_rejects_n_above_one_before_the_upstream(gateway: Gateway, placement: str) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_upstream()) as wire, gateway.scenario() as scenario:
        pinned: Final = _deployment(scenario, wire, _MANTLE_GPT, "chat")
        answer: Final = await _http(
            gateway, _Call(pinned, "chat", False, marker), {**_NO_RETRIES, **_placed(placement, {"n": 2})}
        )
        assert answer.status == 400, answer.text
        assert "does not support n=2" in answer.text and "drop_params" in answer.text, answer.text
        assert _attempts(wire) == ()
        _assert_logged(answer, pinned, "failure")


@pytest.mark.parametrize("placement", ("top-level", "extra-body"))
async def test_a_mantle_chat_pin_drops_n_above_one_under_drop_params(gateway: Gateway, placement: str) -> None:
    marker: Final = uuid.uuid4().hex
    extra: Final[Mapping[str, JsonValue]] = {**_NO_RETRIES, **_placed(placement, {"n": 2}), "drop_params": True}
    with wire_server(_upstream()) as wire, gateway.scenario() as scenario:
        pinned: Final = _deployment(scenario, wire, _MANTLE_GPT, "chat")
        answer: Final = _assert_answered(await _http(gateway, _Call(pinned, "chat", False, marker), extra), marker)
        (attempt,) = _attempts(wire)
        assert attempt.api == "chat" and "n" not in attempt.body, attempt
        _assert_logged(answer, pinned, "success")


@pytest.mark.parametrize("placement", ("top-level", "extra-body"))
async def test_a_no_provider_mantle_chat_pin_sends_an_effort_object_as_its_level(
    gateway: Gateway, placement: str
) -> None:
    marker: Final = uuid.uuid4().hex
    effort: Final[Mapping[str, JsonValue]] = {"reasoning_effort": {"effort": "low"}}
    extra: Final[Mapping[str, JsonValue]] = {**_NO_RETRIES, **_placed(placement, effort)}
    with wire_server(_upstream()) as wire, gateway.scenario() as scenario:
        pinned: Final = _deployment(scenario, wire, _MANTLE_GPT, "chat", provider_pin=False)
        answer: Final = _assert_answered(await _http(gateway, _Call(pinned, "chat", False, marker), extra), marker)
        (attempt,) = _attempts(wire)
        assert (attempt.api, attempt.body.get("reasoning_effort")) == ("chat", "low"), attempt
        _assert_logged(answer, pinned, "success")


@pytest.mark.parametrize(
    ("raw_model_info", "own_route"),
    (
        ('{"mode": ""}', "responses"),
        ('{"mode": "embedding"}', "responses"),
        ('{"mode": null}', "responses"),
        ('{"mode": 5}', None),
        ('{"mode": ["chat"]}', None),
        ('{"mode": "' + "c" * 5000 + '"}', "responses"),
        ('{"mode": "responses", "mode": "chat"}', "chat"),
    ),
    ids=("empty", "unroutable", "null", "int", "list", "5kb", "twice"),
)
async def test_a_malformed_mode_never_moves_an_unpinned_sibling(
    gateway: Gateway, raw_model_info: str, own_route: Api | None
) -> None:
    odd_marker: Final = uuid.uuid4().hex
    sibling_markers: Final = tuple(uuid.uuid4().hex for _ in range(4))
    with wire_server(_upstream()) as wire, gateway.scenario() as scenario:
        odd: Final = _raw_model_info_deployment(scenario, wire, raw_model_info)
        sibling: Final = _deployment(scenario, wire, _MANTLE_GPT, None)
        if own_route is not None:
            _assert_answered(await _http(gateway, _Call(odd, "chat", False, odd_marker)), odd_marker)
        for marker in sibling_markers:
            _assert_answered(await _http(gateway, _Call(sibling, "chat", False, marker)), marker)
        own: Final[dict[str, tuple[Api, ...]]] = {} if own_route is None else {odd_marker: (own_route,)}
        assert _routes(_attempts(wire)) == own | {marker: ("responses",) for marker in sibling_markers}


async def test_a_bridged_sibling_reports_the_upstream_401_after_one_attempt(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_upstream()) as wire, gateway.scenario() as scenario:
        _deployment(scenario, wire, _MANTLE_GPT, "chat")
        sibling: Final = _deployment(scenario, wire, _MANTLE_GPT, None, api_key="wrong-bearer")
        answer: Final = await _http(gateway, _Call(sibling, "chat", False, marker))
        assert answer.status == 401, answer.text
        assert "Invalid bearer token." in answer.text, answer.text
        assert _routes(_attempts(wire)) == {marker: ("responses",)}
        _assert_logged(answer, sibling, "failure")


async def test_adding_and_deleting_a_chat_pin_under_traffic_never_moves_the_sibling(gateway: Gateway) -> None:
    with wire_server(_upstream()) as wire, gateway.scenario() as scenario:
        sibling: Final = _deployment(scenario, wire, _MANTLE_GPT, None)
        before: Final = _waves(sibling, 12)
        during: Final = _waves(sibling, 12)
        after: Final = _waves(sibling, 12)
        first: Final = asyncio.create_task(_burst(_proxy_url(gateway), gateway.key, before))
        with gateway.scenario() as pin_scenario:
            await asyncio.to_thread(_deployment, pin_scenario, wire, _MANTLE_GPT, "chat")
            answered_before: Final = await first
            answered_during: Final = await _burst(_proxy_url(gateway), gateway.key, during)
        answered_after: Final = await _burst(_proxy_url(gateway), gateway.key, after)
        calls: Final = (*before, *during, *after)
        answers: Final = tuple(
            _assert_answered(answer, call.marker)
            for call, answer in zip(calls, (*answered_before, *answered_during, *answered_after), strict=True)
        )
        assert _routes(_attempts(wire)) == {call.marker: ("responses",) for call in calls}
        for answer in answers:
            _assert_logged(answer, sibling, "success")


async def test_an_upstream_outage_mid_burst_reaches_callers_and_routing_recovers(gateway: Gateway) -> None:
    down: Final = threading.Event()
    with wire_server(_upstream(down)) as wire, gateway.scenario() as scenario:
        pinned: Final = _deployment(scenario, wire, _MANTLE_GPT, "chat")
        sibling: Final = _deployment(scenario, wire, _MANTLE_GPT, None)
        outage: Final = (*_waves(pinned, 10), *_waves(sibling, 10))
        recovery: Final = (*_waves(pinned, 10), *_waves(sibling, 10))
        down.set()
        failed: Final = await _burst(_proxy_url(gateway), gateway.key, outage)
        down.clear()
        recovered: Final = tuple(
            _assert_answered(answer, call.marker)
            for call, answer in zip(recovery, await _burst(_proxy_url(gateway), gateway.key, recovery), strict=True)
        )
        for call, answer in zip(outage, failed, strict=True):
            assert answer is not None and answer.status == 503, (call, answer)
            assert "The server is overloaded." in answer.text, answer.text
            _assert_logged(answer, call.model, "failure")
        assert _routes(_attempts(wire)) == {
            call.marker: (_expected_route(call, pinned),) for call in (*outage, *recovery)
        }
        for call, answer in zip(recovery, recovered, strict=True):
            _assert_logged(answer, call.model, "success")


def _yaml_config(wire: Wire, tmp_path: Path, order: tuple[str, ...], label: str) -> Path:
    base: Final = _JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    params: Final[dict[str, JsonValue]] = {
        "model": _MANTLE_GPT,
        "custom_llm_provider": "bedrock_mantle",
        "api_base": wire.url + "/openai/v1",
        "api_key": _API_KEY,
    }
    entries: Final[Mapping[str, dict[str, JsonValue]]] = {
        _PINNED: {"model_name": _PINNED, "litellm_params": params, "model_info": {"mode": "chat"}},
        _SIBLING: {"model_name": _SIBLING, "litellm_params": params},
    }
    config: Final = {
        **base,
        "model_list": [entries[name] for name in order],
        "router_settings": {**object_value(base["router_settings"]), "num_retries": 0},
    }
    path: Final = tmp_path / f"deployment-mode-routing-{label}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _holding_upstream(held: SimpleQueue[str], release: threading.Event) -> Callable[[Request], Reply]:
    respond: Final = _upstream()

    def hold(request: Request) -> Reply:
        if request.method == "POST":
            held.put(_marker_in(request.body.decode()))
            assert release.wait(timeout=60), "The burst was never released"
        return respond(request)

    return hold


def _held_upstream_calls(pid: int, wire: Wire) -> int:
    port: Final = urlsplit(wire.url).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


async def _follow_up(gateway: Gateway) -> tuple[_Call, ...]:
    calls: Final = (*_waves(_PINNED, 6), *_waves(_SIBLING, 6))
    answers: Final = await _burst(_proxy_url(gateway), gateway.key, calls)
    for call, answer in zip(calls, answers, strict=True):
        _assert_logged(_assert_answered(answer, call.marker), call.model, "success")
    return calls


@pytest.mark.timeout(5 * graceful_stop_seconds() + 120)
async def test_a_worker_kill_and_a_restart_in_either_order_keep_each_deployment_on_its_mode(
    gateway: Gateway, tmp_path: Path
) -> None:
    held: Final[SimpleQueue[str]] = SimpleQueue()
    release: Final = threading.Event()
    burst_calls: Final = (*_waves(_PINNED, 10), *_waves(_SIBLING, 10))
    with wire_server(_holding_upstream(held, release)) as wire:
        sibling_first: Final = _yaml_config(wire, tmp_path, (_SIBLING, _PINNED), "sibling-first")
        with owned_proxy_process(gateway, tmp_path, {}, config=sibling_first, workers=2) as owned:
            workers: Final = eventually(
                lambda: tuple(int(found.group(1)) for found in _STARTED_WORKER.finditer(owned.log.read_text())),
                lambda pids: len(pids) == 2,
                seconds=30,
            )
            burst: Final = asyncio.create_task(
                _burst(_proxy_url(owned.gateway), owned.gateway.key, burst_calls, tolerate_drops=True)
            )
            await asyncio.to_thread(eventually, held.qsize, lambda size: size == len(burst_calls), 60)
            held_by_worker: Final = MappingProxyType({pid: _held_upstream_calls(pid, wire) for pid in workers})
            assert sum(held_by_worker.values()) == len(burst_calls), held_by_worker
            victim: Final = max(workers, key=held_by_worker.__getitem__)
            with owned.replacing_worker(victim) as kill:
                kill.freeze()
                await asyncio.to_thread(kill)
                release.set()
                served: Final = await burst
            survivors: Final = tuple(
                _assert_answered(answer, call.marker)
                for call, answer in zip(burst_calls, served, strict=True)
                if answer is not None
            )
            assert len(survivors) == len(burst_calls) - held_by_worker[victim], (held_by_worker, served)
            after_kill: Final = await _follow_up(owned.gateway)
        pinned_first: Final = _yaml_config(wire, tmp_path, (_PINNED, _SIBLING), "pinned-first")
        with owned_proxy_process(gateway, tmp_path, {}, config=pinned_first, workers=2) as restarted:
            after_restart: Final = await _follow_up(restarted.gateway)
        assert _routes(_attempts(wire)) == {
            call.marker: (_expected_route(call, _PINNED),) for call in (*burst_calls, *after_kill, *after_restart)
        }
