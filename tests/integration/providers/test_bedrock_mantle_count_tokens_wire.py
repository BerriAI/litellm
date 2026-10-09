import asyncio
import json
import re
import signal
import socket
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final, cast

import anthropic
import httpx
import psutil
import pytest
import yaml
from anthropic.types import MessageCountTokensToolParam, MessageParam
from integration._support.bedrock_runtime_peer import answer, marker_of, target_of
from integration._support.bedrock_runtime_peer import respond as runtime_generation
from integration._support.client import Gateway, Scenario, eventually, gateway_from_environment, object_value
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers._count_tokens_system_lift import (
    ASSISTANT,
    FOLLOW_UP,
    INSTRUCTION,
    LEADING,
    LIFT_CASES,
    LIFTED,
    MID_SYSTEM,
    REMINDER,
    STRING_CASE,
    USER,
    USER_TEXT,
    LiftCase,
    accepts_count_body,
    async_openai_client,
    count_request,
    expected_count_body,
    openai_client,
)
from pydantic import JsonValue, TypeAdapter

pytestmark = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_OPUS: Final = "global.anthropic.claude-opus-4-8"
_OPUS_BASE: Final = "anthropic.claude-opus-4-8"
_SONNET: Final = "anthropic.claude-sonnet-4-6"
_NOVA: Final = "amazon.nova-lite-v1:0"
_REGION: Final = "us-east-1"
_OWNED_OPUS: Final = "mantle-opus"
_OWNED_SONNET: Final = "mantle-sonnet"
_OWNED_NOVA: Final = "mantle-nova"
_OWNED_MODELS: Final = MappingProxyType({_OWNED_OPUS: _OPUS, _OWNED_SONNET: _SONNET, _OWNED_NOVA: _NOVA})
_ACCESS_KEY: Final = "AKIAINTEGRATIONMANTLE"
_SECRET_KEY: Final = "integration-mantle-secret"
_SIGV4_SCOPE: Final = f"/{_REGION}/bedrock/aws4_request"
_MANTLE_TARGET: Final = "/anthropic/v1/messages/count_tokens"
_MANTLE_VERSION: Final = "2023-06-01"
_MANTLE_COUNT: Final = 4242
_RUNTIME_COUNT: Final = 1345
_UNSUPPORTED: Final = "The provided model doesn't support counting tokens."
_REJECTION: Final = "scripted mantle rejection"
_COUNT_TARGET: Final = re.compile(r"^/model/(.+)/count-tokens$")
_INVOKE_TARGET: Final = re.compile(r"^/model/(.+)/invoke$")
_SCRIPTED_STATUS: Final = re.compile(r"status=(\d{3})")
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_JSON_LIST: Final = TypeAdapter(list[JsonValue])

_SDK_MESSAGES: Final[list[MessageParam]] = [{"role": "user", "content": "Count this message"}]
_MESSAGES: Final = _JSON_LIST.validate_json(json.dumps(_SDK_MESSAGES))
_SYSTEM: Final = "You are a terse assistant that answers in one sentence"
_SYSTEM_BLOCKS: Final[list[JsonValue]] = [
    {"type": "text", "text": "You are a terse assistant"},
    {"type": "text", "text": "Answer in one sentence"},
]
_SDK_TOOLS: Final[list[MessageCountTokensToolParam]] = [
    {
        "name": "get_weather",
        "description": "Look up the current weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City to look up"}},
            "required": ["city"],
        },
    }
]
_TOOLS: Final = _JSON_LIST.validate_json(json.dumps(_SDK_TOOLS))
_GEMINI_BODY: Final[dict[str, JsonValue]] = {"contents": [{"role": "user", "parts": [{"text": "Count this"}]}]}
_GEMINI_MESSAGES: Final[list[JsonValue]] = [{"role": "user", "content": "Count this"}]


def _mantle_body(**fields: JsonValue) -> dict[str, JsonValue]:
    return {"model": _OPUS_BASE, "messages": _MESSAGES, **fields}


_MANTLE_BARE: Final = _mantle_body()
_MANTLE_LIFTED: Final = _mantle_body(system=[LIFTED])
_LIFT_SDK_MESSAGES: Final = cast(
    list[MessageParam], [LEADING, USER]
)  # cast-ok: the SDK types reject the role the proxy lifts
_MANTLE_FULL: Final = _mantle_body(system=_SYSTEM, tools=_TOOLS)


def _json_reply(status: int, payload: Mapping[str, JsonValue]) -> Reply:
    return Reply(status=status, body=json.dumps(payload).encode())


def _runtime_count(request: Request, model: str) -> Reply:
    scripted: Final = _SCRIPTED_STATUS.search(request.body.decode(errors="replace"))
    if scripted is not None:
        return _json_reply(int(scripted.group(1)), {"message": f"scripted {scripted.group(1)}"})
    if "sonnet" in model:
        return _json_reply(200, {"inputTokens": _RUNTIME_COUNT})
    return _json_reply(400, {"message": _UNSUPPORTED})


def _invoke_reply(marker: str) -> Reply:
    return _json_reply(
        200,
        {
            "id": f"msg_{marker}",
            "type": "message",
            "role": "assistant",
            "model": _OPUS_BASE,
            "content": [{"type": "text", "text": answer(marker)}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 5, "output_tokens": 3},
        },
    )


def _runtime(request: Request) -> Reply:
    target: Final = target_of(request)
    counted: Final = _COUNT_TARGET.match(target)
    if counted is not None:
        return _runtime_count(request, counted.group(1))
    if _INVOKE_TARGET.match(target):
        return _invoke_reply(marker_of(request))
    return runtime_generation(request)


def _mantle_counted(_request: Request) -> Reply:
    return _json_reply(200, {"input_tokens": _MANTLE_COUNT})


def _rejected(status: int) -> Reply:
    return _json_reply(status, {"type": "error", "error": {"type": "invalid_request_error", "message": _REJECTION}})


def _rejecting(status: int) -> Callable[[Request], Reply]:
    def count(_request: Request) -> Reply:
        return _rejected(status)

    return count


def _strict(request: Request) -> Reply:
    accepted: Final = accepts_count_body(_JSON_OBJECT.validate_json(request.body))
    return _mantle_counted(request) if accepted else _rejected(400)


def _mantle(count: Callable[[Request], Reply] = _mantle_counted) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.target == _MANTLE_TARGET:
            return count(request)
        return _json_reply(404, {"error": f"unscripted mantle target {request.target}"})

    return respond


def _mantle_environment(port: int) -> Mapping[str, str]:
    return {"BEDROCK_MANTLE_API_BASE": f"http://127.0.0.1:{port}"}


_INHERITED_BEARER: Final = ("AWS_BEARER_TOKEN_BEDROCK",)


def _reserved_port() -> int:
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        return reserve.getsockname()[1]


def _closed_port_url() -> str:
    return f"http://127.0.0.1:{_reserved_port()}"


@pytest.fixture(scope="module")
def mantle_port() -> int:
    return _reserved_port()


@pytest.fixture(scope="module")
def counting_proxy(tmp_path_factory: pytest.TempPathFactory, mantle_port: int) -> Iterator[OwnedProxy]:
    with (
        gateway_from_environment() as gateway,
        owned_proxy_process(
            gateway,
            tmp_path_factory.mktemp("mantle-count"),
            _mantle_environment(mantle_port),
            workers=1,
            remove_environment=_INHERITED_BEARER,
        ) as owned,
    ):
        yield owned


def _litellm_params(model: str, api_base: str) -> dict[str, JsonValue]:
    return {
        "model": f"bedrock/{model}",
        "api_base": api_base,
        "aws_access_key_id": _ACCESS_KEY,
        "aws_secret_access_key": _SECRET_KEY,
        "aws_region_name": _REGION,
    }


def _owned_config(path: Path, runtime_url: str, settings: Mapping[str, JsonValue]) -> Path:
    config: Final = object_value(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    litellm_settings: Final = object_value(config["litellm_settings"])
    path.write_text(
        yaml.safe_dump(
            {
                **config,
                "model_list": [
                    {"model_name": name, "litellm_params": _litellm_params(model, runtime_url)}
                    for name, model in _OWNED_MODELS.items()
                ],
                "litellm_settings": {**litellm_settings, **settings},
            }
        )
    )
    return path


def _deployment(scenario: Scenario, api_base: str, model: str = _OPUS) -> str:
    return scenario.model(model_info=None, api_key=None, **_litellm_params(model, api_base))


def _bare(model: str) -> dict[str, JsonValue]:
    return {"model": model, "messages": _MESSAGES}


def _full(model: str) -> dict[str, JsonValue]:
    return {**_bare(model), "system": _SYSTEM, "tools": _TOOLS}


def _count(gateway: Gateway, body: Mapping[str, JsonValue]) -> httpx.Response:
    return gateway.request("POST", "/v1/messages/count_tokens", body)


def _payload(response: httpx.Response) -> dict[str, JsonValue]:
    assert response.status_code == 200, response.text
    return _JSON_OBJECT.validate_json(response.content)


def _local_count(gateway: Gateway, body: Mapping[str, JsonValue]) -> int:
    response: Final = gateway.request("POST", "/utils/token_counter", body, params={"call_endpoint": "false"})
    payload: Final = _payload(response)
    total: Final = payload["total_tokens"]
    assert payload["tokenizer_type"] not in ("bedrock_api", "bedrock_mantle_api"), response.text
    assert isinstance(total, int) and total > 0 and total not in (_MANTLE_COUNT, _RUNTIME_COUNT), response.text
    return total


def _assert_sigv4(request: Request) -> None:
    authorization: Final = request.headers.get("authorization", "")
    assert authorization.startswith(f"AWS4-HMAC-SHA256 Credential={_ACCESS_KEY}/"), request.headers
    assert _SIGV4_SCOPE in authorization, authorization


def _mantle_requests(requests: Sequence[Request]) -> tuple[dict[str, JsonValue], ...]:
    for request in requests:
        assert (request.method, request.target) == ("POST", _MANTLE_TARGET), request.target
        assert request.headers["anthropic-version"] == _MANTLE_VERSION, request.headers
        assert request.headers["content-type"] == "application/json", request.headers
        _assert_sigv4(request)
    return tuple(_JSON_OBJECT.validate_json(request.body) for request in requests)


def _mantle_bodies(wire: Wire) -> tuple[dict[str, JsonValue], ...]:
    return _mantle_requests(wire.drain())


def _runtime_count_targets(wire: Wire) -> tuple[str, ...]:
    counts: Final = tuple(request for request in wire.drain() if _COUNT_TARGET.match(target_of(request)))
    for request in counts:
        assert request.method == "POST", request.method
        _assert_sigv4(request)
    return tuple(target_of(request) for request in counts)


def _clients(stack: ExitStack, base_url: str, count: int, timeout: float = 30) -> tuple[httpx.Client, ...]:
    return tuple(
        stack.enter_context(httpx.Client(base_url=base_url, timeout=timeout, trust_env=False)) for _ in range(count)
    )


def _counted_on(client: httpx.Client, key: str, body: Mapping[str, JsonValue]) -> tuple[int, JsonValue]:
    response: Final = client.post(
        "/v1/messages/count_tokens", json=dict(body), headers={"Authorization": f"Bearer {key}"}
    )
    return response.status_code, _JSON_OBJECT.validate_json(response.content).get("input_tokens")


def _generated_then_counted(
    client: httpx.Client, key: str, model: str, body: Mapping[str, JsonValue]
) -> tuple[int, int, JsonValue]:
    generated: Final = client.post(
        "/v1/messages",
        json={
            "model": model,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": f"Generate before counting marker-{uuid.uuid4().hex}"}],
        },
        headers={"Authorization": f"Bearer {key}"},
    )
    return generated.status_code, *_counted_on(client, key, body)


def _counted_or_dropped(client: httpx.Client, key: str, body: Mapping[str, JsonValue]) -> tuple[int, JsonValue] | None:
    try:
        return _counted_on(client, key, body)
    except httpx.TransportError:
        return None


def _probed_then_counted_or_dropped(
    client: httpx.Client, key: str, body: Mapping[str, JsonValue], probed: SimpleQueue[int]
) -> tuple[int, tuple[int, JsonValue] | None]:
    port: Final = _local_port(client)
    probed.put(port)
    return port, _counted_or_dropped(client, key, body)


def _local_port(client: httpx.Client) -> int:
    with client.stream("GET", "/health/liveliness") as response:
        port: Final = int(response.extensions["network_stream"].get_extra_info("client_addr")[1])
        response.read()
    assert response.status_code == 200, response.text
    return port


def _accepted_client_ports(pid: int, proxy_port: int) -> frozenset[int]:
    return frozenset(
        connection.raddr.port
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.raddr and connection.laddr.port == proxy_port
    )


def _holding(held: SimpleQueue[str], release: threading.Event, seconds: float) -> Callable[[Request], Reply]:
    def hold(request: Request) -> Reply:
        held.put(request.target)
        assert release.wait(timeout=seconds), "Held count was never released"
        return _mantle_counted(request)

    return hold


def _anthropic_client(gateway: Gateway) -> anthropic.Anthropic:
    return anthropic.Anthropic(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0)


def _async_anthropic_client(gateway: Gateway) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0)


@pytest.mark.parametrize("system", [_SYSTEM, _SYSTEM_BLOCKS], ids=["string", "blocks"])
def test_messages_count_tokens_counts_through_mantle_when_the_runtime_cannot(
    counting_proxy: OwnedProxy, mantle_port: int, system: JsonValue
) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url)
        response: Final = _count(gateway, {**_bare(model), "system": system, "tools": _TOOLS})
        assert _payload(response) == {"input_tokens": _MANTLE_COUNT}, response.text
        assert _runtime_count_targets(runtime) == (f"/model/{_OPUS_BASE}/count-tokens",)
        assert _mantle_bodies(mantle) == (_mantle_body(system=system, tools=_TOOLS),)


def test_messages_count_tokens_without_system_or_tools_sends_a_bare_body_to_mantle(
    counting_proxy: OwnedProxy, mantle_port: int
) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url)
        response: Final = _count(gateway, _bare(model))
        assert _payload(response) == {"input_tokens": _MANTLE_COUNT}, response.text
        assert len(_runtime_count_targets(runtime)) == 1
        assert _mantle_bodies(mantle) == (_MANTLE_BARE,)


def test_utils_token_counter_call_endpoint_reports_the_mantle_tokenizer(
    counting_proxy: OwnedProxy, mantle_port: int
) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url)
        response: Final = gateway.request(
            "POST", "/utils/token_counter", _full(model), params={"call_endpoint": "true"}
        )
        payload: Final = _payload(response)
        assert (payload["total_tokens"], payload["tokenizer_type"]) == (_MANTLE_COUNT, "bedrock_mantle_api")
        assert payload["original_response"] == {"input_tokens": _MANTLE_COUNT}, response.text
        assert (payload["request_model"], payload["model_used"]) == (model, _OPUS), response.text
        assert len(_runtime_count_targets(runtime)) == 1
        assert _mantle_bodies(mantle) == (_MANTLE_FULL,)


def test_gemini_count_tokens_route_reports_the_mantle_total(counting_proxy: OwnedProxy, mantle_port: int) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url)
        response: Final = gateway.request("POST", f"/v1beta/models/{model}:countTokens", _GEMINI_BODY)
        assert _payload(response) == {"totalTokens": _MANTLE_COUNT, "promptTokensDetails": []}, response.text
        assert len(_runtime_count_targets(runtime)) == 1
        assert _mantle_bodies(mantle) == ({"model": _OPUS_BASE, "messages": _GEMINI_MESSAGES},)


def test_gemini_count_tokens_route_reports_the_runtime_total_for_a_model_bedrock_counts(
    counting_proxy: OwnedProxy, mantle_port: int
) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url, _SONNET)
        response: Final = gateway.request("POST", f"/v1beta/models/{model}:countTokens", _GEMINI_BODY)
        assert _payload(response) == {"totalTokens": _RUNTIME_COUNT, "promptTokensDetails": []}, response.text
        assert _runtime_count_targets(runtime) == (f"/model/{_SONNET}/count-tokens",)
        assert mantle.drain() == ()


def test_responses_input_tokens_counts_through_mantle(counting_proxy: OwnedProxy, mantle_port: int) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url)
        response: Final = gateway.request(
            "POST", "/v1/responses/input_tokens", {"model": model, "input": "Count this message"}
        )
        assert _payload(response) == {"object": "response.input_tokens", "input_tokens": _MANTLE_COUNT}, response.text
        assert len(_runtime_count_targets(runtime)) == 1
        assert _mantle_bodies(mantle) == (_MANTLE_BARE,)


def test_anthropic_sdk_count_tokens_counts_through_mantle(counting_proxy: OwnedProxy, mantle_port: int) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url)
        counted: Final = _anthropic_client(gateway).messages.count_tokens(
            model=model, messages=_SDK_MESSAGES, system=_SYSTEM, tools=_SDK_TOOLS
        )
        assert counted.input_tokens == _MANTLE_COUNT, counted
        assert len(_runtime_count_targets(runtime)) == 1
        assert _mantle_bodies(mantle) == (_MANTLE_FULL,)


def test_async_anthropic_sdk_count_tokens_counts_through_mantle(counting_proxy: OwnedProxy, mantle_port: int) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url)
        counted: Final = asyncio.run(
            _async_anthropic_client(gateway).messages.count_tokens(model=model, messages=_SDK_MESSAGES, system=_SYSTEM)
        )
        assert counted.input_tokens == _MANTLE_COUNT, counted
        assert len(_runtime_count_targets(runtime)) == 1
        assert _mantle_bodies(mantle) == (_mantle_body(system=_SYSTEM),)


def test_model_the_runtime_counts_never_reaches_mantle(counting_proxy: OwnedProxy, mantle_port: int) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url, _SONNET)
        assert _payload(_count(gateway, _full(model))) == {"input_tokens": _RUNTIME_COUNT}
        detailed: Final = gateway.request(
            "POST", "/utils/token_counter", _full(model), params={"call_endpoint": "true"}
        )
        payload: Final = _payload(detailed)
        assert (payload["total_tokens"], payload["tokenizer_type"]) == (_RUNTIME_COUNT, "bedrock_api"), detailed.text
        assert _runtime_count_targets(runtime) == (f"/model/{_SONNET}/count-tokens",) * 2
        assert mantle.drain() == ()


def test_non_claude_model_the_runtime_cannot_count_falls_back_locally_without_mantle(
    counting_proxy: OwnedProxy, mantle_port: int
) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url, _NOVA)
        response: Final = _count(gateway, _bare(model))
        assert _payload(response) == {"input_tokens": _local_count(gateway, _bare(model))}, response.text
        assert _runtime_count_targets(runtime) == (f"/model/{_NOVA}/count-tokens",)
        assert mantle.drain() == ()


def test_runtime_403_on_a_claude_model_never_reaches_mantle(counting_proxy: OwnedProxy, mantle_port: int) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url)
        body: Final[dict[str, JsonValue]] = {
            "model": model,
            "messages": [{"role": "user", "content": "status=403 Count this message"}],
        }
        response: Final = _count(gateway, body)
        assert _payload(response) == {"input_tokens": _local_count(gateway, body)}, response.text
        assert len(_runtime_count_targets(runtime)) == 1
        assert mantle.drain() == ()


def test_unreachable_runtime_falls_back_locally_without_mantle(counting_proxy: OwnedProxy, mantle_port: int) -> None:
    gateway: Final = counting_proxy.gateway
    with wire_server(_mantle(), port=mantle_port) as mantle, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, _closed_port_url())
        response: Final = _count(gateway, _full(model))
        assert _payload(response) == {"input_tokens": _local_count(gateway, _full(model))}, response.text
        assert mantle.drain() == ()


def test_bedrock_passthrough_count_tokens_still_answers_the_runtime_rejection(
    counting_proxy: OwnedProxy, mantle_port: int
) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url)
        response: Final = gateway.request("POST", "/bedrock/v1/messages/count_tokens", _full(model))
        assert response.status_code == 400, response.text
        assert _UNSUPPORTED in response.text and "input_tokens" not in response.text, response.text
        assert len(_runtime_count_targets(runtime)) == 1
        assert mantle.drain() == ()


def test_responses_input_tokens_with_instructions_counts_through_mantle(
    counting_proxy: OwnedProxy, mantle_port: int
) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(_strict), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url)
        response: Final = gateway.request(
            "POST",
            "/v1/responses/input_tokens",
            {"model": model, "input": "Count this message", "instructions": "Be terse"},
        )
        assert _payload(response) == {"object": "response.input_tokens", "input_tokens": _MANTLE_COUNT}, response.text
        assert len(_runtime_count_targets(runtime)) == 1
        assert _mantle_bodies(mantle) == (_mantle_body(system=[{"type": "text", "text": "Be terse"}]),)


@pytest.mark.parametrize("case", LIFT_CASES.values(), ids=LIFT_CASES.keys())
def test_messages_count_tokens_lifts_the_leading_system_run_for_mantle(
    counting_proxy: OwnedProxy, mantle_port: int, case: LiftCase
) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(_strict), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url)
        response: Final = _count(gateway, count_request(model, case))
        assert _payload(response) == {"input_tokens": _MANTLE_COUNT}, response.text
        assert len(_runtime_count_targets(runtime)) == 1
        assert _mantle_bodies(mantle) == (expected_count_body(_OPUS_BASE, case),)


def test_anthropic_sdk_count_tokens_lifts_the_leading_system_through_mantle(
    counting_proxy: OwnedProxy, mantle_port: int
) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(_strict), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url)
        counted: Final = _anthropic_client(gateway).messages.count_tokens(model=model, messages=_LIFT_SDK_MESSAGES)
        assert counted.input_tokens == _MANTLE_COUNT, counted
        assert len(_runtime_count_targets(runtime)) == 1
        assert _mantle_bodies(mantle) == (_MANTLE_LIFTED,)


def test_async_anthropic_sdk_count_tokens_lifts_the_leading_system_through_mantle(
    counting_proxy: OwnedProxy, mantle_port: int
) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(_strict), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url)
        counted: Final = asyncio.run(
            _async_anthropic_client(gateway).messages.count_tokens(model=model, messages=_LIFT_SDK_MESSAGES)
        )
        assert counted.input_tokens == _MANTLE_COUNT, counted
        assert len(_runtime_count_targets(runtime)) == 1
        assert _mantle_bodies(mantle) == (_MANTLE_LIFTED,)


def test_utils_token_counter_call_endpoint_counts_a_leading_system_through_mantle(
    counting_proxy: OwnedProxy, mantle_port: int
) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(_strict), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url)
        response: Final = gateway.request(
            "POST", "/utils/token_counter", count_request(model, STRING_CASE), params={"call_endpoint": "true"}
        )
        payload: Final = _payload(response)
        assert (payload["total_tokens"], payload["tokenizer_type"]) == (_MANTLE_COUNT, "bedrock_mantle_api")
        assert payload["original_response"] == {"input_tokens": _MANTLE_COUNT}, response.text
        assert (payload["request_model"], payload["model_used"]) == (model, _OPUS), response.text
        assert len(_runtime_count_targets(runtime)) == 1
        assert _mantle_bodies(mantle) == (_MANTLE_LIFTED,)


def test_openai_sdk_input_tokens_lifts_instructions_through_mantle(
    counting_proxy: OwnedProxy, mantle_port: int
) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(_strict), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url)
        counted: Final = openai_client(gateway).responses.input_tokens.count(
            model=model, input=USER_TEXT, instructions=INSTRUCTION
        )
        assert counted.input_tokens == _MANTLE_COUNT, counted
        assert len(_runtime_count_targets(runtime)) == 1
        assert _mantle_bodies(mantle) == (_MANTLE_LIFTED,)


def test_async_openai_sdk_input_tokens_lifts_instructions_through_mantle(
    counting_proxy: OwnedProxy, mantle_port: int
) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(_strict), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url)
        counted: Final = asyncio.run(
            async_openai_client(gateway).responses.input_tokens.count(
                model=model, input=USER_TEXT, instructions=INSTRUCTION
            )
        )
        assert counted.input_tokens == _MANTLE_COUNT, counted
        assert len(_runtime_count_targets(runtime)) == 1
        assert _mantle_bodies(mantle) == (_MANTLE_LIFTED,)


def test_responses_input_tokens_lifts_instructions_ahead_of_a_leading_system_item_through_mantle(
    counting_proxy: OwnedProxy, mantle_port: int
) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(_strict), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url)
        response: Final = gateway.request(
            "POST",
            "/v1/responses/input_tokens",
            {"model": model, "input": [MID_SYSTEM, USER], "instructions": INSTRUCTION},
        )
        assert _payload(response) == {"object": "response.input_tokens", "input_tokens": _MANTLE_COUNT}, response.text
        assert len(_runtime_count_targets(runtime)) == 1
        assert _mantle_bodies(mantle) == (_mantle_body(system=[LIFTED, {"type": "text", "text": REMINDER}]),)


def test_messages_count_tokens_forwards_tools_beside_the_lifted_system_to_mantle(
    counting_proxy: OwnedProxy, mantle_port: int
) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(_strict), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url)
        response: Final = _count(gateway, {**count_request(model, STRING_CASE), "tools": _TOOLS})
        assert _payload(response) == {"input_tokens": _MANTLE_COUNT}, response.text
        assert len(_runtime_count_targets(runtime)) == 1
        assert _mantle_bodies(mantle) == (_mantle_body(system=[LIFTED], tools=_TOOLS),)


def test_messages_count_tokens_keeps_a_mid_conversation_system_in_place_for_mantle(
    counting_proxy: OwnedProxy, mantle_port: int
) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(_strict), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url)
        body: Final[dict[str, JsonValue]] = {
            "model": model,
            "messages": [LEADING, USER, MID_SYSTEM, ASSISTANT, FOLLOW_UP],
        }
        response: Final = _count(gateway, body)
        assert _payload(response) == {"input_tokens": _MANTLE_COUNT}, response.text
        assert len(_runtime_count_targets(runtime)) == 1
        assert _mantle_bodies(mantle) == (
            _mantle_body(messages=[USER, MID_SYSTEM, ASSISTANT, FOLLOW_UP], system=[LIFTED]),
        )


def test_messages_count_tokens_falls_back_locally_when_every_message_is_system_for_mantle(
    counting_proxy: OwnedProxy, mantle_port: int
) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(_strict), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url)
        body: Final[dict[str, JsonValue]] = {"model": model, "messages": [LEADING]}
        local: Final = _local_count(gateway, body)
        response: Final = _count(gateway, body)
        assert _payload(response) == {"input_tokens": local}, response.text
        assert len(_runtime_count_targets(runtime)) == 1
        assert _mantle_bodies(mantle) == (_mantle_body(messages=[], system=[LIFTED]),)


def test_messages_count_tokens_leaves_a_leading_system_in_place_beside_a_non_text_system_for_mantle(
    counting_proxy: OwnedProxy, mantle_port: int
) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(_strict), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url)
        local: Final = _local_count(gateway, count_request(model, STRING_CASE))
        response: Final = _count(gateway, {**count_request(model, STRING_CASE), "system": 5})
        assert _payload(response) == {"input_tokens": local}, response.text
        assert len(_runtime_count_targets(runtime)) == 1
        assert _mantle_bodies(mantle) == (_mantle_body(messages=[LEADING, USER], system=5),)


def test_messages_count_tokens_repeated_request_lifts_the_leading_system_each_time_for_mantle(
    counting_proxy: OwnedProxy, mantle_port: int
) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(_strict), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url)
        answers: Final = tuple(_payload(_count(gateway, count_request(model, STRING_CASE))) for _ in range(2))
        assert answers == ({"input_tokens": _MANTLE_COUNT},) * 2
        assert _runtime_count_targets(runtime) == (f"/model/{_OPUS_BASE}/count-tokens",) * 2
        assert _mantle_bodies(mantle) == (_MANTLE_LIFTED,) * 2


def test_messages_count_tokens_answers_a_leading_system_without_content_before_any_mantle_call(
    counting_proxy: OwnedProxy, mantle_port: int
) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(_strict), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url)
        body: Final[dict[str, JsonValue]] = {"model": model, "messages": [{"role": "system"}, USER]}
        local: Final = _local_count(gateway, body)
        response: Final = _count(gateway, body)
        assert _payload(response) == {"input_tokens": local}, response.text
        assert mantle.drain() == ()
        follow_up: Final = _count(gateway, count_request(model, STRING_CASE))
        assert _payload(follow_up) == {"input_tokens": _MANTLE_COUNT}, follow_up.text
        assert len(_runtime_count_targets(runtime)) == 1
        assert _mantle_bodies(mantle) == (_MANTLE_LIFTED,)


def test_messages_count_tokens_duplicate_messages_key_lifts_the_last_value_for_mantle(
    counting_proxy: OwnedProxy, mantle_port: int
) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(_strict), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url)
        first: Final = json.dumps([USER])
        last: Final = json.dumps([LEADING, USER])
        response: Final = gateway.client.post(
            "/v1/messages/count_tokens",
            content=f'{{"model": "{model}", "messages": {first}, "messages": {last}}}',
            headers={"Authorization": f"Bearer {gateway.key}", "Content-Type": "application/json"},
        )
        assert _payload(response) == {"input_tokens": _MANTLE_COUNT}, response.text
        assert len(_runtime_count_targets(runtime)) == 1
        assert _mantle_bodies(mantle) == (_MANTLE_LIFTED,)


def test_disabled_token_counter_counts_a_leading_system_through_mantle(
    gateway: Gateway, mantle_port: int, tmp_path: Path
) -> None:
    with ExitStack() as stack:
        runtime: Final = stack.enter_context(wire_server(_runtime))
        config: Final = _owned_config(
            tmp_path / "disabled-token-counter-lift.yaml", runtime.url, {"disable_token_counter": True}
        )
        owned: Final = stack.enter_context(
            owned_proxy_process(
                gateway,
                tmp_path,
                _mantle_environment(mantle_port),
                config=config,
                workers=2,
                remove_environment=_INHERITED_BEARER,
            )
        )
        with wire_server(_mantle(_strict), port=mantle_port) as strict:
            counted: Final = _count(owned.gateway, count_request(_OWNED_OPUS, STRING_CASE))
            assert _payload(counted) == {"input_tokens": _MANTLE_COUNT}, counted.text
            assert _mantle_bodies(strict) == (_MANTLE_LIFTED,)
        assert len(_runtime_count_targets(runtime)) == 1


def test_mantle_outage_between_concurrent_lifted_waves_falls_back_then_recovers(
    counting_proxy: OwnedProxy, mantle_port: int
) -> None:
    gateway: Final = counting_proxy.gateway
    with ExitStack() as stack:
        clients: Final = _clients(stack, str(gateway.client.base_url), 8)
        pool: Final = stack.enter_context(ThreadPoolExecutor(max_workers=len(clients)))
        runtime: Final = stack.enter_context(wire_server(_runtime))
        scenario: Final = stack.enter_context(gateway.scenario())
        model: Final = _deployment(scenario, runtime.url)
        body: Final = count_request(model, STRING_CASE)
        local: Final = _local_count(gateway, body)

        def count(client: httpx.Client) -> tuple[int, JsonValue]:
            return _counted_on(client, gateway.key, body)

        with wire_server(_mantle(_strict), port=mantle_port) as mantle:
            assert tuple(pool.map(count, clients)) == ((200, _MANTLE_COUNT),) * len(clients)
            assert _mantle_bodies(mantle) == (_MANTLE_LIFTED,) * len(clients)
        assert tuple(pool.map(count, clients)) == ((200, local),) * len(clients)
        with wire_server(_mantle(_strict), port=mantle_port) as revived:
            assert tuple(pool.map(count, clients)) == ((200, _MANTLE_COUNT),) * len(clients)
            assert _mantle_bodies(revived) == (_MANTLE_LIFTED,) * len(clients)
        assert len(_runtime_count_targets(runtime)) == 3 * len(clients)


@pytest.mark.parametrize("status", [400, 403, 404, 500, 503])
def test_messages_count_tokens_falls_back_locally_when_mantle_errors(
    counting_proxy: OwnedProxy, mantle_port: int, status: int
) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(_rejecting(status)), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url)
        response: Final = _count(gateway, _full(model))
        assert _payload(response) == {"input_tokens": _local_count(gateway, _full(model))}, response.text
        assert len(_runtime_count_targets(runtime)) == 1
        assert _mantle_bodies(mantle) == (_MANTLE_FULL,)


@pytest.mark.parametrize(
    "body",
    [b'{"inputTokens": 7}', b"not json at all", b"{}"],
    ids=["runtime_key", "not_json", "empty_object"],
)
def test_messages_count_tokens_falls_back_locally_when_mantle_answers_without_input_tokens(
    counting_proxy: OwnedProxy, mantle_port: int, body: bytes
) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(lambda _request: Reply(body=body)), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url)
        response: Final = _count(gateway, _bare(model))
        assert _payload(response) == {"input_tokens": _local_count(gateway, _bare(model))}, response.text
        assert len(_runtime_count_targets(runtime)) == 1
        assert _mantle_bodies(mantle) == (_MANTLE_BARE,)


def test_messages_count_tokens_falls_back_locally_when_mantle_is_unreachable(counting_proxy: OwnedProxy) -> None:
    gateway: Final = counting_proxy.gateway
    with wire_server(_runtime) as runtime, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, runtime.url)
        response: Final = _count(gateway, _full(model))
        assert _payload(response) == {"input_tokens": _local_count(gateway, _full(model))}, response.text
        assert len(_runtime_count_targets(runtime)) == 1


def test_messages_count_tokens_falls_back_locally_when_mantle_never_answers(
    counting_proxy: OwnedProxy, mantle_port: int
) -> None:
    gateway: Final = counting_proxy.gateway
    held: Final[SimpleQueue[str]] = SimpleQueue()
    release: Final = threading.Event()

    def hold(request: Request) -> Reply:
        held.put(request.target)
        release.wait(timeout=120)
        return Reply(drop_connection=True)

    with ExitStack() as stack:
        (client,) = _clients(stack, str(gateway.client.base_url), 1, timeout=120)
        pool: Final = stack.enter_context(ThreadPoolExecutor(max_workers=1))
        runtime: Final = stack.enter_context(wire_server(_runtime))
        mantle: Final = stack.enter_context(wire_server(_mantle(hold), port=mantle_port))
        stack.callback(release.set)
        scenario: Final = stack.enter_context(gateway.scenario())
        model: Final = _deployment(scenario, runtime.url)
        body: Final = _full(model)
        local: Final = _local_count(gateway, body)
        future: Final = pool.submit(_counted_on, client, gateway.key, body)
        eventually(held.qsize, lambda size: size == 1, seconds=30)
        assert gateway.request("GET", "/health/liveliness").status_code == 200
        assert _local_count(gateway, body) == local
        assert not future.done()
        assert future.result(timeout=120) == (200, local)
        assert len(_runtime_count_targets(runtime)) == 1
        assert len(mantle.drain()) == 1


def test_messages_count_tokens_falls_back_locally_when_mantle_rejects_a_non_text_system(
    counting_proxy: OwnedProxy, mantle_port: int
) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(_strict), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url)
        response: Final = _count(gateway, {**_bare(model), "system": 5})
        assert _payload(response) == {"input_tokens": _local_count(gateway, _bare(model))}, response.text
        assert len(_runtime_count_targets(runtime)) == 1
        assert _mantle_bodies(mantle) == (_mantle_body(system=5),)


@pytest.mark.parametrize(
    "tools",
    [5, "", "x" * 5120, ["get_weather"]],
    ids=["int", "empty_string", "5kb_string", "list_of_strings"],
)
def test_messages_count_tokens_rejects_malformed_tools_without_calling_either_peer(
    counting_proxy: OwnedProxy, mantle_port: int, tools: JsonValue
) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url)
        refused: Final = _count(gateway, {**_bare(model), "tools": tools})
        assert 400 <= refused.status_code < 600, refused.text
        assert "input_tokens" not in refused.text, refused.text
        assert runtime.drain() == () and mantle.drain() == ()
        assert _generated_then_counted(gateway.client, gateway.key, model, _bare(model)) == (200, 200, _MANTLE_COUNT)
        assert _mantle_bodies(mantle) == (_MANTLE_BARE,)


@pytest.mark.parametrize("fields", [{}, {"messages": []}], ids=["missing", "empty"])
def test_messages_count_tokens_without_messages_is_rejected(
    counting_proxy: OwnedProxy, mantle_port: int, fields: dict[str, JsonValue]
) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url)
        response: Final = _count(gateway, {"model": model, "system": _SYSTEM, "tools": _TOOLS, **fields})
        assert response.status_code == 400, response.text
        assert "messages parameter is required" in response.text, response.text
        assert runtime.drain() == () and mantle.drain() == ()


def test_messages_count_tokens_unauthenticated_request_never_reaches_either_peer(
    counting_proxy: OwnedProxy, mantle_port: int
) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url)
        response: Final = gateway.request(
            "POST", "/v1/messages/count_tokens", _full(model), key="sk-not-a-key-this-proxy-issued"
        )
        assert response.status_code == 401, response.text
        assert "input_tokens" not in response.text, response.text
        assert runtime.drain() == () and mantle.drain() == ()


def test_messages_count_tokens_forwards_a_5kb_system_verbatim_to_mantle(
    counting_proxy: OwnedProxy, mantle_port: int
) -> None:
    gateway: Final = counting_proxy.gateway
    system: Final = "Answer in one sentence. " * 214
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url)
        response: Final = _count(gateway, {**_bare(model), "system": system})
        assert _payload(response) == {"input_tokens": _MANTLE_COUNT}, response.text
        assert len(_runtime_count_targets(runtime)) == 1
        assert _mantle_bodies(mantle) == (_mantle_body(system=system),)


def test_messages_count_tokens_duplicate_system_and_tools_keys_forward_one_value_each_to_mantle(
    counting_proxy: OwnedProxy, mantle_port: int
) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url)
        fields: Final = f'"system": {json.dumps(_SYSTEM)}, "tools": {json.dumps(_TOOLS)}'
        response: Final = gateway.client.post(
            "/v1/messages/count_tokens",
            content=f'{{"model": "{model}", "messages": {json.dumps(_MESSAGES)}, {fields}, {fields}}}',
            headers={"Authorization": f"Bearer {gateway.key}", "Content-Type": "application/json"},
        )
        assert _payload(response) == {"input_tokens": _MANTLE_COUNT}, response.text
        assert len(_runtime_count_targets(runtime)) == 1
        (sent,) = mantle.drain()
        assert _mantle_requests((sent,)) == (_MANTLE_FULL,)
        assert (sent.body.count(b'"system"'), sent.body.count(b'"tools"')) == (1, 1), sent.body


def test_messages_count_tokens_leaves_empty_tools_and_system_out_of_the_mantle_body(
    counting_proxy: OwnedProxy, mantle_port: int
) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url)
        response: Final = _count(gateway, {**_bare(model), "tools": [], "system": ""})
        assert _payload(response) == {"input_tokens": _MANTLE_COUNT}, response.text
        assert len(_runtime_count_targets(runtime)) == 1
        assert _mantle_bodies(mantle) == (_MANTLE_BARE,)


def test_messages_count_tokens_repeated_request_reaches_both_peers_each_time(
    counting_proxy: OwnedProxy, mantle_port: int
) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url)
        answers: Final = tuple(_payload(_count(gateway, _bare(model))) for _ in range(2))
        assert answers == ({"input_tokens": _MANTLE_COUNT},) * 2
        assert _runtime_count_targets(runtime) == (f"/model/{_OPUS_BASE}/count-tokens",) * 2
        assert _mantle_bodies(mantle) == (_MANTLE_BARE,) * 2


def test_disabled_token_counter_surfaces_the_mantle_error_instead_of_counting_locally(
    gateway: Gateway, mantle_port: int, tmp_path: Path
) -> None:
    with ExitStack() as stack:
        runtime: Final = stack.enter_context(wire_server(_runtime))
        config: Final = _owned_config(
            tmp_path / "disabled-token-counter.yaml", runtime.url, {"disable_token_counter": True}
        )
        owned: Final = stack.enter_context(
            owned_proxy_process(
                gateway,
                tmp_path,
                _mantle_environment(mantle_port),
                config=config,
                workers=2,
                remove_environment=_INHERITED_BEARER,
            )
        )
        with wire_server(_mantle(_rejecting(403)), port=mantle_port) as refusing:
            refused: Final = _count(owned.gateway, _full(_OWNED_OPUS))
            assert refused.status_code == 403, refused.text
            assert _REJECTION in refused.text and "input_tokens" not in refused.text, refused.text
            assert _mantle_bodies(refusing) == (_MANTLE_FULL,)
        with wire_server(_mantle(), port=mantle_port) as counting:
            assert _payload(_count(owned.gateway, _full(_OWNED_OPUS))) == {"input_tokens": _MANTLE_COUNT}
            assert _mantle_bodies(counting) == (_MANTLE_FULL,)
            assert _payload(_count(owned.gateway, _full(_OWNED_SONNET))) == {"input_tokens": _RUNTIME_COUNT}
            rejected: Final = _count(owned.gateway, _full(_OWNED_NOVA))
            assert rejected.status_code == 400, rejected.text
            assert _UNSUPPORTED in rejected.text and "input_tokens" not in rejected.text, rejected.text
            assert counting.drain() == ()
        assert len(_runtime_count_targets(runtime)) == 4


def test_mantle_outage_between_concurrent_waves_falls_back_then_recovers(
    counting_proxy: OwnedProxy, mantle_port: int
) -> None:
    gateway: Final = counting_proxy.gateway
    with ExitStack() as stack:
        clients: Final = _clients(stack, str(gateway.client.base_url), 8)
        pool: Final = stack.enter_context(ThreadPoolExecutor(max_workers=len(clients)))
        runtime: Final = stack.enter_context(wire_server(_runtime))
        scenario: Final = stack.enter_context(gateway.scenario())
        model: Final = _deployment(scenario, runtime.url)
        body: Final = _full(model)
        local: Final = _local_count(gateway, body)

        def generate_then_count(client: httpx.Client) -> tuple[int, int, JsonValue]:
            return _generated_then_counted(client, gateway.key, model, body)

        def count_only(client: httpx.Client) -> tuple[int, JsonValue]:
            return _counted_on(client, gateway.key, body)

        with wire_server(_mantle(), port=mantle_port) as mantle:
            assert tuple(pool.map(generate_then_count, clients)) == ((200, 200, _MANTLE_COUNT),) * len(clients)
            assert _mantle_bodies(mantle) == (_MANTLE_FULL,) * len(clients)
        outage: Final = tuple(pool.map(count_only, clients))
        assert outage == ((200, local),) * len(clients)
        with wire_server(_mantle(), port=mantle_port) as revived:
            assert tuple(pool.map(generate_then_count, clients)) == ((200, 200, _MANTLE_COUNT),) * len(clients)
            assert _mantle_bodies(revived) == (_MANTLE_FULL,) * len(clients)
        assert len(_runtime_count_targets(runtime)) == 3 * len(clients)


def test_slow_mantle_holds_concurrent_counts_without_stalling_the_proxy(
    counting_proxy: OwnedProxy, mantle_port: int
) -> None:
    gateway: Final = counting_proxy.gateway
    held: Final[SimpleQueue[str]] = SimpleQueue()
    release: Final = threading.Event()
    with ExitStack() as stack:
        clients: Final = _clients(stack, str(gateway.client.base_url), 6)
        runtime: Final = stack.enter_context(wire_server(_runtime))
        mantle: Final = stack.enter_context(wire_server(_mantle(_holding(held, release, 20)), port=mantle_port))
        scenario: Final = stack.enter_context(gateway.scenario())
        pool: Final = stack.enter_context(ThreadPoolExecutor(max_workers=len(clients)))
        stack.callback(release.set)
        model: Final = _deployment(scenario, runtime.url)
        futures: Final = tuple(
            pool.submit(_generated_then_counted, client, gateway.key, model, _full(model)) for client in clients
        )
        eventually(held.qsize, lambda size: size == len(clients), seconds=30)
        assert gateway.request("GET", "/health/liveliness").status_code == 200
        assert _local_count(gateway, _full(model)) > 0
        assert not any(future.done() for future in futures)
        release.set()
        assert tuple(future.result(timeout=30) for future in futures) == ((200, 200, _MANTLE_COUNT),) * len(clients)
        assert _mantle_bodies(mantle) == (_MANTLE_FULL,) * len(clients)
        assert len(_runtime_count_targets(runtime)) == len(clients)


def test_worker_sigkill_mid_burst_leaves_the_sibling_counting(
    gateway: Gateway, mantle_port: int, tmp_path: Path
) -> None:
    held: Final[SimpleQueue[str]] = SimpleQueue()
    release: Final = threading.Event()
    with ExitStack() as stack:
        runtime: Final = stack.enter_context(wire_server(_runtime))
        mantle: Final = stack.enter_context(wire_server(_mantle(_holding(held, release, 60)), port=mantle_port))
        config: Final = _owned_config(tmp_path / "worker-kill.yaml", runtime.url, {})
        owned: Final = stack.enter_context(
            owned_proxy_process(
                gateway,
                tmp_path,
                _mantle_environment(mantle_port),
                config=config,
                workers=2,
                remove_environment=_INHERITED_BEARER,
            )
        )
        body: Final = _full(_OWNED_OPUS)
        proxy_url: Final = owned.gateway.client.base_url
        workers: Final = eventually(
            lambda: tuple(int(pid) for pid in _STARTED_WORKER.findall(owned.log.read_text())),
            lambda pids: len(pids) == 2,
            seconds=30,
        )
        clients: Final = _clients(stack, str(proxy_url), 12)
        pool: Final = stack.enter_context(ThreadPoolExecutor(max_workers=len(clients)))
        stack.callback(release.set)
        probed: Final[SimpleQueue[int]] = SimpleQueue()
        futures: Final[tuple[Future[tuple[int, tuple[int, JsonValue] | None]], ...]] = tuple(
            pool.submit(_probed_then_counted_or_dropped, client, owned.gateway.key, body, probed) for client in clients
        )
        eventually(held.qsize, lambda size: size == len(clients), seconds=30)
        ports: Final = frozenset(probed.get_nowait() for _ in clients)
        shares: Final = {pid: _accepted_client_ports(pid, proxy_url.port or 0) & ports for pid in workers}
        assert sum(map(len, shares.values())) == len(clients), shares
        victim: Final = min((pid for pid in workers if shares[pid]), key=lambda pid: len(shares[pid]))
        psutil.Process(victim).send_signal(signal.SIGKILL)
        release.set()
        for port, result in (future.result(timeout=60) for future in futures):
            assert result == (None if port in shares[victim] else (200, _MANTLE_COUNT)), (port, result, shares)
        second_wave: Final = _clients(stack, str(proxy_url), 6)
        assert tuple(_counted_on(client, owned.gateway.key, body) for client in second_wave) == (
            (200, _MANTLE_COUNT),
        ) * len(second_wave)
        assert len(_mantle_bodies(mantle)) == len(clients) + len(second_wave)
        eventually(lambda: len(_STARTED_WORKER.findall(owned.log.read_text())), lambda started: started >= 3, 120)
        assert owned.process.poll() is None


def _streamed_text(text: str) -> str:
    events: Final = tuple(
        _JSON_OBJECT.validate_json(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: {")
    )
    return "".join(_delta_content(event) for event in events)


def _delta_content(event: dict[str, JsonValue]) -> str:
    choices: Final = event.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    delta: Final = object_value(choices[0]).get("delta")
    if not isinstance(delta, dict):
        return ""
    content: Final = delta.get("content")
    return content if isinstance(content, str) else ""


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_completions_on_the_same_deployment_still_generate(
    counting_proxy: OwnedProxy, mantle_port: int, stream: bool
) -> None:
    gateway: Final = counting_proxy.gateway
    marker: Final = uuid.uuid4().hex
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"chat control marker-{marker}"}],
                "stream": stream,
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        generated: Final = _streamed_text(response.text) if stream else response.text
        assert answer(marker) in generated, response.text
        assert not stream or response.text.rstrip().endswith("data: [DONE]"), response.text
        (sent,) = runtime.drain()
        assert target_of(sent) == f"/model/{_OPUS}/{'converse-stream' if stream else 'converse'}", sent.target
        assert mantle.drain() == ()


def test_messages_endpoint_on_the_same_deployment_still_generates(counting_proxy: OwnedProxy, mantle_port: int) -> None:
    gateway: Final = counting_proxy.gateway
    marker: Final = uuid.uuid4().hex
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": f"marker-{marker}"}]},
        )
        assert response.status_code == 200, response.text
        assert answer(marker) in response.text, response.text
        (sent,) = runtime.drain()
        assert target_of(sent) == f"/model/{_OPUS}/invoke", sent.target
        assert mantle.drain() == ()


def test_responses_endpoint_on_the_same_deployment_still_generates(
    counting_proxy: OwnedProxy, mantle_port: int
) -> None:
    gateway: Final = counting_proxy.gateway
    marker: Final = uuid.uuid4().hex
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url)
        response: Final = gateway.request("POST", "/v1/responses", {"model": model, "input": f"marker-{marker}"})
        assert response.status_code == 200, response.text
        assert answer(marker) in response.text, response.text
        (sent,) = runtime.drain()
        assert target_of(sent) == f"/model/{_OPUS}/converse", sent.target
        assert mantle.drain() == ()


def test_utils_token_counter_local_mode_never_calls_either_peer(counting_proxy: OwnedProxy, mantle_port: int) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime) as runtime,
        wire_server(_mantle(), port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime.url)
        assert _local_count(gateway, _full(model)) > 0
        assert runtime.drain() == () and mantle.drain() == ()
