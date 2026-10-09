import asyncio
import json
import socket
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final, cast

import httpcore
import httpx
import psutil
import pytest
from anthropic.types import MessageParam
from integration._support.client import Gateway, Scenario, eventually, gateway_from_environment
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers._count_tokens_system_lift import (
    ANTHROPIC_VERSION,
    ASSISTANT,
    EPHEMERAL,
    FOLLOW_UP,
    IMAGE_PART,
    INSTRUCTION,
    JSON_OBJECT,
    LEADING,
    LIFT_CASES,
    LIFTED,
    MID_SYSTEM,
    REJECTION,
    REMINDER,
    STRING_CASE,
    TOKEN_COUNTING_BETA,
    TOOLS,
    USER,
    USER_TEXT,
    LiftCase,
    accepts_count_body,
    anthropic_client,
    async_anthropic_client,
    async_openai_client,
    count_request,
    expected_count_body,
    openai_client,
)
from pydantic import JsonValue, TypeAdapter

_MODEL: Final = "claude-opus-5-5"
_API_KEY: Final = "synthetic-count-tokens-key"
_COUNT: Final = 3131
_PROVIDER_TOKENIZERS: Final = frozenset({"anthropic_api", "azure_ai_anthropic_api"})
_SDK_MESSAGES: Final = cast(
    list[MessageParam], [LEADING, USER]
)  # cast-ok: the SDK types reject the role the proxy lifts
_STRING_BODY: Final = expected_count_body(_MODEL, STRING_CASE)
_PROXY_MODULE: Final = "integration._support.proxy"
_PROBES_PER_ROUND: Final = 8
_CLIENT_ADDRESS: Final = TypeAdapter(tuple[str, int])
_ARGUMENTS: Final = TypeAdapter(tuple[str, ...])
_NAME: Final = TypeAdapter(str)


@dataclass(frozen=True, slots=True)
class _Provider:
    prefix: str
    target: str
    tokenizer: str
    azure: bool


@dataclass(frozen=True, slots=True)
class _Deployment:
    provider: _Provider
    port: int
    model: str


_ANTHROPIC: Final = _Provider("anthropic", "/v1/messages/count_tokens", "anthropic_api", False)
_AZURE: Final = _Provider("azure_ai", "/anthropic/v1/messages/count_tokens", "azure_ai_anthropic_api", True)
_PROVIDERS: Final = MappingProxyType({"anthropic": _ANTHROPIC, "azure_ai": _AZURE})


@pytest.fixture(params=_PROVIDERS.keys())
def provider(request: pytest.FixtureRequest) -> _Provider:
    name: Final[object] = request.param  # pyright: ignore[reportAny]  # pytest types the fixture param as Any
    return _PROVIDERS[_NAME.validate_python(name)]


def _json_reply(status: int, payload: Mapping[str, JsonValue]) -> Reply:
    return Reply(status=status, body=json.dumps(payload).encode())


def _rejected(status: int) -> Reply:
    return _json_reply(status, {"type": "error", "error": {"type": "invalid_request_error", "message": REJECTION}})


def _counted(request: Request) -> Reply:
    accepted: Final = accepts_count_body(JSON_OBJECT.validate_json(request.body))
    return _json_reply(200, {"input_tokens": _COUNT}) if accepted else _rejected(400)


def _rejecting(status: int) -> Callable[[Request], Reply]:
    def count(_request: Request) -> Reply:
        return _rejected(status)

    return count


def _holding(held: SimpleQueue[str], release: threading.Event, seconds: float) -> Callable[[Request], Reply]:
    def hold(request: Request) -> Reply:
        held.put(request.target)
        assert release.wait(timeout=seconds), "Held count was never released"
        return _counted(request)

    return hold


def _peer(provider: _Provider, count: Callable[[Request], Reply] = _counted) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.target == provider.target:
            return count(request)
        return _json_reply(404, {"error": f"unscripted target {request.target}"})

    return respond


def _listening(deployment: _Deployment, count: Callable[[Request], Reply] = _counted) -> AbstractContextManager[Wire]:
    return wire_server(_peer(deployment.provider, count), port=deployment.port)


def _reserved_port() -> int:
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        return _CLIENT_ADDRESS.validate_python(reserve.getsockname())[1]


def _cmdline(process: psutil.Process) -> tuple[str, ...]:
    try:
        return _ARGUMENTS.validate_python(process.cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return ()


def _serves(cmdline: tuple[str, ...], proxy_port: int) -> bool:
    if _PROXY_MODULE not in cmdline or "--port" not in cmdline:
        return False
    return cmdline[cmdline.index("--port") + 1] == str(proxy_port)


def _listens(process: psutil.Process, proxy_port: int) -> bool:
    return any(
        connection.status == psutil.CONN_LISTEN and connection.laddr and connection.laddr.port == proxy_port
        for connection in process.net_connections(kind="tcp")
    )


def _proxy_workers(proxy_port: int) -> frozenset[int]:
    (master,) = tuple(process for process in psutil.process_iter() if _serves(_cmdline(process), proxy_port))
    spawned: Final = frozenset(child.pid for child in master.children() if _listens(child, proxy_port))
    return spawned or frozenset({master.pid})


def _holder(workers: frozenset[int], client_port: int) -> int | None:
    def holds(pid: int) -> bool:
        return any(
            connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == client_port
            for connection in psutil.Process(pid).net_connections(kind="tcp")
        )

    return next((pid for pid in sorted(workers) if holds(pid)), None)


def _client_port(response: httpx.Response) -> int:
    stream: Final[object] = response.extensions["network_stream"]  # pyright: ignore[reportAny]  # httpx types extensions as Any
    assert isinstance(stream, httpcore.NetworkStream), stream
    return _CLIENT_ADDRESS.validate_python(stream.get_extra_info("client_addr"))[1]


def _probe(gateway: Gateway, body: Mapping[str, JsonValue], workers: frozenset[int]) -> tuple[int | None, JsonValue]:
    with httpx.Client(base_url=str(gateway.client.base_url), timeout=30, trust_env=False) as client:
        response: Final = client.post(
            "/v1/messages/count_tokens", json=dict(body), headers={"Authorization": f"Bearer {gateway.key}"}
        )
        return _holder(workers, _client_port(response)), JSON_OBJECT.validate_json(response.content).get("input_tokens")


def _round(
    gateway: Gateway, body: Mapping[str, JsonValue], workers: frozenset[int]
) -> tuple[tuple[int | None, JsonValue], ...]:
    def probe(_index: int) -> tuple[int | None, JsonValue]:
        return _probe(gateway, body, workers)

    with ThreadPoolExecutor(max_workers=_PROBES_PER_ROUND) as pool:
        return tuple(pool.map(probe, range(_PROBES_PER_ROUND)))


def _settled_on_every_worker(gateway: Gateway, body: Mapping[str, JsonValue]) -> None:
    proxy_port: Final = gateway.client.base_url.port
    assert proxy_port is not None, gateway.client.base_url
    workers: Final = _proxy_workers(proxy_port)
    eventually(
        lambda: _round(gateway, body, workers),
        lambda observed: (
            frozenset(pid for pid, _ in observed) == workers and all(count == _COUNT for _, count in observed)
        ),
        seconds=60,
    )


def _deploy(gateway: Gateway, scenario: Scenario, provider: _Provider) -> _Deployment:
    port: Final = _reserved_port()
    model: Final = scenario.model(
        model_info=None, model=f"{provider.prefix}/{_MODEL}", api_base=f"http://127.0.0.1:{port}", api_key=_API_KEY
    )
    with wire_server(_peer(provider), port=port):
        _settled_on_every_worker(gateway, {"model": model, "messages": [USER]})
    return _Deployment(provider, port, model)


@pytest.fixture(scope="module")
def deployments() -> Iterator[Mapping[str, _Deployment]]:
    with gateway_from_environment() as gateway, gateway.scenario() as scenario:
        yield MappingProxyType({chosen.prefix: _deploy(gateway, scenario, chosen) for chosen in (_ANTHROPIC, _AZURE)})


@pytest.fixture
def deployment(provider: _Provider, deployments: Mapping[str, _Deployment]) -> _Deployment:
    return deployments[provider.prefix]


def _count(gateway: Gateway, body: Mapping[str, JsonValue], key: str | None = None) -> httpx.Response:
    return gateway.request("POST", "/v1/messages/count_tokens", body, key=key)


def _payload(response: httpx.Response) -> dict[str, JsonValue]:
    assert response.status_code == 200, response.text
    return JSON_OBJECT.validate_json(response.content)


def _local_count(gateway: Gateway, body: Mapping[str, JsonValue]) -> int:
    response: Final = gateway.request("POST", "/utils/token_counter", body, params={"call_endpoint": "false"})
    payload: Final = _payload(response)
    total: Final = payload["total_tokens"]
    assert payload["tokenizer_type"] not in _PROVIDER_TOKENIZERS, response.text
    assert isinstance(total, int) and total > 0 and total != _COUNT, response.text
    return total


def _bodies(wire: Wire, provider: _Provider) -> tuple[dict[str, JsonValue], ...]:
    received: Final = wire.drain()
    for request in received:
        assert (request.method, request.target) == ("POST", provider.target), request.target
        assert request.headers["anthropic-version"] == ANTHROPIC_VERSION, request.headers
        assert TOKEN_COUNTING_BETA in request.headers["anthropic-beta"], request.headers
        assert request.headers["content-type"] == "application/json", request.headers
        assert request.headers["x-api-key"] == _API_KEY, request.headers
        assert (request.headers.get("api-key") == _API_KEY) is provider.azure, request.headers
    return tuple(JSON_OBJECT.validate_json(request.body) for request in received)


def _clients(stack: ExitStack, base_url: str, count: int) -> tuple[httpx.Client, ...]:
    return tuple(
        stack.enter_context(httpx.Client(base_url=base_url, timeout=30, trust_env=False)) for _ in range(count)
    )


def _counted_on(client: httpx.Client, key: str, body: Mapping[str, JsonValue]) -> tuple[int, JsonValue]:
    response: Final = client.post(
        "/v1/messages/count_tokens", json=dict(body), headers={"Authorization": f"Bearer {key}"}
    )
    return response.status_code, JSON_OBJECT.validate_json(response.content).get("input_tokens")


@pytest.mark.parametrize("case", LIFT_CASES.values(), ids=LIFT_CASES.keys())
def test_messages_count_tokens_lifts_the_leading_system_run(
    deployment: _Deployment, gateway: Gateway, case: LiftCase
) -> None:
    with _listening(deployment) as peer:
        response: Final = _count(gateway, count_request(deployment.model, case))
        assert _payload(response) == {"input_tokens": _COUNT}, response.text
        assert _bodies(peer, deployment.provider) == (expected_count_body(_MODEL, case),)


def test_anthropic_sdk_count_tokens_lifts_the_leading_system_message(deployment: _Deployment, gateway: Gateway) -> None:
    with _listening(deployment) as peer:
        counted: Final = anthropic_client(gateway).messages.count_tokens(model=deployment.model, messages=_SDK_MESSAGES)
        assert counted.input_tokens == _COUNT, counted
        assert _bodies(peer, deployment.provider) == (_STRING_BODY,)


def test_async_anthropic_sdk_count_tokens_lifts_the_leading_system_message(
    deployment: _Deployment, gateway: Gateway
) -> None:
    with _listening(deployment) as peer:
        counted: Final = asyncio.run(
            async_anthropic_client(gateway).messages.count_tokens(model=deployment.model, messages=_SDK_MESSAGES)
        )
        assert counted.input_tokens == _COUNT, counted
        assert _bodies(peer, deployment.provider) == (_STRING_BODY,)


def test_utils_token_counter_call_endpoint_counts_a_leading_system_through_the_provider(
    deployment: _Deployment, gateway: Gateway
) -> None:
    with _listening(deployment) as peer:
        response: Final = gateway.request(
            "POST",
            "/utils/token_counter",
            count_request(deployment.model, STRING_CASE),
            params={"call_endpoint": "true"},
        )
        payload: Final = _payload(response)
        expected_tokenizer: Final = deployment.provider.tokenizer
        assert (payload["total_tokens"], payload["tokenizer_type"]) == (_COUNT, expected_tokenizer), response.text
        assert payload["original_response"] == {"input_tokens": _COUNT}, response.text
        assert (payload["request_model"], payload["model_used"]) == (deployment.model, _MODEL), response.text
        assert _bodies(peer, deployment.provider) == (_STRING_BODY,)


def test_utils_token_counter_local_mode_never_calls_the_peer_for_a_leading_system(
    deployment: _Deployment, gateway: Gateway
) -> None:
    with _listening(deployment) as peer:
        assert _local_count(gateway, count_request(deployment.model, STRING_CASE)) > 0
        assert peer.drain() == ()


def test_responses_input_tokens_lifts_instructions(deployment: _Deployment, gateway: Gateway) -> None:
    with _listening(deployment) as peer:
        response: Final = gateway.request(
            "POST",
            "/v1/responses/input_tokens",
            {"model": deployment.model, "input": USER_TEXT, "instructions": INSTRUCTION},
        )
        assert _payload(response) == {"object": "response.input_tokens", "input_tokens": _COUNT}, response.text
        assert _bodies(peer, deployment.provider) == (_STRING_BODY,)


def test_responses_input_tokens_lifts_instructions_ahead_of_a_leading_system_item(
    deployment: _Deployment, gateway: Gateway
) -> None:
    with _listening(deployment) as peer:
        response: Final = gateway.request(
            "POST",
            "/v1/responses/input_tokens",
            {"model": deployment.model, "input": [MID_SYSTEM, USER], "instructions": INSTRUCTION},
        )
        assert _payload(response) == {"object": "response.input_tokens", "input_tokens": _COUNT}, response.text
        assert _bodies(peer, deployment.provider) == (
            {"model": _MODEL, "messages": [USER], "system": [LIFTED, {"type": "text", "text": REMINDER}]},
        )


def test_openai_sdk_input_tokens_lifts_instructions(deployment: _Deployment, gateway: Gateway) -> None:
    with _listening(deployment) as peer:
        counted: Final = openai_client(gateway).responses.input_tokens.count(
            model=deployment.model, input=USER_TEXT, instructions=INSTRUCTION
        )
        assert counted.input_tokens == _COUNT, counted
        assert _bodies(peer, deployment.provider) == (_STRING_BODY,)


def test_async_openai_sdk_input_tokens_lifts_instructions(deployment: _Deployment, gateway: Gateway) -> None:
    with _listening(deployment) as peer:
        counted: Final = asyncio.run(
            async_openai_client(gateway).responses.input_tokens.count(
                model=deployment.model, input=USER_TEXT, instructions=INSTRUCTION
            )
        )
        assert counted.input_tokens == _COUNT, counted
        assert _bodies(peer, deployment.provider) == (_STRING_BODY,)


def test_messages_count_tokens_forwards_tools_beside_the_lifted_system(
    deployment: _Deployment, gateway: Gateway
) -> None:
    with _listening(deployment) as peer:
        response: Final = _count(gateway, {**count_request(deployment.model, STRING_CASE), "tools": TOOLS})
        assert _payload(response) == {"input_tokens": _COUNT}, response.text
        assert _bodies(peer, deployment.provider) == ({**_STRING_BODY, "tools": TOOLS},)


def test_messages_count_tokens_keeps_a_mid_conversation_system_in_place(
    deployment: _Deployment, gateway: Gateway
) -> None:
    with _listening(deployment) as peer:
        body: Final[dict[str, JsonValue]] = {
            "model": deployment.model,
            "messages": [LEADING, USER, MID_SYSTEM, ASSISTANT, FOLLOW_UP],
        }
        response: Final = _count(gateway, body)
        assert _payload(response) == {"input_tokens": _COUNT}, response.text
        assert _bodies(peer, deployment.provider) == (
            {"model": _MODEL, "messages": [USER, MID_SYSTEM, ASSISTANT, FOLLOW_UP], "system": [LIFTED]},
        )


def test_messages_count_tokens_falls_back_locally_when_every_message_is_system(
    deployment: _Deployment, gateway: Gateway
) -> None:
    with _listening(deployment) as peer:
        body: Final[dict[str, JsonValue]] = {"model": deployment.model, "messages": [LEADING]}
        local: Final = _local_count(gateway, body)
        response: Final = _count(gateway, body)
        assert _payload(response) == {"input_tokens": local}, response.text
        assert _bodies(peer, deployment.provider) == ({"model": _MODEL, "messages": [], "system": [LIFTED]},)


def test_messages_count_tokens_leaves_the_request_untouched_for_a_non_text_system(
    deployment: _Deployment, gateway: Gateway
) -> None:
    with _listening(deployment) as peer:
        local: Final = _local_count(gateway, count_request(deployment.model, STRING_CASE))
        response: Final = _count(gateway, {**count_request(deployment.model, STRING_CASE), "system": 5})
        assert _payload(response) == {"input_tokens": local}, response.text
        assert _bodies(peer, deployment.provider) == ({"model": _MODEL, "messages": [LEADING, USER], "system": 5},)


def test_messages_count_tokens_repeated_request_lifts_each_time(deployment: _Deployment, gateway: Gateway) -> None:
    with _listening(deployment) as peer:
        answers: Final = tuple(
            _payload(_count(gateway, count_request(deployment.model, STRING_CASE))) for _ in range(2)
        )
        assert answers == ({"input_tokens": _COUNT},) * 2
        assert _bodies(peer, deployment.provider) == (_STRING_BODY,) * 2


def test_messages_count_tokens_answers_a_leading_system_without_content_before_any_peer_call(
    deployment: _Deployment, gateway: Gateway
) -> None:
    with _listening(deployment) as peer:
        body: Final[dict[str, JsonValue]] = {"model": deployment.model, "messages": [{"role": "system"}, USER]}
        local: Final = _local_count(gateway, body)
        response: Final = _count(gateway, body)
        assert _payload(response) == {"input_tokens": local}, response.text
        assert peer.drain() == ()
        follow_up: Final = _count(gateway, count_request(deployment.model, STRING_CASE))
        assert _payload(follow_up) == {"input_tokens": _COUNT}, follow_up.text
        assert _bodies(peer, deployment.provider) == (_STRING_BODY,)


def test_messages_count_tokens_duplicate_messages_key_lifts_the_last_value(
    deployment: _Deployment, gateway: Gateway
) -> None:
    with _listening(deployment) as peer:
        first: Final = json.dumps([USER])
        last: Final = json.dumps([LEADING, USER])
        response: Final = gateway.client.post(
            "/v1/messages/count_tokens",
            content=f'{{"model": "{deployment.model}", "messages": {first}, "messages": {last}}}',
            headers={"Authorization": f"Bearer {gateway.key}", "Content-Type": "application/json"},
        )
        assert _payload(response) == {"input_tokens": _COUNT}, response.text
        assert _bodies(peer, deployment.provider) == (_STRING_BODY,)


@pytest.mark.parametrize("status", [400, 403, 404, 500, 503])
def test_messages_count_tokens_falls_back_locally_when_the_peer_rejects_the_lifted_body(
    deployment: _Deployment, gateway: Gateway, status: int
) -> None:
    with _listening(deployment, _rejecting(status)) as peer:
        body: Final = count_request(deployment.model, STRING_CASE)
        local: Final = _local_count(gateway, body)
        response: Final = _count(gateway, body)
        assert _payload(response) == {"input_tokens": local}, response.text
        assert _bodies(peer, deployment.provider) == (_STRING_BODY,)


def test_messages_count_tokens_unauthenticated_request_never_reaches_the_peer(
    deployment: _Deployment, gateway: Gateway
) -> None:
    with _listening(deployment) as peer:
        response: Final = _count(gateway, count_request(deployment.model, STRING_CASE), key="sk-not-a-key")
        assert response.status_code == 401, response.text
        assert peer.drain() == ()


def test_peer_outage_between_concurrent_waves_falls_back_then_recovers(
    deployment: _Deployment, gateway: Gateway
) -> None:
    with ExitStack() as stack:
        clients: Final = _clients(stack, str(gateway.client.base_url), 8)
        pool: Final = stack.enter_context(ThreadPoolExecutor(max_workers=len(clients)))
        body: Final = count_request(deployment.model, STRING_CASE)
        local: Final = _local_count(gateway, body)

        def count(client: httpx.Client) -> tuple[int, JsonValue]:
            return _counted_on(client, gateway.key, body)

        with _listening(deployment) as peer:
            assert tuple(pool.map(count, clients)) == ((200, _COUNT),) * len(clients)
            assert _bodies(peer, deployment.provider) == (_STRING_BODY,) * len(clients)
        assert tuple(pool.map(count, clients)) == ((200, local),) * len(clients)
        with _listening(deployment) as revived:
            assert tuple(pool.map(count, clients)) == ((200, _COUNT),) * len(clients)
            assert _bodies(revived, deployment.provider) == (_STRING_BODY,) * len(clients)


def test_slow_peer_holds_concurrent_lifted_counts_without_stalling_the_proxy(
    deployment: _Deployment, gateway: Gateway
) -> None:
    held: Final[SimpleQueue[str]] = SimpleQueue()
    release: Final = threading.Event()
    with ExitStack() as stack:
        clients: Final = _clients(stack, str(gateway.client.base_url), 6)
        peer: Final = stack.enter_context(_listening(deployment, _holding(held, release, 20)))
        pool: Final = stack.enter_context(ThreadPoolExecutor(max_workers=len(clients)))
        stack.callback(release.set)
        body: Final = count_request(deployment.model, STRING_CASE)
        futures: Final = tuple(pool.submit(_counted_on, client, gateway.key, body) for client in clients)
        eventually(held.qsize, lambda size: size == len(clients), seconds=30)
        assert gateway.request("GET", "/health/liveliness").status_code == 200
        assert _local_count(gateway, body) > 0
        assert not any(future.done() for future in futures)
        release.set()
        assert tuple(future.result(timeout=30) for future in futures) == ((200, _COUNT),) * len(clients)
        assert _bodies(peer, deployment.provider) == (_STRING_BODY,) * len(clients)


def test_chat_completions_on_the_same_deployment_keeps_a_mid_conversation_system_in_place(
    deployments: Mapping[str, _Deployment], gateway: Gateway
) -> None:
    identity: Final = f"msg_{uuid.uuid4().hex}"
    anthropic_deployment: Final = deployments[_ANTHROPIC.prefix]

    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", "/v1/messages"), request.target
        return _json_reply(
            200,
            {
                "id": identity,
                "type": "message",
                "role": "assistant",
                "model": _MODEL,
                "content": [{"type": "text", "text": "done"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 12, "output_tokens": 3},
            },
        )

    with wire_server(respond, port=anthropic_deployment.port) as peer:
        reminder: Final[dict[str, JsonValue]] = {
            "role": "system",
            "content": [
                {"type": "text", "text": REMINDER, "cache_control": EPHEMERAL},
                {"type": "text", "text": ""},
                IMAGE_PART,
            ],
        }
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": anthropic_deployment.model,
                "max_tokens": 16,
                "messages": [LEADING, USER, reminder, ASSISTANT, FOLLOW_UP],
            },
        )
        assert response.status_code == 200, response.text
        (sent,) = peer.drain()
        body: Final = JSON_OBJECT.validate_json(sent.body)
        assert body["system"] == [LIFTED], body
        assert body["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": USER_TEXT}]},
            {"role": "system", "content": [{"type": "text", "text": REMINDER, "cache_control": EPHEMERAL}]},
            {"role": "assistant", "content": [{"type": "text", "text": "One."}]},
            {"role": "user", "content": [{"type": "text", "text": "Again"}]},
        ], body
