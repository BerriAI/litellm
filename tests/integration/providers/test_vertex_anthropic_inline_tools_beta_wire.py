import asyncio
import json
import re
import signal
import threading
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from typing import Final

import anthropic
import httpx
import openai
import psutil
import pytest
import yaml
from integration._support.client import Gateway, Scenario, eventually
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_INLINE_TOOLS: Final = "inline-tools-2026-09-15"
_THINKING: Final = "interleaved-thinking-2025-05-14"
_VERTEX_UNSUPPORTED: Final = "effort-2025-11-24"
_UNKNOWN: Final = "not-a-beta-2099-01-01"
_BACKEND: Final = "claude-opus-5-5"
_PROJECT: Final = "scripted-project"
_LOCATION: Final = "us-east5"
_MODEL_PATH: Final = f"/v1/projects/{_PROJECT}/locations/{_LOCATION}/publishers/anthropic/models/{_BACKEND}"
_VERTEX_TARGET: Final = f"{_MODEL_PATH}:rawPredict"
_VERTEX_STREAM_TARGET: Final = f"{_MODEL_PATH}:streamRawPredict?alt=sse"
_ANTHROPIC_TARGET: Final = "/v1/messages"
_ANTHROPIC_KEY: Final = "synthetic-anthropic-key"
_REPLY_TEXT: Final = "inline tools beta control"
_OWNED_MODEL: Final = "partner-claude"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_OWNED_PROXY_CELL_SECONDS: Final = 2 * graceful_stop_seconds() + 120

_REPLY: Final[dict[str, JsonValue]] = {
    "id": "msg_inline_tools_beta",
    "type": "message",
    "role": "assistant",
    "model": _BACKEND,
    "content": [{"type": "text", "text": _REPLY_TEXT}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 5, "output_tokens": 3},
}
_EVENTS: Final[tuple[tuple[str, dict[str, JsonValue]], ...]] = (
    (
        "message_start",
        {
            "type": "message_start",
            "message": {**_REPLY, "content": [], "stop_reason": None, "usage": {"input_tokens": 5, "output_tokens": 1}},
        },
    ),
    ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
    (
        "content_block_delta",
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": _REPLY_TEXT}},
    ),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    (
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 3},
        },
    ),
    ("message_stop", {"type": "message_stop"}),
)
_SSE: Final = tuple(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode() for name, data in _EVENTS)
_REPLY_BODY: Final = json.dumps(_REPLY).encode()


@dataclass(frozen=True, slots=True)
class _Delivery:
    target: str
    beta: str | None
    body: str


def _streams(request: Request) -> bool:
    return request.target == _VERTEX_STREAM_TARGET or _JSON_OBJECT.validate_json(request.body).get("stream") is True


def _peer(request: Request) -> Reply:
    if request.target in (_VERTEX_TARGET, _VERTEX_STREAM_TARGET):
        assert request.headers["authorization"] == "Bearer scripted-token", request.headers
    elif request.target == _ANTHROPIC_TARGET:
        assert request.headers["x-api-key"] == _ANTHROPIC_KEY, request.headers
    else:
        return Reply(status=404, body=json.dumps({"error": f"unscripted target {request.target}"}).encode())
    return Reply(content_type="text/event-stream", chunks=_SSE) if _streams(request) else Reply(body=_REPLY_BODY)


def _holding(held: SimpleQueue[str], release: threading.Event) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        held.put(request.target)
        assert release.wait(timeout=60), "Held request was never released"
        return _peer(request)

    return respond


def _delivered(wire: Wire) -> tuple[_Delivery, ...]:
    return tuple(
        _Delivery(request.target, request.headers.get("anthropic-beta"), request.body.decode())
        for request in wire.drain()
    )


def _one_delivery(wire: Wire, target: str, marker: str) -> str | None:
    (sent,) = _delivered(wire)
    assert sent.target == target, sent.target
    assert marker in sent.body, sent.body
    return sent.beta


def _vertex_deployment(gateway: Gateway, scenario: Scenario, api_base: str) -> str:
    return scenario.model(
        model=f"vertex_ai/{_BACKEND}",
        api_base=api_base,
        api_key=None,
        vertex_project=_PROJECT,
        vertex_location=_LOCATION,
        vertex_credentials=service_account_json(_PROJECT, gateway.upstream_url.rstrip("/")),
    )


def _anthropic_deployment(scenario: Scenario, api_base: str) -> str:
    return scenario.model(model=f"anthropic/{_BACKEND}", api_base=api_base, api_key=_ANTHROPIC_KEY)


def _marker(label: str) -> str:
    return f"{label} {uuid.uuid4().hex}"


def _messages_body(model: str, marker: str, stream: bool = False) -> dict[str, JsonValue]:
    return {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": marker}], "stream": stream}


def _chat_body(model: str, marker: str, stream: bool = False) -> dict[str, JsonValue]:
    return {"model": model, "messages": [{"role": "user", "content": marker}], "stream": stream}


def _responses_body(model: str, marker: str, stream: bool = False) -> dict[str, JsonValue]:
    return {"model": model, "input": marker, "stream": stream}


def _beta_headers(value: str | None) -> dict[str, str]:
    return {} if value is None else {"anthropic-beta": value}


def _generated(gateway: Gateway, path: str, body: Mapping[str, JsonValue], beta: str | None) -> httpx.Response:
    response: Final = gateway.request("POST", path, body, headers=_beta_headers(beta))
    assert response.status_code == 200, response.text
    assert _REPLY_TEXT in response.text, response.text
    return response


def _vertex_target(stream: bool) -> str:
    return _VERTEX_STREAM_TARGET if stream else _VERTEX_TARGET


def _clients(stack: ExitStack, base_url: str, count: int) -> tuple[httpx.Client, ...]:
    return tuple(
        stack.enter_context(httpx.Client(base_url=base_url, timeout=60, trust_env=False)) for _ in range(count)
    )


def _post(client: httpx.Client, key: str, path: str, body: Mapping[str, JsonValue], beta: str) -> int:
    return client.post(
        path, json=dict(body), headers={"Authorization": f"Bearer {key}", **_beta_headers(beta)}
    ).status_code


def _post_or_dropped(client: httpx.Client, key: str, path: str, body: Mapping[str, JsonValue], beta: str) -> int | None:
    try:
        return _post(client, key, path, body, beta)
    except httpx.TransportError:
        return None


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


def _burst_bodies(model: str, label: str, count: int) -> tuple[tuple[str, dict[str, JsonValue], str], ...]:
    markers: Final = tuple(_marker(label) for _ in range(count))
    return tuple(
        (
            "/v1/messages" if index % 2 == 0 else "/v1/chat/completions",
            _messages_body(model, marker, stream=index % 4 == 2)
            if index % 2 == 0
            else _chat_body(model, marker, stream=index % 4 == 3),
            marker,
        )
        for index, marker in enumerate(markers)
    )


def _owned_config(path: Path, gateway: Gateway, api_base: str) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    path.write_text(
        yaml.safe_dump(
            {
                **config,
                "model_list": [
                    {
                        "model_name": _OWNED_MODEL,
                        "litellm_params": {
                            "model": f"vertex_ai/{_BACKEND}",
                            "api_base": api_base,
                            "vertex_project": _PROJECT,
                            "vertex_location": _LOCATION,
                            "vertex_credentials": service_account_json(_PROJECT, gateway.upstream_url.rstrip("/")),
                        },
                    }
                ],
            }
        )
    )
    return path


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_vertex_messages_forwards_the_inline_tools_beta(gateway: Gateway, stream: bool) -> None:
    marker: Final = _marker("vertex messages")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url)
        response: Final = _generated(gateway, "/v1/messages", _messages_body(model, marker, stream), _INLINE_TOOLS)
        assert not stream or "event: message_stop" in response.text, response.text
        assert _one_delivery(wire, _vertex_target(stream), marker) == _INLINE_TOOLS


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_vertex_chat_completions_forwards_the_inline_tools_beta(gateway: Gateway, stream: bool) -> None:
    marker: Final = _marker("vertex chat")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url)
        response: Final = _generated(gateway, "/v1/chat/completions", _chat_body(model, marker, stream), _INLINE_TOOLS)
        assert not stream or response.text.rstrip().endswith("data: [DONE]"), response.text
        assert _one_delivery(wire, _vertex_target(stream), marker) == _INLINE_TOOLS


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_vertex_responses_forwards_the_inline_tools_beta(gateway: Gateway, stream: bool) -> None:
    marker: Final = _marker("vertex responses")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url)
        _generated(gateway, "/v1/responses", _responses_body(model, marker, stream), _INLINE_TOOLS)
        assert _one_delivery(wire, _vertex_target(stream), marker) == _INLINE_TOOLS


def test_vertex_messages_through_the_anthropic_sdk_forwards_the_inline_tools_beta(gateway: Gateway) -> None:
    marker: Final = _marker("anthropic sdk sync")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url)
        with anthropic.Anthropic(
            base_url=str(gateway.client.base_url),
            api_key=gateway.key,
            max_retries=0,
            http_client=httpx.Client(trust_env=False, timeout=60),
        ) as client:
            reply: Final = client.beta.messages.create(
                model=model, max_tokens=16, messages=[{"role": "user", "content": marker}], betas=[_INLINE_TOOLS]
            )
        assert _REPLY_TEXT in reply.model_dump_json(), reply
        assert _one_delivery(wire, _VERTEX_TARGET, marker) == _INLINE_TOOLS


def test_vertex_messages_through_the_async_anthropic_sdk_forwards_the_inline_tools_beta(gateway: Gateway) -> None:
    marker: Final = _marker("anthropic sdk async")

    async def generate(model: str) -> str:
        async with anthropic.AsyncAnthropic(
            base_url=str(gateway.client.base_url),
            api_key=gateway.key,
            max_retries=0,
            http_client=httpx.AsyncClient(trust_env=False, timeout=60),
        ) as client:
            reply: Final = await client.beta.messages.create(
                model=model, max_tokens=16, messages=[{"role": "user", "content": marker}], betas=[_INLINE_TOOLS]
            )
        return reply.model_dump_json()

    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url)
        assert _REPLY_TEXT in asyncio.run(generate(model))
        assert _one_delivery(wire, _VERTEX_TARGET, marker) == _INLINE_TOOLS


def test_vertex_chat_through_the_openai_sdk_forwards_the_inline_tools_beta(gateway: Gateway) -> None:
    marker: Final = _marker("openai sdk sync")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url)
        with openai.OpenAI(
            base_url=f"{gateway.client.base_url}/v1",
            api_key=gateway.key,
            max_retries=0,
            http_client=httpx.Client(trust_env=False, timeout=60),
        ) as client:
            reply: Final = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": marker}], extra_headers=_beta_headers(_INLINE_TOOLS)
            )
        assert reply.choices[0].message.content == _REPLY_TEXT, reply
        assert _one_delivery(wire, _VERTEX_TARGET, marker) == _INLINE_TOOLS


def test_vertex_chat_through_the_async_openai_sdk_forwards_the_inline_tools_beta(gateway: Gateway) -> None:
    marker: Final = _marker("openai sdk async")

    async def generate(model: str) -> str | None:
        async with openai.AsyncOpenAI(
            base_url=f"{gateway.client.base_url}/v1",
            api_key=gateway.key,
            max_retries=0,
            http_client=httpx.AsyncClient(trust_env=False, timeout=60),
        ) as client:
            reply: Final = await client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": marker}], extra_headers=_beta_headers(_INLINE_TOOLS)
            )
        return reply.choices[0].message.content

    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url)
        assert asyncio.run(generate(model)) == _REPLY_TEXT
        assert _one_delivery(wire, _VERTEX_TARGET, marker) == _INLINE_TOOLS


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_anthropic_chat_completions_forwards_the_inline_tools_beta(gateway: Gateway, stream: bool) -> None:
    marker: Final = _marker("anthropic chat")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire.url)
        response: Final = _generated(gateway, "/v1/chat/completions", _chat_body(model, marker, stream), _INLINE_TOOLS)
        assert not stream or response.text.rstrip().endswith("data: [DONE]"), response.text
        assert _one_delivery(wire, _ANTHROPIC_TARGET, marker) == _INLINE_TOOLS


def test_anthropic_messages_passes_the_inline_tools_beta_through_unfiltered(gateway: Gateway) -> None:
    marker: Final = _marker("anthropic messages")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire.url)
        _generated(gateway, "/v1/messages", _messages_body(model, marker), f"{_INLINE_TOOLS},{_UNKNOWN}")
        assert _one_delivery(wire, _ANTHROPIC_TARGET, marker) == f"{_INLINE_TOOLS},{_UNKNOWN}"


def test_vertex_messages_drops_an_unsupported_beta_sent_next_to_inline_tools(gateway: Gateway) -> None:
    marker: Final = _marker("vertex unsupported sibling")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url)
        _generated(gateway, "/v1/messages", _messages_body(model, marker), f"{_VERTEX_UNSUPPORTED},{_INLINE_TOOLS}")
        assert _one_delivery(wire, _VERTEX_TARGET, marker) == _INLINE_TOOLS


def test_vertex_messages_forwards_inline_tools_with_another_supported_beta(gateway: Gateway) -> None:
    marker: Final = _marker("vertex supported sibling")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url)
        _generated(gateway, "/v1/messages", _messages_body(model, marker), f"{_THINKING},{_INLINE_TOOLS}")
        assert _one_delivery(wire, _VERTEX_TARGET, marker) == f"{_INLINE_TOOLS},{_THINKING}"


@pytest.mark.parametrize(
    "beta",
    [None, _UNKNOWN, "", "5", f"{_UNKNOWN}," * 240, _INLINE_TOOLS.upper()],
    ids=["absent", "unknown", "empty", "int", "5kb_junk", "upper_case"],
)
def test_vertex_messages_sends_no_beta_header_when_nothing_survives_the_filter(
    gateway: Gateway, beta: str | None
) -> None:
    marker: Final = _marker("vertex nothing survives")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url)
        _generated(gateway, "/v1/messages", _messages_body(model, marker), beta)
        assert _one_delivery(wire, _VERTEX_TARGET, marker) is None


@pytest.mark.parametrize(
    "beta",
    [f"{_INLINE_TOOLS} , {_INLINE_TOOLS}", f"{_UNKNOWN}," * 240 + _INLINE_TOOLS],
    ids=["duplicated_with_spaces", "inside_5kb_junk"],
)
def test_vertex_messages_forwards_inline_tools_once_from_a_hostile_header_value(gateway: Gateway, beta: str) -> None:
    marker: Final = _marker("vertex hostile value")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url)
        _generated(gateway, "/v1/messages", _messages_body(model, marker), beta)
        assert _one_delivery(wire, _VERTEX_TARGET, marker) == _INLINE_TOOLS


def test_vertex_messages_forwards_inline_tools_once_from_a_repeated_header_line(gateway: Gateway) -> None:
    marker: Final = _marker("vertex repeated line")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url)
        response: Final = gateway.client.post(
            "/v1/messages",
            json=_messages_body(model, marker),
            headers=httpx.Headers(
                [
                    ("Authorization", f"Bearer {gateway.key}"),
                    ("anthropic-beta", _INLINE_TOOLS),
                    ("anthropic-beta", _INLINE_TOOLS),
                ]
            ),
        )
        assert response.status_code == 200, response.text
        assert _one_delivery(wire, _VERTEX_TARGET, marker) == _INLINE_TOOLS


def test_vertex_messages_unauthenticated_request_with_the_beta_never_reaches_the_peer(gateway: Gateway) -> None:
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            _messages_body(model, _marker("unauthenticated")),
            key="sk-not-a-key-this-proxy-issued",
            headers=_beta_headers(_INLINE_TOOLS),
        )
        assert response.status_code == 401, response.text
        assert wire.drain() == ()


def test_reloading_the_allowlist_keeps_forwarding_inline_tools(gateway: Gateway) -> None:
    marker: Final = _marker("after reload")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url)
        reloaded: Final = gateway.request("POST", "/reload/anthropic_beta_headers")
        assert reloaded.status_code == 200, reloaded.text
        status: Final = gateway.request("GET", "/schedule/anthropic_beta_headers_reload/status")
        assert status.status_code == 200, status.text
        assert _JSON_OBJECT.validate_json(status.content)["scheduled"] is False, status.text
        _generated(gateway, "/v1/messages", _messages_body(model, marker), _INLINE_TOOLS)
        assert _one_delivery(wire, _VERTEX_TARGET, marker) == _INLINE_TOOLS


def test_vertex_messages_repeated_requests_each_carry_the_inline_tools_beta(gateway: Gateway) -> None:
    markers: Final = tuple(_marker("repeated") for _ in range(2))
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url)
        for marker in markers:
            _generated(gateway, "/v1/messages", _messages_body(model, marker), _INLINE_TOOLS)
        delivered: Final = _delivered(wire)
        assert tuple((sent.target, sent.beta) for sent in delivered) == ((_VERTEX_TARGET, _INLINE_TOOLS),) * 2
        assert tuple(marker in sent.body for marker, sent in zip(markers, delivered, strict=True)) == (True, True)


def test_concurrent_burst_across_both_workers_forwards_inline_tools_on_every_request(gateway: Gateway) -> None:
    with ExitStack() as stack:
        wire: Final = stack.enter_context(wire_server(_peer))
        scenario: Final = stack.enter_context(gateway.scenario())
        model: Final = _vertex_deployment(gateway, scenario, wire.url)
        burst: Final = _burst_bodies(model, "burst", 12)
        clients: Final = _clients(stack, str(gateway.client.base_url), len(burst))
        pool: Final = stack.enter_context(ThreadPoolExecutor(max_workers=len(burst)))
        statuses: Final = tuple(
            pool.map(
                lambda pair: _post(pair[0], gateway.key, pair[1][0], pair[1][1], _INLINE_TOOLS),
                zip(clients, burst, strict=True),
            )
        )
        assert statuses == (200,) * len(burst), statuses
        delivered: Final = _delivered(wire)
        assert tuple(sent.beta for sent in delivered) == (_INLINE_TOOLS,) * len(burst), delivered
        seen: Final = tuple(sum(marker in sent.body for sent in delivered) for _, _, marker in burst)
        assert seen == (1,) * len(burst), seen


def test_upstream_outage_mid_run_recovers_with_inline_tools_still_forwarded(gateway: Gateway) -> None:
    with ExitStack() as stack:
        clients: Final = _clients(stack, str(gateway.client.base_url), 8)
        pool: Final = stack.enter_context(ThreadPoolExecutor(max_workers=len(clients)))
        scenario: Final = stack.enter_context(gateway.scenario())

        def wave(label: str) -> tuple[tuple[int | None, str], ...]:
            markers: Final = tuple(_marker(label) for _ in clients)
            statuses: Final = tuple(
                pool.map(
                    lambda pair: _post_or_dropped(
                        pair[0], gateway.key, "/v1/messages", _messages_body(model, pair[1]), _INLINE_TOOLS
                    ),
                    zip(clients, markers, strict=True),
                )
            )
            return tuple(zip(statuses, markers, strict=True))

        def forwarded(wire: Wire, results: tuple[tuple[int | None, str], ...]) -> None:
            assert tuple(status for status, _ in results) == (200,) * len(clients), results
            delivered: Final = _delivered(wire)
            assert tuple(sent.beta for sent in delivered) == (_INLINE_TOOLS,) * len(clients), delivered
            seen: Final = tuple(sum(marker in sent.body for sent in delivered) for _, marker in results)
            assert seen == (1,) * len(clients), seen

        with wire_server(_peer) as wire:
            port: Final = int(wire.url.rsplit(":", 1)[1])
            model: Final = _vertex_deployment(gateway, scenario, wire.url)
            forwarded(wire, wave("before outage"))
        outage: Final = wave("during outage")
        assert all(status is not None and status >= 500 for status, _ in outage), outage
        assert gateway.request("GET", "/health/liveliness").status_code == 200
        with wire_server(_peer, port=port) as revived:
            forwarded(revived, wave("after outage"))


@pytest.mark.timeout(_OWNED_PROXY_CELL_SECONDS)
def test_worker_sigkill_mid_burst_leaves_the_replacement_forwarding_inline_tools(
    gateway: Gateway, tmp_path: Path
) -> None:
    held: Final[SimpleQueue[str]] = SimpleQueue()
    release: Final = threading.Event()
    with ExitStack() as stack:
        wire: Final = stack.enter_context(wire_server(_holding(held, release)))
        stack.callback(release.set)
        config: Final = _owned_config(tmp_path / "inline-tools-worker-kill.yaml", gateway, wire.url)
        owned: Final = stack.enter_context(
            owned_proxy_process(
                gateway, tmp_path, {"LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS": "true"}, config=config, workers=2
            )
        )
        proxy_url: Final = owned.gateway.client.base_url
        workers: Final = eventually(
            lambda: tuple(int(pid) for pid in _STARTED_WORKER.findall(owned.log.read_text())),
            lambda pids: len(pids) == 2,
            seconds=30,
        )
        clients: Final = _clients(stack, str(proxy_url), 12)
        pool: Final = stack.enter_context(ThreadPoolExecutor(max_workers=len(clients)))
        ports: Final = tuple(_local_port(client) for client in clients)
        markers: Final = tuple(_marker("worker kill") for _ in clients)
        futures: Final = tuple(
            pool.submit(
                _post_or_dropped,
                client,
                gateway.key,
                "/v1/messages",
                _messages_body(_OWNED_MODEL, marker),
                _INLINE_TOOLS,
            )
            for client, marker in zip(clients, markers, strict=True)
        )
        eventually(held.qsize, lambda size: size == len(clients), seconds=30)
        shares: Final = {pid: _accepted_client_ports(pid, proxy_url.port or 0) & frozenset(ports) for pid in workers}
        assert sum(map(len, shares.values())) == len(clients), shares
        victim: Final = min((pid for pid in workers if shares[pid]), key=lambda pid: len(shares[pid]))
        psutil.Process(victim).send_signal(signal.SIGKILL)
        release.set()
        results: Final = tuple(future.result(timeout=60) for future in futures)
        for port, result in zip(ports, results, strict=True):
            assert result == (None if port in shares[victim] else 200), (port, result, shares)
        eventually(lambda: len(_STARTED_WORKER.findall(owned.log.read_text())), lambda started: started >= 3, 120)
        second_wave: Final = tuple(_marker("after kill") for _ in range(6))
        for marker in second_wave:
            assert (
                _post(
                    owned.gateway.client,
                    gateway.key,
                    "/v1/messages",
                    _messages_body(_OWNED_MODEL, marker),
                    _INLINE_TOOLS,
                )
                == 200
            )
        delivered: Final = _delivered(wire)
        assert tuple(sent.beta for sent in delivered) == (_INLINE_TOOLS,) * (len(clients) + len(second_wave)), delivered
        seen: Final = tuple(sum(marker in sent.body for sent in delivered) for marker in markers + second_wave)
        assert seen == (1,) * len(seen), seen
        assert owned.process.poll() is None
