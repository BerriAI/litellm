import asyncio
import json
import os
import re
import signal
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import chain, count
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final, Literal
from urllib.parse import urlsplit

import httpx
import psutil
import pytest
import yaml
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process
from integration._support.redis_process import owned_redis
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_CONFIG_MODEL: Final = "gpt-6-bridge-chaos"
_API_KEY: Final = "synthetic-azure-key"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MARKER: Final = re.compile(r"marker-([0-9a-f]{32})")
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_CHAT_TOOL: Final = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
_MESSAGES_TOOL: Final = {
    "name": "get_weather",
    "description": "Get the weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}
_RESPONSES_TOOL: Final = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the weather for a city.",
    "strict": None,
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}

Endpoint = Literal["chat", "messages", "responses"]


@dataclass(frozen=True, slots=True)
class _Call:
    endpoint: Endpoint
    stream: bool
    marker: str


@dataclass(frozen=True, slots=True)
class _Served:
    call: _Call
    status: int
    text: str


@dataclass(frozen=True, slots=True)
class _Arrival:
    ordinal: int
    marker: str


def _text(marker: str) -> str:
    return f"Answer marker-{marker}."


def _prompt(marker: str) -> str:
    return f"Question marker-{marker}"


def _tool_arguments(marker: str) -> str:
    return json.dumps({"city": marker}, separators=(",", ":"))


def _path(endpoint: Endpoint) -> str:
    match endpoint:
        case "chat":
            return "/v1/chat/completions"
        case "messages":
            return "/v1/messages"
        case "responses":
            return "/v1/responses"


def _request_body(call: _Call) -> dict[str, JsonValue]:
    common: Final[dict[str, JsonValue]] = {"model": _CONFIG_MODEL, "num_retries": 0}
    match call.endpoint:
        case "chat":
            return {
                **common,
                "messages": [{"role": "user", "content": _prompt(call.marker)}],
                "tools": [_CHAT_TOOL],
                "stream": call.stream,
            }
        case "messages":
            return {
                **common,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": _prompt(call.marker)}],
                "tools": [_MESSAGES_TOOL],
                "stream": call.stream,
            }
        case "responses":
            return {
                **common,
                "input": _prompt(call.marker),
                "tools": [_RESPONSES_TOOL],
                "stream": call.stream,
            }


def _marker_of(request: Request) -> str:
    found: Final = _MARKER.search(request.body.decode())
    assert found is not None, request.body
    return found.group(1)


def _calls_by_marker(calls: tuple[_Call, ...]) -> Mapping[str, _Call]:
    return MappingProxyType({call.marker: call for call in calls})


def _assert_wire_request(request: Request, expected_calls: Mapping[str, _Call]) -> str:
    marker: Final = _marker_of(request)
    body: Final = _JSON_OBJECT.validate_json(request.body)
    call: Final = expected_calls[marker]
    assert body["model"] == "gpt-6-sol", body
    assert body["tools"] == [_RESPONSES_TOOL], body
    expected_input: Final = (
        _prompt(marker)
        if call.endpoint == "responses"
        else [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": _prompt(marker)}],
            }
        ]
    )
    assert body["input"] == expected_input, body
    assert body.get("stream", False) is call.stream, body
    assert "reasoning" not in body, body
    assert request.target == "/openai/responses?api-version=2025-04-01-preview", request.target
    return marker


def _responses_reply(marker: str) -> bytes:
    return json.dumps(
        {
            "id": f"resp_{marker}",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "gpt-6-sol",
            "output": [
                {
                    "id": f"msg_{marker}",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": _text(marker), "annotations": []}],
                },
                {
                    "id": f"fc_{marker}",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": f"call_{marker}",
                    "name": "get_weather",
                    "arguments": _tool_arguments(marker),
                },
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }
    ).encode()


def _responses_stream(marker: str) -> tuple[bytes, ...]:
    text: Final = _text(marker)
    arguments: Final = _tool_arguments(marker)
    response: Final = {
        "id": f"resp_{marker}",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-6-sol",
        "output": [
            {
                "id": f"msg_{marker}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            },
            {
                "id": f"fc_{marker}",
                "type": "function_call",
                "status": "completed",
                "call_id": f"call_{marker}",
                "name": "get_weather",
                "arguments": arguments,
            },
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }
    events: Final = (
        {
            "type": "response.created",
            "response": {**response, "status": "in_progress", "output": []},
        },
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": f"msg_{marker}",
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        },
        {
            "type": "response.output_text.delta",
            "item_id": f"msg_{marker}",
            "output_index": 0,
            "content_index": 0,
            "delta": text,
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": response["output"][0],
        },
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {
                "id": f"fc_{marker}",
                "type": "function_call",
                "status": "in_progress",
                "call_id": f"call_{marker}",
                "name": "get_weather",
                "arguments": "",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": f"fc_{marker}",
            "output_index": 1,
            "delta": arguments,
        },
        {
            "type": "response.output_item.done",
            "output_index": 1,
            "item": response["output"][1],
        },
        {"type": "response.completed", "response": response},
    )
    return tuple(f"data: {json.dumps(event)}\n\n".encode() for event in events)


def _reply(request: Request, expected_calls: Mapping[str, _Call], *, pause_between_chunks: float = 0) -> Reply:
    if request.method == "GET" and request.target == "/v1/models":
        return Reply(body=b'{"object":"list","data":[{"id":"gpt-6-sol","object":"model"}]}')
    marker: Final = _assert_wire_request(request, expected_calls)
    body: Final = _JSON_OBJECT.validate_json(request.body)
    if body.get("stream") is True:
        return Reply(
            content_type="text/event-stream",
            chunks=_responses_stream(marker),
            pause_between_chunks=pause_between_chunks,
        )
    return Reply(body=_responses_reply(marker))


def _proxy_config(wire: Wire, directory: Path) -> Path:
    configuration: Final = _JSON_OBJECT.validate_python(
        yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    )
    proxy_config: Final = {
        **configuration,
        "model_list": [
            {
                "model_name": _CONFIG_MODEL,
                "litellm_params": {
                    "model": "azure/gpt-6-sol",
                    "api_base": wire.url,
                    "api_key": _API_KEY,
                    "api_version": "2025-04-01-preview",
                },
            },
        ],
    }
    path: Final = directory / "gpt-6-responses-bridge-chaos.yaml"
    path.write_text(yaml.safe_dump(proxy_config))
    return path


def _spend_row(response_id: str) -> dict[str, JsonValue]:
    assert response_id, "Caller response had no id"
    rows: Final = eventually(
        lambda: read_rows(
            """
            SELECT
                request_id,
                status,
                CASE
                    WHEN left(request_id, 16) = 'resp_bGl0ZWxsbTp'
                    THEN convert_from(
                        decode(
                            translate(
                                regexp_replace(substr(request_id, 6), '_cache_hit[0-9]+[.][0-9]+$', ''),
                                '-_',
                                '+/'
                            ),
                            'base64'
                        ),
                        'UTF8'
                    )
                    ELSE request_id
                END AS decoded_request_id
            FROM "LiteLLM_SpendLogs"
            WHERE request_id=%s
               OR CASE
                    WHEN left(request_id, 16) = 'resp_bGl0ZWxsbTp'
                    THEN split_part(
                        convert_from(
                            decode(
                                translate(
                                    regexp_replace(substr(request_id, 6), '_cache_hit[0-9]+[.][0-9]+$', ''),
                                    '-_',
                                    '+/'
                                ),
                                'base64'
                            ),
                            'UTF8'
                        ),
                        ';response_id:',
                        2
                    ) = %s
                    ELSE false
                  END
            """,
            (response_id, response_id),
        ),
        lambda values: len(values) == 1,
        seconds=70,
    )
    decoded_request_id: Final = rows[0]["decoded_request_id"]
    assert rows[0]["request_id"] == response_id or (
        isinstance(decoded_request_id, str) and decoded_request_id.endswith(f";response_id:{response_id}")
    ), rows
    return rows[0]


def _record_audit_cell(row_id: str, node_id: str, response_id: str, spend_found: bool) -> None:
    results_dir: Final = os.environ.get("INTEGRATION_RESULTS_DIR")
    assert results_dir is not None, "INTEGRATION_RESULTS_DIR is required for audit cell evidence"
    artifact: Final = Path(results_dir) / "audit-cells.jsonl"
    record: Final = {
        "row_id": row_id,
        "node_id": node_id,
        "response_id": response_id,
        "upstream_path": "/openai/responses?api-version=2025-04-01-preview",
        "spend_row_found": spend_found,
        "leg": os.environ.get("LITAUDIT_LEG", "head"),
    }
    with artifact.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")


async def _send(client: httpx.AsyncClient, key: str, call: _Call) -> _Served:
    async with client.stream(
        "POST",
        _path(call.endpoint),
        json=_request_body(call),
        headers={"Authorization": f"Bearer {key}", "anthropic-version": "2023-06-01"},
    ) as response:
        raw: Final = await response.aread()
    return _Served(call=call, status=response.status_code, text=raw.decode())


async def _burst(
    base_url: str,
    key: str,
    calls: tuple[_Call, ...],
    *,
    tolerate_transport_errors: bool = False,
) -> tuple[_Served, ...]:
    async with httpx.AsyncClient(base_url=base_url, timeout=90, trust_env=False) as client:
        results: Final = await asyncio.gather(
            *(_send(client, key, call) for call in calls),
            return_exceptions=tolerate_transport_errors,
        )
    for result in results:
        assert not isinstance(result, BaseException) or isinstance(result, httpx.TransportError), repr(result)
    return tuple(result for result in results if isinstance(result, _Served))


def _calls(count_calls: int, endpoint_order: tuple[Endpoint, ...]) -> tuple[_Call, ...]:
    return tuple(
        _Call(
            endpoint=endpoint_order[index % len(endpoint_order)],
            stream=endpoint_order[index % len(endpoint_order)] == "chat" and index % 8 == 0,
            marker=uuid.uuid4().hex,
        )
        for index in range(count_calls)
    )


def _chat_stream_response_id(served: _Served) -> str:
    chunks: Final = tuple(
        _JSON_OBJECT.validate_json(line.removeprefix("data: "))
        for line in served.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    )
    assert chunks, served.text
    return str(chunks[0]["id"])


def _response_id(served: _Served) -> str:
    if served.call.endpoint == "chat" and served.call.stream:
        return _chat_stream_response_id(served)
    body: Final = _JSON_OBJECT.validate_json(served.text)
    return str(body["id"])


def _assert_chat_choice(body: dict[str, JsonValue], marker: str, response_text: str) -> None:
    assert body["choices"] == [
        {
            "finish_reason": "tool_calls",
            "index": 0,
            "message": {
                "role": "assistant",
                "content": _text(marker),
                "tool_calls": [
                    {
                        "id": f"call_{marker}",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": _tool_arguments(marker)},
                        "index": 0,
                    }
                ],
            },
        }
    ], response_text


def _assert_served(served: _Served) -> None:
    assert served.status == 200, served.text
    marker: Final = served.call.marker
    match served.call.endpoint:
        case "chat" if served.call.stream:
            chunks: Final = tuple(
                _JSON_OBJECT.validate_json(line.removeprefix("data: "))
                for line in served.text.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            )
            choices: Final = tuple(chain.from_iterable(chunk["choices"] for chunk in chunks if chunk.get("choices")))
            assert chunks, served.text
            assert choices, served.text
            assert all(choice["index"] == 0 for choice in choices), served.text
            assert "".join(str(choice["delta"].get("content") or "") for choice in choices) == _text(marker), (
                served.text
            )
            tool_call_chunks: Final = tuple(
                chain.from_iterable(choice["delta"].get("tool_calls", []) for choice in choices)
            )
            assert all(tool_call.get("index") == 0 for tool_call in tool_call_chunks), served.text
            assert "".join(str(tool_call["function"].get("name") or "") for tool_call in tool_call_chunks) == (
                "get_weather"
            ), served.text
            assert "".join(
                str(tool_call["function"].get("arguments") or "") for tool_call in tool_call_chunks
            ) == _tool_arguments(marker), served.text
            assert tuple(
                choice.get("finish_reason") for choice in choices if choice.get("finish_reason") is not None
            ) == ("tool_calls",), served.text
        case "chat":
            body: Final = _JSON_OBJECT.validate_json(served.text)
            _assert_chat_choice(body, marker, served.text)
        case "messages":
            body: Final = _JSON_OBJECT.validate_json(served.text)
            assert body["content"] == [
                {"type": "text", "text": _text(marker)},
                {
                    "type": "tool_use",
                    "id": f"call_{marker}",
                    "name": "get_weather",
                    "input": {"city": marker},
                },
            ], served.text
            assert body["stop_reason"] == "tool_use", served.text
        case "responses":
            body: Final = _JSON_OBJECT.validate_json(served.text)
            assert body["output"] == [
                {
                    "type": "message",
                    "id": f"msg_{marker}",
                    "status": "completed",
                    "role": "assistant",
                    "phase": None,
                    "content": [
                        {
                            "type": "output_text",
                            "text": _text(marker),
                            "annotations": [],
                            "logprobs": None,
                        }
                    ],
                },
                {
                    "type": "function_call",
                    "id": f"fc_{marker}",
                    "call_id": f"call_{marker}",
                    "name": "get_weather",
                    "arguments": _tool_arguments(marker),
                    "status": "completed",
                    "namespace": None,
                },
            ], served.text


def _record_served(row_id: str, node_id: str, served: tuple[_Served, ...]) -> None:
    response_ids: Final = tuple(_response_id(item) for item in served)
    for response_id in response_ids:
        spend: Final = _spend_row(response_id)
        assert spend["status"] == "success", spend
        _record_audit_cell(row_id, node_id, response_id, True)
    for item in served:
        _assert_served(item)


def _assert_wire_markers(wire: Wire, calls: tuple[_Call, ...]) -> None:
    received: Final = wire.drain()
    posts: Final = tuple(request for request in received if request.method == "POST")
    probes: Final = tuple((request.method, request.target) for request in received if request.method != "POST")
    markers: Final = tuple(_marker_of(request) for request in posts)
    assert len(posts) == len(calls), received
    assert sorted(markers) == sorted(call.marker for call in calls), received
    assert set(probes) <= {("GET", "/v1/models")}, probes
    assert all(request.target == "/openai/responses?api-version=2025-04-01-preview" for request in posts), received


def _open_upstream_connections(pid: int, upstream: str) -> int:
    port: Final = urlsplit(upstream).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


@pytest.mark.timeout(180)
async def test_f1_concurrent_gpt_6_endpoints_keep_each_marker_and_spend_row(
    gateway: Gateway,
    tmp_path: Path,
) -> None:
    calls: Final = _calls(30, ("chat", "messages", "responses", "chat"))
    expected_calls: Final = _calls_by_marker(calls)
    with wire_server(lambda request: _reply(request, expected_calls)) as wire, owned_redis(tmp_path) as redis:
        config: Final = _proxy_config(wire, tmp_path)
        with owned_proxy_process(
            gateway,
            tmp_path,
            {
                "LITELLM_DISABLE_NO_REDIS_WARNING": "true",
                "REDIS_HOST": redis.host,
                "REDIS_PORT": str(redis.port),
            },
            config=config,
            workers=2,
        ) as owned:
            served: Final = await _burst(str(owned.gateway.client.base_url), owned.gateway.key, calls)
            assert len(served) == 30
            _record_served(
                "F1",
                "test_f1_concurrent_gpt_6_endpoints_keep_each_marker_and_spend_row",
                served,
            )
            _assert_wire_markers(wire, calls)


@pytest.mark.timeout(180)
async def test_f2_scripted_503_window_recovers_for_concurrent_gpt_6_requests(
    gateway: Gateway,
    tmp_path: Path,
) -> None:
    calls: Final = _calls(30, ("chat", "messages", "responses", "chat"))
    request_sequence: Final = count()
    arrivals: Final[SimpleQueue[_Arrival]] = SimpleQueue()
    unavailable_requests: Final = 8
    expected_calls: Final = _calls_by_marker(calls)

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return _reply(request, expected_calls)
        marker: Final = _assert_wire_request(request, expected_calls)
        ordinal: Final = next(request_sequence)
        arrivals.put(_Arrival(ordinal=ordinal, marker=marker))
        if ordinal < unavailable_requests:
            return Reply(status=503, body=b'{"error":{"message":"scripted 503 window"}}')
        return _reply(request, expected_calls)

    with wire_server(respond) as wire, owned_redis(tmp_path) as redis:
        config: Final = _proxy_config(wire, tmp_path)
        with owned_proxy_process(
            gateway,
            tmp_path,
            {
                "LITELLM_DISABLE_NO_REDIS_WARNING": "true",
                "REDIS_HOST": redis.host,
                "REDIS_PORT": str(redis.port),
            },
            config=config,
            workers=2,
        ) as owned:
            served: Final = await _burst(str(owned.gateway.client.base_url), owned.gateway.key, calls)
            arrivals_ordered: Final = tuple(
                sorted(
                    (arrivals.get_nowait() for _ in range(arrivals.qsize())),
                    key=lambda arrival: arrival.ordinal,
                )
            )
            assert len(served) == 30
            assert len(arrivals_ordered) == 30
            expected_status: Final = MappingProxyType(
                {arrival.marker: 503 if arrival.ordinal < unavailable_requests else 200 for arrival in arrivals_ordered}
            )
            assert sum(item.status != 200 for item in served) == unavailable_requests, served
            failures: Final = tuple(item for item in served if expected_status[item.call.marker] == 503)
            for item in failures:
                assert item.status != 200, item.text
                assert "scripted 503 window" in item.text, item.text
            successes: Final = tuple(item for item in served if expected_status[item.call.marker] == 200)
            _record_served(
                "F2",
                "test_f2_scripted_503_window_recovers_for_concurrent_gpt_6_requests",
                successes,
            )
            for item in served:
                expected: Final = expected_status[item.call.marker]
                assert (item.status == 200) == (expected == 200), item.text
            health: Final = owned.gateway.request("GET", "/health/liveliness")
            assert health.status_code == 200, health.text
            _assert_wire_markers(wire, calls)


@pytest.mark.timeout(180)
async def test_f3_killing_one_gpt_6_proxy_worker_preserves_survivor_and_followup_requests(
    gateway: Gateway,
    tmp_path: Path,
) -> None:
    calls: Final = tuple(_Call(endpoint="chat", stream=False, marker=uuid.uuid4().hex) for _ in range(20))
    follow_up: Final = tuple(_Call(endpoint="chat", stream=False, marker=uuid.uuid4().hex) for _ in range(8))
    all_calls: Final = (*calls, *follow_up)
    expected_calls: Final = _calls_by_marker(all_calls)
    release: Final = threading.Event()
    held_markers: Final[SimpleQueue[str]] = SimpleQueue()

    def held(request: Request) -> Reply:
        if request.target == "/v1/models":
            return _reply(request, expected_calls)
        held_markers.put(_assert_wire_request(request, expected_calls))
        assert release.wait(timeout=60), "The request burst was not released"
        return _reply(request, expected_calls)

    with wire_server(held) as wire, owned_redis(tmp_path) as redis:
        config: Final = _proxy_config(wire, tmp_path)
        with owned_proxy_process(
            gateway,
            tmp_path,
            {
                "LITELLM_DISABLE_NO_REDIS_WARNING": "true",
                "REDIS_HOST": redis.host,
                "REDIS_PORT": str(redis.port),
            },
            config=config,
            workers=2,
        ) as owned:
            workers: Final = eventually(
                lambda: tuple(int(pid) for pid in _STARTED_WORKER.findall(owned.log.read_text())),
                lambda pids: len(pids) == 2,
                seconds=30,
            )
            burst: Final = asyncio.create_task(
                _burst(
                    str(owned.gateway.client.base_url),
                    owned.gateway.key,
                    calls,
                    tolerate_transport_errors=True,
                )
            )
            try:
                await asyncio.to_thread(eventually, held_markers.qsize, lambda size: size == len(calls), 60)
                held_by: Final = MappingProxyType({pid: _open_upstream_connections(pid, wire.url) for pid in workers})
                assert sum(held_by.values()) == len(calls), held_by
                victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
                victim: Final = psutil.Process(victim_pid)
                victim.suspend()
                victim.send_signal(signal.SIGKILL)
            finally:
                release.set()
            served: Final = await burst
            successful: Final = tuple(item for item in served if item.status == 200)
            assert len(successful) == held_by[survivor_pid], (held_by, len(successful))
            _record_served(
                "F3",
                "test_f3_killing_one_gpt_6_proxy_worker_preserves_survivor_and_followup_requests",
                successful,
            )
            follow_up_served: Final = await _burst(
                str(owned.gateway.client.base_url),
                owned.gateway.key,
                follow_up,
            )
            assert len(follow_up_served) == len(follow_up)
            _record_served(
                "F3",
                "test_f3_killing_one_gpt_6_proxy_worker_preserves_survivor_and_followup_requests",
                follow_up_served,
            )
            assert all(item.status == 200 for item in follow_up_served), follow_up_served
            _assert_wire_markers(wire, (*calls, *follow_up))


@pytest.mark.timeout(180)
async def test_f4_slow_concurrent_gpt_6_tool_replies_do_not_mix_markers_or_calls(
    gateway: Gateway,
    tmp_path: Path,
) -> None:
    calls: Final = tuple(_Call(endpoint="chat", stream=False, marker=uuid.uuid4().hex) for _ in range(20))
    expected_calls: Final = _calls_by_marker(calls)
    barrier: Final = threading.Barrier(len(calls))

    def slow_reply(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return _reply(request, expected_calls)
        marker: Final = _assert_wire_request(request, expected_calls)
        barrier.wait(timeout=60)
        return Reply(body=_responses_reply(marker))

    with wire_server(slow_reply) as wire, owned_redis(tmp_path) as redis:
        config: Final = _proxy_config(wire, tmp_path)
        with owned_proxy_process(
            gateway,
            tmp_path,
            {
                "LITELLM_DISABLE_NO_REDIS_WARNING": "true",
                "REDIS_HOST": redis.host,
                "REDIS_PORT": str(redis.port),
            },
            config=config,
            workers=2,
        ) as owned:
            served: Final = await _burst(str(owned.gateway.client.base_url), owned.gateway.key, calls)
            assert len(served) == len(calls)
            _record_served(
                "F4",
                "test_f4_slow_concurrent_gpt_6_tool_replies_do_not_mix_markers_or_calls",
                served,
            )
            _assert_wire_markers(wire, calls)
