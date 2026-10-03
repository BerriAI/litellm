import asyncio
import base64
import json
import os
import signal
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import httpx
import psutil
import pytest
import yaml
from integration._support.client import Gateway, Scenario, eventually
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

FREE: Final = {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0}
PRICED: Final = {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002}
_DROPPED_IDS: Final = frozenset(range(0, 30, 3))
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


@dataclass(frozen=True, slots=True)
class _BurstResult:
    request_id: str
    endpoint: str
    tier: str
    status: int | None
    text: str
    transport_error: str | None
    call_id: str | None = None
    spend_row_id: str | None = None


@dataclass(frozen=True, slots=True)
class _ModelDeployment:
    name: str
    model_id: str
    tier: str


def _identity(request: Request) -> tuple[str, str]:
    body: Final = _JSON_OBJECT.validate_json(request.body)
    if request.target.endswith("/responses"):
        return str(body["input"]), str(body["model"])
    messages: Final = TypeAdapter(list[JsonValue]).validate_python(body["messages"])
    prompt: Final = _JSON_OBJECT.validate_python(messages[0])
    return str(prompt["content"]), str(body["model"])


def _model_list_reply(request: Request) -> Reply | None:
    if request.method != "GET" or request.target != "/v1/models":
        return None
    return Reply(
        body=b'{"object":"list","data":[{"id":"gpt-4o-mini","object":"model","created":1,"owned_by":"litellm"},'
        b'{"id":"litellm-unpriced-9097","object":"model","created":1,"owned_by":"litellm"}]}',
        content_type="application/json",
    )


def _proxy_model_probe(gateway: Gateway) -> tuple[int, frozenset[str], str]:
    try:
        with httpx.Client(base_url=str(gateway.client.base_url), timeout=5, trust_env=False) as client:
            response: Final = client.get(
                "/v1/models",
                headers={"Authorization": f"Bearer {gateway.key}", "Connection": "close"},
            )
    except httpx.HTTPError as error:
        return 0, frozenset(), f"{type(error).__name__}: {error}"
    if response.status_code != 200:
        return response.status_code, frozenset(), response.text
    payload: Final = _JSON_OBJECT.validate_json(response.content)
    entries: Final = TypeAdapter(list[JsonValue]).validate_python(payload["data"])
    model_names: Final = frozenset(str(_JSON_OBJECT.validate_python(entry)["id"]) for entry in entries)
    return response.status_code, model_names, response.text


def _wait_for_proxy_models(gateway: Gateway, expected_model_names: tuple[str, ...]) -> None:
    expected: Final = frozenset(expected_model_names)
    assert expected, expected_model_names
    eventually(
        lambda: tuple(_proxy_model_probe(gateway) for _ in range(10)),
        lambda probes: (
            len(probes) == 10
            and all(status == 200 and expected.issubset(model_names) for status, model_names, _ in probes)
        ),
        seconds=70,
    )


def _chat_stream(identity: str, model: str, *, dropped: bool = False) -> tuple[bytes, ...]:
    initial: Final = (
        "data: "
        + json.dumps(
            {
                "id": identity,
                "object": "chat.completion.chunk",
                "created": 1,
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello"}, "finish_reason": None}],
            }
        )
        + "\n\n"
    ).encode()
    if dropped:
        return initial, b"unused"
    final: Final = (
        "data: "
        + json.dumps(
            {
                "id": identity,
                "object": "chat.completion.chunk",
                "created": 1,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
        + "\n\n"
    ).encode()
    usage: Final = (
        "data: "
        + json.dumps(
            {
                "id": identity,
                "object": "chat.completion.chunk",
                "created": 1,
                "model": model,
                "choices": [],
                "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
            }
        )
        + "\n\n"
    ).encode()
    return initial, final, usage, b"data: [DONE]\n\n"


def _responses_stream(identity: str, model: str, *, dropped: bool = False) -> tuple[bytes, ...]:
    initial: Final = (
        "event: response.created\n"
        + "data: "
        + json.dumps(
            {
                "type": "response.created",
                "response": {
                    "id": identity,
                    "object": "response",
                    "created_at": 1,
                    "model": model,
                    "status": "in_progress",
                    "output": [],
                },
            }
        )
        + "\n\n"
    ).encode()
    if dropped:
        return initial, b"unused"
    completed: Final = (
        "event: response.completed\n"
        + "data: "
        + json.dumps(
            {
                "type": "response.completed",
                "response": {
                    "id": identity,
                    "object": "response",
                    "created_at": 1,
                    "model": model,
                    "status": "completed",
                    "output": [],
                    "usage": {
                        "input_tokens": 11,
                        "output_tokens": 4,
                        "total_tokens": 15,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens_details": {"reasoning_tokens": 0},
                    },
                },
            }
        )
        + "\n\n"
    ).encode()
    return initial, completed


def _upstream_reply(request: Request, *, dropped: bool = False) -> Reply:
    model_list: Final = _model_list_reply(request)
    if model_list is not None:
        return model_list
    identity, model = _identity(request)
    chunks: Final = (
        _responses_stream(identity, model, dropped=dropped)
        if request.target.endswith("/responses")
        else _chat_stream(identity, model, dropped=dropped)
    )
    return Reply(
        content_type="text/event-stream",
        chunks=chunks,
        abort_after=1 if dropped else None,
    )


def _gated_upstream_reply(request: Request, gate: threading.Event) -> Reply:
    response: Final = _upstream_reply(request)
    if request.method == "GET":
        return response
    return Reply(
        content_type=response.content_type,
        chunks=response.chunks,
        gate_after_first=gate,
        gate_timeout_seconds=30,
    )


def _dropping_upstream_reply(request: Request) -> Reply:
    if request.method == "GET":
        return _upstream_reply(request)
    identity, _ = _identity(request)
    dropped: Final = int(identity.split("-")[1]) in _DROPPED_IDS
    return _upstream_reply(request, dropped=dropped)


def _sse_events(text: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        _JSON_OBJECT.validate_json(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    )


def _response_usage(event: dict[str, JsonValue]) -> dict[str, JsonValue]:
    response: Final = _JSON_OBJECT.validate_python(event["response"])
    return _JSON_OBJECT.validate_python(response["usage"])


def _usage_values(result: _BurstResult) -> tuple[dict[str, JsonValue], ...]:
    events: Final = _sse_events(result.text)
    if result.endpoint == "/v1/responses":
        return tuple(_response_usage(event) for event in events if event.get("type") == "response.completed")
    return tuple(_JSON_OBJECT.validate_python(event["usage"]) for event in events if event.get("usage") is not None)


def _responses_spend_request_id(model_id: str, response_id: str) -> str:
    identity: Final = (f"litellm:custom_llm_provider:openai;model_id:{model_id};response_id:{response_id}").encode()
    return "resp_" + base64.b64encode(identity).decode()


def _spend_row_request_id(endpoint: str, model_id: str, response_id: str, text: str) -> str | None:
    if endpoint == "/v1/responses":
        return _responses_spend_request_id(model_id, response_id)
    events: Final = _sse_events(text)
    event_ids: Final = tuple(event["id"] for event in events if event.get("id") is not None)
    return str(event_ids[0]) if event_ids else None


def _cost_field(usage: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {key: value for key, value in usage.items() if key == "cost"}


def _expected_cost_field(tier: str) -> dict[str, JsonValue]:
    if tier == "free":
        return {"cost": 0.0}
    if tier == "priced":
        return {"cost": pytest.approx(0.019)}
    return {}


def _expected_spend(tier: str) -> float:
    return 0.019 if tier == "priced" else 0.0


async def _read_async_response(response: httpx.Response) -> tuple[str, str | None]:
    try:
        return (await response.aread()).decode(), None
    except httpx.HTTPError as error:
        return "", f"{type(error).__name__}: {error}"


def _read_sync_response(response: httpx.Response) -> tuple[str, str | None]:
    try:
        return response.read().decode(), None
    except httpx.HTTPError as error:
        return "", f"{type(error).__name__}: {error}"


async def _request_stream(
    client: httpx.AsyncClient,
    model: str,
    model_id: str,
    tier: str,
    endpoint: str,
    request_id: str,
    *,
    dropped: bool = False,
) -> _BurstResult:
    body: Final = (
        {"model": model, "input": request_id, "stream": True}
        if endpoint == "/v1/responses"
        else {
            "model": model,
            "messages": [{"role": "user", "content": request_id}],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    )
    try:
        async with client.stream("POST", endpoint, json=body) as response:
            status: Final = response.status_code
            text, error = await _read_async_response(response)
            call_id: Final = response.headers.get("x-litellm-call-id")
            spend_row_id: Final = _spend_row_request_id(endpoint, model_id, request_id, text)
            return _BurstResult(request_id, endpoint, tier, status, text, error, call_id, spend_row_id)
    except httpx.HTTPError as request_error:
        return _BurstResult(
            request_id,
            endpoint,
            tier,
            None,
            "",
            f"{type(request_error).__name__}: {request_error}",
        )


async def _request_for_index(
    client: httpx.AsyncClient,
    deployments: tuple[_ModelDeployment, _ModelDeployment, _ModelDeployment],
    index: int,
    dropped_ids: frozenset[int],
) -> _BurstResult:
    deployment: Final = deployments[(index // 2) % 3]
    endpoint: Final = "/v1/responses" if index % 2 else "/v1/chat/completions"
    request_id: Final = f"burst-{index}-{uuid.uuid4().hex}"
    return await _request_stream(
        client,
        deployment.name,
        deployment.model_id,
        deployment.tier,
        endpoint,
        request_id,
        dropped=index in dropped_ids,
    )


async def _burst(
    gateway: Gateway,
    deployments: tuple[_ModelDeployment, _ModelDeployment, _ModelDeployment],
    count: int,
    dropped_ids: frozenset[int] = frozenset(),
) -> tuple[_BurstResult, ...]:
    async with httpx.AsyncClient(
        base_url=str(gateway.client.base_url),
        headers={"Authorization": f"Bearer {gateway.key}", "Connection": "close"},
        timeout=30,
        limits=httpx.Limits(max_connections=40, max_keepalive_connections=0),
    ) as client:
        requests: Final = tuple(_request_for_index(client, deployments, index, dropped_ids) for index in range(count))
        return tuple(await asyncio.gather(*requests))


def _model_names(scenario: Scenario, api_base: str) -> tuple[_ModelDeployment, _ModelDeployment, _ModelDeployment]:
    free_model_id: Final = str(uuid.uuid4())
    priced_model_id: Final = str(uuid.uuid4())
    unpriced_model_id: Final = str(uuid.uuid4())
    return (
        _ModelDeployment(
            scenario.model(
                model="openai/gpt-4o-mini",
                api_base=api_base,
                model_info={"id": free_model_id},
                **FREE,
            ),
            free_model_id,
            "free",
        ),
        _ModelDeployment(
            scenario.model(
                model="openai/gpt-4o-mini",
                api_base=api_base,
                model_info={"id": priced_model_id},
                **PRICED,
            ),
            priced_model_id,
            "priced",
        ),
        _ModelDeployment(
            scenario.model(
                model="openai/litellm-unpriced-9097",
                api_base=api_base,
                model_info={"id": unpriced_model_id},
            ),
            unpriced_model_id,
            "unpriced",
        ),
    )


def _assert_spend_rows(results: tuple[_BurstResult, ...]) -> None:
    completed: Final = tuple(result for result in results if result.status == 200 and len(_usage_values(result)) == 1)
    spend_row_ids: Final = tuple(result.spend_row_id for result in completed)
    assert spend_row_ids and all(spend_row_id is not None for spend_row_id in spend_row_ids), results
    ids: Final = tuple(spend_row_id for spend_row_id in spend_row_ids if spend_row_id is not None)
    call_ids: Final = tuple(result.call_id for result in completed)
    assert all(call_id is not None for call_id in call_ids), results
    expected_call_ids: Final = dict(
        zip(
            ids,
            (call_id for call_id in call_ids if call_id is not None),
            strict=True,
        )
    )
    assert len(set(ids)) == len(ids), results
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT request_id, litellm_call_id, spend FROM "LiteLLM_SpendLogs" WHERE request_id = ANY(%s::text[])',
            ("{" + ",".join(ids) + "}",),
        ),
        lambda values: len(values) == len(ids),
        seconds=70,
    )
    assert sorted(str(row["request_id"]) for row in rows) == sorted(ids), rows
    assert len(rows) == len(ids), rows
    observed_call_ids: Final = {str(row["request_id"]): str(row["litellm_call_id"]) for row in rows}
    assert observed_call_ids == expected_call_ids, rows
    observed_spend: Final = {str(row["request_id"]): float(row["spend"]) for row in rows}
    expected_spend: Final = {
        spend_row_id: _expected_spend(result.tier) for result, spend_row_id in zip(completed, ids, strict=True)
    }
    assert observed_spend == expected_spend, rows


def _proxy_worker_pids(gateway: Gateway) -> tuple[int, ...]:
    proxy_port: Final = gateway.client.base_url.port or 80
    masters: Final = tuple(process for process in psutil.process_iter() if _is_proxy_master(process, proxy_port))
    process_groups: Final = tuple(_worker_processes(process) for process in masters)
    processes: Final = chain.from_iterable(process_groups)
    return tuple(sorted({process.pid for process in processes}))


def _is_proxy_master(process: psutil.Process, proxy_port: int) -> bool:
    try:
        command: Final = process.cmdline()
        port_index: Final = command.index("--port")
        return "integration._support.proxy" in command and command[port_index + 1] == str(proxy_port)
    except (psutil.ZombieProcess, psutil.NoSuchProcess, psutil.AccessDenied, ValueError, IndexError):
        return False


def _worker_processes_for_pid(pid: int) -> tuple[psutil.Process, ...]:
    try:
        return _worker_processes(psutil.Process(pid))
    except (psutil.ZombieProcess, psutil.NoSuchProcess):
        return ()


def _worker_processes(process: psutil.Process) -> tuple[psutil.Process, ...]:
    try:
        candidates: Final = (process, *process.children(recursive=True))
    except (psutil.ZombieProcess, psutil.NoSuchProcess, psutil.AccessDenied):
        return ()
    return tuple(candidate for candidate in candidates if _is_worker_process(candidate))


def _is_worker_process(process: psutil.Process) -> bool:
    try:
        return process.status() != psutil.STATUS_ZOMBIE and "spawn_main" in " ".join(process.cmdline())
    except (psutil.ZombieProcess, psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def _active_workers(workers: tuple[int, ...], upstream_port: int) -> tuple[int, ...]:
    return tuple(pid for pid in workers if _worker_has_connection(pid, upstream_port))


def _worker_has_connection(pid: int, upstream_port: int) -> bool:
    try:
        return any(
            connection.status == psutil.CONN_ESTABLISHED and connection.raddr.port == upstream_port
            for connection in psutil.Process(pid).net_connections(kind="tcp")
        )
    except (psutil.ZombieProcess, psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def _assert_success_cost(result: _BurstResult) -> None:
    assert result.status == 200 and result.transport_error is None, result
    usages: Final = _usage_values(result)
    assert len(usages) == 1, result.text
    usage: Final = usages[0]
    input_tokens: Final = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens: Final = usage.get("output_tokens", usage.get("completion_tokens"))
    assert (input_tokens, output_tokens, usage["total_tokens"]) == (11, 4, 15), result.text
    assert _cost_field(usage) == _expected_cost_field(result.tier), result.text


def _assert_success_costs(results: tuple[_BurstResult, ...]) -> None:
    for result in results:
        _assert_success_cost(result)


def _assert_dropped_stream(result: _BurstResult) -> None:
    events: Final = _sse_events(result.text)
    failure_frames: Final = tuple(
        event for event in events if event.get("type") in {"response.failed", "error"} or "error" in event
    )
    assert (
        result.transport_error is not None
        or (result.status is not None and result.status >= 400)
        or failure_frames
        or "[DONE]" not in result.text
    ), result
    event_usages: Final = chain.from_iterable(_event_usage_objects(event) for event in events)
    body_usages: Final = (
        _event_usage_objects(_JSON_OBJECT.validate_json(result.text)) if result.text.lstrip().startswith("{") else ()
    )
    usages: Final = tuple(chain(event_usages, body_usages))
    assert all(_cost_field(usage) == {} for usage in usages), result.text


def _event_usage_objects(event: dict[str, JsonValue]) -> tuple[dict[str, JsonValue], ...]:
    top_level: Final = (_JSON_OBJECT.validate_python(event["usage"]),) if isinstance(event.get("usage"), dict) else ()
    response: Final = event.get("response")
    response_body: Final = _JSON_OBJECT.validate_python(response) if isinstance(response, dict) else {}
    response_usage: Final = response_body.get("usage")
    nested: Final = (_JSON_OBJECT.validate_python(response_usage),) if isinstance(response_usage, dict) else ()
    return top_level + nested


def test_burst_streams_report_cost_and_persist_once_on_two_workers(
    gateway: Gateway,
) -> None:
    gate: Final = threading.Event()
    with wire_server(lambda request: _gated_upstream_reply(request, gate)) as wire, gateway.scenario() as scenario:
        models: Final = _model_names(scenario, wire.url + "/v1")
        _wait_for_proxy_models(gateway, tuple(model.name for model in models))
        worker_pids: Final = _proxy_worker_pids(gateway)
        assert len(set(worker_pids)) == 2, worker_pids
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending: Final = pool.submit(asyncio.run, _burst(gateway, models, 30))
            eventually(lambda: wire.received.qsize(), lambda value: value >= 10, seconds=10)
            active: Final = eventually(
                lambda: _active_workers(worker_pids, urlsplit(wire.url).port or 0),
                lambda pids: len(pids) == 2,
                seconds=3,
            )
            assert set(active) == set(worker_pids), active
            gate.set()
            results: Final = pending.result(timeout=40)
        _assert_success_costs(results)
        _assert_spend_rows(results)
        requests: Final = wire.drain()
    posts: Final = tuple(request for request in requests if request.method == "POST")
    gets: Final = tuple(request for request in requests if request.method == "GET")
    assert len(posts) == 30, requests
    assert all(request.target == "/v1/models" for request in gets), requests
    assert gateway.client.get("/health/liveliness").status_code == 200


def test_burst_streams_with_one_third_dropped_preserve_completed_costs(
    gateway: Gateway,
) -> None:
    with (
        wire_server(_dropping_upstream_reply) as wire,
        gateway.scenario() as scenario,
    ):
        models: Final = _model_names(scenario, wire.url + "/v1")
        _wait_for_proxy_models(gateway, tuple(model.name for model in models))
        results: Final = asyncio.run(_burst(gateway, models, 30, _DROPPED_IDS))
        dropped: Final = tuple(result for index, result in enumerate(results) if index in _DROPPED_IDS)
        completed: Final = tuple(result for index, result in enumerate(results) if index not in _DROPPED_IDS)
        assert len(dropped) == 10 and len(completed) == 20, results
        for result in dropped:
            _assert_dropped_stream(result)
        _assert_success_costs(completed)
        _assert_spend_rows(completed)
        requests: Final = wire.drain()
    posts: Final = tuple(request for request in requests if request.method == "POST")
    gets: Final = tuple(request for request in requests if request.method == "GET")
    assert len(posts) == 30, requests
    assert all(request.target == "/v1/models" for request in gets), requests
    assert gateway.client.get("/health/liveliness").status_code == 200


def _worker_pids(owned: OwnedProxy) -> tuple[int, ...]:
    return tuple(process.pid for process in _worker_processes_for_pid(owned.process.pid))


def _sync_stream(candidate: Gateway, model: str, model_id: str, request_id: str) -> _BurstResult:
    endpoint: Final = "/v1/chat/completions"
    try:
        with candidate.client.stream(
            "POST",
            endpoint,
            json={
                "model": model,
                "messages": [{"role": "user", "content": request_id}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            headers={"Authorization": f"Bearer {candidate.key}", "Connection": "close"},
        ) as response:
            status: Final = response.status_code
            text, error = _read_sync_response(response)
            call_id: Final = response.headers.get("x-litellm-call-id")
            spend_row_id: Final = _spend_row_request_id(endpoint, model_id, request_id, text)
            return _BurstResult(request_id, endpoint, "free", status, text, error, call_id, spend_row_id)
    except httpx.HTTPError as request_error:
        return _BurstResult(request_id, endpoint, "free", None, "", f"{type(request_error).__name__}: {request_error}")


def test_killing_one_worker_during_free_stream_burst_leaves_survivor_serving(gateway: Gateway, tmp_path: Path) -> None:
    gate: Final = threading.Event()
    base_config: Final = _JSON_OBJECT.validate_python(
        yaml.safe_load((Path(__file__).resolve().parents[1] / "proxy_config.yaml").read_text())
    )
    existing_router: Final = _JSON_OBJECT.validate_python(base_config.get("router_settings", {}))
    configuration: Final = {
        **base_config,
        "router_settings": {**existing_router, "num_retries": 0},
    }
    config_path: Final = tmp_path / "worker-kill.yaml"
    config_path.write_text(yaml.safe_dump(configuration))

    def respond(request: Request) -> Reply:
        model_list: Final = _model_list_reply(request)
        if model_list is not None:
            return model_list
        identity, model = _identity(request)
        return Reply(
            content_type="text/event-stream",
            chunks=_chat_stream(identity, model),
            gate_after_first=gate,
            gate_timeout_seconds=30,
        )

    with (
        wire_server(respond) as wire,
        owned_proxy_process(gateway, tmp_path, {}, config=config_path, workers=2) as owned,
    ):
        with owned.gateway.scenario() as scenario:
            model_id: Final = str(uuid.uuid4())
            model: Final = scenario.model(
                model="openai/gpt-4o-mini",
                api_base=wire.url + "/v1",
                model_info={"id": model_id},
                **FREE,
            )
            _wait_for_proxy_models(owned.gateway, (model,))
            workers: Final = _worker_pids(owned)
            assert len(workers) == 2, workers
            identities: Final = tuple(f"worker-kill-{index}-{uuid.uuid4().hex}" for index in range(12))
            with ThreadPoolExecutor(max_workers=12) as pool:
                futures: Final = tuple(
                    pool.submit(_sync_stream, owned.gateway, model, model_id, identity) for identity in identities
                )
                eventually(lambda: wire.received.qsize(), lambda value: value == len(identities), seconds=15)
                active: Final = eventually(
                    lambda: _active_workers(workers, urlsplit(wire.url).port or 0),
                    lambda pids: len(pids) == 2,
                    seconds=5,
                )
                assert set(active) == set(workers), active
                victim: Final = workers[0]
                survivor: Final = workers[1]
                os.kill(victim, signal.SIGKILL)
                gate.set()
            first_results: Final = tuple(future.result(timeout=40) for future in futures)
            eventually(
                lambda: _worker_pids(owned),
                lambda pids: survivor in pids and victim not in pids,
                seconds=5,
            )
            follow_up: Final = tuple(
                _sync_stream(
                    owned.gateway,
                    model,
                    model_id,
                    f"worker-survivor-{index}-{uuid.uuid4().hex}",
                )
                for index in range(6)
            )
            results: Final = first_results + follow_up
            _assert_success_costs(follow_up)
            completed: Final = tuple(
                result
                for result in results
                if result.status == 200 and result.transport_error is None and len(_usage_values(result)) == 1
            )
            assert len(completed) >= len(follow_up), results
            _assert_success_costs(completed)
            _assert_spend_rows(completed)
            assert psutil.pid_exists(survivor), f"surviving worker {survivor} exited"
            assert owned.gateway.client.get("/health/liveliness").status_code == 200
            requests: Final = wire.drain()
    posts: Final = tuple(request for request in requests if request.method == "POST")
    gets: Final = tuple(request for request in requests if request.method == "GET")
    assert len(posts) == 18, requests
    assert all(request.target == "/v1/models" for request in gets), requests
