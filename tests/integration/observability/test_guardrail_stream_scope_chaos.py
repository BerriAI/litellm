from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import uuid
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from threading import Barrier, Event
from types import MappingProxyType
from typing import Final, Literal, TypeAlias, cast

import httpx
import psutil
import pytest
import yaml
from integration._support.client import Gateway, JsonValue, eventually
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import TypeAdapter

ChaosEndpoint: TypeAlias = Literal["chat", "messages", "responses"]
CHAOS_ENDPOINTS: Final[tuple[ChaosEndpoint, ...]] = ("chat", "messages", "responses")
CHAOS_MODELS: Final = MappingProxyType(
    {"chat": "chaos-chat", "messages": "chaos-messages", "responses": "chaos-responses"}
)
GUARDRAIL_PATH: Final = "/beta/litellm_basic_guardrail_api"
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
POSTGRES_IMAGE: Final = "postgres:16@sha256:e17e86066e5ef83e0952a9347f5c792b7ece00972e2aa787a6986f471b3dd3d5"


def _json(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode()


def _texts(value: JsonValue) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(chain.from_iterable(_texts(item) for item in value))
    if isinstance(value, dict):
        return tuple(chain.from_iterable(_texts(item) for item in value.values()))
    return ()


def _marker(body: Mapping[str, JsonValue]) -> str:
    return next((text for text in _texts(dict(body)) if text.startswith("audit-")), "audit-chaos")


def _sse(events: Sequence[Mapping[str, JsonValue]]) -> tuple[bytes, ...]:
    return tuple(f"data: {json.dumps(event, separators=(',', ':'))}\n\n".encode() for event in events) + (
        b"data: [DONE]\n\n",
    )


def _messages_stream(message: Mapping[str, JsonValue]) -> tuple[bytes, ...]:
    content: Final = cast(list[JsonValue], message["content"])
    text: Final = cast(dict[str, JsonValue], content[0])["text"]
    assert isinstance(text, str)
    return (
        f"event: message_start\ndata: {json.dumps({**message, 'content': [], 'stop_reason': None, 'usage': {'input_tokens': 2, 'output_tokens': 0}})}\n\n".encode(),
        b'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
        f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': text}})}\n\n".encode(),
        b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
        f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': 2}})}\n\n".encode(),
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    )


def _responses_stream(
    response: Mapping[str, JsonValue],
    output: Mapping[str, JsonValue],
    marker: str,
) -> tuple[bytes, ...]:
    events: Final[tuple[dict[str, JsonValue], ...]] = (
        {"type": "response.created", "response": {**response, "status": "in_progress", "output": []}},
        {"type": "response.in_progress", "response": {**response, "status": "in_progress", "output": []}},
        {"type": "response.output_item.added", "item": dict(output), "output_index": 0},
        {
            "type": "response.content_part.added",
            "item_id": f"msg-{marker}",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        },
        {
            "type": "response.output_text.delta",
            "item_id": f"msg-{marker}",
            "output_index": 0,
            "content_index": 0,
            "delta": marker,
        },
        {
            "type": "response.output_text.done",
            "item_id": f"msg-{marker}",
            "output_index": 0,
            "content_index": 0,
            "text": marker,
        },
        {
            "type": "response.content_part.done",
            "item_id": f"msg-{marker}",
            "output_index": 0,
            "content_index": 0,
            "part": cast(list[JsonValue], output["content"])[0],
        },
        {"type": "response.output_item.done", "item": dict(output), "output_index": 0},
        {"type": "response.completed", "response": dict(response)},
    )
    return tuple(
        f"event: {event['type']}\ndata: {json.dumps({**event, 'sequence_number': index}, separators=(',', ':'))}\n\n".encode()
        for index, event in enumerate(events)
    )


def _provider(request: Request) -> Reply:
    if request.method == "GET" and request.target.partition("?")[0] == "/v1/models":
        return Reply(body=_json({"object": "list", "data": [{"id": "gpt-4o-mini", "object": "model"}]}))
    if not request.body:
        return Reply(status=400, body=_json({"error": "request body is required"}))
    body: Final = JSON_OBJECT.validate_json(request.body)
    marker: Final = _marker(body)
    streamed: Final = bool(body.get("stream"))
    if request.target == "/v1/chat/completions":
        if streamed:
            return Reply(
                content_type="text/event-stream",
                chunks=_sse(
                    (
                        {
                            "id": f"chatcmpl-{marker}",
                            "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": {"content": marker}, "finish_reason": None}],
                        },
                    )
                ),
            )
        return Reply(
            body=_json(
                {
                    "id": f"chatcmpl-{marker}",
                    "object": "chat.completion",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": marker}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
                }
            )
        )
    if request.target == "/v1/messages":
        message: Final = {
            "id": f"msg-{marker}",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5-20250929",
            "content": [{"type": "text", "text": marker}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 2, "output_tokens": 2},
        }
        return (
            Reply(
                content_type="text/event-stream",
                chunks=_messages_stream(message),
            )
            if streamed
            else Reply(body=_json(message))
        )
    if request.target == "/v1/responses":
        response: Final = {
            "id": f"resp-{marker}",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "gpt-4o-mini",
            "output": [
                {
                    "type": "message",
                    "id": f"msg-{marker}",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": marker, "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 2, "output_tokens": 2, "total_tokens": 4},
        }
        return (
            Reply(
                content_type="text/event-stream",
                chunks=_responses_stream(response, cast(dict[str, JsonValue], response["output"][0]), marker),
            )
            if streamed
            else Reply(body=_json(response))
        )
    return Reply(status=404, body=_json({"error": f"unexpected provider target {request.target}"}))


def _sink(request: Request) -> Reply:
    assert request.target.endswith(GUARDRAIL_PATH), request.target
    body: Final = JSON_OBJECT.validate_json(request.body)
    assert body.get("litellm_call_id") is not None or any("audit-" in text for text in _texts(body)), (
        request.body.decode()
    )
    return Reply(body=_json({"action": "NONE"}))


def _rail(
    name: str,
    sink_url: str,
    scope: Literal["streaming", "non_streaming"],
    *,
    default_on: bool = False,
) -> dict[str, JsonValue]:
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "generic_guardrail_api",
            "mode": "pre_call",
            "default_on": default_on,
            "stream_scope": scope,
            "api_base": f"{sink_url}/{name}",
            "api_key": "synthetic-chaos-key",
        },
    }


def _config(provider_url: str, rails: Sequence[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    base: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    return cast(
        dict[str, JsonValue],
        {
            **base,
            "guardrails": list(rails),
            "model_list": [
                {
                    "model_name": CHAOS_MODELS["chat"],
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_base": f"{provider_url}/v1",
                        "api_key": "synthetic-provider-key",
                        "num_retries": 0,
                    },
                },
                {
                    "model_name": CHAOS_MODELS["messages"],
                    "litellm_params": {
                        "model": "anthropic/claude-sonnet-4-5-20250929",
                        "api_base": provider_url,
                        "api_key": "synthetic-provider-key",
                        "num_retries": 0,
                    },
                },
                {
                    "model_name": CHAOS_MODELS["responses"],
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_base": f"{provider_url}/v1",
                        "api_key": "synthetic-provider-key",
                        "num_retries": 0,
                    },
                },
            ],
            "environment_variables": {
                **base.get("environment_variables", {}),
                "OPENAI_API_BASE": provider_url,
                "OPENAI_API_KEY": "synthetic-provider-key",
                "ANTHROPIC_API_BASE": provider_url,
                "ANTHROPIC_API_KEY": "synthetic-provider-key",
            },
        },
    )


@dataclass(frozen=True, slots=True)
class ChaosRig:
    gateway: Gateway
    provider: Wire
    directory: Path


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ChaosRig]:
    with (
        httpx.Client(
            base_url=os.environ["INTEGRATION_PROXY_URL"],
            timeout=30,
            trust_env=False,
        ) as root_client,
        wire_server(_provider) as provider,
    ):
        root_gateway: Final = Gateway(
            root_client,
            os.environ.get("INTEGRATION_MASTER_KEY", "sk-integration-master"),
            os.environ["INTEGRATION_UPSTREAM_URL"],
        )
        yield ChaosRig(root_gateway, provider, tmp_path_factory.mktemp("stream-scope-chaos"))


@dataclass(frozen=True, slots=True)
class CallPlan:
    marker: str
    call_id: str
    endpoint: ChaosEndpoint
    streamed: bool


def _plans(prefix: str, count: int) -> tuple[CallPlan, ...]:
    return tuple(
        CallPlan(
            f"audit-{prefix}-{index}-{uuid.uuid4().hex}",
            f"{prefix}-{uuid.uuid4().hex}",
            CHAOS_ENDPOINTS[index % len(CHAOS_ENDPOINTS)],
            index % 2 == 1,
        )
        for index in range(count)
    )


def _request(gateway: Gateway, plan: CallPlan, rails: Sequence[str]) -> httpx.Response:
    guardrail_field: Final[dict[str, JsonValue]] = {"guardrails": list(rails)} if rails else {}
    match plan.endpoint:
        case "chat":
            return gateway.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": CHAOS_MODELS["chat"],
                    "messages": [{"role": "user", "content": plan.marker}],
                    **guardrail_field,
                    **({"stream": True} if plan.streamed else {}),
                },
                headers={"x-litellm-call-id": plan.call_id},
            )
        case "messages":
            return gateway.request(
                "POST",
                "/v1/messages",
                {
                    "model": CHAOS_MODELS["messages"],
                    "max_tokens": 32,
                    "messages": [{"role": "user", "content": plan.marker}],
                    **guardrail_field,
                    **({"stream": True} if plan.streamed else {}),
                },
                headers={"x-litellm-call-id": plan.call_id},
            )
        case "responses":
            return gateway.request(
                "POST",
                "/v1/responses",
                {
                    "model": CHAOS_MODELS["responses"],
                    "input": plan.marker,
                    **guardrail_field,
                    **({"stream": True} if plan.streamed else {}),
                },
                headers={"x-litellm-call-id": plan.call_id},
            )
    raise AssertionError(plan.endpoint)


def _rows_for_marker(rows: Sequence[Request], marker: str) -> tuple[Request, ...]:
    return tuple(request for request in rows if marker.encode() in request.body)


def _rail_scans(rows: Sequence[Request], rail_name: str, marker: str) -> tuple[Request, ...]:
    return tuple(
        request for request in rows if request.target.startswith(f"/{rail_name}/") and marker.encode() in request.body
    )


def _spend_rows(call_id: str, database_url: str | None = None) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT request_id, litellm_call_id FROM "LiteLLM_SpendLogs" WHERE request_id=%s OR litellm_call_id=%s',
        (call_id, call_id),
        database_url=database_url,
    )


def _one_spend_row(
    call_id: str,
    database_url: str | None = None,
    *,
    seconds: float = 70,
) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: _spend_rows(call_id, database_url),
        lambda values: len(values) >= 1,
        seconds=seconds,
    )
    assert len(rows) == 1, (call_id, rows)
    return rows[0]


def _expected_in_scope(plan: CallPlan, sink_a_name: str, sink_b_name: str) -> tuple[str, str]:
    return (sink_a_name, sink_b_name) if plan.streamed else (sink_b_name, sink_a_name)


def _assert_successful_calls(
    plans: Sequence[CallPlan],
    responses: Sequence[httpx.Response],
    provider_rows: Sequence[Request],
    sink_a_rows: Sequence[Request],
    sink_b_rows: Sequence[Request],
    sink_a_name: str,
    sink_b_name: str,
) -> None:
    for plan, response in zip(plans, responses):
        if response.status_code != 200:
            continue
        assert plan.marker in response.text, (plan, response.text)
        provider_match: Final = _rows_for_marker(provider_rows, plan.marker)
        assert len(provider_match) == 1, (plan, provider_match)
        in_sink, out_sink = _expected_in_scope(plan, sink_a_name, sink_b_name)
        in_rows: Final = _rail_scans(
            sink_a_rows if in_sink == sink_a_name else sink_b_rows,
            in_sink,
            plan.marker,
        )
        out_rows: Final = _rail_scans(
            sink_a_rows if out_sink == sink_a_name else sink_b_rows,
            out_sink,
            plan.marker,
        )
        assert len(in_rows) == 1 and len(out_rows) == 0, (plan, in_rows, out_rows)
        spend: Final = _one_spend_row(plan.call_id)
        assert plan.call_id in (spend.get("request_id"), spend.get("litellm_call_id")), (plan, spend)


def _call_wave(gateway: Gateway, plans: Sequence[CallPlan], rails: Sequence[str]) -> tuple[httpx.Response, ...]:
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures: Final[tuple[Future[httpx.Response], ...]] = tuple(
            pool.submit(_request, gateway, plan, rails) for plan in plans
        )
        return tuple(future.result() for future in futures)


@contextmanager
def _owned_proxy(
    rig: ChaosRig,
    directory: Path,
    rails: Sequence[dict[str, JsonValue]],
    *,
    workers: int = 1,
) -> Iterator[OwnedProxy]:
    config_path: Final = directory / f"chaos-{uuid.uuid4().hex}.yaml"
    config_path.write_text(yaml.safe_dump(_config(rig.provider.url, rails)))
    with owned_proxy_process(rig.gateway, directory, {}, config=config_path, workers=workers) as owned:
        yield owned


def _free_port() -> int:
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        return int(reserve.getsockname()[1])


def _assert_down_wave(
    plans: Sequence[CallPlan],
    responses: Sequence[httpx.Response],
    provider_rows: Sequence[Request],
    sink_a_rows: Sequence[Request],
    sink_b_rows: Sequence[Request],
    sink_a_name: str,
    sink_b_name: str,
) -> None:
    for plan, response in zip(plans, responses):
        if plan.streamed:
            assert response.status_code >= 500 and response.content, (plan, response.status_code, response.text)
            assert _rows_for_marker(provider_rows, plan.marker) == (), (plan, provider_rows)
            assert _rail_scans(sink_b_rows, sink_b_name, plan.marker) == (), (plan, sink_b_name)
            continue
        assert response.status_code == 200 and plan.marker in response.text, (plan, response.status_code, response.text)
        assert len(_rows_for_marker(provider_rows, plan.marker)) == 1, (plan, provider_rows)
        assert _rail_scans(sink_a_rows, sink_a_name, plan.marker) == (), (plan, sink_a_name)
        assert len(_rail_scans(sink_b_rows, sink_b_name, plan.marker)) == 1, (plan, sink_b_name)
        spend: Final = _one_spend_row(plan.call_id)
        assert plan.call_id in (spend.get("request_id"), spend.get("litellm_call_id")), (plan, spend)


def test_h1_sink_outage_keeps_scope_isolated_through_recovery(rig: ChaosRig, tmp_path: Path) -> None:
    port_a: Final = _free_port()
    started: Final = Event()
    unavailable: Final = Event()
    release: Final = Event()

    def gated_sink(request: Request) -> Reply:
        started.set()
        assert release.wait(timeout=45), "sink outage gate was not released"
        if unavailable.is_set():
            return Reply(status=503, body=_json({"error": "synthetic sink outage"}))
        return _sink(request)

    with wire_server(_sink) as sink_b, ExitStack() as sink_a_stack:
        sink_a: Final = sink_a_stack.enter_context(wire_server(gated_sink, port=port_a))
        name_a: Final = f"h1-stream-{uuid.uuid4().hex}"
        name_b: Final = f"h1-non-stream-{uuid.uuid4().hex}"
        rails: Final = (
            _rail(name_a, sink_a.url, "streaming"),
            _rail(name_b, sink_b.url, "non_streaming"),
        )
        with _owned_proxy(rig, tmp_path, rails) as owned, ThreadPoolExecutor(max_workers=30) as pool:
            outage_plans: Final = _plans("h1-burst", 30)
            futures: Final[tuple[Future[httpx.Response], ...]] = tuple(
                pool.submit(_request, owned.gateway, plan, (name_a, name_b)) for plan in outage_plans
            )
            try:
                assert started.wait(timeout=30), "streaming rail did not reach sink A"
                unavailable.set()
            finally:
                release.set()
            sink_a_stack.close()
            outage_responses: Final = tuple(future.result(timeout=70) for future in futures)
            outage_provider: Final = rig.provider.drain()
            outage_a: Final = sink_a.drain()
            outage_b: Final = sink_b.drain()
            _assert_down_wave(outage_plans, outage_responses, outage_provider, outage_a, outage_b, name_a, name_b)

            with wire_server(_sink, port=port_a) as recovered_sink_a:
                recovery_plans: Final = _plans("h1-recovery", 20)
                recovery_responses: Final = _call_wave(owned.gateway, recovery_plans, (name_a, name_b))
                recovery_provider: Final = rig.provider.drain()
                recovery_a: Final = recovered_sink_a.drain()
                recovery_b: Final = sink_b.drain()
                assert tuple(response.status_code for response in recovery_responses) == (200,) * 20, recovery_responses
                _assert_successful_calls(
                    recovery_plans,
                    recovery_responses,
                    recovery_provider,
                    recovery_a,
                    recovery_b,
                    name_a,
                    name_b,
                )


def test_h2_stream_sink_gate_does_not_block_out_of_scope_calls(rig: ChaosRig, tmp_path: Path) -> None:
    started: Final = Event()
    release: Final = Event()
    blocked_marker: Final = f"audit-h2-stream-{uuid.uuid4().hex}"

    def gated_sink(request: Request) -> Reply:
        if blocked_marker.encode() in request.body:
            started.set()
            assert release.wait(timeout=45), "stream sink gate was not released"
        return _sink(request)

    with wire_server(gated_sink) as sink_a, wire_server(_sink) as sink_b:
        name_a: Final = f"h2-stream-{uuid.uuid4().hex}"
        name_b: Final = f"h2-non-stream-{uuid.uuid4().hex}"
        rails: Final = (_rail(name_a, sink_a.url, "streaming"), _rail(name_b, sink_b.url, "non_streaming"))
        with _owned_proxy(rig, tmp_path, rails) as owned, ThreadPoolExecutor(max_workers=2) as pool:
            streaming_plan: Final = CallPlan(blocked_marker, f"h2-stream-{uuid.uuid4().hex}", "chat", True)
            non_streaming_plan: Final = CallPlan(
                f"audit-h2-non-stream-{uuid.uuid4().hex}",
                f"h2-non-stream-{uuid.uuid4().hex}",
                "messages",
                False,
            )
            streaming_future: Final = pool.submit(_request, owned.gateway, streaming_plan, (name_a, name_b))
            assert started.wait(timeout=30), "stream request did not reach the gated sink"
            try:
                non_streaming_future: Final = pool.submit(
                    _request,
                    owned.gateway,
                    non_streaming_plan,
                    (name_a, name_b),
                )
                non_streaming_response: Final = non_streaming_future.result(timeout=15)
                assert non_streaming_response.status_code == 200, non_streaming_response.text
                assert non_streaming_plan.marker in non_streaming_response.text, non_streaming_response.text
            finally:
                release.set()
            streaming_response: Final = streaming_future.result(timeout=30)
            assert streaming_response.status_code == 200, streaming_response.text
            assert streaming_plan.marker in streaming_response.text, streaming_response.text
            provider_rows: Final = rig.provider.drain()
            sink_a_rows: Final = sink_a.drain()
            sink_b_rows: Final = sink_b.drain()
            _assert_successful_calls(
                (streaming_plan, non_streaming_plan),
                (streaming_response, non_streaming_response),
                provider_rows,
                sink_a_rows,
                sink_b_rows,
                name_a,
                name_b,
            )


def _worker_processes(owned: OwnedProxy) -> tuple[psutil.Process, ...]:
    return tuple(
        child for child in psutil.Process(owned.process.pid).children() if "spawn_main" in _process_command(child)
    )


def _process_command(process: psutil.Process) -> str:
    try:
        return " ".join(process.cmdline())
    except psutil.Error:
        return ""


def _safe_response(future: Future[httpx.Response]) -> httpx.Response | None:
    try:
        return future.result(timeout=70)
    except (httpx.HTTPError, TimeoutError):
        return None


def test_h3_worker_kill_mid_burst_keeps_remaining_worker_serving(rig: ChaosRig, tmp_path: Path) -> None:
    plans: Final = _plans("h3-burst", 30)
    gate_markers: Final = frozenset(plan.marker for plan in plans if plan.streamed)
    started: Final = Event()
    release: Final = Event()

    def gated_sink(request: Request) -> Reply:
        if any(marker.encode() in request.body for marker in gate_markers):
            started.set()
            assert release.wait(timeout=60), "worker-kill sink gate was not released"
        return _sink(request)

    with wire_server(gated_sink) as sink_a, wire_server(_sink) as sink_b:
        name_a: Final = f"h3-stream-{uuid.uuid4().hex}"
        name_b: Final = f"h3-non-stream-{uuid.uuid4().hex}"
        rails: Final = (
            _rail(name_a, sink_a.url, "streaming", default_on=True),
            _rail(name_b, sink_b.url, "non_streaming", default_on=True),
        )
        with _owned_proxy(rig, tmp_path, rails, workers=2) as owned, ThreadPoolExecutor(max_workers=30) as pool:
            workers: Final = _worker_processes(owned)
            assert len(workers) == 2, tuple(worker.pid for worker in workers)
            futures: Final[tuple[Future[httpx.Response], ...]] = tuple(
                pool.submit(_request, owned.gateway, plan, ()) for plan in plans
            )
            assert started.wait(timeout=30), "stream requests did not reach the owned sink"
            victim: Final = workers[0]
            survivor: Final = workers[1]
            survivor_plan: Final = CallPlan(
                f"audit-h3-survivor-{uuid.uuid4().hex}",
                f"h3-survivor-{uuid.uuid4().hex}",
                "messages",
                False,
            )
            try:
                victim.send_signal(signal.SIGKILL)
                survivor_response: Final = _request(owned.gateway, survivor_plan, ())
                assert survivor_response.status_code == 200 and survivor_plan.marker in survivor_response.text, (
                    survivor_response.status_code,
                    survivor_response.text,
                )
                assert survivor.is_running(), survivor.pid
            finally:
                release.set()
            responses: Final = tuple(_safe_response(future) for future in futures)
            assert owned.gateway.request("GET", "/health/liveliness").status_code == 200
            successful: Final = tuple(
                (plan, response)
                for plan, response in zip(plans, responses)
                if response is not None and response.status_code == 200
            )
            provider_rows: Final = rig.provider.drain()
            sink_a_rows: Final = sink_a.drain()
            sink_b_rows: Final = sink_b.drain()
            successful_plans: Final = (*tuple(plan for plan, _ in successful), survivor_plan)
            successful_responses: Final = (*tuple(response for _, response in successful), survivor_response)
            _assert_successful_calls(
                successful_plans,
                successful_responses,
                provider_rows,
                sink_a_rows,
                sink_b_rows,
                name_a,
                name_b,
            )


def _create_stored_rail(
    gateway: Gateway,
    name: str,
    sink_url: str,
    stream_scope: Literal["streaming"] | None = "streaming",
) -> str:
    response: Final = gateway.request(
        "POST",
        "/guardrails",
        {
            "guardrail": {
                "guardrail_name": name,
                "litellm_params": {
                    "guardrail": "generic_guardrail_api",
                    "mode": "pre_call",
                    "default_on": False,
                    "api_base": f"{sink_url}/{name}",
                    "api_key": "synthetic-chaos-key",
                    **({"stream_scope": stream_scope} if stream_scope is not None else {}),
                },
            }
        },
    )
    assert response.status_code == 200, response.text
    identity: Final = JSON_OBJECT.validate_json(response.content).get("guardrail_id")
    assert isinstance(identity, str), response.text
    return identity


def _assert_one_scope_wave(
    plans: Sequence[CallPlan],
    responses: Sequence[httpx.Response],
    provider_rows: Sequence[Request],
    sink_rows: Sequence[Request],
    sink_name: str,
    *,
    database_url: str | None = None,
) -> None:
    for plan, response in zip(plans, responses):
        assert response.status_code == 200 and plan.marker in response.text, (plan, response.status_code, response.text)
        provider_match: Final = _rows_for_marker(provider_rows, plan.marker)
        sink_match: Final = _rail_scans(sink_rows, sink_name, plan.marker)
        assert len(provider_match) == 1, (plan, provider_match)
        assert len(sink_match) == int(plan.streamed), (plan, sink_match)
        spend: Final = _one_spend_row(plan.call_id, database_url)
        assert plan.call_id in (spend.get("request_id"), spend.get("litellm_call_id")), (plan, spend)


def test_h4_stored_scope_survives_owned_proxy_restart(rig: ChaosRig, tmp_path: Path) -> None:
    with wire_server(_sink) as sink:
        name: Final = f"h4-stored-{uuid.uuid4().hex}"
        identity: Final = _create_stored_rail(rig.gateway, name, sink.url)
        try:
            with ExitStack() as first_stack:
                first: Final = first_stack.enter_context(_owned_proxy(rig, tmp_path, ()))
                first_plans: Final = _plans("h4-before", 20)
                first_responses: Final = _call_wave(first.gateway, first_plans, (name,))
                first_provider: Final = rig.provider.drain()
                first_sink: Final = sink.drain()
                assert tuple(response.status_code for response in first_responses) == (200,) * 20, first_responses
                _assert_one_scope_wave(first_plans, first_responses, first_provider, first_sink, name)
                first_stack.close()
                with _owned_proxy(rig, tmp_path, ()) as restarted:
                    recovery_plans: Final = _plans("h4-after", 20)
                    recovery_responses: Final = _call_wave(restarted.gateway, recovery_plans, (name,))
                    recovery_provider: Final = rig.provider.drain()
                    recovery_sink: Final = sink.drain()
                    assert tuple(response.status_code for response in recovery_responses) == (200,) * 20, (
                        recovery_responses,
                    )
                    _assert_one_scope_wave(
                        recovery_plans,
                        recovery_responses,
                        recovery_provider,
                        recovery_sink,
                        name,
                    )
        finally:
            deleted: Final = rig.gateway.request("DELETE", f"/guardrails/{identity}")
            assert deleted.status_code == 200, deleted.text


@contextmanager
def _owned_postgres(directory: Path) -> Iterator[PostgresCluster]:
    docker: Final = shutil.which("docker")
    assert docker is not None, "Docker CLI is required for the H5 PostgreSQL outage test"
    docker_info: Final = subprocess.run(
        [docker, "info", "--format", "{{.ServerVersion}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert docker_info.returncode == 0, docker_info.stderr
    directory.mkdir(parents=True, exist_ok=True)
    port: Final = _free_port()
    container_name: Final = f"litellm-stream-scope-h5-{uuid.uuid4().hex}"
    password: Final = uuid.uuid4().hex
    created: Final = subprocess.run(
        [
            docker,
            "create",
            "--name",
            container_name,
            "--env",
            "POSTGRES_USER=postgres",
            "--env",
            "POSTGRES_PASSWORD",
            "--env",
            "POSTGRES_DB=postgres",
            "--publish",
            f"127.0.0.1:{port}:5432/tcp",
            POSTGRES_IMAGE,
        ],
        env=os.environ | {"POSTGRES_PASSWORD": password},
        capture_output=True,
        text=True,
        check=False,
    )
    assert created.returncode == 0, created.stderr
    cluster: Final = PostgresCluster(
        f"postgresql://postgres:{password}@127.0.0.1:{port}/postgres?sslmode=disable",
        container_name,
        docker,
        directory / "postgres.log",
    )
    try:
        started: Final = _start_postgres(cluster)
        assert started.returncode == 0, started.stderr
        assert eventually(lambda: _postgres_is_ready(cluster), bool, seconds=70)
        yield cluster
    finally:
        logs: Final = subprocess.run(
            [docker, "logs", container_name],
            capture_output=True,
            text=True,
            check=False,
        )
        cluster.log_path.write_text(logs.stdout + logs.stderr)
        removed: Final = subprocess.run(
            [docker, "rm", "-f", container_name],
            capture_output=True,
            text=True,
            check=False,
        )
        assert removed.returncode == 0, removed.stderr
        remaining: Final = subprocess.run(
            [docker, "ps", "--all", "--quiet", "--filter", f"name={container_name}"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert remaining.returncode == 0, remaining.stderr
        assert not remaining.stdout.strip(), remaining.stdout


@dataclass(frozen=True, slots=True)
class PostgresCluster:
    database_url: str
    container_name: str
    docker: str
    log_path: Path


def _start_postgres(cluster: PostgresCluster) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [cluster.docker, "start", cluster.container_name],
        capture_output=True,
        text=True,
        check=False,
    )


def _postgres_is_ready(cluster: PostgresCluster) -> bool:
    readiness: Final = subprocess.run(
        [
            cluster.docker,
            "exec",
            cluster.container_name,
            "pg_isready",
            "-h",
            "127.0.0.1",
            "-p",
            "5432",
            "-d",
            "postgres",
            "-U",
            "postgres",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return readiness.returncode == 0


def _postgres_is_running(cluster: PostgresCluster) -> bool:
    state: Final = subprocess.run(
        [cluster.docker, "inspect", "--format", "{{.State.Running}}", cluster.container_name],
        capture_output=True,
        text=True,
        check=False,
    )
    return state.returncode == 0 and state.stdout.strip() == "true"


@pytest.mark.timeout(180)
def test_h5_stored_scope_survives_owned_postgres_outage_and_recovers_once(
    rig: ChaosRig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    postgres_directory: Final = Path(os.environ["INTEGRATION_RESULTS_DIR"]) / f"owned-postgres-{uuid.uuid4().hex}"
    with _owned_postgres(postgres_directory) as database, wire_server(_sink) as sink:
        monkeypatch.setenv("INTEGRATION_PROXY_DATABASE_URL", database.database_url)
        name: Final = f"h5-stored-{uuid.uuid4().hex}"
        with _owned_proxy(rig, tmp_path, ()) as registrar:
            _create_stored_rail(registrar.gateway, name, sink.url)
        with _owned_proxy(rig, tmp_path, ()) as owned:
            try:
                preflight_plans: Final = (
                    CallPlan(
                        f"audit-h5-preflight-{uuid.uuid4().hex}",
                        f"h5-preflight-{uuid.uuid4().hex}",
                        "chat",
                        True,
                    ),
                )
                preflight_responses: Final = _call_wave(owned.gateway, preflight_plans, (name,))
                preflight_provider: Final = rig.provider.drain()
                preflight_sink: Final = sink.drain()
                assert tuple(response.status_code for response in preflight_responses) == (200,), (preflight_responses,)
                _assert_one_scope_wave(
                    preflight_plans,
                    preflight_responses,
                    preflight_provider,
                    preflight_sink,
                    name,
                    database_url=database.database_url,
                )
                outage_result: Final = subprocess.run(
                    [database.docker, "stop", database.container_name],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                assert outage_result.returncode == 0, outage_result.stderr
                outage_plans: Final = _plans("h5-outage", 20)
                outage_responses: Final = _call_wave(owned.gateway, outage_plans, (name,))
                outage_provider: Final = rig.provider.drain()
                outage_sink: Final = sink.drain()
                for plan, response in zip(outage_plans, outage_responses):
                    assert response.status_code == 200 and plan.marker in response.text, (
                        plan,
                        response.status_code,
                        response.text,
                    )
                    assert len(_rows_for_marker(outage_provider, plan.marker)) == 1, (plan, outage_provider)
                    sink_match: Final = _rail_scans(outage_sink, name, plan.marker)
                    assert len(sink_match) == int(plan.streamed), (plan, sink_match)

                recovered_database: Final = _start_postgres(database)
                assert recovered_database.returncode == 0, recovered_database.stderr
                postgres_ready: Final = eventually(lambda: _postgres_is_ready(database), bool, seconds=70)
                assert postgres_ready
                readiness: Final = eventually(
                    lambda: owned.gateway.client.get("/health/readiness"),
                    lambda response: response.status_code == 200 and response.json().get("db") == "connected",
                    seconds=70,
                )
                assert readiness.json().get("db") == "connected", readiness.text
                recovery_plans: Final = _plans("h5-recovery", 20)
                recovery_responses: Final = _call_wave(owned.gateway, recovery_plans, (name,))
                recovery_provider: Final = rig.provider.drain()
                recovery_sink: Final = sink.drain()
                assert tuple(response.status_code for response in recovery_responses) == (200,) * 20, recovery_responses
                _assert_one_scope_wave(
                    recovery_plans,
                    recovery_responses,
                    recovery_provider,
                    recovery_sink,
                    name,
                    database_url=database.database_url,
                )
            finally:
                if not _postgres_is_running(database):
                    restarted: Final = _start_postgres(database)
                    assert restarted.returncode == 0, restarted.stderr
                    assert eventually(lambda: _postgres_is_ready(database), bool, seconds=70)


@pytest.mark.timeout(360)
def test_h6_spend_rows_for_requests_served_during_postgres_restart_land_once(
    rig: ChaosRig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.skip("BUG: LIT-9050 spend rows for requests served during a Postgres restart are dropped")
    postgres_directory: Final = Path(os.environ["INTEGRATION_RESULTS_DIR"]) / f"owned-postgres-{uuid.uuid4().hex}"
    with _owned_postgres(postgres_directory) as database, wire_server(_sink) as sink:
        monkeypatch.setenv("INTEGRATION_PROXY_DATABASE_URL", database.database_url)
        name: Final = f"h6-stored-{uuid.uuid4().hex}"
        plans: Final = _plans("h6-postgres-restart", 20)
        outage_markers: Final = frozenset(plan.marker for plan in plans[:10])
        arrivals: Final = Barrier(len(plans) + 1)
        during_outage: Final = Event()
        after_restart: Final = Event()

        def _gated_provider(request: Request) -> Reply:
            if request.method == "GET" and request.target.partition("?")[0] == "/v1/models":
                return _provider(request)
            body: Final = JSON_OBJECT.validate_json(request.body)
            marker: Final = _marker(body)
            arrivals.wait(timeout=70)
            gate: Final = during_outage if marker in outage_markers else after_restart
            assert gate.wait(timeout=70), marker
            return _provider(request)

        with wire_server(_gated_provider) as provider:
            h6_rig: Final = ChaosRig(rig.gateway, provider, rig.directory)
            with _owned_proxy(h6_rig, tmp_path, ()) as registrar:
                _create_stored_rail(registrar.gateway, name, sink.url, stream_scope=None)
            with _owned_proxy(h6_rig, tmp_path, ()) as owned, ThreadPoolExecutor(max_workers=len(plans)) as pool:
                futures: Final = tuple(
                    pool.submit(_request, owned.gateway, plan, (name,)) for plan in plans
                )
                try:
                    arrivals.wait(timeout=70)
                    stopped: Final = subprocess.run(
                        [database.docker, "stop", database.container_name],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    assert stopped.returncode == 0, stopped.stderr
                    during_outage.set()
                    outage_responses: Final = tuple(future.result(timeout=70) for future in futures[:10])
                    for plan, response in zip(plans[:10], outage_responses):
                        assert response.status_code == 200 and plan.marker in response.text, (
                            plan,
                            response.status_code,
                            response.text,
                        )

                    restarted: Final = _start_postgres(database)
                    assert restarted.returncode == 0, restarted.stderr
                    postgres_ready: Final = eventually(lambda: _postgres_is_ready(database), bool, seconds=70)
                    assert postgres_ready
                    readiness: Final = eventually(
                        lambda: owned.gateway.client.get("/health/readiness"),
                        lambda response: response.status_code == 200
                        and JSON_OBJECT.validate_python(cast(object, response.json())).get("db") == "connected",
                        seconds=70,
                    )
                    readiness_body: Final = JSON_OBJECT.validate_python(cast(object, readiness.json()))
                    assert readiness_body.get("db") == "connected", readiness.text
                    after_restart.set()
                    responses: Final = tuple(future.result(timeout=70) for future in futures)
                finally:
                    during_outage.set()
                    after_restart.set()
                    if not _postgres_is_running(database):
                        recovered: Final = _start_postgres(database)
                        assert recovered.returncode == 0, recovered.stderr
                        assert eventually(lambda: _postgres_is_ready(database), bool, seconds=70)

                provider_rows: Final = provider.drain()
                sink_rows: Final = sink.drain()
                for plan, response in zip(plans, responses):
                    assert response.status_code == 200 and plan.marker in response.text, (
                        plan,
                        response.status_code,
                        response.text,
                    )
                    assert len(_rows_for_marker(provider_rows, plan.marker)) == 1, (plan, provider_rows)
                    assert len(_rail_scans(sink_rows, name, plan.marker)) == 1, (plan, sink_rows)

                spend_rows: Final = eventually(
                    lambda: tuple(_spend_rows(plan.call_id, database.database_url) for plan in plans),
                    lambda values: all(len(rows) == 1 for rows in values),
                    seconds=70,
                    return_last_on_timeout=True,
                )
                counts: Final = tuple(len(rows) for rows in spend_rows)
                missing_ids: Final = tuple(
                    plan.call_id for plan, rows in zip(plans, spend_rows) if not rows
                )
                duplicate_ids: Final = tuple(
                    plan.call_id for plan, rows in zip(plans, spend_rows) if len(rows) > 1
                )
                assert counts == (1,) * len(plans), {
                    "missing_ids": missing_ids,
                    "duplicate_ids": duplicate_ids,
                    "counts": counts,
                }
