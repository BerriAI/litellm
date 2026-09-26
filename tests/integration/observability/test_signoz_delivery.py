import asyncio
import base64
import json
import os
import re
import signal
import threading
import uuid
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Final

import anthropic
import httpx
import openai
import psutil
import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue
from pydantic import JsonValue, TypeAdapter

MARKER: Final = re.compile(rb"signoz-[0-9a-f]{32}")
JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
RESPONSE_ID: Final = "gen_ai.response.id"
INGESTION_HEADER: Final = "signoz-ingestion-key"
OPERATOR_KEY: Final = "operator-ingestion-" + uuid.uuid4().hex
TENANT_KEY: Final = "tenant-ingestion-" + uuid.uuid4().hex


def _marker() -> str:
    return "signoz-" + uuid.uuid4().hex


def _chat_reply(identity: str, stream: bool) -> Reply:
    if not stream:
        return Reply(
            body=json.dumps(
                {
                    "id": identity,
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "signoz ok"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
                }
            ).encode()
        )
    chunk: Final = {"id": identity, "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini"}
    return Reply(
        content_type="text/event-stream",
        chunks=(
            b"data: "
            + json.dumps(
                {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "signoz"}}]}
            ).encode()
            + b"\n\n",
            b"data: "
            + json.dumps(
                {**chunk, "choices": [{"index": 0, "delta": {"content": " ok"}, "finish_reason": "stop"}]}
            ).encode()
            + b"\n\n",
            b"data: "
            + json.dumps(
                {**chunk, "choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}}
            ).encode()
            + b"\n\n",
            b"data: [DONE]\n\n",
        ),
    )


def _responses_reply(identity: str, stream: bool) -> Reply:
    response: Final[dict[str, JsonValue]] = {
        "id": identity,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-4o-mini",
        "output": [
            {
                "id": "msg_" + identity,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "signoz ok", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9},
    }
    if not stream:
        return Reply(body=json.dumps(response).encode())
    events: Final[tuple[dict[str, JsonValue], ...]] = (
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {**response, "status": "in_progress", "output": []},
        },
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": "msg_" + identity,
            "output_index": 0,
            "content_index": 0,
            "delta": "signoz ok",
        },
        {"type": "response.completed", "sequence_number": 2, "response": response},
    )
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events),
    )


def _upstream(request: Request) -> Reply:
    found: Final = MARKER.search(request.body)
    if found is None:
        return Reply(status=404, body=b'{"error":"no marker"}')
    if request.headers.get("authorization") == "Bearer revoked-provider-key":
        return Reply(
            status=401, body=b'{"error":{"message":"Incorrect API key provided","type":"invalid_request_error"}}'
        )
    marker: Final = found.group(0).decode()
    stream: Final = object_value(JSON.validate_json(request.body)).get("stream") is True
    if request.target.endswith("/responses"):
        return _responses_reply(f"resp_{marker}", stream)
    return _chat_reply(f"chatcmpl-{marker}", stream)


def _decoded_responses_id(identity: str) -> str:
    try:
        return base64.b64decode(identity.removeprefix("resp_").encode()).decode()
    except (ValueError, UnicodeDecodeError):
        return identity


def _canonical_id(identity: str) -> str:
    return _decoded_responses_id(identity).rpartition("response_id:")[2]


def _sse_events(text: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        object_value(JSON.validate_json(line[6:]))
        for line in text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    )


def _text_at(payload: JsonValue, *path: str) -> str:
    if not path:
        return string_value(payload)
    return _text_at(object_value(payload)[path[0]], *path[1:])


def _body_id(response: httpx.Response) -> str:
    return _text_at(JSON.validate_json(response.content), "id")


@dataclass(frozen=True, slots=True)
class Span:
    target: str
    ingestion_key: str | None
    attributes: Mapping[str, str]


def _attribute_text(value: AnyValue) -> str:
    match value.WhichOneof("value"):
        case "string_value":
            return value.string_value
        case "int_value":
            return str(value.int_value)
        case "double_value":
            return str(value.double_value)
        case "bool_value":
            return str(value.bool_value)
        case _:
            return ""


@dataclass(frozen=True, slots=True)
class Collector:
    wire: Wire
    outage: threading.Event
    rejection: threading.Event
    missing: threading.Event
    slow: threading.Event
    release: threading.Event
    accepted: Sequence[Request]
    refused: Sequence[Request]
    guard: threading.Lock

    def refused_batch_carrying(self, response_id: str) -> Request:
        def carrying() -> tuple[Request, ...]:
            with self.guard:
                return tuple(batch for batch in self.refused if response_id.encode() in batch.body)

        return eventually(carrying, lambda found: len(found) >= 1, seconds=30)[0]

    def refused_batches(self) -> tuple[Request, ...]:
        def refused() -> tuple[Request, ...]:
            with self.guard:
                return tuple(self.refused)

        return eventually(refused, lambda found: len(found) >= 1, seconds=30)

    def spans(self) -> tuple[Span, ...]:
        with self.guard:
            batches: Final = tuple(self.accepted)
        return tuple(
            Span(
                batch.target,
                batch.headers.get(INGESTION_HEADER),
                {attribute.key: _attribute_text(attribute.value) for attribute in span.attributes},
            )
            for batch in batches
            for resource in ExportTraceServiceRequest.FromString(batch.body).resource_spans
            for scope in resource.scope_spans
            for span in scope.spans
        )

    def spans_for(self, response_id: str) -> tuple[Span, ...]:
        return tuple(
            span
            for span in self.spans()
            if RESPONSE_ID in span.attributes
            and _canonical_id(span.attributes[RESPONSE_ID]) == _canonical_id(response_id)
        )

    def single_span(self, response_id: str, *, elsewhere: "Collector | None" = None) -> Span:
        found: Final = eventually(
            lambda: self.spans_for(response_id), lambda spans: len(spans) == 1, seconds=30, return_last_on_timeout=True
        )
        assert len(found) == 1, (
            f"{len(found)} spans for {response_id} at this sink; other sink saw "
            f"{elsewhere.landed((response_id,)) if elsewhere else 'n/a'}"
        )
        return found[0]

    def landed(self, response_ids: Sequence[str]) -> dict[str, int]:
        spans: Final = self.spans()
        return {
            _canonical_id(identity): sum(
                1
                for span in spans
                if RESPONSE_ID in span.attributes
                and _canonical_id(span.attributes[RESPONSE_ID]) == _canonical_id(identity)
            )
            for identity in response_ids
        }


def _collector() -> Iterator[Collector]:
    outage: Final = threading.Event()
    rejection: Final = threading.Event()
    missing: Final = threading.Event()
    slow: Final = threading.Event()
    release: Final = threading.Event()
    accepted: Final[deque[Request]] = deque()  # mutable-ok: the sink thread records each accepted batch as it arrives
    refused: Final[deque[Request]] = deque()  # mutable-ok: the sink thread records each refused batch as it arrives
    guard: Final = threading.Lock()

    def refuse(request: Request, status: int, body: bytes) -> Reply:
        with guard:
            refused.append(request)
        return Reply(status=status, body=body)

    def sink(request: Request) -> Reply:
        if slow.is_set():
            release.wait(timeout=30)
        if outage.is_set():
            return refuse(request, 503, b'{"error":"sink down"}')
        if rejection.is_set():
            return refuse(request, 403, b'{"error":"forbidden"}')
        if missing.is_set():
            return refuse(request, 404, b'{"error":"not found"}')
        with guard:
            accepted.append(request)
        return Reply()

    with wire_server(sink) as wire:
        yield Collector(wire, outage, rejection, missing, slow, release, accepted, refused, guard)


@pytest.fixture(scope="session")
def operator_sink() -> Iterator[Collector]:
    yield from _collector()


@pytest.fixture(scope="session")
def tenant_sink() -> Iterator[Collector]:
    yield from _collector()


@pytest.fixture(scope="session")
def provider() -> Iterator[Wire]:
    with wire_server(_upstream) as wire:
        yield wire


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    process: OwnedProxy
    model: str
    upstream: Wire
    sink: Collector
    tenant_sink: Collector

    def openai_client(self) -> openai.OpenAI:
        return openai.OpenAI(base_url=str(self.proxy.client.base_url) + "/v1", api_key=self.proxy.key, max_retries=0)

    def async_openai_client(self) -> openai.AsyncOpenAI:
        return openai.AsyncOpenAI(
            base_url=str(self.proxy.client.base_url) + "/v1", api_key=self.proxy.key, max_retries=0
        )

    def anthropic_client(self) -> anthropic.Anthropic:
        return anthropic.Anthropic(base_url=str(self.proxy.client.base_url), api_key=self.proxy.key, max_retries=0)

    def async_anthropic_client(self) -> anthropic.AsyncAnthropic:
        return anthropic.AsyncAnthropic(base_url=str(self.proxy.client.base_url), api_key=self.proxy.key, max_retries=0)

    def chat(
        self, marker: str, *, headers: Mapping[str, str] | None = None, key: str | None = None, **extra: JsonValue
    ) -> httpx.Response:
        return self.proxy.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": self.model,
                "messages": [{"role": "user", "content": marker}],
                "cache": {"no-cache": True},
                **extra,
            },
            headers=headers,
            key=key,
        )

    def upstream_bodies(self, marker: str) -> tuple[dict[str, JsonValue], ...]:
        return tuple(
            object_value(JSON.validate_json(request.body))
            for request in self.upstream.drain()
            if marker.encode() in request.body
        )

    def spend_rows(self, response_id: str) -> tuple[dict[str, JsonValue], ...]:
        return tuple(
            eventually(
                lambda: read_rows(
                    'SELECT request_id, spend FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (response_id,)
                ),
                lambda values: len(values) == 1,
                seconds=70,
            )
        )

    def tenant_logging(self, endpoint: str | None, key: str | None) -> JsonValue:
        variables: Final[dict[str, JsonValue]] = {
            **({"signoz_ingestion_endpoint": endpoint} if endpoint is not None else {}),
            **({"signoz_ingestion_key": key} if key is not None else {}),
        }
        return [{"callback_name": "signoz", "callback_type": "success", "callback_vars": variables}]


@dataclass(frozen=True, slots=True)
class RigFactory:
    provider: Wire
    sink: Collector
    tenant_sink: Collector
    directory: Path
    otel_v2: bool
    workers: int
    endpoint: str | None
    ingestion_key: str | None = OPERATOR_KEY

    def config_path(self) -> Path:
        loaded: Final = object_value(
            JSON.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
        )
        config: Final = {
            **loaded,
            "litellm_settings": {
                **object_value(loaded["litellm_settings"]),
                "callbacks": ["signoz"],
                "provider_url_destination_allowed_hosts": [self.tenant_sink.wire.url],
            },
            "general_settings": {**object_value(loaded["general_settings"]), "disable_model_info_refresh": True},
        }
        path: Final = self.directory / f"signoz-{uuid.uuid4().hex}.yaml"
        path.write_text(yaml.safe_dump(config))
        return path

    def overrides(self) -> dict[str, str]:
        return {
            "LITELLM_OTEL_V2": "1" if self.otel_v2 else "0",
            "OTEL_BSP_SCHEDULE_DELAY": "300",
            **({"SIGNOZ_INGESTION_ENDPOINT": self.endpoint} if self.endpoint is not None else {}),
            **({"SIGNOZ_INGESTION_KEY": self.ingestion_key} if self.ingestion_key is not None else {}),
        }

    def start(self) -> Iterator[Rig]:
        with (
            gateway_from_environment() as gateway,
            owned_proxy_process(
                gateway,
                self.directory,
                self.overrides(),
                config=self.config_path(),
                remove_environment=("SIGNOZ_INGESTION_ENDPOINT", "SIGNOZ_INGESTION_KEY"),
                workers=self.workers,
            ) as owned,
            owned.gateway.scenario() as scenario,
        ):
            model: Final = scenario.model(api_base=self.provider.url + "/v1")
            yield Rig(owned.gateway, owned, model, self.provider, self.sink, self.tenant_sink)


@pytest.fixture(scope="session")
def rig(
    provider: Wire, operator_sink: Collector, tenant_sink: Collector, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Rig]:
    factory: Final = RigFactory(
        provider, operator_sink, tenant_sink, tmp_path_factory.mktemp("signoz"), False, 2, operator_sink.wire.url
    )
    yield from factory.start()


@pytest.fixture(scope="session")
def v2_rig(
    provider: Wire, operator_sink: Collector, tenant_sink: Collector, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Rig]:
    factory: Final = RigFactory(
        provider, operator_sink, tenant_sink, tmp_path_factory.mktemp("signoz-v2"), True, 2, operator_sink.wire.url
    )
    yield from factory.start()


def _assert_operator_span(rig: Rig, response_id: str, marker: str) -> Span:
    span: Final = rig.sink.single_span(response_id)
    assert span.target == "/v1/traces", span
    assert span.ingestion_key == OPERATOR_KEY, span
    assert rig.tenant_sink.landed((response_id,)) == {_canonical_id(response_id): 0}
    bodies: Final = rig.upstream_bodies(marker)
    assert len(bodies) == 1, bodies
    assert "signoz" not in json.dumps(bodies[0]).replace(marker, ""), bodies[0]
    return span


def test_signoz_is_registered_as_an_opentelemetry_callback(rig: Rig) -> None:
    listed: Final = rig.proxy.request("GET", "/active/callbacks")
    assert listed.status_code == 200, listed.text
    assert "OpenTelemetry" in json.dumps(listed.json()), listed.text
    log: Final = rig.process.log.read_text()
    assert "SIGNOZ_INGESTION_ENDPOINT not found" not in log


def test_chat_completion_sdk_span_lands_at_the_operator_sink_with_the_ingestion_key(rig: Rig) -> None:
    marker: Final = _marker()
    completion: Final = rig.openai_client().chat.completions.create(
        model=rig.model, messages=[{"role": "user", "content": marker}]
    )
    assert completion.id == f"chatcmpl-{marker}"
    span: Final = _assert_operator_span(rig, completion.id, marker)
    assert span.attributes.get("gen_ai.request.model") or span.attributes.get("llm.request.model"), span
    rows: Final = rig.spend_rows(completion.id)
    assert rows[0]["request_id"] == completion.id, rows


def test_chat_stream_async_sdk_span_lands_once_after_the_stream_is_consumed(rig: Rig) -> None:
    marker: Final = _marker()

    async def consume() -> frozenset[str]:
        stream: Final = await rig.async_openai_client().chat.completions.create(
            model=rig.model, messages=[{"role": "user", "content": marker}], stream=True
        )
        return frozenset([chunk.id async for chunk in stream])

    identities: Final = asyncio.run(consume())
    assert identities == {f"chatcmpl-{marker}"}, identities
    _assert_operator_span(rig, f"chatcmpl-{marker}", marker)


def test_messages_sdk_span_lands_at_the_operator_sink(rig: Rig) -> None:
    marker: Final = _marker()
    message: Final = rig.anthropic_client().messages.create(
        model=rig.model, max_tokens=16, messages=[{"role": "user", "content": marker}]
    )
    _assert_operator_span(rig, message.id, marker)


def test_messages_stream_async_sdk_span_lands_once_after_the_stream_is_consumed(rig: Rig) -> None:
    marker: Final = _marker()

    async def consume() -> str:
        async with rig.async_anthropic_client().messages.stream(
            model=rig.model, max_tokens=16, messages=[{"role": "user", "content": marker}]
        ) as stream:
            async for _ in stream:
                pass
            return (await stream.get_final_message()).id

    identity: Final = asyncio.run(consume())
    _assert_operator_span(rig, identity, marker)


def test_responses_sdk_span_lands_at_the_operator_sink(rig: Rig) -> None:
    marker: Final = _marker()
    response: Final = rig.openai_client().responses.create(model=rig.model, input=marker)
    _assert_operator_span(rig, response.id, marker)


def test_responses_stream_raw_httpx_span_lands_once_after_the_stream_is_consumed(rig: Rig) -> None:
    marker: Final = _marker()
    response: Final = rig.proxy.request("POST", "/v1/responses", {"model": rig.model, "input": marker, "stream": True})
    assert response.status_code == 200, response.text
    _assert_operator_span(rig, _responses_id(response, marker), marker)


def test_v2_flag_on_still_delivers_the_operator_span_with_the_ingestion_key(v2_rig: Rig) -> None:
    marker: Final = _marker()
    response: Final = v2_rig.chat(marker)
    assert response.status_code == 200, response.text
    _assert_operator_span(v2_rig, _body_id(response), marker)


def test_endpoint_already_ending_in_v1_traces_is_not_doubled(
    provider: Wire, operator_sink: Collector, tenant_sink: Collector, tmp_path_factory: pytest.TempPathFactory
) -> None:
    factory: Final = RigFactory(
        provider,
        operator_sink,
        tenant_sink,
        tmp_path_factory.mktemp("signoz-suffixed"),
        False,
        2,
        operator_sink.wire.url + "/v1/traces",
    )
    suffixed: Final = next(started := factory.start())
    marker: Final = _marker()
    response: Final = suffixed.chat(marker)
    assert response.status_code == 200, response.text
    span: Final = suffixed.sink.single_span(_body_id(response))
    assert span.target == "/v1/traces", span
    assert tuple(started) == ()


def test_three_identical_requests_produce_one_span_each(rig: Rig) -> None:
    markers: Final = tuple(_marker() for _ in range(3))
    responses: Final = tuple(rig.chat(marker) for marker in markers)
    assert all(response.status_code == 200 for response in responses), [response.text for response in responses]
    identities: Final = tuple(_body_id(response) for response in responses)
    landed: Final = eventually(
        lambda: rig.sink.landed(identities), lambda seen: all(count >= 1 for count in seen.values()), seconds=30
    )
    assert landed == {identity: 1 for identity in identities}, landed
    assert rig.sink.landed(identities) == landed


def test_unauthenticated_request_is_rejected_without_an_upstream_call_and_any_span_records_the_401(rig: Rig) -> None:
    marker: Final = _marker()
    response: Final = rig.chat(marker, key="sk-not-a-real-key")
    assert response.status_code == 401, response.text
    later: Final = rig.chat(_marker())
    assert later.status_code == 200, later.text
    rig.sink.single_span(_body_id(later))
    assert rig.upstream_bodies(marker) == ()
    marker_spans: Final = tuple(span for span in rig.sink.spans() if marker in json.dumps(span.attributes))
    assert all(span.attributes.get("error.code") == "401" for span in marker_spans), marker_spans
    assert not any(span.attributes.get(RESPONSE_ID, "").startswith("chatcmpl-") for span in marker_spans), marker_spans


def test_request_supplied_signoz_variables_are_refused_before_the_upstream_is_called(rig: Rig) -> None:
    marker: Final = _marker()
    response: Final = rig.chat(
        marker,
        metadata={"signoz_ingestion_endpoint": rig.tenant_sink.wire.url, "signoz_ingestion_key": TENANT_KEY},
    )
    assert response.status_code == 401, response.text
    assert "signoz_ingestion_endpoint is not allowed in request body" in response.text
    assert rig.upstream_bodies(marker) == ()
    assert not any(marker in json.dumps(span.attributes) for span in rig.tenant_sink.spans())


def test_upstream_401_reaches_the_caller_and_unrelated_traffic_keeps_landing(rig: Rig) -> None:
    marker: Final = _marker()
    with rig.proxy.scenario() as scenario:
        broken: Final = scenario.model(api_base=rig.upstream.url + "/v1", api_key="revoked-provider-key")
        failed: Final = rig.proxy.request(
            "POST", "/v1/chat/completions", {"model": broken, "messages": [{"role": "user", "content": marker}]}
        )
    assert failed.status_code == 401, failed.text
    assert "Incorrect API key provided" in failed.text
    healthy_marker: Final = _marker()
    healthy: Final = rig.chat(healthy_marker)
    assert healthy.status_code == 200, healthy.text
    _assert_operator_span(rig, _body_id(healthy), healthy_marker)


def test_health_services_accepts_signoz(rig: Rig) -> None:
    response: Final = rig.proxy.request("GET", "/health/services", params={"service": "signoz"})
    assert response.status_code == 200, response.text


def test_sink_answering_403_drops_those_spans_and_later_spans_still_land(rig: Rig) -> None:
    rig.sink.rejection.set()
    try:
        rejected: Final = rig.chat(_marker())
        assert rejected.status_code == 200, rejected.text
        rig.sink.refused_batch_carrying(_body_id(rejected))
    finally:
        rig.sink.rejection.clear()
    later: Final = rig.chat(_marker())
    assert later.status_code == 200, later.text
    rig.sink.single_span(_body_id(later))
    assert rig.sink.landed((_body_id(rejected),)) == {_body_id(rejected): 0}


def test_sink_answering_404_drops_those_spans_and_later_spans_still_land(rig: Rig) -> None:
    rig.sink.missing.set()
    try:
        dropped: Final = rig.chat(_marker())
        assert dropped.status_code == 200, dropped.text
        rig.sink.refused_batch_carrying(_body_id(dropped))
    finally:
        rig.sink.missing.clear()
    later: Final = rig.chat(_marker())
    assert later.status_code == 200, later.text
    rig.sink.single_span(_body_id(later))
    assert rig.sink.landed((_body_id(dropped),)) == {_body_id(dropped): 0}


def test_key_level_signoz_destination_routes_the_span_to_the_tenant_sink(v2_rig: Rig) -> None:
    marker: Final = _marker()
    with v2_rig.proxy.scenario() as scenario:
        token: Final = scenario.key(
            metadata={"logging": v2_rig.tenant_logging(v2_rig.tenant_sink.wire.url, TENANT_KEY)}
        )
        response: Final = v2_rig.chat(marker, key=token)
    assert response.status_code == 200, response.text
    identity: Final = _body_id(response)
    span: Final = v2_rig.tenant_sink.single_span(identity, elsewhere=v2_rig.sink)
    assert span.ingestion_key == TENANT_KEY, span
    assert v2_rig.sink.landed((identity,)) == {identity: 0}, "operator sink also received the tenant span"


def test_team_level_signoz_destination_routes_the_span_to_the_tenant_sink(v2_rig: Rig) -> None:
    marker: Final = _marker()
    with v2_rig.proxy.scenario() as scenario:
        team: Final = scenario.team(
            metadata={"logging": v2_rig.tenant_logging(v2_rig.tenant_sink.wire.url, TENANT_KEY)}
        )
        token: Final = scenario.key(team_id=team)
        response: Final = v2_rig.chat(marker, key=token)
    assert response.status_code == 200, response.text
    identity: Final = _body_id(response)
    span: Final = v2_rig.tenant_sink.single_span(identity, elsewhere=v2_rig.sink)
    assert span.ingestion_key == TENANT_KEY, span
    assert v2_rig.sink.landed((identity,)) == {identity: 0}, "operator sink also received the tenant span"


def test_key_level_destination_wins_over_the_team_level_destination(v2_rig: Rig) -> None:
    marker: Final = _marker()
    team_key: Final = "team-" + TENANT_KEY
    with v2_rig.proxy.scenario() as scenario:
        team: Final = scenario.team(metadata={"logging": v2_rig.tenant_logging(v2_rig.tenant_sink.wire.url, team_key)})
        token: Final = scenario.key(
            team_id=team, metadata={"logging": v2_rig.tenant_logging(v2_rig.tenant_sink.wire.url, TENANT_KEY)}
        )
        response: Final = v2_rig.chat(marker, key=token)
    assert response.status_code == 200, response.text
    span: Final = v2_rig.tenant_sink.single_span(_body_id(response))
    assert span.ingestion_key == TENANT_KEY, span


def test_team_endpoint_without_an_ingestion_key_routes_the_span_without_the_operator_key(v2_rig: Rig) -> None:
    marker: Final = _marker()
    with v2_rig.proxy.scenario() as scenario:
        team: Final = scenario.team(metadata={"logging": v2_rig.tenant_logging(v2_rig.tenant_sink.wire.url, None)})
        token: Final = scenario.key(team_id=team)
        response: Final = v2_rig.chat(marker, key=token)
    assert response.status_code == 200, response.text
    identity: Final = _body_id(response)
    span: Final = v2_rig.tenant_sink.single_span(identity, elsewhere=v2_rig.sink)
    assert span.ingestion_key is None, f"operator ingestion key leaked to the team collector: {span}"
    assert v2_rig.sink.landed((identity,)) == {identity: 0}, "operator sink also received the tenant span"


def test_team_endpoint_off_the_allowlist_keeps_the_span_at_the_operator_sink(v2_rig: Rig) -> None:
    marker: Final = _marker()
    with v2_rig.proxy.scenario() as scenario:
        team: Final = scenario.team(
            metadata={"logging": v2_rig.tenant_logging("http://tenant.invalid:4318/v1/traces", TENANT_KEY)}
        )
        token: Final = scenario.key(team_id=team)
        response: Final = v2_rig.chat(marker, key=token)
    assert response.status_code == 200, response.text
    _assert_operator_span(v2_rig, _body_id(response), marker)
    eventually(
        lambda: v2_rig.process.log.read_text(),
        lambda text: "provider_url_destination_allowed_hosts" in text,
        seconds=30,
    )


def test_legacy_mode_ignores_key_level_signoz_destination_and_keeps_the_operator_sink(rig: Rig) -> None:
    marker: Final = _marker()
    with rig.proxy.scenario() as scenario:
        token: Final = scenario.key(metadata={"logging": rig.tenant_logging(rig.tenant_sink.wire.url, TENANT_KEY)})
        response: Final = rig.chat(marker, key=token)
    assert response.status_code == 200, response.text
    _assert_operator_span(rig, _body_id(response), marker)


@pytest.mark.parametrize(
    "endpoint",
    ["", "not-a-url", "ftp://tenant.invalid", "x" * 5000, 12345, ["http://tenant.invalid"]],
    ids=["empty", "bare", "ftp", "5kb", "int", "list"],
)
def test_hostile_tenant_endpoint_never_breaks_the_request_or_the_operator_sink(
    v2_rig: Rig, endpoint: JsonValue
) -> None:
    marker: Final = _marker()
    created: Final = v2_rig.proxy.request(
        "POST",
        "/key/generate",
        {
            "metadata": {
                "logging": [
                    {
                        "callback_name": "signoz",
                        "callback_type": "success",
                        "callback_vars": {"signoz_ingestion_endpoint": endpoint, "signoz_ingestion_key": TENANT_KEY},
                    }
                ]
            }
        },
    )
    assert created.status_code in (200, 400, 422), created.text
    if created.status_code != 200:
        return
    try:
        response: Final = v2_rig.chat(marker, key=_text_at(JSON.validate_json(created.content), "key"))
        assert response.status_code == 200, response.text
        identity: Final = _body_id(response)
        eventually(
            lambda: v2_rig.sink.landed((identity,))[identity] + v2_rig.tenant_sink.landed((identity,))[identity],
            lambda total: total >= 1,
            seconds=30,
        )
        assert v2_rig.sink.landed((identity,))[identity] + v2_rig.tenant_sink.landed((identity,))[identity] == 1
        assert v2_rig.tenant_sink.landed((identity,)) == {identity: 0}, "unusable endpoint reached the tenant sink"
    finally:
        v2_rig.proxy.post("/key/delete", {"keys": [created.json()["key"]]})


def test_missing_ingestion_endpoint_fails_loudly_at_boot(
    provider: Wire, operator_sink: Collector, tenant_sink: Collector, tmp_path_factory: pytest.TempPathFactory
) -> None:
    factory: Final = RigFactory(
        provider, operator_sink, tenant_sink, tmp_path_factory.mktemp("signoz-noenv"), False, 2, None
    )
    started: Final = factory.start()
    broken: Final = next(started)
    try:
        response: Final = broken.chat(_marker())
        assert response.status_code == 200, response.text
        eventually(
            lambda: broken.process.log.read_text(),
            lambda text: "SIGNOZ_INGESTION_ENDPOINT not found" in text,
            seconds=30,
        )
        assert operator_sink.landed((_body_id(response),)) == {_body_id(response): 0}
    finally:
        with pytest.raises(StopIteration):
            next(started)


def test_empty_ingestion_endpoint_is_treated_as_missing(
    provider: Wire, operator_sink: Collector, tenant_sink: Collector, tmp_path_factory: pytest.TempPathFactory
) -> None:
    factory: Final = RigFactory(
        provider, operator_sink, tenant_sink, tmp_path_factory.mktemp("signoz-empty"), False, 2, ""
    )
    started: Final = factory.start()
    broken: Final = next(started)
    try:
        response: Final = broken.chat(_marker())
        assert response.status_code == 200, response.text
        eventually(
            lambda: broken.process.log.read_text(),
            lambda text: "SIGNOZ_INGESTION_ENDPOINT not found" in text,
            seconds=30,
        )
    finally:
        with pytest.raises(StopIteration):
            next(started)


def _is_event_stream(response: httpx.Response) -> bool:
    return "content-type" in response.headers and response.headers["content-type"].startswith("text/event-stream")


def _chat_id(response: httpx.Response) -> str:
    if not _is_event_stream(response):
        return _body_id(response)
    identities: Final = frozenset(_text_at(event, "id") for event in _sse_events(response.text))
    assert len(identities) == 1, response.text
    return next(iter(identities))


def _responses_id(response: httpx.Response, marker: str) -> str:
    if not _is_event_stream(response):
        return _body_id(response)
    completed: Final = tuple(
        _text_at(event, "response", "id")
        for event in _sse_events(response.text)
        if event.get("type") == "response.completed"
    )
    assert len(completed) == 1 and completed[0].startswith("resp_"), response.text
    return f"resp_{marker}"


def _message_id(response: httpx.Response) -> str:
    if not _is_event_stream(response):
        return _body_id(response)
    starts: Final = tuple(
        _text_at(event, "message", "id") for event in _sse_events(response.text) if event.get("type") == "message_start"
    )
    assert len(starts) == 1, response.text
    return starts[0]


def _burst(rig: Rig, count: int) -> tuple[tuple[int, str, str | None], ...]:
    markers: Final = tuple(_marker() for _ in range(count))

    def one(index: int) -> tuple[int, str, str | None]:
        marker: Final = markers[index]
        headers: Final = {"Authorization": f"Bearer {rig.proxy.key}"}
        stream: Final = index % 2 == 0
        path, body, identity_of = (
            ("/v1/chat/completions", {"model": rig.model, "messages": [{"role": "user", "content": marker}]}, _chat_id),
            ("/v1/responses", {"model": rig.model, "input": marker}, partial(_responses_id, marker=marker)),
            (
                "/v1/messages",
                {"model": rig.model, "max_tokens": 16, "messages": [{"role": "user", "content": marker}]},
                _message_id,
            ),
        )[index % 3]
        try:
            response: Final = rig.proxy.client.post(path, json={**body, "stream": stream}, headers=headers)
            response.read()
        except httpx.HTTPError as error:
            return index, marker, repr(error)
        return (index, marker, response.text) if response.status_code != 200 else (index, identity_of(response), None)

    with ThreadPoolExecutor(max_workers=10) as pool:
        return tuple(pool.map(one, range(count)))


def _assert_exactly_once(rig: Rig, identities: Sequence[str]) -> None:
    landed: Final = eventually(
        lambda: rig.sink.landed(identities), lambda seen: all(count >= 1 for count in seen.values()), seconds=80
    )
    assert landed == {_canonical_id(identity): 1 for identity in identities}, landed
    charged: Final = frozenset(identity for identity in identities if not identity.startswith("resp_"))
    spend: Final = eventually(
        lambda: read_rows(
            'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id = ANY(%s::text[])',
            ("{" + ",".join(charged) + "}",),
        ),
        lambda rows: {str(row["request_id"]) for row in rows} >= charged,
        seconds=70,
    )
    assert {str(row["request_id"]) for row in spend} == charged, spend


def test_sink_outage_during_a_mixed_burst_lands_every_response_exactly_once_after_recovery(rig: Rig) -> None:
    rig.sink.outage.set()
    try:
        health_down: Final = rig.proxy.request("GET", "/health/services", params={"service": "signoz"})
        results: Final = _burst(rig, 30)
        assert all(error is None for _, _, error in results), [error for _, _, error in results if error]
        rig.sink.refused_batches()
    finally:
        rig.sink.outage.clear()
    assert health_down.status_code == 200, health_down.text
    _assert_exactly_once(rig, tuple(identity for _, identity, _ in results))


def test_slow_sink_during_a_burst_lands_every_response_exactly_once(rig: Rig) -> None:
    rig.sink.release.clear()
    rig.sink.slow.set()
    try:
        results: Final = _burst(rig, 20)
        assert all(error is None for _, _, error in results), [error for _, _, error in results if error]
        identities: Final = tuple(identity for _, identity, _ in results)
        assert all(count == 0 for count in rig.sink.landed(identities).values()), "sink accepted while held"
    finally:
        rig.sink.slow.clear()
        rig.sink.release.set()
    _assert_exactly_once(rig, identities)


def test_killing_one_of_two_workers_mid_burst_keeps_serving_and_never_duplicates_a_span(rig: Rig) -> None:
    root: Final = psutil.Process(rig.process.process.pid)
    workers: Final = eventually(
        lambda: tuple(child for child in root.children() if "resource_tracker" not in " ".join(child.cmdline())),
        lambda found: len(found) == 2,
        seconds=30,
    )
    markers: Final = tuple(_marker() for _ in range(24))

    def one(index: int) -> tuple[str, str | None]:
        if index == 8:
            os.kill(workers[0].pid, signal.SIGKILL)
        try:
            response: Final = rig.chat(markers[index])
            return f"chatcmpl-{markers[index]}", None if response.status_code == 200 else response.text
        except httpx.HTTPError as error:
            return f"chatcmpl-{markers[index]}", repr(error)

    with ThreadPoolExecutor(max_workers=6) as pool:
        results: Final = tuple(pool.map(one, range(24)))
    assert rig.process.process.poll() is None, "Proxy root exited after a worker was killed"
    after: Final = rig.chat(_marker())
    assert after.status_code == 200, after.text
    rig.sink.single_span(_body_id(after))
    failures: Final = tuple(error for _, error in results if error)
    assert all(error.startswith(("ReadError(", "RemoteProtocolError(", "ConnectError(")) for error in failures), (
        failures
    )
    assert len(failures) <= 6, failures
    served: Final = tuple(identity for identity, error in results if error is None)
    assert len(served) >= 18, results
    settled: Final = tuple(identity for index, (identity, error) in enumerate(results) if index > 14 and not error)
    _assert_exactly_once(rig, settled)
    assert all(count <= 1 for count in rig.sink.landed(served).values()), rig.sink.landed(served)


def test_terminating_the_proxy_right_after_a_burst_flushes_every_span_before_exit(
    provider: Wire, operator_sink: Collector, tenant_sink: Collector, tmp_path_factory: pytest.TempPathFactory
) -> None:
    factory: Final = RigFactory(
        provider,
        operator_sink,
        tenant_sink,
        tmp_path_factory.mktemp("signoz-shutdown"),
        False,
        2,
        operator_sink.wire.url,
    )
    started: Final = factory.start()
    rig: Final = next(started)
    responses: Final = tuple(rig.chat(_marker()) for _ in range(10))
    assert all(response.status_code == 200 for response in responses), [response.text for response in responses]
    identities: Final = tuple(_body_id(response) for response in responses)
    drained: Final = eventually(
        lambda: rig.sink.landed(identities), lambda seen: all(count >= 1 for count in seen.values()), seconds=60
    )
    assert drained == {identity: 1 for identity in identities}, drained
    rig.process.process.terminate()
    assert rig.process.process.wait(timeout=40) in (0, -signal.SIGTERM)
    assert rig.sink.landed(identities) == drained
    with pytest.raises(httpx.ConnectError):
        next(started)
