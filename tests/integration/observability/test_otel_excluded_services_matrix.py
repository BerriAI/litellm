import asyncio
import json
import os
import re
import signal
import uuid
from collections.abc import Callable, Generator, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, Protocol, cast

import anthropic
import httpx
import openai
import psutil
import pytest
import yaml
from integration._support.client import Gateway, Scenario, eventually, gateway_from_environment, object_value
from integration._support.otlp_sink import Span, SpanSinks, configure_sink, recorded_spans, spans_for_trace
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

MARKER: Final = re.compile(rb"excl-[0-9a-f]{32}")
FAILING: Final = re.compile(rb"excl-fail-[0-9a-f]{32}")
JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
UNCONFIGURED_VARIANT: Final[TypeAdapter[Literal["null_block", "bare_env"]]] = TypeAdapter(
    Literal["null_block", "bare_env"]
)
REPLY_TEXT: Final = "excluded ok"
SERVER: Final = 2
INVALID_NAME_LOG: Final = "is not a datastore service"
INVALID_VALUE_LOG: Final = "excluded_services must be"
Endpoint = Literal["chat", "responses", "messages"]
Client = Literal["raw", "sdk", "async_sdk"]
ENDPOINTS: Final[tuple[Endpoint, ...]] = ("chat", "responses", "messages")
CLIENTS: Final[tuple[Client, ...]] = ("raw", "sdk", "async_sdk")
AuditConfigWriter = Callable[[Path, Mapping[str, JsonValue]], Path]


class _FixtureRequestParam(Protocol):
    @property
    def param(self) -> object: ...


def _marker() -> str:
    return "excl-" + uuid.uuid4().hex


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
                        {"index": 0, "message": {"role": "assistant", "content": REPLY_TEXT}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
                }
            ).encode()
        )
    chunk: Final = {"id": identity, "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini"}
    first, _, rest = REPLY_TEXT.partition(" ")
    deltas: Final[tuple[dict[str, JsonValue], ...]] = (
        {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": first}}]},
        {**chunk, "choices": [{"index": 0, "delta": {"content": " " + rest}, "finish_reason": "stop"}]},
        {**chunk, "choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}},
    )
    return Reply(
        content_type="text/event-stream",
        chunks=(*(b"data: " + json.dumps(delta).encode() + b"\n\n" for delta in deltas), b"data: [DONE]\n\n"),
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
                "content": [{"type": "output_text", "text": REPLY_TEXT, "annotations": []}],
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
            "delta": REPLY_TEXT,
        },
        {"type": "response.completed", "sequence_number": 2, "response": response},
    )
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events),
    )


def _upstream(request: Request) -> Reply:
    if FAILING.search(request.body) is not None:
        return Reply(status=500, body=b'{"error":{"message":"scripted upstream failure","type":"server_error"}}')
    found: Final = MARKER.search(request.body)
    if found is None:
        return Reply(status=404, body=b'{"error":"no marker"}')
    marker: Final = found.group(0).decode()
    stream: Final = object_value(JSON.validate_json(request.body)).get("stream") is True
    if request.target.endswith("/responses"):
        return _responses_reply(f"resp_{marker}", stream)
    return _chat_reply(f"chatcmpl-{marker}", stream)


def _at(payload: JsonValue, *path: str | int) -> JsonValue:
    if not path:
        return payload
    step: Final = path[0]
    if isinstance(step, int):
        assert isinstance(payload, list), payload
        return _at(payload[step], *path[1:])
    return _at(object_value(payload)[step], *path[1:])


def _sse(body: str) -> tuple[JsonValue, ...]:
    return tuple(
        JSON.validate_json(line[6:])
        for line in body.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    )


def _raw_text(endpoint: Endpoint, stream: bool, body: str) -> str:
    if not stream:
        path: Final[tuple[str | int, ...]] = {
            "chat": ("choices", 0, "message", "content"),
            "responses": ("output", 0, "content", 0, "text"),
            "messages": ("content", 0, "text"),
        }[endpoint]
        return str(_at(JSON.validate_json(body), *path))
    events: Final = _sse(body)
    if endpoint == "chat":
        return "".join(
            str(object_value(_at(event, "choices", 0, "delta")).get("content") or "")
            for event in events
            if _at(event, "choices")
        )
    if endpoint == "responses":
        return "".join(
            str(_at(event, "delta")) for event in events if _at(event, "type") == "response.output_text.delta"
        )
    return "".join(
        str(_at(event, "delta", "text"))
        for event in events
        if _at(event, "type") == "content_block_delta" and _at(event, "delta", "type") == "text_delta"
    )


def _body(model: str, endpoint: Endpoint, marker: str, stream: bool) -> tuple[str, dict[str, JsonValue]]:
    if endpoint == "chat":
        return "/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": marker}],
            "stream": stream,
        }
    if endpoint == "responses":
        return "/v1/responses", {"model": model, "input": marker, "stream": stream}
    return "/v1/messages", {
        "model": model,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": marker}],
        "stream": stream,
    }


@dataclass(frozen=True, slots=True)
class Sent:
    call_id: str
    text: str


@dataclass(frozen=True, slots=True)
class Cursors:
    operator: int
    tenant: int


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    owned: OwnedProxy
    scenario: Scenario
    model: str
    key: str
    upstream: Wire
    sinks: SpanSinks

    def cursors(self) -> Cursors:
        self.upstream.drain()
        return Cursors(recorded_spans(self.sinks.operator)[0], recorded_spans(self.sinks.tenant)[0])

    def upstream_hits(self, marker: str) -> int:
        return sum(1 for request in self.upstream.drain() if marker.encode() in request.body)

    def base_url(self) -> str:
        return str(self.proxy.client.base_url)

    def raw(
        self, endpoint: Endpoint, marker: str, stream: bool, key: str | None = None, trace_id: str | None = None
    ) -> Sent:
        path, body = _body(self.model, endpoint, marker, stream)
        auth: Final = {"Authorization": f"Bearer {key or self.key}"}
        parent: Final = {} if trace_id is None else {"traceparent": f"00-{trace_id}-{uuid.uuid4().hex[:16]}-01"}
        with self.proxy.client.stream("POST", path, json=body, headers={**auth, **parent}) as response:
            text: Final = response.read().decode()
            assert response.status_code == 200, text
            return Sent(response.headers["x-litellm-call-id"], _raw_text(endpoint, stream, text))

    def sdk(self, endpoint: Endpoint, marker: str, stream: bool) -> Sent:
        if endpoint == "messages":
            messages: Final = anthropic.Anthropic(base_url=self.base_url(), api_key=self.key, max_retries=0).messages
            if not stream:
                reply: Final = messages.with_raw_response.create(
                    model=self.model, max_tokens=16, messages=[{"role": "user", "content": marker}]
                )
                block: Final = reply.parse().content[0]
                assert isinstance(block, anthropic.types.TextBlock), block
                return Sent(reply.headers["x-litellm-call-id"], block.text)
            with messages.with_streaming_response.create(
                model=self.model, max_tokens=16, messages=[{"role": "user", "content": marker}], stream=True
            ) as streamed:
                return Sent(
                    streamed.headers["x-litellm-call-id"],
                    "".join(
                        event.delta.text
                        for event in streamed.parse()
                        if event.type == "content_block_delta" and event.delta.type == "text_delta"
                    ),
                )
        client: Final = openai.OpenAI(base_url=self.base_url() + "/v1", api_key=self.key, max_retries=0)
        if endpoint == "chat":
            if not stream:
                completion: Final = client.chat.completions.with_raw_response.create(
                    model=self.model, messages=[{"role": "user", "content": marker}]
                )
                return Sent(
                    completion.headers["x-litellm-call-id"], completion.parse().choices[0].message.content or ""
                )
            with client.chat.completions.with_streaming_response.create(
                model=self.model, messages=[{"role": "user", "content": marker}], stream=True
            ) as chunks:
                return Sent(
                    chunks.headers["x-litellm-call-id"],
                    "".join(chunk.choices[0].delta.content or "" for chunk in chunks.parse() if chunk.choices),
                )
        if not stream:
            created: Final = client.responses.with_raw_response.create(model=self.model, input=marker)
            return Sent(created.headers["x-litellm-call-id"], created.parse().output_text)
        with client.responses.with_streaming_response.create(model=self.model, input=marker, stream=True) as events:
            return Sent(
                events.headers["x-litellm-call-id"],
                "".join(event.delta for event in events.parse() if event.type == "response.output_text.delta"),
            )

    async def async_sdk(self, endpoint: Endpoint, marker: str, stream: bool) -> Sent:
        if endpoint == "messages":
            messages: Final = anthropic.AsyncAnthropic(
                base_url=self.base_url(), api_key=self.key, max_retries=0
            ).messages
            if not stream:
                reply: Final = await messages.with_raw_response.create(
                    model=self.model, max_tokens=16, messages=[{"role": "user", "content": marker}]
                )
                block: Final = reply.parse().content[0]
                assert isinstance(block, anthropic.types.TextBlock), block
                return Sent(reply.headers["x-litellm-call-id"], block.text)
            async with messages.with_streaming_response.create(
                model=self.model, max_tokens=16, messages=[{"role": "user", "content": marker}], stream=True
            ) as streamed:
                pieces: Final = [
                    event.delta.text
                    async for event in await streamed.parse()
                    if event.type == "content_block_delta" and event.delta.type == "text_delta"
                ]
                return Sent(streamed.headers["x-litellm-call-id"], "".join(pieces))
        client: Final = openai.AsyncOpenAI(base_url=self.base_url() + "/v1", api_key=self.key, max_retries=0)
        if endpoint == "chat":
            if not stream:
                completion: Final = await client.chat.completions.with_raw_response.create(
                    model=self.model, messages=[{"role": "user", "content": marker}]
                )
                return Sent(
                    completion.headers["x-litellm-call-id"], completion.parse().choices[0].message.content or ""
                )
            async with client.chat.completions.with_streaming_response.create(
                model=self.model, messages=[{"role": "user", "content": marker}], stream=True
            ) as chunks:
                deltas: Final = [
                    chunk.choices[0].delta.content or "" async for chunk in await chunks.parse() if chunk.choices
                ]
                return Sent(chunks.headers["x-litellm-call-id"], "".join(deltas))
        if not stream:
            created: Final = await client.responses.with_raw_response.create(model=self.model, input=marker)
            return Sent(created.headers["x-litellm-call-id"], created.parse().output_text)
        async with client.responses.with_streaming_response.create(
            model=self.model, input=marker, stream=True
        ) as events:
            texts: Final = [
                event.delta async for event in await events.parse() if event.type == "response.output_text.delta"
            ]
            return Sent(events.headers["x-litellm-call-id"], "".join(texts))

    def send(self, endpoint: Endpoint, client: Client, marker: str, stream: bool) -> Sent:
        if client == "raw":
            return self.raw(endpoint, marker, stream)
        if client == "sdk":
            return self.sdk(endpoint, marker, stream)
        return asyncio.run(self.async_sdk(endpoint, marker, stream))


def _db_systems(spans: tuple[Span, ...]) -> set[str]:
    return {
        str(system)
        for span in spans
        if (system := span["attributes"].get("db.system.name") or span["attributes"].get("db.system")) is not None
    }


def _post_auth_datastore_spans(spans: tuple[Span, ...]) -> tuple[Span, ...]:
    auth_ids: Final = frozenset(span["span_id"] for span in spans if span["name"].startswith("auth "))
    return tuple(span for span in spans if _db_systems((span,)) and span["parent_span_id"] not in auth_ids)


def _names(spans: tuple[Span, ...]) -> list[str]:
    return sorted(span["name"] for span in spans)


def _has_root(spans: tuple[Span, ...]) -> bool:
    return any(span["kind"] == SERVER for span in spans)


def _trace_of_call(sink: str, call_id: str, since: int) -> tuple[Span, ...]:
    _, spans = recorded_spans(sink, since)
    traces: Final = {span["trace_id"] for span in spans if span["attributes"].get("litellm.call_id") == call_id}
    return tuple(span for span in spans if span["trace_id"] in traces)


def _operator_trace(rig: Rig, sent: Sent, cursors: Cursors) -> tuple[Span, ...]:
    trace: Final = eventually(
        lambda: _trace_of_call(rig.sinks.operator, sent.call_id, cursors.operator),
        lambda spans: _has_root(spans) and "redis" in _db_systems(spans),
        seconds=40,
    )
    assert len({span["trace_id"] for span in trace}) == 1, _names(trace)
    return trace


def _traced_raw(rig: Rig, endpoint: Endpoint, marker: str) -> tuple[str, Sent]:
    trace_id: Final = uuid.uuid4().hex
    return trace_id, rig.raw(endpoint, marker, stream=False, trace_id=trace_id)


def _operator_trace_by_id(rig: Rig, trace_id: str, cursors: Cursors) -> tuple[Span, ...]:
    return eventually(
        lambda: spans_for_trace(recorded_spans(rig.sinks.operator, cursors.operator)[1], trace_id),
        _has_root,
        seconds=40,
    )


def _tenant_mirror(rig: Rig, operator: tuple[Span, ...], cursors: Cursors) -> tuple[Span, ...]:
    kept: Final = frozenset(span["name"] for span in operator if not _db_systems((span,)))
    return eventually(
        lambda: spans_for_trace(recorded_spans(rig.sinks.tenant, cursors.tenant)[1], operator[0]["trace_id"]),
        lambda spans: kept <= {span["name"] for span in spans},
        seconds=40,
    )


def _assert_tenant_mirrors(rig: Rig, operator: tuple[Span, ...], cursors: Cursors) -> tuple[Span, ...]:
    tenant: Final = _tenant_mirror(rig, operator, cursors)
    assert _db_systems(tenant) == set(), f"datastore spans reached the tenant: {_names(tenant)}"
    assert sum(1 for span in tenant if span["kind"] == SERVER) == 1, _names(tenant)
    return tenant


def _assert_withheld(rig: Rig, sent: Sent, cursors: Cursors) -> tuple[Span, ...]:
    tenant: Final = _assert_tenant_mirrors(rig, _operator_trace(rig, sent, cursors), cursors)
    assert any("gen_ai.operation.name" in span["attributes"] for span in tenant), _names(tenant)
    return tenant


def _config(directory: Path, otel_audit_config: AuditConfigWriter, otel: Mapping[str, JsonValue], name: str) -> Path:
    written: Final = otel_audit_config(directory, {})
    loaded: Final = object_value(JSON.validate_python(yaml.safe_load(written.read_text())))
    settings: Final = object_value(loaded["callback_settings"])
    config: Final = {**loaded, "callback_settings": {**settings, "otel": {**object_value(settings["otel"]), **otel}}}
    path: Final = directory / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _null_otel_config(directory: Path, otel_audit_config: AuditConfigWriter, name: str) -> Path:
    written: Final = otel_audit_config(directory, {})
    loaded: Final = object_value(JSON.validate_python(yaml.safe_load(written.read_text())))
    config: Final = {
        **loaded,
        "litellm_settings": {**object_value(loaded["litellm_settings"]), "callbacks": ["langfuse_otel"]},
        "callback_settings": {**object_value(loaded["callback_settings"]), "otel": None},
    }
    path: Final = directory / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _operator_langfuse(sinks: SpanSinks) -> dict[str, str]:
    return {
        "LANGFUSE_HOST": sinks.operator,
        "LANGFUSE_PUBLIC_KEY": "pk-lf-operator",
        "LANGFUSE_SECRET_KEY": "sk-lf-operator",
        "OTEL_EXPORTER": "http/json",
        "OTEL_ENDPOINT": sinks.operator,
    }


@contextmanager
def _started(
    provider: Wire,
    sinks: SpanSinks,
    config: Path,
    directory: Path,
    langfuse_vars: Mapping[str, JsonValue],
    workers: int,
    environment: Mapping[str, str] = MappingProxyType({}),
) -> Generator[Rig]:
    with (
        gateway_from_environment() as gateway,
        owned_proxy_process(
            gateway,
            directory,
            {"LITELLM_OTEL_V2": "1", "OTEL_BSP_SCHEDULE_DELAY": "300", **environment},
            config=config,
            remove_environment=("LITELLM_OTEL_EXCLUDED_SERVICES",),
            workers=workers,
        ) as owned,
        owned.gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=provider.url + "/v1")
        team: Final = scenario.team()
        attached: Final = owned.gateway.request(
            "POST", f"/team/{team}/callback", {"callback_name": "langfuse_otel", "callback_vars": dict(langfuse_vars)}
        )
        assert attached.status_code == 200, attached.text
        yield Rig(owned.gateway, owned, scenario, model, scenario.key(team_id=team), provider, sinks)


@pytest.fixture(scope="module")
def provider() -> Iterator[Wire]:
    with wire_server(_upstream) as wire:
        yield wire


@pytest.fixture(scope="module")
def rig(
    provider: Wire,
    audit_sinks: SpanSinks,
    otel_audit_config: AuditConfigWriter,
    langfuse_vars: dict[str, JsonValue],
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("excluded-matrix")
    config: Final = _config(directory, otel_audit_config, {"excluded_services": ["redis", "postgres"]}, "matrix")
    with _started(provider, audit_sinks, config, directory, langfuse_vars, workers=2) as started:
        yield started


@pytest.fixture(scope="module", params=["null_block", "bare_env"], ids=["null_block", "bare_env"])
def unconfigured_rig(
    request: pytest.FixtureRequest,
    provider: Wire,
    audit_sinks: SpanSinks,
    otel_audit_config: AuditConfigWriter,
    langfuse_vars: dict[str, JsonValue],
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Rig]:
    parameter: Final = cast(_FixtureRequestParam, request).param
    variant: Final = UNCONFIGURED_VARIANT.validate_python(parameter)
    directory: Final = tmp_path_factory.mktemp(f"excluded-{variant}")
    config: Final = (
        _null_otel_config(directory, otel_audit_config, variant)
        if variant == "null_block"
        else _config(directory, otel_audit_config, {}, variant)
    )
    environment: Final = (
        _operator_langfuse(audit_sinks) if variant == "null_block" else {"EXCLUDED_SERVICES": "redis,postgres"}
    )
    with _started(
        provider, audit_sinks, config, directory, langfuse_vars, workers=2, environment=environment
    ) as started:
        yield started


@pytest.mark.timeout(120)
@pytest.mark.parametrize("stream", [False, True], ids=["unary", "stream"])
@pytest.mark.parametrize("client", CLIENTS)
@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_tenant_trace_keeps_request_spans_without_datastore_spans(
    rig: Rig, endpoint: Endpoint, client: Client, stream: bool
) -> None:
    cursors: Final = rig.cursors()
    marker: Final = _marker()
    sent: Final = rig.send(endpoint, client, marker, stream)
    assert sent.text == REPLY_TEXT, sent
    assert rig.upstream_hits(marker) == 1
    _assert_withheld(rig, sent, cursors)


@pytest.mark.timeout(120)
@pytest.mark.parametrize("endpoint", ["chat", "messages"])
def test_cache_hit_twin_keeps_datastore_spans_off_the_tenant(rig: Rig, endpoint: Endpoint) -> None:
    marker: Final = _marker()
    first: Final = rig.raw(endpoint, marker, stream=False)
    assert first.text == REPLY_TEXT, first
    assert rig.upstream_hits(marker) == 1
    cursors: Final = rig.cursors()
    trace_id, hit = eventually(
        lambda: _traced_raw(rig, endpoint, marker), lambda sent: rig.upstream_hits(marker) == 0, seconds=20
    )
    assert hit.text == REPLY_TEXT, hit
    _assert_tenant_mirrors(rig, _operator_trace_by_id(rig, trace_id, cursors), cursors)


@pytest.mark.timeout(120)
@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_failed_upstream_call_keeps_datastore_spans_off_the_tenant(rig: Rig, endpoint: Endpoint) -> None:
    cursors: Final = rig.cursors()
    marker: Final = "excl-fail-" + uuid.uuid4().hex
    trace_id: Final = uuid.uuid4().hex
    path, body = _body(rig.model, endpoint, marker, stream=False)
    failed: Final = rig.proxy.client.post(
        path,
        json=body,
        headers={"Authorization": f"Bearer {rig.key}", "traceparent": f"00-{trace_id}-{uuid.uuid4().hex[:16]}-01"},
    )
    assert failed.status_code == 500, failed.text
    assert rig.upstream_hits(marker) >= 1
    operator: Final = eventually(
        lambda: spans_for_trace(recorded_spans(rig.sinks.operator, cursors.operator)[1], trace_id),
        lambda spans: _has_root(spans) and "redis" in _db_systems(spans),
        seconds=40,
    )
    _assert_tenant_mirrors(rig, operator, cursors)


@pytest.mark.timeout(120)
def test_key_level_callback_vars_destination_is_filtered_too(rig: Rig, langfuse_vars: dict[str, JsonValue]) -> None:
    key: Final = rig.scenario.key(
        metadata={
            "logging": [
                {"callback_name": "langfuse_otel", "callback_type": "success", "callback_vars": dict(langfuse_vars)}
            ]
        }
    )
    cursors: Final = rig.cursors()
    marker: Final = _marker()
    sent: Final = rig.raw("chat", marker, stream=False, key=key)
    assert sent.text == REPLY_TEXT, sent
    assert rig.upstream_hits(marker) == 1
    _assert_withheld(rig, sent, cursors)


@pytest.mark.timeout(120)
@pytest.mark.parametrize("status", [403, 404])
def test_rejecting_tenant_destination_leaves_serving_and_the_operator_trace_intact(rig: Rig, status: int) -> None:
    configure_sink(rig.sinks.tenant, status=status)
    try:
        cursors: Final = rig.cursors()
        marker: Final = _marker()
        sent: Final = rig.raw("chat", marker, stream=True)
        assert sent.text == REPLY_TEXT, sent
        assert rig.upstream_hits(marker) == 1
        _assert_withheld(rig, sent, cursors)
    finally:
        configure_sink(rig.sinks.tenant, status=200)
    after: Final = rig.cursors()
    _assert_withheld(rig, rig.raw("responses", _marker(), stream=False), after)


def _burst(rig: Rig, count: int) -> tuple[Sent | str, ...]:
    def one(index: int) -> Sent | str:
        try:
            return rig.raw(ENDPOINTS[index % 3], _marker(), stream=index % 2 == 0)
        except (httpx.HTTPError, AssertionError) as error:
            return repr(error)

    with ThreadPoolExecutor(max_workers=10) as pool:
        return tuple(pool.map(one, range(count)))


def _served(results: tuple[Sent | str, ...]) -> tuple[Sent, ...]:
    return tuple(result for result in results if isinstance(result, Sent))


def _assert_operator_exactly_once(rig: Rig, served: tuple[Sent, ...], cursors: Cursors) -> set[str]:
    wanted: Final = {sent.call_id for sent in served}

    def roots() -> dict[str, int]:
        _, spans = recorded_spans(rig.sinks.operator, cursors.operator)
        traced: Final = {
            span["trace_id"]: str(span["attributes"]["litellm.call_id"])
            for span in spans
            if span["attributes"].get("litellm.call_id") in wanted
        }
        counts: Final = {call: 0 for call in wanted}
        for span in spans:
            if span["kind"] == SERVER and span["trace_id"] in traced:
                counts[traced[span["trace_id"]]] += 1
        return counts

    landed: Final = eventually(roots, lambda counts: all(count >= 1 for count in counts.values()), seconds=90)
    assert landed == {call: 1 for call in wanted}, landed
    _, spans = recorded_spans(rig.sinks.operator, cursors.operator)
    return {span["trace_id"] for span in spans if span["attributes"].get("litellm.call_id") in wanted}


def _assert_tenant_never_saw_datastore_spans(rig: Rig, cursors: Cursors, traces: set[str]) -> None:
    tenant: Final = eventually(
        lambda: recorded_spans(rig.sinks.tenant, cursors.tenant)[1],
        lambda spans: traces <= {span["trace_id"] for span in spans if span["kind"] == SERVER},
        seconds=90,
    )
    assert _db_systems(tenant) == set(), _names(tenant)


@pytest.mark.timeout(300)
def test_tenant_outage_during_a_mixed_burst_keeps_serving_and_never_leaks_datastore_spans(rig: Rig) -> None:
    cursors: Final = rig.cursors()
    configure_sink(rig.sinks.tenant, status=503)
    try:
        results: Final = _burst(rig, 30)
    finally:
        configure_sink(rig.sinks.tenant, status=200)
    served: Final = _served(results)
    assert len(served) == 30, [result for result in results if isinstance(result, str)]
    assert all(sent.text == REPLY_TEXT for sent in served), served
    traces: Final = _assert_operator_exactly_once(rig, served, cursors)
    _assert_tenant_never_saw_datastore_spans(rig, cursors, traces)
    after: Final = rig.cursors()
    _assert_withheld(rig, rig.raw("messages", _marker(), stream=True), after)


@pytest.mark.timeout(300)
def test_stalled_tenant_destination_during_a_burst_does_not_block_responses(rig: Rig) -> None:
    cursors: Final = rig.cursors()
    configure_sink(rig.sinks.tenant, paused=True)
    try:
        results: Final = _burst(rig, 20)
    finally:
        configure_sink(rig.sinks.tenant, paused=False)
    served: Final = _served(results)
    assert len(served) == 20, [result for result in results if isinstance(result, str)]
    traces: Final = _assert_operator_exactly_once(rig, served, cursors)
    _assert_tenant_never_saw_datastore_spans(rig, cursors, traces)


@pytest.mark.timeout(300)
def test_killing_one_of_two_workers_mid_burst_keeps_the_filter_on_the_survivor(rig: Rig) -> None:
    root: Final = psutil.Process(rig.owned.process.pid)
    workers: Final = eventually(
        lambda: tuple(child for child in root.children() if "resource_tracker" not in " ".join(child.cmdline())),
        lambda found: len(found) == 2,
        seconds=30,
    )
    cursors: Final = rig.cursors()

    def one(index: int) -> Sent | str:
        if index == 6:
            os.kill(workers[0].pid, signal.SIGKILL)
        try:
            return rig.raw("chat", _marker(), stream=index % 2 == 0)
        except (httpx.HTTPError, AssertionError) as error:
            return repr(error)

    with ThreadPoolExecutor(max_workers=6) as pool:
        results: Final = tuple(pool.map(one, range(18)))
    assert rig.owned.process.poll() is None, "Proxy root exited after a worker was killed"
    failures: Final = tuple(result for result in results if isinstance(result, str))
    assert all(failure.startswith(("ReadError(", "RemoteProtocolError(", "ConnectError(")) for failure in failures), (
        failures
    )
    assert len(failures) <= 6, failures
    settled: Final = tuple(result for index, result in enumerate(results) if index > 12 and isinstance(result, Sent))
    traces: Final = _assert_operator_exactly_once(rig, settled, cursors)
    _assert_tenant_never_saw_datastore_spans(rig, cursors, traces)
    after: Final = rig.cursors()
    _assert_withheld(rig, rig.raw("chat", _marker(), stream=False), after)


def _assert_tenant_kept(
    rig: Rig, trace_id: str, cursors: Cursors, *, needs_model_span: bool = False
) -> tuple[Span, ...]:
    def ready(spans: tuple[Span, ...]) -> bool:
        return (
            sum(1 for span in spans if span["kind"] == SERVER) == 1
            and "redis" in _db_systems(spans)
            and (not needs_model_span or any("gen_ai.operation.name" in span["attributes"] for span in spans))
        )

    tenant: Final = eventually(
        lambda: spans_for_trace(recorded_spans(rig.sinks.tenant, cursors.tenant)[1], trace_id),
        ready,
        seconds=40,
        return_last_on_timeout=True,
    )
    assert sum(1 for span in tenant if span["kind"] == SERVER) == 1, _names(tenant)
    assert "redis" in _db_systems(tenant), f"redis spans missing at the tenant: {_names(tenant)}"
    assert not needs_model_span or any("gen_ai.operation.name" in span["attributes"] for span in tenant), _names(tenant)
    return tenant


def _assert_kept(rig: Rig, sent: Sent, cursors: Cursors) -> tuple[Span, ...]:
    operator: Final = _operator_trace(rig, sent, cursors)
    return _assert_tenant_kept(rig, operator[0]["trace_id"], cursors, needs_model_span=True)


@pytest.mark.timeout(120)
@pytest.mark.parametrize("stream", [False, True], ids=["unary", "stream"])
@pytest.mark.parametrize("client", CLIENTS)
@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_unconfigured_tenant_trace_keeps_datastore_spans(
    unconfigured_rig: Rig, endpoint: Endpoint, client: Client, stream: bool
) -> None:
    cursors: Final = unconfigured_rig.cursors()
    marker: Final = _marker()
    sent: Final = unconfigured_rig.send(endpoint, client, marker, stream)
    assert sent.text == REPLY_TEXT, sent
    assert unconfigured_rig.upstream_hits(marker) == 1
    _assert_kept(unconfigured_rig, sent, cursors)


@pytest.mark.timeout(120)
@pytest.mark.parametrize("endpoint", ["chat", "messages"])
def test_unconfigured_cache_hit_twin_keeps_datastore_spans(unconfigured_rig: Rig, endpoint: Endpoint) -> None:
    cursors: Final = unconfigured_rig.cursors()
    marker: Final = _marker()
    first_result: Final = _traced_raw(unconfigured_rig, endpoint, marker)
    first: Final = first_result[1]
    assert first.text == REPLY_TEXT, first
    assert unconfigured_rig.upstream_hits(marker) == 1
    _assert_kept(unconfigured_rig, first, cursors)
    hit_cursors: Final = unconfigured_rig.cursors()

    def read_hit() -> tuple[str, Sent, tuple[Span, ...]]:
        trace_id, sent = _traced_raw(unconfigured_rig, endpoint, marker)
        return trace_id, sent, _operator_trace_by_id(unconfigured_rig, trace_id, hit_cursors)

    trace_id, hit, operator = eventually(
        read_hit,
        lambda result: (
            unconfigured_rig.upstream_hits(marker) == 0
            and "redis" in _db_systems(_post_auth_datastore_spans(result[2]))
        ),
        seconds=60,
    )
    assert hit.text == REPLY_TEXT, hit
    post_auth_datastore: Final = _post_auth_datastore_spans(operator)
    post_auth_span_ids: Final = frozenset(span["span_id"] for span in post_auth_datastore)
    non_datastore_names: Final = frozenset(span["name"] for span in operator if not _db_systems((span,)))
    tenant: Final = eventually(
        lambda: spans_for_trace(recorded_spans(unconfigured_rig.sinks.tenant, hit_cursors.tenant)[1], trace_id),
        lambda spans: (
            non_datastore_names <= frozenset(span["name"] for span in spans)
            and post_auth_span_ids <= frozenset(span["span_id"] for span in spans)
        ),
        seconds=40,
        return_last_on_timeout=True,
    )
    tenant_span_ids: Final = frozenset(span["span_id"] for span in tenant)
    missing_post_auth_names: Final = tuple(
        span["name"] for span in post_auth_datastore if span["span_id"] not in tenant_span_ids
    )
    assert sum(1 for span in tenant if span["kind"] == SERVER) == 1, _names(tenant)
    assert "redis" in _db_systems(tenant), (
        f"operator datastore systems={sorted(_db_systems(post_auth_datastore))}; "
        f"tenant datastore systems={sorted(_db_systems(tenant))}; tenant spans={_names(tenant)}"
    )
    assert not missing_post_auth_names, (
        f"missing post-auth datastore span names={missing_post_auth_names}; "
        f"operator={_names(post_auth_datastore)}; tenant={_names(tenant)}"
    )


@pytest.mark.timeout(120)
@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_unconfigured_failed_upstream_keeps_datastore_spans(unconfigured_rig: Rig, endpoint: Endpoint) -> None:
    cursors: Final = unconfigured_rig.cursors()
    marker: Final = "excl-fail-" + uuid.uuid4().hex
    trace_id: Final = uuid.uuid4().hex
    path, body = _body(unconfigured_rig.model, endpoint, marker, stream=False)
    failed: Final = unconfigured_rig.proxy.client.post(
        path,
        json=body,
        headers={
            "Authorization": f"Bearer {unconfigured_rig.key}",
            "traceparent": f"00-{trace_id}-{uuid.uuid4().hex[:16]}-01",
        },
    )
    assert failed.status_code == 500, failed.text
    assert unconfigured_rig.upstream_hits(marker) >= 1
    operator: Final = eventually(
        lambda: spans_for_trace(recorded_spans(unconfigured_rig.sinks.operator, cursors.operator)[1], trace_id),
        lambda spans: _has_root(spans) and "redis" in _db_systems(spans),
        seconds=40,
    )
    assert "redis" in _db_systems(operator), _names(operator)
    _assert_tenant_kept(unconfigured_rig, trace_id, cursors)


@pytest.mark.timeout(120)
@pytest.mark.parametrize("status", [403, 404])
def test_unconfigured_rejecting_tenant_destination_recovers(unconfigured_rig: Rig, status: int) -> None:
    configure_sink(unconfigured_rig.sinks.tenant, status=status)
    try:
        cursors: Final = unconfigured_rig.cursors()
        marker: Final = _marker()
        sent: Final = unconfigured_rig.raw("chat", marker, stream=True)
        assert sent.text == REPLY_TEXT, sent
        assert unconfigured_rig.upstream_hits(marker) == 1
        _operator_trace(unconfigured_rig, sent, cursors)
    finally:
        configure_sink(unconfigured_rig.sinks.tenant, status=200)
    after: Final = unconfigured_rig.cursors()
    recovered: Final = unconfigured_rig.raw("responses", _marker(), stream=False)
    assert recovered.text == REPLY_TEXT, recovered
    _assert_kept(unconfigured_rig, recovered, after)


@pytest.mark.timeout(120)
def test_unconfigured_key_level_destination_keeps_datastore_spans(
    unconfigured_rig: Rig, langfuse_vars: dict[str, JsonValue]
) -> None:
    key: Final = unconfigured_rig.scenario.key(
        metadata={
            "logging": [
                {"callback_name": "langfuse_otel", "callback_type": "success", "callback_vars": dict(langfuse_vars)}
            ]
        }
    )
    cursors: Final = unconfigured_rig.cursors()
    marker: Final = _marker()
    sent: Final = unconfigured_rig.raw("chat", marker, stream=False, key=key)
    assert sent.text == REPLY_TEXT, sent
    assert unconfigured_rig.upstream_hits(marker) == 1
    _assert_kept(unconfigured_rig, sent, cursors)


def _assert_tenant_kept_the_burst(rig: Rig, cursors: Cursors, traces: set[str]) -> None:
    def ready(spans: tuple[Span, ...]) -> bool:
        def trace_kept(trace: str) -> bool:
            trace_spans: Final = spans_for_trace(spans, trace)
            return any(span["kind"] == SERVER for span in trace_spans) and "redis" in _db_systems(trace_spans)

        return all(trace_kept(trace) for trace in traces)

    tenant: Final = eventually(
        lambda: recorded_spans(rig.sinks.tenant, cursors.tenant)[1],
        ready,
        seconds=90,
        return_last_on_timeout=True,
    )
    burst: Final = tuple(span for span in tenant if span["trace_id"] in traces)
    missing_roots: Final = tuple(
        trace for trace in traces if not any(span["kind"] == SERVER for span in spans_for_trace(burst, trace))
    )
    missing_redis: Final = tuple(trace for trace in traces if "redis" not in _db_systems(spans_for_trace(burst, trace)))
    assert not missing_roots, f"SERVER root missing from tenant burst traces: {missing_roots}, {_names(burst)}"
    assert not missing_redis, f"redis spans missing from tenant burst traces: {missing_redis}, {_names(burst)}"


@pytest.mark.timeout(300)
def test_unconfigured_tenant_outage_during_a_mixed_burst(unconfigured_rig: Rig) -> None:
    cursors: Final = unconfigured_rig.cursors()
    configure_sink(unconfigured_rig.sinks.tenant, status=503)
    try:
        results: Final = _burst(unconfigured_rig, 30)
    finally:
        configure_sink(unconfigured_rig.sinks.tenant, status=200)
    served: Final = _served(results)
    assert len(served) == 30, [result for result in results if isinstance(result, str)]
    assert all(sent.text == REPLY_TEXT for sent in served), served
    traces: Final = _assert_operator_exactly_once(unconfigured_rig, served, cursors)
    _assert_tenant_kept_the_burst(unconfigured_rig, cursors, traces)
    after: Final = unconfigured_rig.cursors()
    _assert_kept(unconfigured_rig, unconfigured_rig.raw("messages", _marker(), stream=True), after)


@pytest.mark.timeout(300)
def test_unconfigured_killing_one_of_two_workers_keeps_the_fan_out(unconfigured_rig: Rig) -> None:
    root: Final = psutil.Process(unconfigured_rig.owned.process.pid)
    workers: Final = eventually(
        lambda: tuple(child for child in root.children() if "resource_tracker" not in " ".join(child.cmdline())),
        lambda found: len(found) == 2,
        seconds=30,
    )
    cursors: Final = unconfigured_rig.cursors()

    def one(index: int) -> Sent | str:
        if index == 6:
            os.kill(workers[0].pid, signal.SIGKILL)
        try:
            return unconfigured_rig.raw("chat", _marker(), stream=index % 2 == 0)
        except (httpx.HTTPError, AssertionError) as error:
            return repr(error)

    with ThreadPoolExecutor(max_workers=6) as pool:
        results: Final = tuple(pool.map(one, range(18)))
    assert unconfigured_rig.owned.process.poll() is None, "Proxy root exited after a worker was killed"
    failures: Final = tuple(result for result in results if isinstance(result, str))
    assert all(failure.startswith(("ReadError(", "RemoteProtocolError(", "ConnectError(")) for failure in failures), (
        failures
    )
    assert len(failures) <= 6, failures
    settled: Final = tuple(result for index, result in enumerate(results) if index > 12 and isinstance(result, Sent))
    traces: Final = _assert_operator_exactly_once(unconfigured_rig, settled, cursors)
    _assert_tenant_kept_the_burst(unconfigured_rig, cursors, traces)
    after: Final = unconfigured_rig.cursors()
    _assert_kept(unconfigured_rig, unconfigured_rig.raw("chat", _marker(), stream=False), after)


@dataclass(frozen=True, slots=True)
class Setting:
    otel: Mapping[str, JsonValue]
    withholds_redis: bool
    logs: str | None


SETTINGS: Final[dict[str, Setting]] = {
    "missing": Setting({}, False, None),
    "null": Setting({"excluded_services": None}, False, None),
    "empty_list": Setting({"excluded_services": []}, False, None),
    "empty_string": Setting({"excluded_services": ""}, False, None),
    "yaml_string": Setting({"excluded_services": "redis"}, True, None),
    "duplicates": Setting({"excluded_services": ["redis", "redis"]}, True, None),
    "case_and_space": Setting({"excluded_services": ["REDIS", " Postgres "]}, True, None),
    "integer": Setting({"excluded_services": 7}, False, INVALID_VALUE_LOG),
    "mapping": Setting({"excluded_services": {"redis": True}}, False, INVALID_VALUE_LOG),
    "non_string_item": Setting({"excluded_services": [7, "redis"]}, True, INVALID_VALUE_LOG),
    "oversized_name": Setting({"excluded_services": "x" * 5000}, False, INVALID_NAME_LOG),
}


@pytest.mark.timeout(180)
@pytest.mark.parametrize("name", SETTINGS)
def test_excluded_services_setting_shapes_boot_and_resolve(
    name: str,
    provider: Wire,
    audit_sinks: SpanSinks,
    otel_audit_config: AuditConfigWriter,
    langfuse_vars: dict[str, JsonValue],
    tmp_path: Path,
) -> None:
    setting: Final = SETTINGS[name]
    config: Final = _config(tmp_path, otel_audit_config, setting.otel, name)
    with _started(provider, audit_sinks, config, tmp_path, langfuse_vars, workers=1) as started:
        cursors: Final = started.cursors()
        marker: Final = _marker()
        sent: Final = started.raw("chat", marker, stream=False)
        assert sent.text == REPLY_TEXT, sent
        assert started.upstream_hits(marker) == 1
        operator: Final = _operator_trace(started, sent, cursors)
        tenant: Final = _tenant_mirror(started, operator, cursors)
        if setting.withholds_redis:
            assert "redis" not in _db_systems(tenant), _names(tenant)
        else:
            eventually(
                lambda: _db_systems(
                    spans_for_trace(recorded_spans(started.sinks.tenant, cursors.tenant)[1], tenant[0]["trace_id"])
                ),
                lambda systems: "redis" in systems,
                seconds=30,
            )
        log: Final = started.owned.log.read_text()
        if setting.logs is None:
            assert INVALID_NAME_LOG not in log and INVALID_VALUE_LOG not in log, log[-2000:]
        else:
            assert setting.logs in log, log[-4000:]
