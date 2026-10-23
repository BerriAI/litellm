import asyncio
import base64
import json
import re
import threading
import uuid
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import anthropic
import httpx
import openai
import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue
from pydantic import JsonValue, TypeAdapter

MARKER: Final = re.compile(rb"nops-[0-9a-f]{32}")
JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
RESPONSE_ID: Final = "gen_ai.response.id"
REPLY_TEXT: Final = "nops upstream reply text"
OPERATOR_KEY: Final = "nops-operator-" + uuid.uuid4().hex


def _marker() -> str:
    return "nops-" + uuid.uuid4().hex


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
    return Reply(
        content_type="text/event-stream",
        chunks=(
            b"data: "
            + json.dumps(
                {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": REPLY_TEXT}}]}
            ).encode()
            + b"\n\n",
            b"data: "
            + json.dumps({**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}).encode()
            + b"\n\n",
            b"data: "
            + json.dumps(
                {**chunk, "choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}}
            ).encode()
            + b"\n\n",
            b"data: [DONE]\n\n",
        ),
    )


def _responses_reply(identity: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
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
        ).encode()
    )


def _upstream(request: Request) -> Reply:
    found: Final = MARKER.search(request.body)
    if found is None:
        return Reply(status=404, body=b'{"error":"no marker"}')
    marker: Final = found.group(0).decode()
    if request.target.endswith("/responses"):
        return _responses_reply(f"resp_{marker}")
    stream: Final = object_value(JSON.validate_json(request.body)).get("stream") is True
    return _chat_reply(f"chatcmpl-{marker}", stream)


def _decoded_responses_id(identity: str) -> str:
    try:
        return base64.b64decode(identity.removeprefix("resp_").encode()).decode()
    except (ValueError, UnicodeDecodeError):
        return identity


def _canonical_id(identity: str) -> str:
    return _decoded_responses_id(identity).rpartition("response_id:")[2]


def _text_at(payload: JsonValue, *path: str) -> str:
    if not path:
        return string_value(payload)
    return _text_at(object_value(payload)[path[0]], *path[1:])


def _body_id(response: httpx.Response) -> str:
    return _text_at(JSON.validate_json(response.content), "id")


@dataclass(frozen=True, slots=True)
class Span:
    target: str
    authorization: str | None
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
    rejection: threading.Event
    missing: threading.Event
    accepted: Sequence[Request]
    refused: Sequence[Request]
    guard: threading.Lock

    def refused_batch_carrying(self, response_id: str) -> Request:
        def carrying() -> tuple[Request, ...]:
            with self.guard:
                return tuple(batch for batch in self.refused if response_id.encode() in batch.body)

        return eventually(carrying, lambda found: len(found) >= 1, seconds=30)[0]

    def bodies(self) -> tuple[bytes, ...]:
        with self.guard:
            return tuple(batch.body for batch in self.accepted)

    def spans(self) -> tuple[Span, ...]:
        with self.guard:
            batches: Final = tuple(batch for batch in self.accepted if batch.target.endswith("/v1/traces"))
        return tuple(
            Span(
                batch.target,
                batch.headers.get("authorization"),
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

    def single_span(self, response_id: str) -> Span:
        found: Final = eventually(
            lambda: self.spans_for(response_id), lambda spans: len(spans) == 1, seconds=30, return_last_on_timeout=True
        )
        assert len(found) == 1, f"{len(found)} spans for {response_id} at the sink"
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


@pytest.fixture(scope="session")
def sink() -> Iterator[Collector]:
    rejection: Final = threading.Event()
    missing: Final = threading.Event()
    accepted: Final[deque[Request]] = deque()  # mutable-ok: the sink thread records each accepted batch as it arrives
    refused: Final[deque[Request]] = deque()  # mutable-ok: the sink thread records each refused batch as it arrives
    guard: Final = threading.Lock()

    def refuse(request: Request, status: int, body: bytes) -> Reply:
        with guard:
            refused.append(request)
        return Reply(status=status, body=body)

    def receive(request: Request) -> Reply:
        if rejection.is_set():
            return refuse(request, 403, b'{"error":"forbidden"}')
        if missing.is_set():
            return refuse(request, 404, b'{"error":"not found"}')
        with guard:
            accepted.append(request)
        return Reply()

    with wire_server(receive) as wire:
        yield Collector(wire, rejection, missing, accepted, refused, guard)


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

    def openai_client(self) -> openai.OpenAI:
        return openai.OpenAI(base_url=str(self.proxy.client.base_url) + "/v1", api_key=self.proxy.key, max_retries=0)

    def async_openai_client(self) -> openai.AsyncOpenAI:
        return openai.AsyncOpenAI(
            base_url=str(self.proxy.client.base_url) + "/v1", api_key=self.proxy.key, max_retries=0
        )

    def anthropic_client(self) -> anthropic.Anthropic:
        return anthropic.Anthropic(base_url=str(self.proxy.client.base_url), api_key=self.proxy.key, max_retries=0)

    def chat(self, content: str) -> httpx.Response:
        return self.proxy.request(
            "POST",
            "/v1/chat/completions",
            {"model": self.model, "messages": [{"role": "user", "content": content}], "cache": {"no-cache": True}},
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


@dataclass(frozen=True, slots=True)
class RigFactory:
    provider: Wire
    sink: Collector
    directory: Path
    otel_v2: bool
    endpoint: str
    api_key: str | None = OPERATOR_KEY

    def config_path(self) -> Path:
        loaded: Final = object_value(
            JSON.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
        )
        config: Final = {
            **loaded,
            "litellm_settings": {**object_value(loaded["litellm_settings"]), "callbacks": ["nops"]},
        }
        path: Final = self.directory / f"nops-{uuid.uuid4().hex}.yaml"
        path.write_text(yaml.safe_dump(config))
        return path

    def overrides(self) -> dict[str, str]:
        return {
            "LITELLM_OTEL_V2": "1" if self.otel_v2 else "0",
            "OTEL_BSP_SCHEDULE_DELAY": "300",
            "NOPS_OTEL_ENDPOINT": self.endpoint,
            **({"NOPS_API_KEY": self.api_key} if self.api_key is not None else {}),
        }

    def start(self) -> Iterator[Rig]:
        with (
            gateway_from_environment() as gateway,
            owned_proxy_process(
                gateway,
                self.directory,
                self.overrides(),
                config=self.config_path(),
                remove_environment=(
                    "NOPS_API_KEY",
                    "NOPS_OTEL_ENDPOINT",
                    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
                ),
                workers=2,
            ) as owned,
            owned.gateway.scenario() as scenario,
        ):
            model: Final = scenario.model(api_base=self.provider.url + "/v1")
            yield Rig(owned.gateway, owned, model, self.provider, self.sink)


@pytest.fixture(scope="session")
def rig(provider: Wire, sink: Collector, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    yield from RigFactory(provider, sink, tmp_path_factory.mktemp("nops"), False, sink.wire.url).start()


@pytest.fixture(scope="session")
def v2_rig(provider: Wire, sink: Collector, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    yield from RigFactory(provider, sink, tmp_path_factory.mktemp("nops-v2"), True, sink.wire.url).start()


def _assert_bearer_span(rig: Rig, response_id: str, marker: str) -> Span:
    span: Final = rig.sink.single_span(response_id)
    assert span.target == "/v1/traces", span
    assert span.authorization == f"Bearer {OPERATOR_KEY}", span
    bodies: Final = rig.upstream_bodies(marker)
    assert len(bodies) == 1, bodies
    assert OPERATOR_KEY not in json.dumps(bodies[0]), bodies[0]
    return span


def test_nops_is_registered_as_an_opentelemetry_callback(rig: Rig) -> None:
    listed: Final = rig.proxy.request("GET", "/active/callbacks")
    assert listed.status_code == 200, listed.text
    assert "OpenTelemetry" in json.dumps(listed.json()), listed.text
    assert "NOPS_API_KEY not found" not in rig.process.log.read_text()


def test_chat_completion_sdk_span_lands_at_the_sink_with_the_bearer_key(rig: Rig) -> None:
    marker: Final = _marker()
    completion: Final = rig.openai_client().chat.completions.create(
        model=rig.model, messages=[{"role": "user", "content": marker}]
    )
    assert completion.id == f"chatcmpl-{marker}"
    span: Final = _assert_bearer_span(rig, completion.id, marker)
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
    _assert_bearer_span(rig, f"chatcmpl-{marker}", marker)


def test_messages_sdk_span_lands_at_the_sink_with_the_bearer_key(rig: Rig) -> None:
    marker: Final = _marker()
    message: Final = rig.anthropic_client().messages.create(
        model=rig.model, max_tokens=16, messages=[{"role": "user", "content": marker}]
    )
    _assert_bearer_span(rig, message.id, marker)


def test_responses_sdk_span_lands_at_the_sink_with_the_bearer_key(rig: Rig) -> None:
    marker: Final = _marker()
    response: Final = rig.openai_client().responses.create(model=rig.model, input=marker)
    _assert_bearer_span(rig, response.id, marker)


def test_v2_flag_on_still_delivers_the_span_with_the_bearer_key(v2_rig: Rig) -> None:
    marker: Final = _marker()
    response: Final = v2_rig.chat(marker)
    assert response.status_code == 200, response.text
    _assert_bearer_span(v2_rig, _body_id(response), marker)


def _assert_span_without_content(rig: Rig) -> None:
    marker: Final = _marker()
    prompt: Final = "prompt-" + uuid.uuid4().hex
    response: Final = rig.chat(f"{marker} {prompt}")
    assert response.status_code == 200, response.text
    span: Final = rig.sink.single_span(_body_id(response))
    assert "gen_ai.input.messages" not in span.attributes, span
    assert "gen_ai.output.messages" not in span.attributes, span
    bodies: Final = rig.sink.bodies()
    assert not any(prompt.encode() in body for body in bodies)
    assert not any(REPLY_TEXT.encode() in body for body in bodies)


def test_legacy_span_carries_no_prompt_or_response_content_by_default(rig: Rig) -> None:
    _assert_span_without_content(rig)


def test_v2_span_carries_no_prompt_or_response_content_by_default(v2_rig: Rig) -> None:
    _assert_span_without_content(v2_rig)


def test_endpoint_already_ending_in_v1_traces_is_not_doubled(
    provider: Wire, sink: Collector, tmp_path_factory: pytest.TempPathFactory
) -> None:
    factory: Final = RigFactory(
        provider, sink, tmp_path_factory.mktemp("nops-suffixed"), False, sink.wire.url + "/v1/traces"
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


def test_health_services_accepts_nops(rig: Rig) -> None:
    response: Final = rig.proxy.request("GET", "/health/services", params={"service": "nops"})
    assert response.status_code == 200, response.text
    assert "nops" in _text_at(JSON.validate_json(response.content), "message"), response.text


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


@pytest.mark.parametrize("api_key", [None, ""], ids=["unset", "empty"])
def test_missing_api_key_fails_loudly_at_boot(
    provider: Wire, sink: Collector, tmp_path_factory: pytest.TempPathFactory, api_key: str | None
) -> None:
    factory: Final = RigFactory(provider, sink, tmp_path_factory.mktemp("nops-nokey"), False, sink.wire.url, api_key)
    started: Final = factory.start()
    broken: Final = next(started)
    try:
        response: Final = broken.chat(_marker())
        assert response.status_code == 200, response.text
        eventually(
            lambda: broken.process.log.read_text(),
            lambda text: "NOPS_API_KEY not found in environment variables" in text,
            seconds=30,
        )
        assert sink.landed((_body_id(response),)) == {_body_id(response): 0}
    finally:
        with pytest.raises(StopIteration):
            next(started)
