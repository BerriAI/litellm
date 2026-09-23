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
from pathlib import Path
from typing import Final

import anthropic
import httpx
import openai
import psutil
import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

MARKER: Final = re.compile(rb"otelconv-[0-9a-f]{32}")
CONVERSATION: Final = "gen_ai.conversation.id"


def _marker() -> str:
    return "otelconv-" + uuid.uuid4().hex


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
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "conversation ok"},
                            "finish_reason": "stop",
                        }
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
                {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "conversation"}}]}
            ).encode()
            + b"\n\n",
            b"data: "
            + json.dumps(
                {
                    **chunk,
                    "choices": [{"index": 0, "delta": {"content": " ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
                }
            ).encode()
            + b"\n\n",
            b"data: [DONE]\n\n",
        ),
    )


def _responses_reply(identity: str, stream: bool) -> Reply:
    response: Final = {
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
                "content": [{"type": "output_text", "text": "conversation ok", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9},
    }
    if not stream:
        return Reply(body=json.dumps(response).encode())
    events: Final = (
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
            "delta": "conversation ok",
        },
        {"type": "response.completed", "sequence_number": 2, "response": response},
    )
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events),
    )


def _decoded_responses_id(identity: str) -> str:
    try:
        return base64.b64decode(identity.removeprefix("resp_").encode()).decode()
    except (ValueError, UnicodeDecodeError):
        return identity


def _canonical_id(identity: str) -> str:
    return _decoded_responses_id(identity).rpartition("response_id:")[2]


def _sse_events(text: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ") and line != "data: [DONE]"
    )


def _upstream(request: Request) -> Reply:
    found: Final = MARKER.search(request.body)
    if found is None:
        return Reply(status=404, body=b'{"error":"no marker"}')
    marker: Final = found.group(0).decode()
    stream: Final = json.loads(request.body).get("stream") is True
    if request.target.endswith("/responses"):
        return _responses_reply(f"resp_{marker}", stream)
    return _chat_reply(f"chatcmpl-{marker}", stream)


@dataclass(frozen=True, slots=True)
class Collector:
    wire: Wire
    outage: threading.Event
    rejection: threading.Event
    slow: threading.Event
    accepted: Sequence[Request]
    guard: threading.Lock

    def attributes(self) -> tuple[dict[str, dict[str, JsonValue]], ...]:
        with self.guard:
            batches: Final = tuple(self.accepted)
        return tuple(
            {attribute["key"]: attribute["value"] for attribute in span.get("attributes", ())}
            for batch in batches
            for resource in json.loads(batch.body)["resourceSpans"]
            for scope in resource["scopeSpans"]
            for span in scope["spans"]
        )

    def spans(self, response_id: str) -> tuple[dict[str, dict[str, JsonValue]], ...]:
        return tuple(
            attributes
            for attributes in self.attributes()
            if isinstance(logged := attributes.get("gen_ai.response.id", {}).get("stringValue"), str)
            and _canonical_id(logged) == _canonical_id(response_id)
        )

    def conversation_ids(self, response_id: str) -> tuple[str | None, ...]:
        return tuple(
            attributes[CONVERSATION]["stringValue"] if CONVERSATION in attributes else None
            for attributes in self.spans(response_id)
        )

    def single_span(self, response_id: str) -> str | None:
        return eventually(lambda: self.conversation_ids(response_id), lambda values: len(values) == 1, seconds=30)[0]

    def logged_id(self, response_id: str) -> str:
        spans: Final = eventually(lambda: self.spans(response_id), lambda values: len(values) == 1, seconds=30)
        return str(spans[0]["gen_ai.response.id"]["stringValue"])


@pytest.fixture(scope="session")
def collector() -> Iterator[Collector]:
    outage: Final = threading.Event()
    rejection: Final = threading.Event()
    slow: Final = threading.Event()
    accepted: Final[deque[Request]] = deque()  # mutable-ok: sink thread appends each accepted batch
    guard: Final = threading.Lock()

    def sink(request: Request) -> Reply:
        if slow.is_set():
            threading.Event().wait(1.5)
        if outage.is_set():
            return Reply(status=503, body=b'{"error":"sink down"}')
        if rejection.is_set():
            return Reply(status=403, body=b'{"error":"forbidden"}')
        with guard:
            accepted.append(request)
        return Reply()

    with wire_server(sink) as wire:
        yield Collector(wire, outage, rejection, slow, accepted, guard)


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

    def async_anthropic_client(self) -> anthropic.AsyncAnthropic:
        return anthropic.AsyncAnthropic(base_url=str(self.proxy.client.base_url), api_key=self.proxy.key, max_retries=0)

    def chat(
        self,
        marker: str,
        *,
        headers: Mapping[str, str] | None = None,
        key: str | None = None,
        **extra: JsonValue,
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
        return tuple(json.loads(request.body) for request in self.upstream.drain() if marker.encode() in request.body)

    def spend_session(self, response_id: str) -> str | None:
        rows: Final = eventually(
            lambda: read_rows('SELECT session_id FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (response_id,)),
            lambda values: len(values) == 1,
            seconds=70,
        )
        value: Final = rows[0]["session_id"]
        assert value is None or isinstance(value, str), rows
        return value

    def spend_request_ids(self, session: str) -> tuple[str, ...]:
        rows: Final = read_rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE session_id=%s', (session,))
        return tuple(str(row["request_id"]) for row in rows)


@dataclass(frozen=True, slots=True)
class RigFactory:
    provider: Wire
    sink: Collector
    directory: Path
    settings: Mapping[str, JsonValue]
    workers: int

    def start(self) -> Iterator[Rig]:
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["litellm_settings"].update({"callbacks": ["otel"]})
        config["general_settings"].update({"disable_model_info_refresh": True, **self.settings})
        config["callback_settings"] = {
            "otel": {"exporter": "http/json", "endpoint": self.sink.wire.url, "mapper_names": ["genai"]},
        }
        path: Final = self.directory / f"otel-{uuid.uuid4().hex}.yaml"
        path.write_text(yaml.safe_dump(config))
        overrides: Final = {"LITELLM_OTEL_V2": "1", "OTEL_BSP_SCHEDULE_DELAY": "300"}
        with (
            gateway_from_environment() as gateway,
            owned_proxy_process(gateway, self.directory, overrides, config=path, workers=self.workers) as owned,
            owned.gateway.scenario() as scenario,
        ):
            model: Final = scenario.model(api_base=self.provider.url + "/v1")
            yield Rig(owned.gateway, owned, model, self.provider, self.sink)


@pytest.fixture(scope="session")
def rig(provider: Wire, collector: Collector, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    yield from RigFactory(provider, collector, tmp_path_factory.mktemp("otel"), {}, 2).start()


@pytest.fixture(scope="session")
def generating_rig(provider: Wire, collector: Collector, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    factory: Final = RigFactory(
        provider, collector, tmp_path_factory.mktemp("otel-generate"), {"missing_session_id": "generate"}, 2
    )
    yield from factory.start()


@pytest.fixture(scope="session")
def two_worker_rig(provider: Wire, collector: Collector, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    yield from RigFactory(provider, collector, tmp_path_factory.mktemp("otel-workers"), {}, 2).start()


def _assert_upstream_clean(rig: Rig, marker: str, session: str) -> None:
    bodies: Final = rig.upstream_bodies(marker)
    assert len(bodies) == 1, bodies
    assert session not in json.dumps(bodies[0]), bodies[0]


@pytest.mark.covers("other.observability.otel.conversation_id_from_body_session_id_chat_sdk")
def test_chat_completion_sdk_body_litellm_session_id_lands_as_conversation_id(rig: Rig) -> None:
    marker: Final = _marker()
    session: Final = "conv-" + uuid.uuid4().hex
    completion: Final = rig.openai_client().chat.completions.create(
        model=rig.model,
        messages=[{"role": "user", "content": marker}],
        extra_body={"litellm_session_id": session, "cache": {"no-cache": True}},
    )
    assert completion.id == f"chatcmpl-{marker}", completion
    assert completion.choices[0].message.content == "conversation ok", completion
    assert rig.sink.single_span(completion.id) == session
    assert rig.spend_session(completion.id) == session
    _assert_upstream_clean(rig, marker, session)


@pytest.mark.covers("other.observability.otel.conversation_id_from_header_chat_stream_async_sdk")
def test_chat_stream_async_sdk_x_litellm_session_id_header_lands_as_conversation_id(rig: Rig) -> None:
    marker: Final = _marker()
    session: Final = "conv-" + uuid.uuid4().hex

    async def consume() -> tuple[str, str]:
        stream: Final = await rig.async_openai_client().chat.completions.create(
            model=rig.model,
            messages=[{"role": "user", "content": marker}],
            stream=True,
            extra_headers={"x-litellm-session-id": session},
            extra_body={"cache": {"no-cache": True}},
        )
        chunks: Final = [chunk async for chunk in stream]
        return chunks[0].id, "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)

    identity, text = asyncio.run(consume())
    assert identity == f"chatcmpl-{marker}", identity
    assert text == "conversation ok", text
    assert rig.sink.single_span(identity) == session
    assert rig.spend_session(identity) == session
    _assert_upstream_clean(rig, marker, session)


@pytest.mark.covers("other.observability.otel.conversation_id_from_header_messages_sdk")
def test_messages_sdk_x_litellm_session_id_header_lands_as_conversation_id(rig: Rig) -> None:
    marker: Final = _marker()
    session: Final = "conv-" + uuid.uuid4().hex
    message: Final = rig.anthropic_client().messages.create(
        model=rig.model,
        max_tokens=16,
        messages=[{"role": "user", "content": marker}],
        extra_headers={"x-litellm-session-id": session},
    )
    assert message.content[0].type == "text" and message.content[0].text == "conversation ok", message
    assert rig.sink.single_span(message.id) == session
    assert rig.spend_session(message.id) == session
    _assert_upstream_clean(rig, marker, session)


@pytest.mark.covers("other.observability.otel.conversation_id_from_langfuse_header_messages_stream_async_sdk")
def test_messages_stream_async_sdk_langfuse_session_id_header_lands_as_conversation_id(rig: Rig) -> None:
    marker: Final = _marker()
    session: Final = "conv-" + uuid.uuid4().hex

    async def consume() -> tuple[str, str]:
        async with rig.async_anthropic_client().messages.stream(
            model=rig.model,
            max_tokens=16,
            messages=[{"role": "user", "content": marker}],
            extra_headers={"langfuse_session_id": session},
        ) as stream:
            text: Final = "".join([piece async for piece in stream.text_stream])
            return (await stream.get_final_message()).id, text

    identity, text = asyncio.run(consume())
    assert text == "conversation ok", text
    assert rig.sink.single_span(identity) == session
    assert rig.spend_session(identity), identity
    _assert_upstream_clean(rig, marker, session)


@pytest.mark.covers("other.observability.otel.conversation_id_from_header_responses_sdk")
def test_responses_sdk_x_litellm_session_id_header_lands_as_conversation_id(rig: Rig) -> None:
    marker: Final = _marker()
    session: Final = "conv-" + uuid.uuid4().hex
    response: Final = rig.openai_client().responses.create(
        model=rig.model, input=marker, extra_headers={"x-litellm-session-id": session}
    )
    assert response.output[0].id == f"msg_resp_{marker}", response
    assert response.output_text == "conversation ok", response
    assert rig.sink.single_span(response.id) == session
    assert rig.spend_session(response.id) == session
    _assert_upstream_clean(rig, marker, session)


@pytest.mark.covers("other.observability.otel.conversation_id_from_metadata_responses_stream_raw")
def test_responses_stream_raw_metadata_session_id_lands_as_conversation_id(rig: Rig) -> None:
    marker: Final = _marker()
    session: Final = "conv-" + uuid.uuid4().hex
    with rig.proxy.client.stream(
        "POST",
        "/v1/responses",
        json={"model": rig.model, "input": marker, "stream": True, "metadata": {"session_id": session}},
        headers={"Authorization": f"Bearer {rig.proxy.key}"},
    ) as response:
        body: Final = response.read().decode()
        assert response.status_code == 200, body
    events: Final = _sse_events(body)
    completed: Final = tuple(event for event in events if event["type"] == "response.completed")
    assert len(completed) == 1, events
    assert completed[0]["response"]["output"][0]["id"] == f"msg_resp_{marker}", completed
    assert str(completed[0]["response"]["id"]).startswith("resp_"), completed
    assert rig.upstream_bodies(marker) == (
        {"model": "gpt-4o-mini", "input": marker, "metadata": {"session_id": session}, "stream": True},
    )
    assert rig.sink.single_span(f"resp_{marker}") == session
    assert rig.spend_session(rig.sink.logged_id(f"resp_{marker}")) == session


@pytest.mark.covers("other.observability.otel.conversation_id_from_metadata_chat_raw")
def test_chat_raw_metadata_session_id_lands_as_conversation_id(rig: Rig) -> None:
    marker: Final = _marker()
    session: Final = "conv-" + uuid.uuid4().hex
    response: Final = rig.chat(marker, metadata={"session_id": session})
    assert response.status_code == 200, response.text
    identity: Final = response.json()["id"]
    assert identity == f"chatcmpl-{marker}", response.text
    assert rig.sink.single_span(identity) == session
    assert rig.spend_session(identity) == session
    _assert_upstream_clean(rig, marker, session)


@pytest.mark.covers("other.observability.otel.conversation_id_non_string_session_ids_match_spend_row")
def test_integer_and_list_litellm_session_id_match_the_spend_row_or_are_dropped_together(rig: Rig) -> None:
    for odd in (123, ["a", "b"]):
        marker: Final = _marker()
        response: Final = rig.chat(marker, litellm_session_id=odd)
        assert response.status_code == 200, response.text
        identity: Final = response.json()["id"]
        assert rig.sink.single_span(identity) == rig.spend_session(identity), (odd, rig.sink.conversation_ids(identity))
        assert len(rig.upstream_bodies(marker)) == 1


@pytest.mark.covers("other.observability.otel.conversation_id_empty_string_session_id_is_omitted")
def test_empty_string_litellm_session_id_leaves_the_span_without_a_conversation_id(rig: Rig) -> None:
    marker: Final = _marker()
    response: Final = rig.chat(marker, litellm_session_id="")
    assert response.status_code == 200, response.text
    identity: Final = response.json()["id"]
    assert rig.sink.single_span(identity) is None
    assert rig.spend_session(identity), response.text


@pytest.mark.covers("other.observability.otel.conversation_id_five_kilobyte_header_round_trips")
def test_five_kilobyte_session_header_round_trips_to_the_span_and_the_spend_row(rig: Rig) -> None:
    marker: Final = _marker()
    session: Final = ("s" * 5000) + uuid.uuid4().hex
    response: Final = rig.chat(marker, headers={"x-litellm-session-id": session})
    assert response.status_code == 200, response.text
    identity: Final = response.json()["id"]
    assert rig.sink.single_span(identity) == session
    assert rig.spend_session(identity) == session
    _assert_upstream_clean(rig, marker, session)


@pytest.mark.covers("other.observability.otel.conversation_id_duplicate_header_lands_once")
def test_duplicate_session_header_lands_once_and_unchanged(rig: Rig) -> None:
    marker: Final = _marker()
    session: Final = "conv-" + uuid.uuid4().hex
    response: Final = rig.proxy.client.post(
        "/v1/chat/completions",
        json={"model": rig.model, "messages": [{"role": "user", "content": marker}], "cache": {"no-cache": True}},
        headers=[
            ("Authorization", f"Bearer {rig.proxy.key}"),
            ("x-litellm-session-id", session),
            ("x-litellm-session-id", session),
        ],
    )
    assert response.status_code == 200, response.text
    identity: Final = response.json()["id"]
    assert rig.sink.single_span(identity) == session
    assert rig.spend_session(identity) == session
    _assert_upstream_clean(rig, marker, session)


@pytest.mark.covers("other.observability.otel.conversation_id_unauthenticated_request_leaves_no_span")
def test_unauthenticated_request_with_session_header_is_rejected_and_leaves_no_span(rig: Rig) -> None:
    marker: Final = _marker()
    response: Final = rig.chat(marker, headers={"x-litellm-session-id": "conv-" + uuid.uuid4().hex}, key="sk-wrong")
    assert response.status_code == 401, response.text
    assert rig.upstream_bodies(marker) == ()
    control: Final = rig.chat(marker)
    assert control.status_code == 200, control.text
    assert rig.sink.single_span(control.json()["id"]) is None
    assert rig.spend_session(control.json()["id"]), control.text


@pytest.mark.covers("other.observability.otel.conversation_id_survives_sink_rejection")
def test_sink_rejecting_with_403_drops_those_spans_and_later_spans_still_land(rig: Rig) -> None:
    rig.sink.rejection.set()
    try:
        rejected: Final = rig.chat(_marker(), headers={"x-litellm-session-id": "conv-rejected"})
        assert rejected.status_code == 200, rejected.text
        eventually(lambda: any(request.body for request in rig.sink.wire.drain()), lambda seen: seen, seconds=30)
    finally:
        rig.sink.rejection.clear()
    marker: Final = _marker()
    session: Final = "conv-" + uuid.uuid4().hex
    response: Final = rig.chat(marker, headers={"x-litellm-session-id": session})
    assert response.status_code == 200, response.text
    assert rig.sink.single_span(response.json()["id"]) == session


@pytest.mark.covers("other.observability.otel.conversation_id_absent_without_caller_session")
def test_request_without_any_session_input_has_no_conversation_id(rig: Rig) -> None:
    marker: Final = _marker()
    response: Final = rig.chat(marker)
    assert response.status_code == 200, response.text
    identity: Final = response.json()["id"]
    assert rig.sink.single_span(identity) is None
    assert rig.spend_session(identity), response.text


@pytest.mark.covers("other.observability.otel.conversation_id_ignores_generated_session_id")
def test_generate_policy_minted_session_id_reaches_the_spend_row_but_not_the_span(generating_rig: Rig) -> None:
    marker: Final = _marker()
    response: Final = generating_rig.chat(marker)
    assert response.status_code == 200, response.text
    identity: Final = response.json()["id"]
    minted: Final = generating_rig.spend_session(identity)
    assert minted, response.text
    assert generating_rig.sink.single_span(identity) is None


@pytest.mark.covers("other.observability.otel.conversation_id_langfuse_header_wins_over_generated")
def test_generate_policy_keeps_the_langfuse_session_header_as_conversation_id(generating_rig: Rig) -> None:
    marker: Final = _marker()
    session: Final = "conv-" + uuid.uuid4().hex
    response: Final = generating_rig.chat(marker, headers={"langfuse_session_id": session})
    assert response.status_code == 200, response.text
    assert generating_rig.sink.single_span(response.json()["id"]) == session


@pytest.mark.covers("other.observability.otel.conversation_id_header_precedence_matches_spend_row")
def test_header_body_and_metadata_session_ids_resolve_to_the_same_id_as_the_spend_row(rig: Rig) -> None:
    marker: Final = _marker()
    header: Final = "conv-header-" + uuid.uuid4().hex
    response: Final = rig.chat(
        marker,
        headers={"x-litellm-session-id": header},
        litellm_session_id="conv-body-" + uuid.uuid4().hex,
        metadata={"session_id": "conv-meta-" + uuid.uuid4().hex},
    )
    assert response.status_code == 200, response.text
    identity: Final = response.json()["id"]
    assert rig.sink.single_span(identity) == header
    assert rig.spend_session(identity) == header


@pytest.mark.covers("other.observability.otel.conversation_id_repeated_requests_log_once_each")
def test_three_identical_requests_produce_one_span_each_with_the_same_conversation_id(rig: Rig) -> None:
    marker: Final = _marker()
    session: Final = "conv-" + uuid.uuid4().hex
    responses: Final = tuple(rig.chat(marker, headers={"x-litellm-session-id": session}) for _ in range(3))
    assert all(response.status_code == 200 for response in responses), [response.text for response in responses]
    identity: Final = f"chatcmpl-{marker}"
    spans: Final = eventually(lambda: rig.sink.conversation_ids(identity), lambda values: len(values) == 3, seconds=30)
    assert spans == (session, session, session), spans
    assert len(rig.upstream_bodies(marker)) == 3


@pytest.mark.covers("other.observability.otel.conversation_id_ignores_trace_id_backfill")
def test_metadata_trace_id_alone_fills_the_spend_row_but_not_the_span(rig: Rig) -> None:
    marker: Final = _marker()
    trace: Final = "trace-" + uuid.uuid4().hex
    response: Final = rig.chat(marker, metadata={"trace_id": trace})
    assert response.status_code == 200, response.text
    identity: Final = response.json()["id"]
    assert rig.sink.single_span(identity) is None
    assert rig.spend_session(identity) == trace


def _chat_id(response: httpx.Response) -> str:
    if not response.headers.get("content-type", "").startswith("text/event-stream"):
        return response.json()["id"]
    identities: Final = frozenset(str(event["id"]) for event in _sse_events(response.text))
    assert len(identities) == 1, response.text
    return next(iter(identities))


def _responses_id(response: httpx.Response) -> str:
    if not response.headers.get("content-type", "").startswith("text/event-stream"):
        return response.json()["id"]
    completed: Final = tuple(
        event["response"]["id"] for event in _sse_events(response.text) if event.get("type") == "response.completed"
    )
    assert len(completed) == 1, response.text
    return str(completed[0])


def _message_id(response: httpx.Response) -> str:
    if not response.headers.get("content-type", "").startswith("text/event-stream"):
        return response.json()["id"]
    starts: Final = tuple(
        event["message"]["id"] for event in _sse_events(response.text) if event.get("type") == "message_start"
    )
    assert len(starts) == 1, response.text
    return starts[0]


def _burst(rig: Rig, count: int, session_for: Mapping[int, str]) -> tuple[tuple[int, str, str | None], ...]:
    markers: Final = tuple(_marker() for _ in range(count))

    def one(index: int) -> tuple[int, str, str | None]:
        marker: Final = markers[index]
        headers: Final = {"Authorization": f"Bearer {rig.proxy.key}", "x-litellm-session-id": session_for[index]}
        route: Final = index % 3
        try:
            if route == 0:
                response: Final = rig.proxy.client.post(
                    "/v1/chat/completions",
                    json={
                        "model": rig.model,
                        "messages": [{"role": "user", "content": marker}],
                        "stream": index % 2 == 0,
                    },
                    headers=headers,
                )
                response.read()
                if response.status_code != 200:
                    return index, marker, response.text
                return index, _chat_id(response), None
            if route == 1:
                response = rig.proxy.client.post(
                    "/v1/responses",
                    json={"model": rig.model, "input": marker, "stream": index % 2 == 0},
                    headers=headers,
                )
                response.read()
                if response.status_code != 200:
                    return index, marker, response.text
                return index, _responses_id(response), None
            response = rig.proxy.client.post(
                "/v1/messages",
                json={
                    "model": rig.model,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": marker}],
                    "stream": index % 2 == 0,
                },
                headers=headers,
            )
            response.read()
            if response.status_code != 200:
                return index, marker, response.text
            return index, _message_id(response), None
        except httpx.HTTPError as error:
            return index, marker, repr(error)

    with ThreadPoolExecutor(max_workers=10) as pool:
        return tuple(pool.map(one, range(count)))


def _is_encrypted_responses_id(identity: str) -> bool:
    return identity.startswith("resp_") and _decoded_responses_id(identity) == identity


def _landed(rig: Rig, expected: Mapping[str, str]) -> dict[str, tuple[str, ...]]:
    spans: Final = rig.sink.attributes()
    return {
        session: tuple(
            _canonical_id(str(attributes["gen_ai.response.id"]["stringValue"]))
            for attributes in spans
            if attributes.get(CONVERSATION, {}).get("stringValue") == session and "gen_ai.response.id" in attributes
        )
        for session in expected.values()
    }


def _assert_exactly_once(rig: Rig, expected: Mapping[str, str], landed: Mapping[str, tuple[str, ...]]) -> None:
    spend: Final = eventually(
        lambda: {
            session: tuple(_canonical_id(identity) for identity in rig.spend_request_ids(session))
            for session in expected.values()
        },
        lambda rows: all(len(values) >= 1 for values in rows.values()),
        seconds=70,
    )
    assert landed == spend, (landed, spend)
    assert all(len(values) == 1 for values in landed.values()), landed
    caller_visible: Final = {
        session: (_canonical_id(identity),)
        for identity, session in expected.items()
        if not _is_encrypted_responses_id(identity)
    }
    assert {session: landed[session] for session in caller_visible} == caller_visible, landed


@pytest.mark.covers("other.observability.otel.conversation_id_sink_outage_recovers_exactly_once")
def test_sink_outage_during_a_mixed_burst_lands_every_response_exactly_once_after_recovery(rig: Rig) -> None:
    sessions: Final = {index: f"conv-{index}-{uuid.uuid4().hex}" for index in range(30)}
    rig.sink.outage.set()
    try:
        health_down: Final = rig.proxy.request("GET", "/health/services", params={"service": "otel"})
        results: Final = _burst(rig, 30, sessions)
        assert all(error is None for _, _, error in results), [error for _, _, error in results if error]
        eventually(lambda: any(True for _ in rig.sink.wire.drain()), lambda seen: seen, seconds=30)
    finally:
        rig.sink.outage.clear()
    assert health_down.status_code == 200, health_down.text
    expected: Final = {identity: sessions[index] for index, identity, _ in results}
    landed: Final = eventually(
        lambda: _landed(rig, expected), lambda seen: all(len(values) >= 1 for values in seen.values()), seconds=80
    )
    _assert_exactly_once(rig, expected, landed)


@pytest.mark.covers("other.observability.otel.conversation_id_slow_sink_no_duplicates")
def test_slow_sink_during_a_burst_lands_every_response_exactly_once(rig: Rig) -> None:
    sessions: Final = {index: f"conv-{index}-{uuid.uuid4().hex}" for index in range(20)}
    rig.sink.slow.set()
    try:
        results: Final = _burst(rig, 20, sessions)
        assert all(error is None for _, _, error in results), [error for _, _, error in results if error]
        expected: Final = {identity: sessions[index] for index, identity, _ in results}
        landed: Final = eventually(
            lambda: _landed(rig, expected), lambda seen: all(len(values) >= 1 for values in seen.values()), seconds=80
        )
    finally:
        rig.sink.slow.clear()
    _assert_exactly_once(rig, expected, landed)


@pytest.mark.covers("other.observability.otel.conversation_id_survives_worker_kill")
def test_killing_one_of_two_workers_mid_burst_keeps_serving_and_never_duplicates_a_span(two_worker_rig: Rig) -> None:
    rig: Final = two_worker_rig
    root: Final = psutil.Process(rig.process.process.pid)
    workers: Final = eventually(
        lambda: tuple(child for child in root.children() if "resource_tracker" not in " ".join(child.cmdline())),
        lambda found: len(found) == 2,
        seconds=30,
    )
    sessions: Final = {index: f"conv-{index}-{uuid.uuid4().hex}" for index in range(24)}
    markers: Final = tuple(_marker() for _ in range(24))

    def one(index: int) -> tuple[str, str | None]:
        if index == 8:
            os.kill(workers[0].pid, signal.SIGKILL)
        try:
            response: Final = rig.chat(markers[index], headers={"x-litellm-session-id": sessions[index]})
            return f"chatcmpl-{markers[index]}", None if response.status_code == 200 else response.text
        except httpx.HTTPError as error:
            return f"chatcmpl-{markers[index]}", repr(error)

    with ThreadPoolExecutor(max_workers=6) as pool:
        results: Final = tuple(pool.map(one, range(24)))
    assert rig.process.process.poll() is None, "Proxy root exited after a worker was killed"
    after: Final = rig.chat(_marker(), headers={"x-litellm-session-id": "conv-after-kill"})
    assert after.status_code == 200, after.text
    assert rig.sink.single_span(after.json()["id"]) == "conv-after-kill"
    failures: Final = tuple(error for _, error in results if error)
    assert all(error.startswith(("ReadError(", "RemoteProtocolError(", "ConnectError(")) for error in failures), (
        failures
    )
    assert len(failures) <= 6, failures
    served: Final = {identity: sessions[index] for index, (identity, error) in enumerate(results) if error is None}
    assert len(served) >= 18, results
    settled: Final = {
        identity: sessions[index] for index, (identity, error) in enumerate(results) if index > 14 and not error
    }
    landed: Final = eventually(
        lambda: _landed(rig, settled), lambda seen: all(len(values) >= 1 for values in seen.values()), seconds=60
    )
    _assert_exactly_once(rig, settled, landed)
    assert all(len(values) <= 1 for values in _landed(rig, served).values()), _landed(rig, served)
    lost: Final = _landed(rig, {identity: sessions[index] for index, (identity, error) in enumerate(results) if error})
    assert all(values == () for values in lost.values()), lost


@pytest.mark.covers("other.observability.otel.conversation_id_flushes_on_shutdown")
def test_terminating_the_proxy_right_after_a_burst_flushes_every_span_before_exit(
    provider: Wire, collector: Collector, tmp_path_factory: pytest.TempPathFactory
) -> None:
    factory: Final = RigFactory(provider, collector, tmp_path_factory.mktemp("otel-shutdown"), {}, 2)
    started: Final = factory.start()
    rig: Final = next(started)
    sessions: Final = {index: f"conv-{index}-{uuid.uuid4().hex}" for index in range(10)}
    markers: Final = tuple(_marker() for _ in range(10))
    responses: Final = tuple(
        rig.chat(markers[index], headers={"x-litellm-session-id": sessions[index]}) for index in range(10)
    )
    assert all(response.status_code == 200 for response in responses), [response.text for response in responses]
    expected: Final = {f"chatcmpl-{markers[index]}": sessions[index] for index in range(10)}
    drained: Final = eventually(
        lambda: _landed(rig, expected), lambda seen: all(len(values) >= 1 for values in seen.values()), seconds=60
    )
    assert drained == {session: (identity,) for identity, session in expected.items()}, drained
    rig.process.process.terminate()
    assert rig.process.process.wait(timeout=40) in (0, -signal.SIGTERM)
    assert _landed(rig, expected) == drained
    with pytest.raises(httpx.ConnectError):
        next(started)
