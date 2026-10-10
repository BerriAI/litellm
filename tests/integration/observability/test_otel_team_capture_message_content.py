import json
import re
import uuid
from collections.abc import Generator, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal
from urllib.parse import urlparse

import httpx
import pytest
import yaml
from integration._support.client import Gateway, Scenario, eventually, gateway_from_environment, object_value
from integration._support.otlp_sink import (
    ConnectSink,
    Span,
    SpanSinks,
    configure_sink,
    owned_connect_sink,
    recorded_requests,
    recorded_spans,
)
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

MARKER: Final = re.compile(rb"cmc-(?:tool-|fail-)?[0-9a-f]{32}")
JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
SERVER: Final = 2
CALL_ID: Final = "litellm.call_id"
OPERATION: Final = "gen_ai.operation.name"
INPUT_TOKENS: Final = "gen_ai.usage.input_tokens"
Endpoint = Literal["chat", "responses", "messages"]
Capture = Literal["no_content", "span_only"]


def _marker(kind: str = "") -> str:
    return f"cmc-{kind}{uuid.uuid4().hex}"


def _usage() -> dict[str, JsonValue]:
    return {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}


def _chat_reply(marker: str, stream: bool) -> Reply:
    identity: Final = f"chatcmpl-{uuid.uuid4().hex}"
    if marker.startswith("cmc-tool-"):
        call: Final[dict[str, JsonValue]] = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": json.dumps({"query": marker})},
        }
        message: Final[dict[str, JsonValue]] = {"role": "assistant", "content": None, "tool_calls": [call]}
        return Reply(
            body=json.dumps(
                {
                    "id": identity,
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
                    "usage": _usage(),
                }
            ).encode()
        )
    if not stream:
        return Reply(
            body=json.dumps(
                {
                    "id": identity,
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": marker}, "finish_reason": "stop"}
                    ],
                    "usage": _usage(),
                }
            ).encode()
        )
    chunk: Final = {"id": identity, "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini"}
    deltas: Final[tuple[dict[str, JsonValue], ...]] = (
        {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": marker}}]},
        {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {**chunk, "choices": [], "usage": _usage()},
    )
    return Reply(
        content_type="text/event-stream",
        chunks=(*(b"data: " + json.dumps(delta).encode() + b"\n\n" for delta in deltas), b"data: [DONE]\n\n"),
    )


def _responses_reply(marker: str, stream: bool) -> Reply:
    identity: Final = uuid.uuid4().hex
    response: Final[dict[str, JsonValue]] = {
        "id": f"resp_{identity}",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-4o-mini",
        "output": [
            {
                "id": f"msg_{identity}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": marker, "annotations": []}],
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
            "item_id": f"msg_{identity}",
            "output_index": 0,
            "content_index": 0,
            "delta": marker,
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
    marker: Final = found.group(0).decode()
    if marker.startswith("cmc-fail-"):
        return Reply(status=500, body=b'{"error":{"message":"scripted upstream failure","type":"server_error"}}')
    stream: Final = object_value(JSON.validate_json(request.body)).get("stream") is True
    if request.target.endswith("/responses"):
        return _responses_reply(marker, stream)
    return _chat_reply(marker, stream)


def _body(model: str, endpoint: Endpoint, marker: str, stream: bool) -> tuple[str, dict[str, JsonValue]]:
    if endpoint == "responses":
        return "/v1/responses", {"model": model, "input": marker, "stream": stream}
    messages: Final[list[JsonValue]] = [{"role": "user", "content": marker}]
    if endpoint == "messages":
        return "/v1/messages", {"model": model, "max_tokens": 16, "messages": messages, "stream": stream}
    return "/v1/chat/completions", {"model": model, "messages": messages, "stream": stream}


@dataclass(frozen=True, slots=True)
class Sent:
    marker: str
    trace_id: str
    call_id: str
    status: int
    text: str


@dataclass(frozen=True, slots=True)
class Cursors:
    operator: int
    tenant: int
    arize: int
    newrelic: int


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    owned: OwnedProxy
    scenario: Scenario
    model: str
    upstream: Wire
    sinks: SpanSinks
    newrelic: ConnectSink

    def cursors(self) -> Cursors:
        self.upstream.drain()
        return Cursors(
            recorded_spans(self.sinks.operator)[0],
            recorded_spans(self.sinks.tenant)[0],
            recorded_spans(self.sinks.arize)[0],
            recorded_spans(self.newrelic.control_url)[0],
        )

    def upstream_hits(self, marker: str) -> int:
        return sum(1 for request in self.upstream.drain() if marker.encode() in request.body)

    def langfuse(self, name: str, host: str | None = None) -> dict[str, str]:
        return {
            "langfuse_public_key": f"pk-lf-{name}",
            "langfuse_secret_key": f"sk-lf-{name}",
            "langfuse_host": host or self.sinks.tenant,
        }

    def attach(self, team: str, callback: str, variables: Mapping[str, str], kind: str | None = None) -> httpx.Response:
        body: Final[dict[str, JsonValue]] = {"callback_name": callback, "callback_vars": dict(variables)}
        return self.proxy.request(
            "POST", f"/team/{team}/callback", body if kind is None else {**body, "callback_type": kind}
        )

    def team_key(
        self,
        capture: Capture | None,
        name: str,
        *,
        callback: str = "langfuse_otel",
        variables: Mapping[str, str] | None = None,
        kind: str | None = None,
    ) -> tuple[str, str]:
        team: Final = self.scenario.team()
        base: Final = dict(variables) if variables is not None else self.langfuse(name)
        attached: Final = self.attach(
            team, callback, base if capture is None else {**base, "capture_message_content": capture}, kind
        )
        assert attached.status_code == 200, attached.text
        return team, self.scenario.key(team_id=team)

    def send(
        self,
        key: str,
        endpoint: Endpoint = "chat",
        stream: bool = False,
        *,
        marker: str | None = None,
        extra: Mapping[str, JsonValue] | None = None,
    ) -> Sent:
        tagged: Final = marker or _marker()
        path, body = _body(self.model, endpoint, tagged, stream)
        trace_id: Final = uuid.uuid4().hex
        call_id: Final = str(uuid.uuid4())
        headers: Final = {
            "Authorization": f"Bearer {key}",
            "traceparent": f"00-{trace_id}-{uuid.uuid4().hex[:16]}-01",
            "x-litellm-call-id": call_id,
        }
        with self.proxy.client.stream("POST", path, json={**body, **(extra or {})}, headers=headers) as response:
            text: Final = response.read().decode()
            return Sent(tagged, trace_id, call_id, response.status_code, text)


def _served(sent: Sent) -> Sent:
    assert sent.status == 200, sent.text
    assert sent.marker in sent.text, sent.text
    return sent


Credential = tuple[str, str]


def _sent_with(url: str, credential: Credential) -> frozenset[str]:
    """Span ids ``url`` received in export requests whose ``credential`` header matched."""
    header, value = credential
    return frozenset(
        span_id
        for request in recorded_requests(url)
        if {name.lower(): seen for name, seen in object_value(request.get("headers") or {}).items()}.get(header)
        == value
        for span_id in request.get("span_ids") or ()
        if isinstance(span_id, str)
    )


def _of_request(url: str, since: int, sent: Sent, credential: Credential | None = None) -> tuple[Span, ...]:
    """Spans of ``sent``'s trace at ``url``, narrowed to one account's export requests when ``credential`` is set."""
    spans: Final = recorded_spans(url, since)[1]
    traces: Final = {sent.trace_id} | {
        span["trace_id"] for span in spans if span["attributes"].get(CALL_ID) == sent.call_id
    }
    owned: Final = None if credential is None else _sent_with(url, credential)
    return tuple(span for span in spans if span["trace_id"] in traces and (owned is None or span["span_id"] in owned))


def _carries(span: Span, marker: str) -> bool:
    return marker in json.dumps(span["attributes"])


def _model_spans(spans: tuple[Span, ...]) -> tuple[Span, ...]:
    return tuple(span for span in spans if OPERATION in span["attributes"])


def _names(spans: tuple[Span, ...]) -> list[str]:
    return sorted(span["name"] for span in spans)


def _with_content(url: str, since: int, sent: Sent) -> tuple[Span, ...]:
    """The copy of ``sent`` at ``url`` once its model span arrived and some span carries the marker."""
    return eventually(
        lambda: _of_request(url, since, sent),
        lambda spans: bool(_model_spans(spans)) and any(_carries(span, sent.marker) for span in spans),
        seconds=40,
    )


def _arrived(
    url: str, since: int, sent: Sent, expected: frozenset[str], credential: Credential | None = None
) -> tuple[Span, ...]:
    """The copy of ``sent`` at ``url`` once every span name in ``expected`` and a usage-bearing model span landed."""
    spans: Final = eventually(
        lambda: _of_request(url, since, sent, credential),
        lambda found: (
            expected <= {span["name"] for span in found}
            and any(INPUT_TOKENS in span["attributes"] for span in _model_spans(found))
        ),
        seconds=40,
    )
    assert all(
        span["attributes"].get(INPUT_TOKENS) == 7 for span in _model_spans(spans) if INPUT_TOKENS in span["attributes"]
    )
    return spans


def _assert_redacted_twin(
    rig: Rig, sent: Sent, cursors: Cursors, sink: str, since: int, credential: Credential | None = None
) -> tuple[Span, ...]:
    operator: Final = _with_content(rig.sinks.operator, cursors.operator, sent)
    carried: Final = frozenset(span["name"] for span in operator if _carries(span, sent.marker))
    tenant: Final = _arrived(sink, since, sent, carried, credential)
    leaked: Final = [span["name"] for span in tenant if _carries(span, sent.marker)]
    assert leaked == [], f"{leaked} carried content; tenant saw {_names(tenant)}"
    model: Final = _model_spans(tenant)[0]["attributes"]
    assert "gen_ai.input.messages" not in model and "gen_ai.output.messages" not in model, sorted(model)
    assert model.get("gen_ai.request.model"), sorted(model)
    return tenant


def _assert_content_twin(rig: Rig, sent: Sent, cursors: Cursors) -> None:
    operator: Final = _with_content(rig.sinks.operator, cursors.operator, sent)
    carried: Final = frozenset(span["name"] for span in operator if _carries(span, sent.marker))
    tenant: Final = _arrived(rig.sinks.tenant, cursors.tenant, sent, carried)
    kept: Final = frozenset(span["name"] for span in tenant if _carries(span, sent.marker))
    assert carried <= kept, f"content lost on {sorted(carried - kept)}"


@pytest.mark.parametrize(
    ("endpoint", "stream"),
    [("chat", False), ("chat", True), ("responses", False), ("messages", True)],
    ids=["chat-unary", "chat-stream", "responses-unary", "messages-stream"],
)
def test_no_content_team_keeps_its_trace_and_metadata_without_content(
    rig: Rig, endpoint: Endpoint, stream: bool
) -> None:
    _, key = rig.team_key("no_content", "a")
    cursors: Final = rig.cursors()
    sent: Final = _served(rig.send(key, endpoint, stream))
    assert rig.upstream_hits(sent.marker) == 1
    _assert_redacted_twin(rig, sent, cursors, rig.sinks.tenant, cursors.tenant)


@pytest.mark.parametrize("capture", ["span_only", None], ids=["span_only", "omitted"])
def test_span_only_and_omitted_teams_keep_the_globally_captured_content(rig: Rig, capture: Capture | None) -> None:
    _, key = rig.team_key(capture, "b")
    cursors: Final = rig.cursors()
    sent: Final = _served(rig.send(key))
    _assert_content_twin(rig, sent, cursors)


def test_tool_call_arguments_never_reach_a_no_content_team(rig: Rig) -> None:
    _, key = rig.team_key("no_content", "a")
    cursors: Final = rig.cursors()
    tool: Final[dict[str, JsonValue]] = {
        "type": "function",
        "function": {"name": "lookup", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}},
    }
    sent: Final = rig.send(key, marker=_marker("tool-"), extra={"tools": [tool]})
    assert sent.status == 200 and sent.marker in sent.text, sent.text
    operator: Final = _with_content(rig.sinks.operator, cursors.operator, sent)
    assert any(sent.marker in str(span["attributes"].get("gen_ai.output.messages", "")) for span in operator), (
        "operator copy lacks the tool-call arguments"
    )
    tenant: Final = _assert_redacted_twin(rig, sent, cursors, rig.sinks.tenant, cursors.tenant)
    assert "lookup" in json.dumps([span["attributes"] for span in tenant]), "tool definitions are metadata and stay"


def test_cache_hit_twin_of_a_no_content_team_is_redacted(rig: Rig) -> None:
    _, key = rig.team_key("no_content", "a")
    marker: Final = _marker()
    _served(rig.send(key, marker=marker))
    assert rig.upstream_hits(marker) == 1
    cursors: Final = rig.cursors()
    hit: Final = eventually(
        lambda: _served(rig.send(key, marker=marker)), lambda _: rig.upstream_hits(marker) == 0, seconds=20
    )
    operator: Final = eventually(
        lambda: _of_request(rig.sinks.operator, cursors.operator, hit),
        lambda spans: any(span["kind"] == SERVER for span in spans),
        seconds=40,
    )
    expected: Final = frozenset(span["name"] for span in operator if span["kind"] == SERVER)
    tenant: Final = eventually(
        lambda: _of_request(rig.sinks.tenant, cursors.tenant, hit),
        lambda spans: expected <= {span["name"] for span in spans},
        seconds=40,
    )
    assert [span["name"] for span in tenant if _carries(span, marker)] == [], _names(tenant)


def test_a_failure_only_entry_rejects_the_setting(rig: Rig) -> None:
    team: Final = rig.scenario.team()
    response: Final = rig.attach(
        team, "langfuse_otel", {**rig.langfuse("fail"), "capture_message_content": "no_content"}, "failure"
    )
    assert response.status_code == 400, response.text
    assert "success_and_failure" in response.text, response.text


def test_key_level_logging_with_no_content_is_redacted(rig: Rig) -> None:
    logging: Final[list[JsonValue]] = [
        {
            "callback_name": "langfuse_otel",
            "callback_type": "success",
            "callback_vars": {**rig.langfuse("key"), "capture_message_content": "no_content"},
        }
    ]
    key: Final = rig.scenario.key(metadata={"logging": logging})
    cursors: Final = rig.cursors()
    sent: Final = _served(rig.send(key))
    _assert_redacted_twin(rig, sent, cursors, rig.sinks.tenant, cursors.tenant)


def test_same_account_destination_redacts_the_shared_copy_only_for_that_team(rig: Rig) -> None:
    _, restricted = rig.team_key("no_content", "operator", variables=rig.langfuse("operator", host=rig.sinks.operator))
    _, other = rig.team_key("span_only", "b")
    cursors: Final = rig.cursors()
    shared: Final = _served(rig.send(restricted))
    assert rig.upstream_hits(shared.marker) == 1
    copy: Final = eventually(
        lambda: _of_request(rig.sinks.operator, cursors.operator, shared),
        lambda spans: any(INPUT_TOKENS in span["attributes"] for span in _model_spans(spans)),
        seconds=40,
    )
    assert [span["name"] for span in copy if _carries(span, shared.marker)] == [], _names(copy)
    assert len([span for span in copy if span["kind"] == SERVER and span["name"].startswith("POST ")]) == 1, _names(
        copy
    )
    _assert_content_twin(rig, _served(rig.send(other)), cursors)


def test_arize_destination_drops_openinference_content(rig: Rig) -> None:
    _, key = rig.team_key(
        "no_content", "arize", callback="arize", variables={"arize_space_key": "space-a", "arize_api_key": "arize-a"}
    )
    cursors: Final = rig.cursors()
    sent: Final = _served(rig.send(key))
    operator: Final = _with_content(rig.sinks.operator, cursors.operator, sent)
    assert any(sent.marker in str(span["attributes"].get("input.value", "")) for span in operator), (
        "operator copy has no OpenInference content to strip"
    )
    tenant: Final = _assert_redacted_twin(
        rig, sent, cursors, rig.sinks.arize, cursors.arize, ("arize-space-id", "space-a")
    )
    model: Final = _model_spans(tenant)[0]["attributes"]
    assert not any(name.startswith(("llm.input_messages.", "llm.output_messages.")) for name in model), sorted(model)
    assert "input.value" not in model and "output.value" not in model, sorted(model)
    assert model.get("llm.model_name"), sorted(model)


def test_weave_destination_drops_its_content(rig: Rig) -> None:
    _, key = rig.team_key(
        "no_content",
        "weave",
        callback="weave_otel",
        variables={"wandb_api_key": "wandb-a", "weave_project_id": "team/a"},
    )
    cursors: Final = rig.cursors()
    sent: Final = _served(rig.send(key))
    operator: Final = eventually(
        lambda: _of_request(rig.sinks.arize, cursors.arize, sent, ("project_id", "operator/weave")),
        lambda spans: any(sent.marker in str(span["attributes"].get("weave.output", "")) for span in spans),
        seconds=40,
    )
    assert operator, "operator weave copy has no weave.output to strip"
    tenant: Final = _assert_redacted_twin(rig, sent, cursors, rig.sinks.arize, cursors.arize, ("project_id", "team/a"))
    assert not any("weave.output" in span["attributes"] for span in tenant), _names(tenant)


def test_newrelic_destination_drops_genai_content(rig: Rig) -> None:
    _, key = rig.team_key("no_content", "newrelic", callback="newrelic", variables={"newrelic_api_key": "nr-team-a"})
    cursors: Final = rig.cursors()
    sent: Final = _served(rig.send(key))
    tenant: Final = _assert_redacted_twin(
        rig, sent, cursors, rig.newrelic.control_url, cursors.newrelic, ("api-key", "nr-team-a")
    )
    hosts: Final = {
        object_value(request.get("headers") or {}).get("Host")
        for request in recorded_requests(rig.newrelic.control_url)
        if object_value(request.get("headers") or {}).get("api-key") == "nr-team-a"
    }
    assert hosts == {"otlp.nr-data.net"}, hosts
    assert tenant


def test_span_only_team_lifts_global_no_content_only_for_its_destination(dark_rig: Rig) -> None:
    """The sibling Arize entry gets a fan-out destination and the SigNoz entry rides the routed tracer."""
    team, key = dark_rig.team_key("span_only", "b")
    siblings: Final = (
        dark_rig.attach(team, "arize", {"arize_space_key": "space-sibling", "arize_api_key": "arize-sibling"}),
        dark_rig.attach(
            team,
            "signoz",
            {"signoz_ingestion_endpoint": dark_rig.sinks.arize + "/v1/traces", "signoz_ingestion_key": "signoz-team"},
        ),
    )
    assert [response.status_code for response in siblings] == [200, 200], [response.text for response in siblings]
    cursors: Final = dark_rig.cursors()
    sent: Final = _served(dark_rig.send(key))
    assert dark_rig.upstream_hits(sent.marker) == 1
    operator: Final = _arrived(dark_rig.sinks.operator, cursors.operator, sent, frozenset())
    tenant: Final = _with_content(dark_rig.sinks.tenant, cursors.tenant, sent)
    sibling: Final = _arrived(
        dark_rig.sinks.arize,
        cursors.arize,
        sent,
        frozenset(),
        ("arize-space-id", "space-sibling"),
    )
    signoz: Final = _arrived(
        dark_rig.sinks.arize,
        cursors.arize,
        sent,
        frozenset(),
        ("signoz-ingestion-key", "signoz-team"),
    )
    assert any(_carries(span, sent.marker) for span in tenant), _names(tenant)
    assert not any(_carries(span, sent.marker) for span in operator), _names(operator)
    assert not any(_carries(span, sent.marker) for span in sibling), _names(sibling)
    assert not any(_carries(span, sent.marker) for span in signoz), _names(signoz)


def test_an_unsupported_value_fails_registration(rig: Rig) -> None:
    team: Final = rig.scenario.team()
    response: Final = rig.attach(team, "langfuse_otel", {**rig.langfuse("a"), "capture_message_content": "bogus"})
    assert response.status_code == 422, response.text
    assert "Invalid capture_message_content 'bogus'" in response.text, response.text


def test_classic_langfuse_rejects_the_setting(rig: Rig) -> None:
    team: Final = rig.scenario.team()
    response: Final = rig.attach(team, "langfuse", {**rig.langfuse("a"), "capture_message_content": "no_content"})
    assert response.status_code == 400, response.text
    assert "capture_message_content" in response.text, response.text


def test_a_second_entry_for_the_same_backend_cannot_disagree(rig: Rig) -> None:
    team, _ = rig.team_key("no_content", "a")
    response: Final = rig.attach(team, "langfuse_otel", {"capture_message_content": "span_only"}, "success_and_failure")
    assert response.status_code == 400, response.text
    assert "already set to 'no_content' by another langfuse_otel entry" in response.text, response.text


def test_replacing_the_registration_switches_the_team_to_content(rig: Rig) -> None:
    team, key = rig.team_key("no_content", "a")
    cursors: Final = rig.cursors()
    _assert_redacted_twin(rig, _served(rig.send(key)), cursors, rig.sinks.tenant, cursors.tenant)
    removed: Final = rig.proxy.request("DELETE", f"/team/{team}/callback/langfuse_otel")
    assert removed.status_code == 200, removed.text
    replaced: Final = rig.attach(team, "langfuse_otel", {**rig.langfuse("a"), "capture_message_content": "span_only"})
    assert replaced.status_code == 200, replaced.text

    def attempt() -> bool:
        since: Final = rig.cursors()
        sent: Final = _served(rig.send(key))
        operator: Final = _with_content(rig.sinks.operator, since.operator, sent)
        carried: Final = frozenset(span["name"] for span in operator if _carries(span, sent.marker))
        tenant: Final = _arrived(rig.sinks.tenant, since.tenant, sent, carried)
        return any(_carries(span, sent.marker) for span in tenant)

    assert eventually(attempt, bool, seconds=60)


def test_request_metadata_cannot_lift_the_team_restriction(rig: Rig) -> None:
    _, key = rig.team_key("no_content", "a")
    cursors: Final = rig.cursors()
    sent: Final = _served(rig.send(key, extra={"metadata": {"capture_message_content": "span_only"}}))
    assert rig.upstream_hits(sent.marker) == 1
    _assert_redacted_twin(rig, sent, cursors, rig.sinks.tenant, cursors.tenant)


def test_stalled_tenant_during_a_mixed_burst_keeps_serving_and_each_teams_policy(rig: Rig) -> None:
    _, restricted = rig.team_key("no_content", "a")
    _, open_key = rig.team_key("span_only", "b")
    shapes: Final[tuple[tuple[Endpoint, bool], ...]] = (
        ("chat", False),
        ("chat", True),
        ("responses", False),
        ("messages", True),
    )
    plan: Final = [(restricted if index % 2 else open_key, *shapes[index % len(shapes)]) for index in range(30)]
    cursors: Final = rig.cursors()
    configure_sink(rig.sinks.tenant, paused=True)
    try:
        with ThreadPoolExecutor(max_workers=10) as pool:
            sent: Final = list(pool.map(lambda job: rig.send(job[0], job[1], job[2]), plan))
    finally:
        configure_sink(rig.sinks.tenant, paused=False)
    assert [result.status for result in sent] == [200] * 30, [result.text for result in sent if result.status != 200]
    for (key, _, _), result in zip(plan, sent, strict=True):
        roots = [
            span
            for span in _with_content(rig.sinks.operator, cursors.operator, result)
            if span["kind"] == SERVER and span["name"].startswith("POST ")
        ]
        assert len(roots) == 1, roots
        if key == restricted:
            _assert_redacted_twin(rig, result, cursors, rig.sinks.tenant, cursors.tenant)
        else:
            _assert_content_twin(rig, result, cursors)


def _write_config(directory: Path, sinks: SpanSinks, name: str, callbacks: tuple[str, ...]) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"] = {
        **config.get("litellm_settings", {}),
        "callbacks": list(callbacks),
        "otel_tenant_destination_mode": "additive",
        "provider_url_destination_allowed_hosts": [
            urlparse(sinks.tenant).netloc,
            urlparse(sinks.operator).netloc,
            urlparse(sinks.arize).netloc,
        ],
    }
    config["general_settings"] = {**config.get("general_settings", {}), "user_api_key_cache_ttl": 2}
    path: Final = directory / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@contextmanager
def _started(
    provider: Wire,
    sinks: SpanSinks,
    newrelic: ConnectSink,
    directory: Path,
    capture: str | None,
    callbacks: tuple[str, ...] = ("langfuse_otel",),
) -> Generator[Rig]:
    environment: Final = {
        "LITELLM_OTEL_V2": "1",
        "OTEL_BSP_SCHEDULE_DELAY": "300",
        "MAPPER_NAMES": "genai,openinference",
        "LANGFUSE_HOST": sinks.operator,
        "LANGFUSE_PUBLIC_KEY": "pk-lf-operator",
        "LANGFUSE_SECRET_KEY": "sk-lf-operator",
        "ARIZE_SPACE_KEY": "space-operator",
        "ARIZE_API_KEY": "arize-operator",
        "ARIZE_HTTP_ENDPOINT": sinks.arize + "/v1/traces",
        "WANDB_API_KEY": "wandb-operator",
        "WANDB_PROJECT_ID": "operator/weave",
        "WANDB_HOST": sinks.arize,
        "SIGNOZ_INGESTION_ENDPOINT": sinks.arize + "/v1/traces",
        "SIGNOZ_INGESTION_KEY": "signoz-operator",
        "HTTPS_PROXY": newrelic.proxy_url,
        "NO_PROXY": "127.0.0.1,localhost",
        "REQUESTS_CA_BUNDLE": newrelic.ca_pem,
        **({} if capture is None else {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": capture}),
    }
    with (
        gateway_from_environment() as gateway,
        owned_proxy_process(
            gateway,
            directory,
            environment,
            config=_write_config(directory, sinks, "capture", callbacks),
            remove_environment=(
                "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
                "LITELLM_OTEL_TENANT_DESTINATION_MODE",
            ),
            workers=2,
        ) as owned,
        owned.gateway.scenario() as scenario,
    ):
        yield Rig(
            owned.gateway, owned, scenario, scenario.model(api_base=provider.url + "/v1"), provider, sinks, newrelic
        )


@pytest.fixture(scope="module")
def provider() -> Iterator[Wire]:
    with wire_server(_upstream) as wire:
        yield wire


@pytest.fixture(scope="module")
def newrelic_sink(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ConnectSink]:
    with owned_connect_sink(tmp_path_factory.mktemp("newrelic-sink")) as sink:
        yield sink


@pytest.fixture(scope="module")
def rig(
    provider: Wire, audit_sinks: SpanSinks, newrelic_sink: ConnectSink, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("capture-span-only")
    with _started(provider, audit_sinks, newrelic_sink, directory, "span_only") as started:
        yield started


@pytest.fixture(scope="module")
def dark_rig(
    provider: Wire, audit_sinks: SpanSinks, newrelic_sink: ConnectSink, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Rig]:
    with _started(
        provider,
        audit_sinks,
        newrelic_sink,
        tmp_path_factory.mktemp("capture-no-content"),
        None,
        ("langfuse_otel", "signoz"),
    ) as started:
        yield started
