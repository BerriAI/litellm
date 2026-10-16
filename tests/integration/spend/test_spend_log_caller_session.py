import itertools
import json
import re
import threading
import uuid
from urllib.parse import urlsplit
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final

import anthropic
import httpx
import openai
import psutil
import pytest
import yaml
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    list_value,
    object_value,
    string_value,
)
from integration._support.budget import readiness_seconds
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(2 * readiness_seconds(2) + 2 * graceful_stop_seconds() + 240)

MARKER: Final = re.compile(rb"callersess-[0-9a-f]{32}")
CALLER_SESSION_KEY: Final = "litellm_caller_session_id"
PATHS: Final = MappingProxyType(
    {"chat": "/v1/chat/completions", "messages": "/v1/messages", "responses": "/v1/responses"}
)
ENDPOINT_STREAMS: Final = tuple(itertools.product(PATHS, (False, True)))
Headers = Mapping[str, str] | Sequence[tuple[str, str]]


def _marker() -> str:
    return "callersess-" + uuid.uuid4().hex


def _token(label: str) -> str:
    return f"{label}-{uuid.uuid4().hex}"


def _frame(payload: Mapping[str, JsonValue]) -> bytes:
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


def _chat_reply(text: str, stream: bool) -> Reply:
    identity: Final = "chatcmpl-" + uuid.uuid4().hex
    usage: Final[dict[str, JsonValue]] = {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}
    if not stream:
        return Reply(
            body=json.dumps(
                {
                    "id": identity,
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
                    ],
                    "usage": usage,
                }
            ).encode()
        )
    chunk: Final[dict[str, JsonValue]] = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
    }
    return Reply(
        content_type="text/event-stream",
        chunks=(
            _frame({**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}}]}),
            _frame({**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage}),
            b"data: [DONE]\n\n",
        ),
    )


def _upstream(request: Request) -> Reply:
    markers: Final = tuple(found.group(0).decode() for found in MARKER.finditer(request.body))
    if request.method != "POST" or not request.target.endswith("/chat/completions") or not markers:
        return Reply(status=404, body=b'{"error":"unexpected request"}')
    return _chat_reply(f"answer {markers[-1]}", JSON_OBJECT.validate_json(request.body).get("stream") is True)


def _user_markers(body: Mapping[str, JsonValue]) -> tuple[str, ...]:
    user_contents: Final = tuple(
        json.dumps(object_value(message)["content"])
        for message in list_value(body.get("messages", []))
        if object_value(message)["role"] == "user"
    )
    return tuple(found.group(0).decode() for content in user_contents if (found := MARKER.search(content.encode())))


def _spans(batch: Request) -> Iterator[dict[str, JsonValue]]:
    for resource in list_value(JSON_OBJECT.validate_json(batch.body)["resourceSpans"]):
        for scope in list_value(object_value(resource)["scopeSpans"]):
            yield from (object_value(span) for span in list_value(object_value(scope)["spans"]))


def _attributes(span: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        string_value(object_value(attribute)["key"]): object_value(attribute)["value"]
        for attribute in list_value(span.get("attributes", []))
    }


@dataclass(frozen=True, slots=True)
class Collector:
    wire: Wire
    accepted: deque[Request]
    guard: threading.Lock

    def span_attributes(self, response_id: str) -> tuple[dict[str, JsonValue], ...]:
        with self.guard:
            batches: Final = tuple(self.accepted)
        attributes: Final = tuple(_attributes(span) for batch in batches for span in _spans(batch))
        return tuple(
            found
            for found in attributes
            if object_value(found.get("gen_ai.response.id", {})).get("stringValue") == response_id
        )

    def conversation_ids(self, conversations: frozenset[str]) -> tuple[str, ...]:
        with self.guard:
            batches: Final = tuple(self.accepted)
        responded: Final = (
            object_value(found.get("gen_ai.conversation.id", {})).get("stringValue")
            for found in (_attributes(span) for batch in batches for span in _spans(batch))
            if "gen_ai.response.id" in found
        )
        return tuple(
            conversation
            for conversation in responded
            if isinstance(conversation, str) and conversation in conversations
        )


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    scenario: Scenario
    model: str
    upstream: Wire
    process: OwnedProxy

    def body(self, endpoint: str, marker: str, extra: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        match endpoint:
            case "chat":
                return {"model": self.model, "messages": [{"role": "user", "content": marker}], **extra}
            case "messages":
                return {
                    "model": self.model,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": marker}],
                    **extra,
                }
            case _:
                return {"model": self.model, "input": marker, **extra}

    def post(
        self,
        endpoint: str,
        marker: str,
        fields: Mapping[str, JsonValue] = MappingProxyType({}),
        *,
        stream: bool = False,
        headers: Headers = (),
        key: str | None = None,
    ) -> httpx.Response:
        payload: Final = self.body(endpoint, marker, fields) | ({"stream": True} if stream else {})
        request_headers: Final = (
            ("Authorization", f"Bearer {self.proxy.key if key is None else key}"),
            *(headers.items() if isinstance(headers, Mapping) else headers),
        )
        with self.proxy.client.stream("POST", PATHS[endpoint], json=payload, headers=request_headers) as response:
            response.read()
        return response

    def send(
        self,
        endpoint: str,
        marker: str,
        fields: Mapping[str, JsonValue] = MappingProxyType({}),
        *,
        stream: bool = False,
        headers: Headers = (),
        key: str | None = None,
    ) -> httpx.Response:
        response: Final = self.post(endpoint, marker, fields, stream=stream, headers=headers, key=key)
        assert response.status_code == 200, response.text
        return response

    def upstream_bodies(self, marker: str) -> tuple[dict[str, JsonValue], ...]:
        bodies: Final = tuple(JSON_OBJECT.validate_json(request.body) for request in self.upstream.drain())
        return tuple(body for body in bodies if _user_markers(body)[-1:] == (marker,))


@dataclass(frozen=True, slots=True)
class Rigs:
    flag_off: Rig
    flag_on: Rig
    sink: Collector


def _spend_rows(call_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT session_id, request_id, metadata::text AS metadata_text FROM "LiteLLM_SpendLogs" '
        "WHERE litellm_call_id=%s",
        (call_id,),
    )


def _spend_row(response: httpx.Response) -> dict[str, JsonValue]:
    call_id: Final = response.headers["x-litellm-call-id"]
    rows: Final = eventually(lambda: _spend_rows(call_id), lambda values: len(values) >= 1, seconds=70)
    assert len(rows) == 1, rows
    return rows[0]


def _session(response: httpx.Response) -> JsonValue:
    return _spend_row(response)["session_id"]


def _model(scenario: Scenario, upstream: Wire) -> str:
    return scenario.model(
        model="hosted_vllm/gpt-4o-mini",
        api_base=f"{upstream.url}/v1",
        use_chat_completions_api=True,
        input_cost_per_token=0.001,
        output_cost_per_token=0.002,
    )


def _rig_config(
    directory: Path, name: str, litellm_settings: Mapping[str, JsonValue], extra: Mapping[str, JsonValue]
) -> Path:
    shared: Final = object_value(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    config: Final = {
        **shared,
        "general_settings": {**object_value(shared["general_settings"]), "store_prompts_in_spend_logs": True},
        "litellm_settings": {**object_value(shared["litellm_settings"]), **litellm_settings},
        **extra,
    }
    path: Final = directory / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture(scope="module")
def rigs(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rigs]:
    accepted: Final[deque[Request]] = deque()  # mutable-ok: sink thread appends each accepted OTLP batch
    guard: Final = threading.Lock()

    def sink(request: Request) -> Reply:
        with guard:
            accepted.append(request)
        return Reply()

    directory: Final = tmp_path_factory.mktemp("caller-session")
    with ExitStack() as stack:
        upstream: Final = stack.enter_context(wire_server(_upstream))
        collector: Final = Collector(stack.enter_context(wire_server(sink)), accepted, guard)
        shared: Final = stack.enter_context(gateway_from_environment())
        flag_off: Final = stack.enter_context(
            owned_proxy_process(shared, directory, {}, config=_rig_config(directory, "flag-off", {}, {}), workers=2)
        )
        flag_on: Final = stack.enter_context(
            owned_proxy_process(
                shared,
                directory,
                {"LITELLM_OTEL_V2": "1", "OTEL_BSP_SCHEDULE_DELAY": "300"},
                config=_rig_config(
                    directory,
                    "flag-on",
                    {"request_correlation_in_logs": True, "callbacks": ["otel"]},
                    {
                        "callback_settings": {
                            "otel": {"exporter": "http/json", "endpoint": collector.wire.url, "mapper_names": ["genai"]}
                        }
                    },
                ),
                workers=2,
            )
        )
        off_scenario: Final = stack.enter_context(flag_off.gateway.scenario())
        on_scenario: Final = stack.enter_context(flag_on.gateway.scenario())
        yield Rigs(
            Rig(flag_off.gateway, off_scenario, _model(off_scenario, upstream), upstream, flag_off),
            Rig(flag_on.gateway, on_scenario, _model(on_scenario, upstream), upstream, flag_on),
            collector,
        )


@pytest.mark.parametrize(("endpoint", "stream"), ENDPOINT_STREAMS)
def test_spend_log_session_is_the_body_session_beside_a_trace_header(rigs: Rigs, endpoint: str, stream: bool) -> None:
    marker: Final = _marker()
    session: Final = _token("convo")
    response: Final = rigs.flag_on.send(
        endpoint,
        marker,
        stream=stream,
        headers={"x-litellm-trace-id": _token("trace")},
        fields={"litellm_session_id": session},
    )

    row: Final = _spend_row(response)
    upstream: Final = rigs.flag_on.upstream_bodies(marker)

    assert row["session_id"] == session, row
    assert CALLER_SESSION_KEY not in string_value(row["metadata_text"]), row
    assert len(upstream) == 1, upstream
    assert CALLER_SESSION_KEY not in json.dumps(upstream[0]), upstream


def test_sdk_clients_record_the_body_session_beside_a_trace_header(rigs: Rigs) -> None:
    rig: Final = rigs.flag_on
    base_url: Final = str(rig.proxy.client.base_url)
    traces: Final = tuple(_token("trace") for _ in range(3))
    sessions: Final = tuple(_token("convo") for _ in range(3))
    markers: Final = tuple(_marker() for _ in range(3))
    chat_client: Final = openai.OpenAI(base_url=base_url + "/v1", api_key=rig.proxy.key, max_retries=0)
    messages_client: Final = anthropic.Anthropic(base_url=base_url, api_key=rig.proxy.key, max_retries=0)

    with chat_client.chat.completions.with_streaming_response.create(
        model=rig.model,
        messages=[{"role": "user", "content": markers[0]}],
        stream=True,
        extra_headers={"x-litellm-trace-id": traces[0]},
        extra_body={"litellm_session_id": sessions[0]},
    ) as chat_stream:
        chat_call: Final = chat_stream.headers["x-litellm-call-id"]
        chat_text: Final = "".join(
            chunk.choices[0].delta.content or "" for chunk in chat_stream.parse() if chunk.choices
        )
    message: Final = messages_client.messages.with_raw_response.create(
        model=rig.model,
        max_tokens=16,
        messages=[{"role": "user", "content": markers[1]}],
        extra_headers={"x-litellm-trace-id": traces[1]},
        extra_body={"litellm_session_id": sessions[1]},
    )
    with chat_client.responses.with_streaming_response.create(
        model=rig.model,
        input=markers[2],
        stream=True,
        extra_headers={"x-litellm-trace-id": traces[2]},
        extra_body={"litellm_session_id": sessions[2]},
    ) as responses_stream:
        responses_call: Final = responses_stream.headers["x-litellm-call-id"]
        completed: Final = tuple(event for event in responses_stream.parse() if event.type == "response.completed")

    first_block: Final = message.parse().content[0]
    assert chat_text == f"answer {markers[0]}", chat_text
    assert isinstance(first_block, anthropic.types.TextBlock) and first_block.text == f"answer {markers[1]}", message
    assert len(completed) == 1, completed
    calls: Final = (chat_call, message.headers["x-litellm-call-id"], responses_call)
    landed: Final = tuple(
        eventually(lambda call=call: _spend_rows(call), lambda values: len(values) == 1, seconds=70)[0]["session_id"]
        for call in calls
    )
    assert landed == sessions, (landed, calls)


def test_conversation_span_keeps_the_trace_header_and_no_internal_key(rigs: Rigs) -> None:
    trace: Final = _token("trace")
    response: Final = rigs.flag_on.send(
        "chat", _marker(), headers={"x-litellm-trace-id": trace}, fields={"litellm_session_id": _token("convo")}
    )
    response_id: Final = string_value(JSON_OBJECT.validate_json(response.content)["id"])

    spans: Final = eventually(lambda: rigs.sink.span_attributes(response_id), lambda found: len(found) == 1, seconds=30)

    assert spans[0]["gen_ai.conversation.id"] == {"stringValue": trace}, spans
    assert CALLER_SESSION_KEY not in json.dumps(spans[0]), spans


def test_a_continued_chains_conversation_spans_keep_each_turns_trace_header(rigs: Rigs) -> None:
    traces: Final = (_token("trace"), _token("trace"))
    first: Final = _turn(rigs.flag_on, _marker(), headers={"x-litellm-trace-id": traces[0]})
    _turn(
        rigs.flag_on,
        _marker(),
        first,
        headers={"x-litellm-trace-id": traces[1]},
        fields={"litellm_session_id": _token("convo")},
    )

    conversations: Final = eventually(
        lambda: rigs.sink.conversation_ids(frozenset(traces)), lambda found: len(found) == 2, seconds=30
    )

    assert sorted(conversations) == sorted(traces), conversations


@pytest.mark.parametrize("endpoint", list(PATHS))
def test_flag_off_keeps_the_trace_id_beside_a_body_session(rigs: Rigs, endpoint: str) -> None:
    trace: Final = _token("trace")

    response: Final = rigs.flag_off.send(
        endpoint, _marker(), headers={"x-litellm-trace-id": trace}, fields={"litellm_session_id": _token("convo")}
    )

    assert _session(response) == trace


@pytest.mark.parametrize("endpoint", list(PATHS))
def test_response_cache_hit_records_the_body_session_beside_a_trace_header(rigs: Rigs, endpoint: str) -> None:
    marker: Final = _marker()
    sessions: Final = (_token("convo"), _token("convo"))
    responses: Final = tuple(
        rigs.flag_on.send(
            endpoint, marker, headers={"x-litellm-trace-id": _token("trace")}, fields={"litellm_session_id": session}
        )
        for session in sessions
    )

    assert tuple(_session(response) for response in responses) == sessions
    assert len(rigs.flag_on.upstream_bodies(marker)) == 1


def test_metadata_session_beside_a_new_trace_per_turn_groups_both_turns(rigs: Rigs) -> None:
    session: Final = _token("convo")
    responses: Final = tuple(
        rigs.flag_on.send(
            "chat",
            _marker(),
            headers={"x-litellm-trace-id": _token("trace")},
            fields={"metadata": {"session_id": session}},
        )
        for _ in range(2)
    )

    assert tuple(_session(response) for response in responses) == (session, session)
    now: Final = datetime.now(timezone.utc)
    page: Final = rigs.flag_on.proxy.get(
        "/spend/logs/ui",
        params={
            "session_id": session,
            "start_date": (now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
            "end_date": (now + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    listed: Final = sorted(string_value(object_value(row)["request_id"]) for row in list_value(page["data"]))
    assert listed == sorted(
        string_value(JSON_OBJECT.validate_json(response.content)["id"]) for response in responses
    ), page


def _shape(shape: str, trace: str, session: str, header: str) -> tuple[Headers, dict[str, JsonValue], str]:
    traced: Final = {"x-litellm-trace-id": trace}
    match shape:
        case "session-header-wins":
            return {"x-litellm-session-id": header}, {"litellm_session_id": session}, header
        case "trace-header-only":
            return traced, {}, trace
        case "body-session-only":
            return {}, {"litellm_session_id": session}, session
        case "int-session":
            return traced, {"litellm_session_id": 7}, trace
        case "list-session":
            return traced, {"litellm_session_id": [session]}, trace
        case "empty-session":
            return traced, {"litellm_session_id": ""}, trace
        case "null-session":
            return traced, {"litellm_session_id": None}, trace
        case "nul-session":
            return traced, {"litellm_session_id": session + "\x00x"}, trace
        case "two-trace-headers":
            return (
                (("x-litellm-trace-id", trace), ("x-litellm-trace-id", header)),
                {"litellm_session_id": session},
                session,
            )
        case _:
            return traced, {"litellm_session_id": session + "s" * 5120}, session + "s" * 5120


@pytest.mark.parametrize(
    "shape",
    [
        "session-header-wins",
        "trace-header-only",
        "body-session-only",
        "int-session",
        "list-session",
        "empty-session",
        "null-session",
        "nul-session",
        "two-trace-headers",
        "5kb-session",
    ],
)
def test_spend_log_session_for_each_session_shape(rigs: Rigs, shape: str) -> None:
    headers, fields, expected = _shape(shape, _token("trace"), _token("convo"), _token("header"))
    neighbor_trace: Final = _token("neighbor")

    response: Final = rigs.flag_on.send("chat", _marker(), fields, headers=headers)
    neighbor: Final = rigs.flag_on.send("chat", _marker(), headers={"x-litellm-trace-id": neighbor_trace})

    assert _session(response) == expected
    assert _session(neighbor) == neighbor_trace


@pytest.mark.parametrize("endpoint", ["chat", "messages"])
def test_client_sent_internal_session_key_is_ignored(rigs: Rigs, endpoint: str) -> None:
    forged: Final = _token("forged")
    trace: Final = _token("trace")
    forged_metadata: Final[dict[str, JsonValue]] = {
        "metadata" if endpoint == "chat" else "litellm_metadata": {CALLER_SESSION_KEY: forged}
    }

    untraced: Final = rigs.flag_on.send(endpoint, _marker(), forged_metadata)
    traced: Final = rigs.flag_on.send(endpoint, _marker(), forged_metadata, headers={"x-litellm-trace-id": trace})

    assert _session(untraced) not in (forged, None)
    assert _session(traced) == trace


def test_identical_requests_each_log_once_with_the_body_session(rigs: Rigs) -> None:
    session: Final = _token("convo")
    trace: Final = _token("trace")
    marker: Final = _marker()

    responses: Final = tuple(
        rigs.flag_on.send("chat", marker, headers={"x-litellm-trace-id": trace}, fields={"litellm_session_id": session})
        for _ in range(3)
    )

    assert len({response.headers["x-litellm-call-id"] for response in responses}) == 3
    assert tuple(_session(response) for response in responses) == (session, session, session)


def _pass_through_route(rig: Rig, target: str) -> str:
    path: Final = f"/caller-session-{uuid.uuid4().hex}"
    created: Final = rig.proxy.post("/config/pass_through_endpoint", {"path": path, "target": target, "auth": True})
    endpoint_id: Final = string_value(object_value(list_value(created["endpoints"])[0])["id"])
    rig.scenario.cleanups.callback(
        lambda: rig.proxy.request("DELETE", "/config/pass_through_endpoint", params={"endpoint_id": endpoint_id})
    )
    return path


@pytest.mark.parametrize("field", ["litellm_session_id", "metadata"])
def test_pass_through_spend_log_session_is_the_body_session_beside_a_trace_header(rigs: Rigs, field: str) -> None:
    marker: Final = _marker()
    session: Final = _token("convo")
    rig: Final = rigs.flag_on
    path: Final = _pass_through_route(rig, f"{rig.upstream.url}/v1/chat/completions")
    key: Final = rig.scenario.key(allowed_passthrough_routes=[path])
    fields: Final[dict[str, JsonValue]] = (
        {"litellm_session_id": session} if field == "litellm_session_id" else {"metadata": {"session_id": session}}
    )

    response: Final = rig.proxy.request(
        "POST",
        path,
        rig.body("chat", marker, fields),
        key=key,
        headers={"x-litellm-trace-id": _token("trace")},
    )

    rows: Final = eventually(
        lambda: read_rows(
            'SELECT session_id FROM "LiteLLM_SpendLogs" WHERE api_key=%s', (sha256(key.encode()).hexdigest(),)
        ),
        lambda found: len(found) >= 1,
        seconds=70,
    )
    assert response.status_code == 200, response.text
    assert f"answer {marker}" in response.text
    assert len(rig.upstream_bodies(marker)) == 1
    assert rows == [{"session_id": session}]


def _turn(
    rig: Rig,
    marker: str,
    previous: str | None = None,
    *,
    key: str | None = None,
    headers: Headers = (),
    fields: Mapping[str, JsonValue] = MappingProxyType({}),
) -> str:
    chained: Final[dict[str, JsonValue]] = {"previous_response_id": previous} if previous else {}
    response: Final = rig.send("responses", marker, {**chained, **fields}, headers=headers, key=key)
    _spend_row(response)
    return string_value(JSON_OBJECT.validate_json(response.content)["id"])


def _turn_fields(shape: str, session: str) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    match shape:
        case "session-every-turn":
            return {"litellm_session_id": session}, {"litellm_session_id": session}
        case "session-from-turn-2":
            return {}, {"litellm_session_id": session}
        case _:
            return {"litellm_session_id": _token("convo")}, {"litellm_session_id": _token("convo")}


@pytest.mark.parametrize(
    ("correlated", "shape", "replays_turn_1"),
    [
        (True, "session-every-turn", True),
        (True, "session-from-turn-2", True),
        (True, "new-session-per-turn", True),
        (False, "session-from-turn-2", False),
    ],
)
def test_bridged_chain_history(rigs: Rigs, correlated: bool, shape: str, replays_turn_1: bool) -> None:
    rig: Final = rigs.flag_on if correlated else rigs.flag_off
    first_fields, second_fields = _turn_fields(shape, _token("convo"))
    markers: Final = (_marker(), _marker(), _marker())
    first: Final = _turn(rig, markers[0], headers={"x-litellm-trace-id": _token("trace")}, fields=first_fields)
    second: Final = _turn(rig, markers[1], first, headers={"x-litellm-trace-id": _token("trace")}, fields=second_fields)

    _turn(rig, markers[2], second)

    replayed: Final = rig.upstream_bodies(markers[2])
    assert len(replayed) == 1, replayed
    assert _user_markers(replayed[0]) == (markers if replays_turn_1 else markers[1:])


def test_bridged_chains_sharing_a_metadata_session_replay_each_others_turns(rigs: Rigs) -> None:
    shared: Final[dict[str, JsonValue]] = {"metadata": {"session_id": _token("convo")}}
    markers: Final = (_marker(), _marker(), _marker())
    first: Final = _turn(rigs.flag_on, markers[0], fields=shared)
    _turn(rigs.flag_on, markers[1], fields=shared)

    _turn(rigs.flag_on, markers[2], first)

    replayed: Final = rigs.flag_on.upstream_bodies(markers[2])
    assert len(replayed) == 1, replayed
    assert _user_markers(replayed[0]) == markers


@dataclass(frozen=True, slots=True)
class Owners:
    first: str
    second: str


def _owners(scenario: Scenario, model: str, shape: str) -> Owners:
    user: Final = _token("user")
    match shape:
        case "other-user":
            return Owners(
                scenario.key(user_id=user, models=[model]), scenario.key(user_id=_token("user"), models=[model])
            )
        case "same-user":
            return Owners(scenario.key(user_id=user, models=[model]), scenario.key(user_id=user, models=[model]))
        case "same-team":
            team: Final = scenario.team()
            return Owners(scenario.key(team_id=team, models=[model]), scenario.key(team_id=team, models=[model]))
        case _:
            return Owners(scenario.key(models=[model]), scenario.key(models=[model]))


@pytest.mark.parametrize(
    ("shape", "replays_first_owner"),
    [("other-user", False), ("same-user", True), ("same-team", True), ("ownerless-keys", False)],
)
def test_bridged_history_follows_the_previous_responses_owner(
    rigs: Rigs, shape: str, replays_first_owner: bool
) -> None:
    rig: Final = rigs.flag_off
    owners: Final = _owners(rig.scenario, rig.model, shape)
    session: Final = _token("convo")
    markers: Final = (_marker(), _marker(), _marker())
    _turn(rig, markers[0], key=owners.first, fields={"litellm_session_id": session})
    second: Final = _turn(rig, markers[1], key=owners.second, fields={"litellm_session_id": session})

    _turn(rig, markers[2], second, key=owners.second)

    replayed: Final = rig.upstream_bodies(markers[2])
    assert len(replayed) == 1, replayed
    assert _user_markers(replayed[0]) == (markers if replays_first_owner else markers[1:])


BURST: Final = ENDPOINT_STREAMS * 4


def test_concurrent_burst_across_endpoints_logs_each_session_once(rigs: Rigs) -> None:
    sessions: Final = tuple(_token("convo") for _ in BURST)

    def one(index: int) -> httpx.Response:
        endpoint, stream = BURST[index]
        return rigs.flag_on.send(
            endpoint,
            _marker(),
            stream=stream,
            headers={"x-litellm-trace-id": _token("trace")},
            fields={"litellm_session_id": sessions[index]},
        )

    with ThreadPoolExecutor(max_workers=12) as pool:
        responses: Final = tuple(pool.map(one, range(len(BURST))))

    assert tuple(_session(response) for response in responses) == sessions


def _open_upstream_connections(pid: int, upstream: Wire) -> int:
    port: Final = urlsplit(upstream.url).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


def _post_or_none(rig: Rig, session: str) -> httpx.Response | None:
    try:
        return rig.post(
            "chat", _marker(), headers={"x-litellm-trace-id": _token("trace")}, fields={"litellm_session_id": session}
        )
    except httpx.HTTPError:
        return None


def test_killing_one_worker_mid_burst_keeps_logging_the_body_session(rigs: Rigs) -> None:
    release: Final = threading.Event()
    arrived: Final[SimpleQueue[bytes]] = SimpleQueue()

    def held(request: Request) -> Reply:
        arrived.put(request.body)
        assert release.wait(timeout=60), "the burst was never released"
        return _upstream(request)

    process: Final = rigs.flag_on.process
    sessions: Final = tuple(_token("convo") for _ in range(20))
    after_session: Final = _token("convo")
    with wire_server(held) as wire:
        rig: Final = replace(rigs.flag_on, model=_model(rigs.flag_on.scenario, wire), upstream=wire)
        workers: Final = eventually(process.worker_pids, lambda found: len(found) == 2, seconds=30)
        with ThreadPoolExecutor(max_workers=len(sessions)) as pool:
            pending: Final = tuple(pool.submit(_post_or_none, rig, session) for session in sessions)
            eventually(arrived.qsize, lambda size: size == len(sessions), seconds=60)
            held_by: Final = MappingProxyType({pid: _open_upstream_connections(pid, wire) for pid in workers})
            victim, survivor = sorted(workers, key=held_by.__getitem__)
            with process.replacing_worker(victim) as kill:
                kill.freeze()
                kill()
                release.set()
                results: Final = tuple(future.result() for future in pending)
                after: Final = rig.send(
                    "chat",
                    _marker(),
                    headers={"x-litellm-trace-id": _token("trace")},
                    fields={"litellm_session_id": after_session},
                )

    served: Final = tuple(
        (response, session)
        for response, session in zip(results, sessions, strict=True)
        if response is not None and response.status_code == 200
    )
    assert sum(held_by.values()) == len(sessions), held_by
    assert len(served) == held_by[survivor] >= len(sessions) // 2, (held_by, len(served))
    assert _session(after) == after_session
    assert tuple(_session(response) for response, _ in served) == tuple(session for _, session in served)
