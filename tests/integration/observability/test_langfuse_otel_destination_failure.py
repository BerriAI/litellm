"""The operator's collector records a failed request the same way whether or not the key or team has its own
``langfuse_otel`` destination, and no key or team destination receives the operator's stack trace, when ``otel`` and
``langfuse_otel`` are both listed and the operator has no Langfuse keys"""

from __future__ import annotations

import json
import uuid
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Final, Literal

import httpx
import pytest
from _langfuse_otel_metadata_support import (
    _chat_body,
    _message_body,
    _response_body,
    _sse_chat,
    _sse_messages,
    _sse_responses,
)
from integration._support.client import Gateway, eventually, gateway_from_environment, held
from integration._support.otlp_sink import (
    AuditConfigWriter,
    Span,
    SpanSinks,
    configure_sink,
    langfuse_callback_variables,
    recorded_requests,
    recorded_spans,
    spans_for_trace,
)
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

SERVER_SPAN_KIND: Final = 2
FAILURE_SPAN: Final = "Failed Proxy Server Request"
STACK_TRACE: Final = "error.stack_trace"
ERROR_ATTRIBUTES: Final = ("error.type", "error.code", "http.response.status_code", "metadata.user_api_key_team_id")
FAILURE_STATUS: Final = {"upstream_500": 500, "unknown_model": 400}
ENDPOINTS: Final = ("/v1/chat/completions", "/v1/messages", "/v1/responses")
Failure = Literal["upstream_500", "unknown_model"]
SCRIPTED_REPLIES: Final = {
    "/chat/completions": (_chat_body, _sse_chat),
    "/messages": (_message_body, _sse_messages),
    "/responses": (_response_body, _sse_responses),
}


@dataclass(frozen=True, slots=True)
class Rig:
    owned: OwnedProxy
    failing_provider: Wire
    working_provider: Wire

    @property
    def proxy(self) -> Gateway:
        return self.owned.gateway


def _upstream_error(_request: Request) -> Reply:
    return Reply(
        status=500, body=json.dumps({"error": {"message": "upstream failed", "type": "server_error"}}).encode()
    )


def _upstream_success(request: Request) -> Reply:
    replies: Final = next((pair for suffix, pair in SCRIPTED_REPLIES.items() if request.target.endswith(suffix)), None)
    if replies is None:
        return Reply(status=404)
    response_id: Final = f"ok-{uuid.uuid4().hex}"
    if json.loads(request.body or b"{}").get("stream"):
        return Reply(content_type="text/event-stream", chunks=replies[1](response_id))
    return Reply(body=json.dumps(replies[0](response_id)).encode())


@pytest.fixture(scope="module")
def rig(
    audit_sinks: SpanSinks, otel_audit_config: AuditConfigWriter, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("langfuse-destination-failure")
    with (
        wire_server(_upstream_error) as failing_provider,
        wire_server(_upstream_success) as working_provider,
        gateway_from_environment() as gateway,
        owned_proxy_process(
            gateway,
            directory,
            {
                "LITELLM_OTEL_V2": "1",
                "OTEL_BSP_SCHEDULE_DELAY": "300",
                "OTEL_EXPORTER": "otlp_http",
                "OTEL_ENDPOINT": audit_sinks.operator,
                "ARIZE_HTTP_ENDPOINT": audit_sinks.arize + "/v1/traces",
            },
            config=otel_audit_config(directory, {"callbacks": ["otel", "langfuse_otel"]}),
            remove_environment=("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"),
            workers=2,
        ) as owned,
    ):
        yield Rig(owned, failing_provider, working_provider)


def _langfuse_destination(sinks: SpanSinks, callback_type: str = "success") -> list[JsonValue]:
    return [
        {
            "callback_name": "langfuse_otel",
            "callback_type": callback_type,
            "callback_vars": langfuse_callback_variables(sinks),
        }
    ]


def _trace(url: str, since: int, trace_id: str) -> tuple[Span, ...]:
    return spans_for_trace(recorded_spans(url, since)[1], trace_id)


def _has_server_span(spans: tuple[Span, ...]) -> bool:
    return any(span["kind"] == SERVER_SPAN_KIND for span in spans)


def _has_failure_record(spans: tuple[Span, ...]) -> bool:
    return _has_server_span(spans) and any(span["name"] == FAILURE_SPAN for span in spans)


def _body(path: str, model: str, marker: str, stream: bool) -> dict[str, JsonValue]:
    if path == "/v1/responses":
        return {"model": model, "input": marker, "stream": stream}
    messages: Final[list[JsonValue]] = [{"role": "user", "content": marker}]
    if path == "/v1/messages":
        return {"model": model, "messages": messages, "max_tokens": 16, "stream": stream}
    return {"model": model, "messages": messages, "stream": stream}


def _send(proxy: Gateway, key: str, model: str, path: str, stream: bool) -> tuple[str, httpx.Response]:
    trace_id: Final = uuid.uuid4().hex
    response: Final = proxy.request(
        "POST",
        path,
        _body(path, model, f"failure-{trace_id}", stream),
        key=key,
        headers={"traceparent": f"00-{trace_id}-{uuid.uuid4().hex[:16]}-01"},
    )
    return trace_id, response


def _expect_status(
    proxy: Gateway, key: str, model: str, status: int, path: str = "/v1/chat/completions", stream: bool = False
) -> str:
    trace_id, response = _send(proxy, key, model, path, stream)
    assert response.status_code == status, response.text
    return trace_id


def _failing_model(scenario_model: Callable[..., str], failure: Failure, provider: Wire) -> str:
    return (
        scenario_model(api_base=provider.url + "/v1")
        if failure == "upstream_500"
        else f"not-deployed-{uuid.uuid4().hex}"
    )


def _operator_record(sinks: SpanSinks, since: int, trace_id: str) -> tuple[Mapping[str, JsonValue], tuple[str, ...]]:
    held(lambda: _trace(sinks.operator, since, trace_id), _has_failure_record, holding=3, seconds=40)
    spans: Final = _trace(sinks.operator, since, trace_id)
    server: Final = next(span for span in spans if span["kind"] == SERVER_SPAN_KIND)
    failures: Final = tuple(str(span["name"]) for span in spans if span["name"] == FAILURE_SPAN)
    return {
        **{name: server["attributes"].get(name) for name in ERROR_ATTRIBUTES},
        STACK_TRACE: STACK_TRACE in server["attributes"],
    }, failures


def _has_failed_call(spans: tuple[Span, ...]) -> bool:
    return any("error.type" in span["attributes"] for span in spans)


def _assert_operator_reads_like_plain(
    sinks: SpanSinks, since: int, traces: tuple[str, str], teams: tuple[str, str], failure: Failure
) -> Mapping[str, JsonValue]:
    plain: Final = _operator_record(sinks, since, traces[0])
    destination: Final = _operator_record(sinks, since, traces[1])

    assert plain[1] == (FAILURE_SPAN,), plain
    assert None not in plain[0].values(), plain[0]
    assert plain[0]["metadata.user_api_key_team_id"] == teams[0], plain[0]
    assert plain[0][STACK_TRACE] is (failure == "upstream_500"), plain[0]
    assert destination[1] == plain[1], (destination, plain)
    assert {**destination[0], "metadata.user_api_key_team_id": None} == {
        **plain[0],
        "metadata.user_api_key_team_id": None,
    }, (destination[0], plain[0])
    assert destination[0]["metadata.user_api_key_team_id"] == teams[1], destination[0]
    return destination[0]


def _assert_destination_reads_like_plain(
    sinks: SpanSinks,
    since: tuple[int, int],
    traces: tuple[str, str],
    teams: tuple[str, str],
    failure: Failure,
) -> None:
    _assert_operator_reads_like_plain(sinks, since[0], traces, teams, failure)
    held(lambda: _trace(sinks.tenant, since[1], traces[1]), _has_server_span, holding=3, seconds=40)
    tenant: Final = _trace(sinks.tenant, since[1], traces[1])

    assert FAILURE_SPAN not in {span["name"] for span in tenant}, [span["name"] for span in tenant]
    assert [span["name"] for span in tenant if STACK_TRACE in span["attributes"]] == [], [
        span["name"] for span in tenant
    ]


@pytest.mark.parametrize("failure", ["upstream_500", "unknown_model"])
@pytest.mark.parametrize("callback_type", ["success", "success_and_failure"])
@pytest.mark.parametrize("level", ["key", "team"])
def test_the_operator_collector_records_a_destination_requests_failure_like_any_other(
    level: Literal["key", "team"],
    callback_type: str,
    failure: Failure,
    rig: Rig,
    audit_sinks: SpanSinks,
) -> None:
    logging: Final = _langfuse_destination(audit_sinks, callback_type)
    with rig.proxy.scenario() as scenario:
        model: Final = _failing_model(scenario.model, failure, rig.failing_provider)
        alias: Final = f"lf-destination-failure-{uuid.uuid4().hex}"
        plain_team: Final = scenario.team()
        destination_team: Final = scenario.team(metadata={"logging": logging} if level == "team" else {})
        plain_key: Final = scenario.key(key_alias=f"{alias}-plain", team_id=plain_team)
        destination_key: Final = scenario.key(
            key_alias=alias,
            team_id=destination_team,
            **({"metadata": {"logging": logging}} if level == "key" else {}),
        )
        since: Final = (recorded_spans(audit_sinks.operator)[0], recorded_spans(audit_sinks.tenant)[0])

        traces: Final = (
            _expect_status(rig.proxy, plain_key, model, FAILURE_STATUS[failure]),
            _expect_status(rig.proxy, destination_key, model, FAILURE_STATUS[failure]),
        )
        _assert_destination_reads_like_plain(audit_sinks, since, traces, (plain_team, destination_team), failure)


@pytest.mark.parametrize("failure", ["upstream_500", "unknown_model"])
@pytest.mark.parametrize("level", ["key", "team"])
def test_a_failure_only_destination_gets_the_failed_call_and_the_operator_record_stays_plain(
    level: Literal["key", "team"], failure: Failure, rig: Rig, audit_sinks: SpanSinks
) -> None:
    logging: Final = _langfuse_destination(audit_sinks, "failure")
    with rig.proxy.scenario() as scenario:
        model: Final = _failing_model(scenario.model, failure, rig.failing_provider)
        plain_team: Final = scenario.team()
        destination_team: Final = scenario.team(metadata={"logging": logging} if level == "team" else {})
        plain_key: Final = scenario.key(team_id=plain_team)
        destination_key: Final = scenario.key(
            team_id=destination_team, **({"metadata": {"logging": logging}} if level == "key" else {})
        )
        since: Final = (recorded_spans(audit_sinks.operator)[0], recorded_spans(audit_sinks.tenant)[0])

        traces: Final = (
            _expect_status(rig.proxy, plain_key, model, FAILURE_STATUS[failure]),
            _expect_status(rig.proxy, destination_key, model, FAILURE_STATUS[failure]),
        )
        destination: Final = _assert_operator_reads_like_plain(
            audit_sinks, since[0], traces, (plain_team, destination_team), failure
        )
        held(lambda: _trace(audit_sinks.tenant, since[1], traces[1]), _has_failed_call, holding=3, seconds=40)
        tenant: Final = _trace(audit_sinks.tenant, since[1], traces[1])

    assert FAILURE_SPAN not in {span["name"] for span in tenant}, [span["name"] for span in tenant]
    assert {str(span["attributes"]["error.type"]) for span in tenant if "error.type" in span["attributes"]} == {
        str(destination["error.type"])
    }, tenant


@pytest.mark.parametrize("failure", ["upstream_500", "unknown_model"])
@pytest.mark.parametrize("stream", [False, True], ids=["unary", "stream"])
@pytest.mark.parametrize("path", ENDPOINTS)
def test_every_endpoint_records_a_destination_requests_failure_like_any_other(
    path: str,
    stream: bool,
    failure: Failure,
    rig: Rig,
    audit_sinks: SpanSinks,
) -> None:
    with rig.proxy.scenario() as scenario:
        model: Final = _failing_model(scenario.model, failure, rig.failing_provider)
        plain_team: Final = scenario.team()
        destination_team: Final = scenario.team()
        plain_key: Final = scenario.key(team_id=plain_team)
        destination_key: Final = scenario.key(
            team_id=destination_team, metadata={"logging": _langfuse_destination(audit_sinks)}
        )
        since: Final = (recorded_spans(audit_sinks.operator)[0], recorded_spans(audit_sinks.tenant)[0])

        traces: Final = (
            _expect_status(rig.proxy, plain_key, model, FAILURE_STATUS[failure], path, stream),
            _expect_status(rig.proxy, destination_key, model, FAILURE_STATUS[failure], path, stream),
        )
        _assert_destination_reads_like_plain(audit_sinks, since, traces, (plain_team, destination_team), failure)


@pytest.mark.parametrize("stream", [False, True], ids=["unary", "stream"])
@pytest.mark.parametrize("path", ENDPOINTS)
def test_a_successful_destination_request_records_no_failure(
    path: str, stream: bool, rig: Rig, audit_sinks: SpanSinks
) -> None:
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=rig.working_provider.url + "/v1")
        key: Final = scenario.key(metadata={"logging": _langfuse_destination(audit_sinks, "success_and_failure")})
        since: Final = (recorded_spans(audit_sinks.operator)[0], recorded_spans(audit_sinks.tenant)[0])

        trace_id, response = _send(rig.proxy, key, model, path, stream)
        assert response.status_code == 200, response.text
        assert "langfuse-echo" in response.text, response.text
        held(lambda: _trace(audit_sinks.operator, since[0], trace_id), _has_server_span, holding=3, seconds=40)
        held(lambda: _trace(audit_sinks.tenant, since[1], trace_id), _has_server_span, holding=3, seconds=40)
        operator: Final = _trace(audit_sinks.operator, since[0], trace_id)
        tenant: Final = _trace(audit_sinks.tenant, since[1], trace_id)

    server: Final = next(span for span in operator if span["kind"] == SERVER_SPAN_KIND)
    assert "error.type" not in server["attributes"], server["attributes"]
    assert FAILURE_SPAN not in {span["name"] for span in (*operator, *tenant)}, [
        span["name"] for span in (*operator, *tenant)
    ]


@pytest.mark.parametrize("failure", ["upstream_500", "unknown_model"])
def test_an_arize_destination_gets_the_failure_without_the_operators_stack_trace(
    failure: Failure,
    rig: Rig,
    audit_sinks: SpanSinks,
) -> None:
    logging: Final[list[JsonValue]] = [
        {
            "callback_name": "arize",
            "callback_type": "success",
            "callback_vars": {"arize_api_key": "arize-test-key", "arize_space_id": "arize-test-space"},
        }
    ]
    with rig.proxy.scenario() as scenario:
        model: Final = _failing_model(scenario.model, failure, rig.failing_provider)
        key: Final = scenario.key(
            key_alias=f"arize-destination-failure-{uuid.uuid4().hex}", metadata={"logging": logging}
        )
        since: Final = recorded_spans(audit_sinks.arize)[0]

        trace_id: Final = _expect_status(rig.proxy, key, model, FAILURE_STATUS[failure])
        held(lambda: _trace(audit_sinks.arize, since, trace_id), _has_server_span, holding=3, seconds=40)
        spans: Final = _trace(audit_sinks.arize, since, trace_id)

    server: Final = next(span for span in spans if span["kind"] == SERVER_SPAN_KIND)
    assert "error.type" in server["attributes"], server["attributes"]
    assert [span["name"] for span in spans if STACK_TRACE in span["attributes"]] == [], [span["name"] for span in spans]


def test_a_destination_outage_and_a_worker_kill_mid_burst_keep_one_failure_record_per_request(
    rig: Rig, audit_sinks: SpanSinks
) -> None:
    proxy: Final = rig.proxy
    with proxy.scenario() as scenario:
        failing_model: Final = scenario.model(api_base=rig.failing_provider.url + "/v1")
        teams: Final = (scenario.team(), scenario.team())
        keys: Final = (
            scenario.key(team_id=teams[0]),
            scenario.key(team_id=teams[1], metadata={"logging": _langfuse_destination(audit_sinks)}),
        )
        since: Final = (recorded_spans(audit_sinks.operator)[0], recorded_spans(audit_sinks.tenant)[0])

        def call(index: int) -> tuple[str, int, int | None]:
            failure: Final[Failure] = "upstream_500" if index % 4 < 2 else "unknown_model"
            model: Final = failing_model if failure == "upstream_500" else f"not-deployed-{uuid.uuid4().hex}"
            side: Final = index % 2
            try:
                trace_id, response = _send(proxy, keys[side], model, ENDPOINTS[index % 3], index % 6 < 3)
            except httpx.TransportError:
                return "", side, None
            assert response.status_code == FAILURE_STATUS[failure], response.text
            return trace_id, side, response.status_code

        exports_before_outage: Final = len(recorded_requests(audit_sinks.tenant))
        configure_sink(audit_sinks.tenant, status=503)
        try:
            with ThreadPoolExecutor(max_workers=12) as pool:
                outage: Final = tuple(pool.map(call, range(24)))
            eventually(
                lambda: len(recorded_requests(audit_sinks.tenant)),
                lambda exports: exports > exports_before_outage,
                seconds=30,
            )
        finally:
            configure_sink(audit_sinks.tenant, status=200)
        assert all(status is not None for _, _, status in outage), outage
        eventually(
            lambda: recorded_spans(audit_sinks.operator, since[0])[1],
            lambda spans: all(_has_failure_record(spans_for_trace(spans, trace)) for trace, _, _ in outage),
            seconds=60,
        )
        victim: Final = min(rig.owned.worker_pids())
        configure_sink(audit_sinks.tenant, delay_seconds=1)
        try:
            with ThreadPoolExecutor(max_workers=12) as pool:
                futures: Final = tuple(pool.submit(call, 24 + index) for index in range(12))
                while sum(future.done() for future in futures) < 3:
                    wait(futures, timeout=0.05)
                replacement: Final = rig.owned.kill_worker(victim)
                killed: Final = tuple(future.result() for future in futures)
        finally:
            configure_sink(audit_sinks.tenant, delay_seconds=0)
        after: Final = tuple(call(36 + index) for index in range(6))
        settled: Final = (*outage, *after)

        held(
            lambda: recorded_spans(audit_sinks.operator, since[0])[1],
            lambda spans: all(_has_failure_record(spans_for_trace(spans, trace)) for trace, _, _ in settled),
            holding=3,
            seconds=60,
        )
        held(
            lambda: recorded_spans(audit_sinks.tenant, since[1])[1],
            lambda spans: all(_has_server_span(spans_for_trace(spans, trace)) for trace, side, _ in after if side),
            holding=3,
            seconds=60,
        )
        operator: Final = recorded_spans(audit_sinks.operator, since[0])[1]
        tenant: Final = recorded_spans(audit_sinks.tenant, since[1])[1]

    assert replacement != victim
    answered: Final = tuple(entry for entry in killed if entry[2] is not None)
    assert answered, killed
    every: Final = (*settled, *answered)
    server_spans: Final = Counter(span["trace_id"] for span in operator if span["kind"] == SERVER_SPAN_KIND)
    failure_spans: Final = Counter(span["trace_id"] for span in operator if span["name"] == FAILURE_SPAN)
    assert [trace for trace, _, _ in settled if (server_spans[trace], failure_spans[trace]) != (1, 1)] == [], (
        server_spans,
        failure_spans,
    )
    assert [trace for trace, _, _ in answered if max(server_spans[trace], failure_spans[trace]) > 1] == []
    for trace, side, _ in settled:
        server = next(span for span in operator if span["trace_id"] == trace and span["kind"] == SERVER_SPAN_KIND)
        assert server["attributes"].get("metadata.user_api_key_team_id") == teams[side], server["attributes"]
        assert None not in (server["attributes"].get(name) for name in ERROR_ATTRIBUTES), server["attributes"]
    traces: Final = {trace for trace, _, _ in every}
    leaked: Final = [
        span["name"]
        for span in tenant
        if span["trace_id"] in traces and (span["name"] == FAILURE_SPAN or STACK_TRACE in span["attributes"])
    ]
    assert leaked == [], leaked
    tenant_servers: Final = Counter(span["trace_id"] for span in tenant if span["kind"] == SERVER_SPAN_KIND)
    assert [trace for trace, side, _ in after if side and tenant_servers[trace] != 1] == [], tenant_servers
    assert [trace for trace, side, _ in every if not side and tenant_servers[trace]] == [], tenant_servers
