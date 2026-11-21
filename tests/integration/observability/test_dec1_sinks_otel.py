"""OpenTelemetry OTLP/HTTP sink contract for the v1 `otel` callback: span tree, attributes, resource and redaction.

The rig sets `callbacks: ["otel"]` with `OTEL_EXPORTER=otlp_http` and `OTEL_ENDPOINT` pointing at a scripted OTLP
collector that records protobuf `ExportTraceServiceRequest` bodies. Spans are exported by a BatchSpanProcessor
(`OTEL_BSP_SCHEDULE_DELAY=500`), never on the request path.

The contract (litellm/integrations/opentelemetry.py): every proxied call yields a root span named
"Received Proxy Server Request", a child "litellm_request" span carrying the `gen_ai.*`, `llm.*`, `metadata.*` and
`litellm.*` attributes, and, when content capture is on, a "raw_gen_ai_request" grandchild with the provider's raw
fields. A W3C `traceparent` from the caller makes the root span a child of the caller's span in the caller's trace.
The resource carries `service.name`, `deployment.environment` and `model_id`. `turn_off_message_logging` drops every
message attribute and the raw request span. A collector outage never fails or slows the caller. The Rust callback
port must emit the same tree and names, because trace backends key dashboards and alerts on them.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from typing import Final

import pytest
from integration._support.client import eventually
from integration._support.otlp_sink import Span
from integration.observability.dec1_callbacks_support import (
    CHAT,
    CHAT_REJECTED,
    PROVIDER_KEY,
    PROVIDER_REJECTION,
    REPLY_TEXT,
)
from integration.observability.dec1_sinks_support import (
    OTLP_TRACES_PATH,
    SinkRig,
    assert_not_held,
    attribute,
    chat,
    healthy_seconds,
    marker,
    new_call_id,
    sinks_rig,
    stream_chat,
    timed_chat,
)

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

ROOT: Final = "Received Proxy Server Request"
PRIMARY: Final = "litellm_request"
RAW: Final = "raw_gen_ai_request"
FAILED: Final = "Failed Proxy Server Request"


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[SinkRig]:
    with sinks_rig(
        tmp_path_factory.mktemp("dec1-sinks-otel"),
        {
            "callbacks": ["otel"],
            "cache": True,
            "cache_params": {"type": "redis", "host": "os.environ/REDIS_HOST", "port": "os.environ/REDIS_PORT"},
        },
        environment={"OTEL_RESOURCE_ATTRIBUTES": "dec1.owner=observability"},
    ) as created:
        yield created


@pytest.fixture(scope="module")
def dark(tmp_path_factory: pytest.TempPathFactory) -> Iterator[SinkRig]:
    with sinks_rig(
        tmp_path_factory.mktemp("dec1-sinks-otel-dark"),
        {"callbacks": ["otel"], "turn_off_message_logging": True},
    ) as created:
        yield created


def _named(spans: tuple[Span, ...], name: str) -> Span:
    found: Final = tuple(span for span in spans if span["name"] == name)
    assert len(found) == 1, [span["name"] for span in spans]
    return found[0]


def _primary(rig: SinkRig, call_id: str) -> Span:
    return _named(rig.otel_trace(call_id), PRIMARY)


def test_a_chat_exports_a_root_a_litellm_request_child_and_a_raw_request_grandchild(rig: SinkRig) -> None:
    """One chat must export, in one trace, a "Received Proxy Server Request" root with no parent, a "litellm_request"
    span whose parent is the root, and a "raw_gen_ai_request" span whose parent is "litellm_request"."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id, marker()).status_code == 200
    spans: Final = rig.otel_trace(call_id)
    root, primary, raw = _named(spans, ROOT), _named(spans, PRIMARY), _named(spans, RAW)
    assert root["parent_span_id"] == ""
    assert primary["parent_span_id"] == root["span_id"]
    assert raw["parent_span_id"] == primary["span_id"]


def test_spans_are_posted_as_otlp_protobuf_to_the_configured_traces_path(rig: SinkRig) -> None:
    """Every export must be a POST to the configured endpoint path /v1/traces with content-type
    application/x-protobuf."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id, marker()).status_code == 200
    rig.otel_trace(call_id)
    posts: Final = rig.peer("otlp").posts()
    assert posts
    assert {request.target.split("?")[0] for request in posts} == {OTLP_TRACES_PATH}
    assert {request.headers.get("content-type") for request in posts} == {"application/x-protobuf"}


def test_litellm_request_carries_the_gen_ai_request_and_usage_attributes(rig: SinkRig) -> None:
    """The "litellm_request" span of a chat with temperature 0.3 and max_tokens 50 must carry gen_ai.system
    "openai", llm.request.type "acompletion", gen_ai.request.temperature 0.3, gen_ai.request.max_tokens 50,
    gen_ai.usage.input_tokens 11, gen_ai.usage.output_tokens 4 and gen_ai.usage.total_tokens 15."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id, marker(), temperature=0.3, max_tokens=50).status_code == 200
    span: Final = _primary(rig, call_id)
    assert attribute(span, "gen_ai.system") == "openai"
    assert attribute(span, "llm.request.type") == "acompletion"
    assert attribute(span, "gen_ai.request.temperature") == 0.3
    assert attribute(span, "gen_ai.request.max_tokens") == 50
    assert attribute(span, "gen_ai.usage.input_tokens") == 11
    assert attribute(span, "gen_ai.usage.output_tokens") == 4
    assert attribute(span, "gen_ai.usage.total_tokens") == 15


def test_litellm_request_names_the_requested_and_response_model(rig: SinkRig) -> None:
    """gen_ai.request.model must be the deployment model the provider was asked for ("gpt-4o-mini"), while
    gen_ai.response.model is the client's model group alias "dec1-chat": the proxy restamps the response `model` to
    the name the client asked for (litellm/proxy/common_request_processing.py:4282-4288) and the span reads that
    field (litellm/integrations/opentelemetry.py:2650-2664), the same contract
    spend/test_stream_alias_billing.py pins for the non-streamed and streamed paths."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id, marker()).status_code == 200
    span: Final = _primary(rig, call_id)
    assert attribute(span, "gen_ai.request.model") == "gpt-4o-mini"
    assert attribute(span, "gen_ai.response.model") == CHAT


def test_litellm_request_ids_match_the_response_and_the_call_id_header(rig: SinkRig) -> None:
    """gen_ai.response.id must equal the `id` of the chat completion the caller received and litellm.call_id must
    equal the `x-litellm-call-id` the caller sent."""
    call_id: Final = new_call_id()
    response: Final = chat(rig, call_id, marker())
    span: Final = _primary(rig, call_id)
    assert attribute(span, "gen_ai.response.id") == response.json()["id"]
    assert attribute(span, "litellm.call_id") == call_id


def test_litellm_request_identifies_the_key_by_hash_and_alias_under_metadata(rig: SinkRig) -> None:
    """A call made with a key aliased "<alias>" must put `metadata.user_api_key_hash` (sha256 hex of the key) and
    `metadata.user_api_key_alias` "<alias>" on the "litellm_request" span, and never the key itself."""
    alias: Final = "dec1-otel-" + uuid.uuid4().hex[:8]
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key(key_alias=alias)
        call_id = new_call_id()
        assert chat(rig, call_id, marker(), key=key).status_code == 200
        span = _primary(rig, call_id)
        assert attribute(span, "metadata.user_api_key_hash") == sha256(key.encode()).hexdigest()
        assert attribute(span, "metadata.user_api_key_alias") == alias
        assert key not in json.dumps(span["attributes"])


def test_team_id_and_alias_are_stamped_on_every_span_of_the_trace(rig: SinkRig) -> None:
    """A call made with a key of team "<alias>" must carry `metadata.user_api_key_team_id` and
    `metadata.user_api_key_team_alias` on the root, the litellm_request and the raw request span alike
    (`_set_team_attributes_on_span`)."""
    team_alias: Final = "dec1-otel-team-" + uuid.uuid4().hex[:8]
    with rig.proxy.scenario() as scenario:
        team: Final = scenario.team(team_alias=team_alias)
        key: Final = scenario.key(team_id=team)
        call_id = new_call_id()
        assert chat(rig, call_id, marker(), key=key).status_code == 200
        spans = rig.otel_trace(call_id)
        for name in (ROOT, PRIMARY, RAW):
            span = _named(spans, name)
            assert attribute(span, "metadata.user_api_key_team_id") == team, name
            assert attribute(span, "metadata.user_api_key_team_alias") == team_alias, name


def test_input_messages_are_captured_by_default(rig: SinkRig) -> None:
    """With no redaction configured, "litellm_request" must carry gen_ai.input.messages containing the caller's
    user text, and gen_ai.operation.name "acompletion"."""
    call_id, text = new_call_id(), marker()
    assert chat(rig, call_id, text).status_code == 200
    span: Final = _primary(rig, call_id)
    assert text in str(attribute(span, "gen_ai.input.messages"))
    assert attribute(span, "gen_ai.operation.name") == "acompletion"


def test_raw_request_span_carries_provider_prefixed_raw_fields(rig: SinkRig) -> None:
    """The "raw_gen_ai_request" span must carry the provider's raw fields under `llm.openai.*`, for example the raw
    response's `llm.openai.id` equal to the response id. UNVERIFIED: which raw keys exist depends on
    `original_response` being a JSON string on the proxy path (set_raw_request_attributes)."""
    call_id: Final = new_call_id()
    response: Final = chat(rig, call_id, marker())
    raw: Final = _named(rig.otel_trace(call_id), RAW)
    assert any(name.startswith("llm.openai.") for name in raw["attributes"])
    assert attribute(raw, "llm.openai.id") == response.json()["id"]


def test_generation_name_renames_the_litellm_request_and_raw_spans(rig: SinkRig) -> None:
    """`metadata.generation_name` "<name>" must replace both "litellm_request" and "raw_gen_ai_request" as span
    names (`_get_span_name`, `_maybe_log_raw_request`), under the same root."""
    call_id, name = new_call_id(), marker()
    assert chat(rig, call_id, marker(), metadata={"generation_name": name}).status_code == 200
    spans: Final = rig.otel_trace(call_id)
    assert [span["name"] for span in spans].count(name) == 2
    assert all(span["name"] not in (PRIMARY, RAW) for span in spans)


def test_traceparent_header_joins_the_callers_trace_under_the_callers_span(rig: SinkRig) -> None:
    """A request with `traceparent: 00-<trace>-<span>-01` must export its root span in trace <trace> with parent
    span id <span>, so the proxy appears inside the caller's distributed trace."""
    trace_id, parent = uuid.uuid4().hex, uuid.uuid4().hex[:16]
    call_id: Final = new_call_id()
    headers: Final = {"traceparent": f"00-{trace_id}-{parent}-01"}
    assert chat(rig, call_id, marker(), headers=headers).status_code == 200
    spans: Final = rig.otel_trace(call_id)
    root: Final = _named(spans, ROOT)
    assert root["trace_id"] == trace_id
    assert root["parent_span_id"] == parent
    assert {span["trace_id"] for span in spans} == {trace_id}


def test_request_without_traceparent_starts_a_fresh_trace_per_call(rig: SinkRig) -> None:
    """Two calls without a traceparent must land in two different traces, each with its own parentless root."""
    first, second = new_call_id(), new_call_id()
    assert chat(rig, first, marker()).status_code == 200
    assert chat(rig, second, marker()).status_code == 200
    one, two = _named(rig.otel_trace(first), ROOT), _named(rig.otel_trace(second), ROOT)
    assert one["trace_id"] != two["trace_id"]
    assert one["parent_span_id"] == two["parent_span_id"] == ""


def test_resource_names_the_service_environment_and_model_id(rig: SinkRig) -> None:
    """Every exported span's resource must carry service.name "dec1-sinks" (OTEL_SERVICE_NAME),
    deployment.environment "dec1-env" (OTEL_ENVIRONMENT_NAME) and model_id "dec1-sinks" (defaults to the service
    name)."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id, marker()).status_code == 200
    for span in rig.otel_trace(call_id):
        assert span["resource"].get("service.name") == "dec1-sinks"
        assert span["resource"].get("deployment.environment") == "dec1-env"
        assert span["resource"].get("model_id") == "dec1-sinks"


def test_otel_resource_attributes_env_is_merged_into_the_resource(rig: SinkRig) -> None:
    """`OTEL_RESOURCE_ATTRIBUTES=dec1.owner=observability` must add `dec1.owner` "observability" to the resource
    next to the litellm defaults (OTELResourceDetector merge in `_get_litellm_resource`)."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id, marker()).status_code == 200
    assert _named(rig.otel_trace(call_id), ROOT)["resource"].get("dec1.owner") == "observability"


def test_spans_of_a_burst_are_batched_into_fewer_exports_than_spans(rig: SinkRig) -> None:
    """Five concurrent chats produce at least fifteen spans, and the batch processor must ship them in fewer POSTs
    than spans, with some POST carrying more than one span. UNVERIFIED: exact grouping depends on
    OTEL_BSP_SCHEDULE_DELAY timing, so only "fewer posts than spans" is asserted."""
    calls: Final = tuple(new_call_id() for _ in range(5))
    before: Final = len(rig.peer("otlp").posts())
    with ThreadPoolExecutor(max_workers=5) as pool:
        statuses = list(pool.map(lambda call_id: chat(rig, call_id, marker()).status_code, calls))
    assert statuses == [200] * 5
    spans: Final = sum(len(rig.otel_trace(call_id)) for call_id in calls)
    posts: Final = len(rig.peer("otlp").posts()) - before
    assert posts < spans


def test_provider_failure_exports_a_failed_proxy_request_span_with_the_exception(rig: SinkRig) -> None:
    """A provider 400 must export a "Failed Proxy Server Request" span under the root whose `exception` attribute
    contains the provider's error text."""
    call_id: Final = new_call_id()
    headers: Final = {"traceparent": f"00-{uuid.uuid4().hex}-{uuid.uuid4().hex[:16]}-01"}
    assert chat(rig, call_id, marker(), model=CHAT_REJECTED, headers=headers).status_code == 400
    trace: Final = headers["traceparent"].split("-")[1]
    spans: Final = eventually(
        lambda: tuple(span for span in rig.spans("otlp") if span["trace_id"] == trace and span["name"] == FAILED),
        lambda found: len(found) >= 1,
        seconds=30,
    )
    assert PROVIDER_REJECTION in str(attribute(spans[0], "exception"))


def test_streamed_chat_exports_one_litellm_request_marked_streaming(rig: SinkRig) -> None:
    """A streamed chat must export exactly one "litellm_request" span, with llm.is_streaming "True" (a string, as
    set_attributes writes `str(stream)`) and the full usage 11 in and 4 out."""
    call_id: Final = new_call_id()
    status, _ = stream_chat(rig, call_id, marker())
    assert status == 200
    span: Final = _primary(rig, call_id)
    assert attribute(span, "llm.is_streaming") == "True"
    assert attribute(span, "gen_ai.usage.input_tokens") == 11
    assert attribute(span, "gen_ai.usage.output_tokens") == 4


def test_cache_hit_span_says_so_and_reports_the_saved_cost(rig: SinkRig) -> None:
    """The second of two identical chats is a cache hit, and its "litellm_request" span must carry
    litellm.cache_hit true and a positive litellm.saved_cache_cost, while the first carries litellm.cache_hit false."""
    text: Final = marker()
    first, second = new_call_id(), new_call_id()
    assert chat(rig, first, text).status_code == 200
    assert attribute(_primary(rig, first), "litellm.cache_hit") is False
    assert chat(rig, second, text).status_code == 200
    hit: Final = _primary(rig, second)
    assert attribute(hit, "litellm.cache_hit") is True
    assert float(attribute(hit, "litellm.saved_cache_cost") or 0) > 0  # type: ignore[arg-type]


def test_no_credential_reaches_the_otlp_collector(rig: SinkRig) -> None:
    """No export body may contain the master key or the provider key, on success or failure."""
    ok, failed = new_call_id(), new_call_id()
    assert chat(rig, ok, marker()).status_code == 200
    assert chat(rig, failed, marker(), model=CHAT_REJECTED).status_code == 400
    rig.otel_trace(ok)
    for request in rig.peer("otlp").posts():
        assert rig.proxy.key.encode() not in request.body
        assert PROVIDER_KEY.encode() not in request.body


def test_turn_off_message_logging_drops_messages_and_the_raw_request_span(dark: SinkRig) -> None:
    """With `turn_off_message_logging: true`, a chat must still export the root and "litellm_request" with usage,
    but no "raw_gen_ai_request" span and no gen_ai.input.messages, and neither the user text nor the reply text may
    appear in any export body (`_resolve_capture_mode` forces NO_CONTENT)."""
    call_id, text = new_call_id(), marker()
    assert chat(dark, call_id, text).status_code == 200
    spans: Final = dark.otel_trace(call_id)
    primary: Final = _named(spans, PRIMARY)
    assert attribute(primary, "gen_ai.usage.input_tokens") == 11
    assert attribute(primary, "gen_ai.input.messages") is None
    assert RAW not in [span["name"] for span in spans]
    for request in dark.peer("otlp").posts():
        assert text.encode() not in request.body
        assert REPLY_TEXT.encode() not in request.body


def test_collector_returning_500_never_fails_or_slows_the_request(rig: SinkRig) -> None:
    """With the collector answering 500, a chat must still return 200 within the healthy control time plus the 30 second stall, and spans of a call made
    after recovery must still be exported."""
    control: Final = healthy_seconds(rig)
    peer: Final = rig.peer("otlp")
    peer.switch.status = 500
    try:
        response, elapsed = timed_chat(rig, new_call_id(), marker())
        assert response.status_code == 200
        assert_not_held(elapsed, control)
    finally:
        peer.switch.status = 200
    call_id: Final = new_call_id()
    assert chat(rig, call_id, marker()).status_code == 200
    rig.otel_trace(call_id, seconds=60)


def test_stalled_collector_never_holds_the_response(rig: SinkRig) -> None:
    """With the collector holding every export open, three sequential chats must each return 200 within the healthy control time plus the 30 second stall,
    because the BatchSpanProcessor exports on its own thread."""
    control: Final = healthy_seconds(rig)
    peer: Final = rig.peer("otlp")
    peer.switch.stall()
    try:
        for _ in range(3):
            response, elapsed = timed_chat(rig, new_call_id(), marker())
            assert response.status_code == 200
            assert_not_held(elapsed, control)
    finally:
        peer.switch.release()
