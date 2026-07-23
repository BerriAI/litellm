"""Tests for the V2 ``OpenTelemetryV2`` CustomLogger adapter.

Exercises the callback surface the existing call sites use: the LLM-call span
opened at the ``pre_call`` boundary and closed at async success/failure, service
hooks, proxy SERVER span lifecycle (start + setters), parent-context resolution
(ambient context), and Baggage promotion onto child spans.
"""

import asyncio
import contextlib
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("opentelemetry")

from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)
from opentelemetry.trace import SpanKind  # noqa: E402
from opentelemetry.trace.status import StatusCode  # noqa: E402

from litellm.integrations.otel import (  # noqa: E402
    GenAI,
    LiteLLM,
    OpenTelemetryV2Config,
)
from litellm.integrations.otel.plumbing import providers  # noqa: E402
from litellm.integrations.otel.plumbing.context import (  # noqa: E402
    _request_destinations,
    reset_mcp_message_trace_carrier,
    set_mcp_message_trace_carrier,
    set_request_destinations,
    set_request_root_span,
)
from litellm.integrations.otel.logger import OpenTelemetryV2  # noqa: E402
from litellm.integrations.otel.model.spans import (  # noqa: E402
    LITELLM_PROXY_REQUEST_SPAN_NAME,
    SpanRole,
)
from litellm.integrations.otel.model.utils import to_ns, to_seconds  # noqa: E402


def _anchor(dests):
    """Anchor admin destinations on the server-only ContextVar the v2 router reads
    (the proxy sets this at auth time; there is no request-carried carrier)."""
    from litellm.integrations.otel.model.destination import OtelDestination

    set_request_destinations(
        tuple(d if isinstance(d, OtelDestination) else OtelDestination.model_validate(d) for d in dests)
    )


@pytest.fixture(autouse=True)
def _reset_request_destinations():
    token = _request_destinations.set(())
    try:
        yield
    finally:
        _request_destinations.reset(token)

# --------------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _reset_request_root_span():
    """Clear the request-root-span anchor around every test.

    In production each request runs in its own asyncio task whose context is a
    fresh copy, so the anchor never leaks between requests. The test process
    shares one context, so reset it explicitly to keep tests order-independent.
    """
    from litellm.integrations.otel.plumbing import context as _otel_context

    _otel_context._request_root_span.set(None)
    _otel_context._mcp_message_trace_carrier.set(None)
    yield
    _otel_context._request_root_span.set(None)
    _otel_context._mcp_message_trace_carrier.set(None)


def _payload(**overrides):
    payload = {
        "call_type": "acompletion",
        "custom_llm_provider": "openai",
        "model": "gpt-4o",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "stream": False,
        "model_parameters": {"temperature": 0.7, "max_tokens": 256},
        "response": {
            "id": "resp_1",
            "model": "gpt-4o-2024",
            "choices": [{"finish_reason": "stop"}],
        },
        "metadata": {
            "team_id": "t1",
            "team_alias": "team one",
            "user_api_key_hash": "hsh",
        },
        "api_base": "https://api.openai.com:443/v1",
        "status": "success",
        "litellm_call_id": "call_1",
        "response_cost": 0.002,
        "hidden_params": {},
    }
    payload.update(overrides)
    return payload


def _kwargs(payload=None):
    return {
        # ``litellm_call_id`` (here carried inside the payload) correlates the
        # pre_call boundary with the close callback — the carrier is keyed by it.
        "standard_logging_object": payload if payload is not None else _payload(),
        "litellm_params": {"metadata": {}},
    }


def _logger(legacy_compat=True, team_metadata_keys=None):
    cfg = OpenTelemetryV2Config(
        exporter="in_memory",
        legacy_compat=legacy_compat,
        baggage_team_metadata_keys=team_metadata_keys or [],
    )
    exporter = InMemorySpanExporter()
    tracer_provider = providers.build_tracer_provider(cfg, exporter=exporter)
    return OpenTelemetryV2(config=cfg, tracer_provider=tracer_provider), exporter


def _emit_llm(logger, kwargs=None, *, ambient=None, fail=False):
    """Drive the real boundary flow: open at ``pre_call`` then close at the async
    callback. ``ambient``, if given, is the span that is the active OTel context
    while ``pre_call`` runs (the server span) so the LLM span parents to it."""
    if kwargs is None:
        kwargs = _kwargs()
    payload = kwargs.get("standard_logging_object") or {}
    with (
        trace.use_span(ambient, end_on_exit=False)
        if ambient is not None
        else contextlib.nullcontext()
    ):
        logger.log_pre_api_call(model=payload.get("model"), messages=[], kwargs=kwargs)
    hook = logger.async_log_failure_event if fail else logger.async_log_success_event
    asyncio.run(hook(kwargs, None, None, None))
    return kwargs


# --------------------------------------------------------------------------- #
#  Time helpers
# --------------------------------------------------------------------------- #


def test_to_ns_handles_datetime_and_float():
    dt = datetime(2026, 5, 26, 12, 0, 0, tzinfo=timezone.utc)
    assert to_ns(dt) == int(dt.timestamp() * 1e9)
    assert to_ns(1.5) == 1_500_000_000
    assert to_ns(None) is None
    assert to_ns(True) is None  # bool is rejected — not a real epoch value


def test_to_seconds_parses_string_formats():
    assert to_seconds("2026-05-26 12:00:00.123") is not None
    assert to_seconds("2026-05-26 12:00:00") is not None
    assert to_seconds("nonsense") is None
    assert to_seconds(None) is None
    assert to_seconds(1.5) == 1.5


# --------------------------------------------------------------------------- #
#  LLM-call callbacks
# --------------------------------------------------------------------------- #


def test_async_log_success_event_emits_llm_call_span():
    logger, exporter = _logger()
    _emit_llm(logger)
    (span,) = exporter.get_finished_spans()
    assert span.name == "chat gpt-4o"
    assert span.kind is SpanKind.CLIENT
    assert span.attributes[GenAI.OPERATION_NAME] == "chat"
    assert span.attributes[GenAI.REQUEST_MODEL] == "gpt-4o"
    assert span.attributes[LiteLLM.CALL_ID] == "call_1"
    # Success leaves status UNSET (semconv default), not forced OK.
    assert span.status.status_code is StatusCode.UNSET


def test_streaming_span_carries_time_to_first_chunk():
    logger, exporter = _logger()
    kwargs = {
        **_kwargs(payload=_payload(stream=True)),
        "optional_params": {"stream": True},
        "api_call_start_time": datetime(2026, 5, 26, 12, 0, 0, tzinfo=timezone.utc),
        "completion_start_time": datetime(2026, 5, 26, 12, 0, 0, 750000, tzinfo=timezone.utc),
    }
    _emit_llm(logger, kwargs)
    (span,) = exporter.get_finished_spans()
    assert span.attributes[GenAI.RESPONSE_TIME_TO_FIRST_CHUNK] == pytest.approx(0.75)


def test_non_streaming_span_has_no_time_to_first_chunk():
    logger, exporter = _logger()
    kwargs = {
        **_kwargs(),
        "optional_params": {},
        "api_call_start_time": datetime(2026, 5, 26, 12, 0, 0, tzinfo=timezone.utc),
        "completion_start_time": datetime(2026, 5, 26, 12, 0, 5, tzinfo=timezone.utc),
    }
    _emit_llm(logger, kwargs)
    (span,) = exporter.get_finished_spans()
    assert GenAI.RESPONSE_TIME_TO_FIRST_CHUNK not in span.attributes


def test_streaming_span_without_timing_omits_time_to_first_chunk():
    logger, exporter = _logger()
    kwargs = {
        **_kwargs(payload=_payload(stream=True)),
        "optional_params": {"stream": True},
    }
    _emit_llm(logger, kwargs)
    (span,) = exporter.get_finished_spans()
    assert GenAI.RESPONSE_TIME_TO_FIRST_CHUNK not in span.attributes


def test_async_log_failure_event_marks_error_status():
    logger, exporter = _logger()
    payload = _payload(
        status="failure",
        error_information={"error_class": "RateLimitError", "error_message": "429"},
    )
    _emit_llm(logger, _kwargs(payload=payload), fail=True)
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["error.type"] == "RateLimitError"


def _logger_with_events(enable_events):
    from opentelemetry.sdk._logs.export import InMemoryLogExporter

    cfg = OpenTelemetryV2Config(exporter="in_memory", enable_events=enable_events)
    span_exporter = InMemorySpanExporter()
    tracer_provider = providers.build_tracer_provider(cfg, exporter=span_exporter)
    log_exporter = InMemoryLogExporter()
    logger_provider = providers.build_logger_provider(cfg, log_exporter=log_exporter)
    logger = OpenTelemetryV2(config=cfg, tracer_provider=tracer_provider, logger_provider=logger_provider)
    return logger, span_exporter, log_exporter


def test_enable_events_records_operation_exception_through_failure_callback():
    """With ``enable_events`` on, a real failure callback records the GenAI
    ``gen_ai.client.operation.exception`` log event, carrying the traceback from
    the standard logging payload and correlated to the LLM-call span."""
    from litellm.integrations.otel.model.semconv import ExceptionEvent, GenAIEvent

    logger, span_exporter, log_exporter = _logger_with_events(enable_events=True)
    payload = _payload(
        status="failure",
        error_information={
            "error_class": "RateLimitError",
            "error_message": "429 rate limited",
            "traceback": "Traceback (most recent call last) ...",
        },
    )
    _emit_llm(logger, _kwargs(payload=payload), fail=True)

    (span,) = span_exporter.get_finished_spans()
    (log,) = log_exporter.get_finished_logs()
    record = log.log_record
    assert record.attributes["event.name"] == GenAIEvent.OPERATION_EXCEPTION
    assert record.attributes[ExceptionEvent.TYPE] == "RateLimitError"
    assert record.attributes[ExceptionEvent.MESSAGE] == "429 rate limited"
    assert record.attributes[ExceptionEvent.STACKTRACE] == "Traceback (most recent call last) ..."
    assert record.trace_id == span.context.trace_id
    assert record.span_id == span.context.span_id


def test_events_off_by_default_records_no_log_event_on_failure():
    """``enable_events`` defaults to off: even with a logs pipeline injected, a
    failure records only the span-side error surface, no log event."""
    logger, span_exporter, log_exporter = _logger_with_events(enable_events=False)
    payload = _payload(
        status="failure",
        error_information={"error_class": "RateLimitError", "error_message": "429"},
    )
    _emit_llm(logger, _kwargs(payload=payload), fail=True)

    assert len(span_exporter.get_finished_spans()) == 1
    assert log_exporter.get_finished_logs() == ()
    assert OpenTelemetryV2Config(exporter="in_memory").enable_events is False


def test_sync_log_event_is_noop():
    """V2 closes the span async-only; the sync callback runs out-of-context, so
    it no-ops (the span stays open on the carrier until the async callback)."""
    logger, exporter = _logger()
    kwargs = _kwargs()
    logger.log_pre_api_call(model="gpt-4o", messages=[], kwargs=kwargs)
    logger.log_success_event(kwargs, None, None, None)
    logger.log_failure_event(kwargs, None, None, None)
    assert exporter.get_finished_spans() == ()


def test_missing_standard_logging_object_is_noop():
    """No carrier (``pre_call`` never ran) → the callback emits nothing."""
    logger, exporter = _logger()
    asyncio.run(
        logger.async_log_success_event({"litellm_params": {}}, None, None, None)
    )
    assert exporter.get_finished_spans() == ()


def test_no_span_when_pre_call_never_ran():
    """A request rejected before the upstream call — at the auth/budget gate, or
    blocked by a pre-call guardrail — never reaches ``pre_call``, so there is no
    carrier and the failure log produces no phantom CLIENT span. This replaces the
    old post-hoc heuristics: "did pre_call run?" is the only signal needed."""
    logger, exporter = _logger()
    payload = _payload(
        status="failure",
        error_information={"error_class": "ProxyException", "error_code": "401"},
    )
    # No log_pre_api_call: the call never started.
    asyncio.run(
        logger.async_log_failure_event(_kwargs(payload=payload), None, None, None)
    )
    assert exporter.get_finished_spans() == ()  # no phantom LLM span


def test_real_llm_failure_still_emitted():
    """A genuine LLM failure: ``pre_call`` ran (the call was attempted), so the
    CLIENT span is opened at the boundary and closed ERROR."""
    logger, exporter = _logger()
    payload = _payload(
        status="failure",
        error_information={"error_class": "RateLimitError", "error_code": "429"},
    )
    _emit_llm(logger, _kwargs(payload=payload), fail=True)
    (span,) = exporter.get_finished_spans()
    assert span.name == "chat gpt-4o"
    assert span.status.status_code is StatusCode.ERROR


def test_idempotent_on_repeat_callback():
    """The carrier is the dedup: once the async callback closes the span and
    clears the carrier, a second callback firing emits nothing."""
    logger, exporter = _logger()
    kwargs = _kwargs()
    logger.log_pre_api_call(model="gpt-4o", messages=[], kwargs=kwargs)
    asyncio.run(logger.async_log_success_event(kwargs, None, None, None))
    asyncio.run(logger.async_log_success_event(kwargs, None, None, None))
    assert len(exporter.get_finished_spans()) == 1


# --------------------------------------------------------------------------- #
#  MCP tool-call spans
# --------------------------------------------------------------------------- #


def _mcp_payload(**overrides):
    payload = {
        "call_type": "call_mcp_tool",
        "status": "success",
        "litellm_call_id": "mcp_1",
        "response_cost": 0.01,
        "metadata": {
            "user_api_key_team_id": "t1",
            "mcp_tool_call_metadata": {
                "name": "get_weather",
                "arguments": {"city": "Paris"},
                "result": {"temp_c": 21},
                "mcp_server_name": "weather-mcp",
                "mcp_session_id": "sess-abc123",
            },
        },
        "hidden_params": {},
    }
    payload.update(overrides)
    return payload


def _logger_capturing():
    from litellm.integrations.otel.model.config import CaptureMessageContent

    cfg = OpenTelemetryV2Config(
        exporter="in_memory",
        legacy_compat=False,
        capture_message_content=CaptureMessageContent.SPAN_ONLY,
    )
    exporter = InMemorySpanExporter()
    tracer_provider = providers.build_tracer_provider(cfg, exporter=exporter)
    return OpenTelemetryV2(config=cfg, tracer_provider=tracer_provider), exporter


def test_mcp_tool_call_emits_client_span():
    """A closed MCP tool call becomes a CLIENT span named ``tools/call {tool}``,
    carrying the MCP semconv method/operation and the vendor server name."""
    logger, exporter = _logger()
    kwargs = {"standard_logging_object": _mcp_payload()}
    asyncio.run(logger.async_log_success_event(kwargs, None, None, None))
    (span,) = exporter.get_finished_spans()
    assert span.name == "tools/call get_weather"
    assert span.kind is SpanKind.CLIENT
    assert span.attributes["mcp.method.name"] == "tools/call"
    assert span.attributes["mcp.session.id"] == "sess-abc123"
    assert span.attributes[GenAI.OPERATION_NAME] == "execute_tool"
    assert span.attributes["gen_ai.tool.name"] == "get_weather"
    assert span.attributes[LiteLLM.MCP_SERVER_NAME] == "weather-mcp"
    assert span.attributes[LiteLLM.CALL_ID] == "mcp_1"
    assert span.status.status_code is StatusCode.UNSET
    # Tool I/O is content: withheld while capture is off (the default).
    assert "gen_ai.tool.call.arguments" not in span.attributes
    assert "gen_ai.tool.call.result" not in span.attributes


def test_mcp_tool_call_stateless_omits_session_id():
    """A stateless MCP call carries no ``mcp-session-id``, so the span must omit
    ``mcp.session.id`` rather than stamping an empty or ``None`` value."""
    logger, exporter = _logger()
    payload = _mcp_payload()
    del payload["metadata"]["mcp_tool_call_metadata"]["mcp_session_id"]
    asyncio.run(
        logger.async_log_success_event(
            {"standard_logging_object": payload}, None, None, None
        )
    )
    (span,) = exporter.get_finished_spans()
    assert "mcp.session.id" not in span.attributes
    assert span.attributes["mcp.method.name"] == "tools/call"


def test_mcp_tool_call_is_not_logged_as_llm_call():
    """The MCP branch must short-circuit the LLM-call path: even if ``pre_call``
    opened a stray carrier for this id, the result is one MCP span, never an LLM
    ``chat`` span."""
    logger, exporter = _logger()
    kwargs = {"standard_logging_object": _mcp_payload()}
    logger.log_pre_api_call(model="MCP: get_weather", messages=[], kwargs=kwargs)
    asyncio.run(logger.async_log_success_event(kwargs, None, None, None))
    (span,) = exporter.get_finished_spans()
    assert span.attributes["mcp.method.name"] == "tools/call"
    assert "gen_ai.request.model" not in span.attributes


def test_mcp_tool_call_captures_io_when_enabled():
    logger, exporter = _logger_capturing()
    kwargs = {"standard_logging_object": _mcp_payload()}
    asyncio.run(logger.async_log_success_event(kwargs, None, None, None))
    (span,) = exporter.get_finished_spans()
    assert '"Paris"' in span.attributes["gen_ai.tool.call.arguments"]
    assert "21" in span.attributes["gen_ai.tool.call.result"]


def test_mcp_tool_call_failure_marks_error():
    logger, exporter = _logger()
    payload = _mcp_payload(
        status="failure",
        error_information={"error_class": "MCPError", "error_message": "upstream 500"},
    )
    asyncio.run(
        logger.async_log_failure_event(
            {"standard_logging_object": payload}, None, None, None
        )
    )
    (span,) = exporter.get_finished_spans()
    assert span.name == "tools/call get_weather"
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["error.type"] == "MCPError"


def test_mcp_tool_call_deduped_on_repeat():
    logger, exporter = _logger()
    kwargs = {"standard_logging_object": _mcp_payload()}
    asyncio.run(logger.async_log_success_event(kwargs, None, None, None))
    asyncio.run(logger.async_log_success_event(kwargs, None, None, None))
    assert len(exporter.get_finished_spans()) == 1


def test_mcp_tool_call_metadata_read_from_nested_metadata_not_top_level():
    """``mcp_tool_call_metadata`` lives under ``StandardLoggingPayload.metadata``;
    a top-level copy (the pre-fix shape the reader used to look at) must be ignored
    so the reader can't silently regress to producing an empty ``tools/call`` span
    with no session id, tool name, or server name."""
    logger, exporter = _logger()
    payload = _mcp_payload()
    # Move the real metadata to the top level only, mirroring the old buggy read
    # location. ``call_type`` still classifies this as an MCP call, so the span is
    # emitted, but none of its fields are reachable from the wrong nesting level.
    payload["mcp_tool_call_metadata"] = payload["metadata"].pop(
        "mcp_tool_call_metadata"
    )
    asyncio.run(
        logger.async_log_success_event(
            {"standard_logging_object": payload}, None, None, None
        )
    )
    (span,) = exporter.get_finished_spans()
    assert span.name == "tools/call"
    assert "mcp.session.id" not in span.attributes
    assert "gen_ai.tool.name" not in span.attributes
    assert LiteLLM.MCP_SERVER_NAME not in span.attributes


def _mcp_list_payload(**overrides):
    payload = {
        "call_type": "list_mcp_tools",
        "status": "success",
        "litellm_call_id": "mcp_list_1",
        "metadata": {
            "user_api_key_team_id": "t1",
            "spend_logs_metadata": {"mcp_operation": "list_tools"},
        },
        "hidden_params": {},
    }
    payload.update(overrides)
    return payload


def test_mcp_list_tools_emits_client_span():
    """An MCP ``tools/list`` discovery call becomes a CLIENT span named ``tools/list``,
    carrying only the MCP method and the call id. Per the GenAI MCP semconv the list
    span omits ``gen_ai.operation.name`` and ``gen_ai.tool.name`` (tool-call-only) and
    ``mcp.session.id`` (the list path threads no session id), so a naive reuse of the
    tool-call mapper would wrongly stamp them, and the pre-fix code emitted no span at
    all for a ``list_mcp_tools`` payload."""
    logger, exporter = _logger()
    kwargs = {"standard_logging_object": _mcp_list_payload()}
    asyncio.run(logger.async_log_success_event(kwargs, None, None, None))
    (span,) = exporter.get_finished_spans()
    assert span.name == "tools/list"
    assert span.kind is SpanKind.CLIENT
    assert span.attributes["mcp.method.name"] == "tools/list"
    assert span.attributes[LiteLLM.CALL_ID] == "mcp_list_1"
    assert span.status.status_code is StatusCode.UNSET
    # Bug-killers: no span pre-fix (empty exporter -> the unpack above raises), and a
    # tool-call-shaped fix would leak execute_tool / tool name / session id here.
    assert GenAI.OPERATION_NAME not in span.attributes
    assert "gen_ai.tool.name" not in span.attributes
    assert "mcp.session.id" not in span.attributes


_MCP_SPAN_CASES = [
    (_mcp_payload, "tools/call get_weather"),
    (_mcp_list_payload, "tools/list"),
]


@pytest.mark.parametrize("make_payload, span_name", _MCP_SPAN_CASES)
def test_mcp_span_roots_and_links_transport_without_propagated_context(
    make_payload, span_name
):
    """MCP and the HTTP transport are independent lifecycles (one streamable-HTTP
    session multiplexes many messages), so per the MCP semconv the message span
    must NOT nest under the session/transport span — that is what made it render
    skewed at the session's start. With no propagated ``params._meta`` context it
    starts its own root trace and records the transport span as a *link*, never
    the parent."""
    logger, exporter = _logger()
    transport = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    set_request_root_span(transport)
    asyncio.run(
        logger.async_log_success_event(
            {"standard_logging_object": make_payload()}, None, None, None
        )
    )
    transport.end()
    span = next(s for s in exporter.get_finished_spans() if s.name == span_name)
    assert span.parent is None
    assert span.context.trace_id != transport.get_span_context().trace_id
    assert [link.context.span_id for link in span.links] == [
        transport.get_span_context().span_id
    ]


@pytest.mark.parametrize("make_payload, span_name", _MCP_SPAN_CASES)
def test_mcp_span_parents_to_propagated_meta_trace_context(make_payload, span_name):
    """When the client propagates W3C trace context in the request's
    ``params._meta`` (SEP-414), the MCP span parents to it (one distributed trace)
    and still links the transport span — never falling through to the
    ambient/session span."""
    logger, exporter = _logger()
    transport = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    set_request_root_span(transport)
    token = set_mcp_message_trace_carrier(
        {"traceparent": "00-11111111111111111111111111111111-2222222222222222-01"}
    )
    try:
        asyncio.run(
            logger.async_log_success_event(
                {"standard_logging_object": make_payload()}, None, None, None
            )
        )
    finally:
        reset_mcp_message_trace_carrier(token)
    transport.end()
    span = next(s for s in exporter.get_finished_spans() if s.name == span_name)
    assert span.context.trace_id == 0x11111111111111111111111111111111
    assert span.parent is not None
    assert span.parent.span_id == 0x2222222222222222
    assert [link.context.span_id for link in span.links] == [
        transport.get_span_context().span_id
    ]


@pytest.mark.parametrize("make_payload, span_name", _MCP_SPAN_CASES)
def test_mcp_span_ignores_client_supplied_baggage(make_payload, span_name):
    """The MCP span must NOT honor W3C Baggage from the client's ``params._meta``.

    ``params._meta`` is caller-controlled and the baggage processor stamps
    allowlisted baggage keys onto every span, so extracting remote baggage would
    let a client spoof a span's identity (e.g. ``litellm.team.id``). The propagator
    extracts trace context only, so the spoofed keys never reach the span while the
    legitimate traceparent parenting still works."""
    logger, exporter = _logger()
    transport = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    set_request_root_span(transport)
    token = set_mcp_message_trace_carrier(
        {
            "traceparent": "00-11111111111111111111111111111111-2222222222222222-01",
            "baggage": "litellm.team.id=spoofed-team,litellm.metadata.user_api_key_user_id=attacker",
        }
    )
    try:
        asyncio.run(
            logger.async_log_success_event(
                {"standard_logging_object": make_payload()}, None, None, None
            )
        )
    finally:
        reset_mcp_message_trace_carrier(token)
    transport.end()
    span = next(s for s in exporter.get_finished_spans() if s.name == span_name)
    # Trace context still honored: proves the carrier was processed, not dropped wholesale.
    assert span.parent is not None and span.parent.span_id == 0x2222222222222222
    # Identity is the authenticated payload's team, never the client's spoofed value.
    assert span.attributes[LiteLLM.TEAM_ID] == "t1"
    assert "litellm.metadata.user_api_key_user_id" not in span.attributes


@pytest.mark.parametrize("make_payload, span_name", _MCP_SPAN_CASES)
def test_mcp_span_carries_authenticated_identity(make_payload, span_name):
    """An MCP span is labeled with the authenticated request's identity (team/key),
    seeded from the parsed payload like the LLM-call span. Without this seeding the
    span — parented to an empty remote context — would carry no team/key attribute at
    all, so it couldn't be attributed or filtered by team in the traces backend."""
    logger, exporter = _logger()
    asyncio.run(
        logger.async_log_success_event(
            {"standard_logging_object": make_payload()}, None, None, None
        )
    )
    span = next(s for s in exporter.get_finished_spans() if s.name == span_name)
    assert span.attributes[LiteLLM.TEAM_ID] == "t1"


def test_mcp_span_malformed_traceparent_starts_root():
    """A malformed traceparent in ``params._meta`` must not crash or parent to a
    bogus span: the propagator ignores it, so the span starts its own root trace and
    still links the transport span."""
    logger, exporter = _logger()
    transport = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    set_request_root_span(transport)
    token = set_mcp_message_trace_carrier({"traceparent": "not-a-valid-traceparent"})
    try:
        asyncio.run(
            logger.async_log_success_event(
                {"standard_logging_object": _mcp_list_payload()}, None, None, None
            )
        )
    finally:
        reset_mcp_message_trace_carrier(token)
    transport.end()
    span = next(s for s in exporter.get_finished_spans() if s.name == "tools/list")
    assert span.parent is None
    assert [link.context.span_id for link in span.links] == [
        transport.get_span_context().span_id
    ]


def test_pre_call_idempotent_keeps_first_span():
    """A retried call may re-enter ``pre_call`` with the same call id; the first
    span (with the true start time) is kept, not replaced."""
    logger, _ = _logger()
    kwargs = _kwargs()
    server = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    with trace.use_span(server, end_on_exit=False):
        logger.log_pre_api_call(model="gpt-4o", messages=[], kwargs=kwargs)
        first = logger._open_llm_calls["call_1"]
        logger.log_pre_api_call(model="gpt-4o", messages=[], kwargs=kwargs)
        second = logger._open_llm_calls["call_1"]
    server.end()
    assert first is second  # not overwritten


# --------------------------------------------------------------------------- #
#  Parent resolution — ambient context at the boundary (no metadata threading)
# --------------------------------------------------------------------------- #


def test_llm_span_parents_to_ambient_server_span():
    """The span is opened at ``pre_call`` while the server span is the active
    context, so it nests under it natively (no ``litellm_parent_otel_span``)."""
    logger, exporter = _logger()
    server = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    _emit_llm(logger, ambient=server)
    server.end()
    by_name = {s.name: s for s in exporter.get_finished_spans()}
    llm_span = by_name["chat gpt-4o"]
    assert llm_span.parent is not None
    assert llm_span.parent.span_id == server.get_span_context().span_id


def test_llm_span_is_root_without_ambient_server_span():
    """No server span at ``pre_call`` → creation is deferred and the span is a
    root of its own trace (the SDK / no-proxy path)."""
    logger, exporter = _logger()
    _emit_llm(logger)
    (span,) = exporter.get_finished_spans()
    assert span.parent is None  # standalone (no proxy server span) → root


# --------------------------------------------------------------------------- #
#  Explicit request-root-span anchor — request-level spans (LLM call, guardrail)
#  parent to the captured server span, NOT to whatever span is momentarily
#  active. Regression cover for the two ambient-only failure modes:
#    * auth: the LLM/guardrail span must not nest under the live ``auth`` span;
#    * pass-through: the span must not orphan when closed off the request task.
# --------------------------------------------------------------------------- #


def test_llm_span_anchors_to_root_even_inside_active_phase_span():
    """Bug 1: a synthetic error log can fire ``pre_call`` while the ``auth`` phase
    span is the *active* context. The LLM span must still parent to the request
    root (the server span), never to the auth span it happens to be nested in."""
    logger, exporter = _logger()
    server = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    set_request_root_span(server)
    kwargs = _kwargs()
    # ``auth`` phase span is the active span when pre_call + close run.
    with trace.use_span(server, end_on_exit=False):
        with logger.start_phase_span("auth /chat/completions"):
            logger.log_pre_api_call(model="gpt-4o", messages=[], kwargs=kwargs)
            asyncio.run(logger.async_log_success_event(kwargs, None, None, None))
    server.end()
    by_name = {s.name: s for s in exporter.get_finished_spans()}
    llm_span = by_name["chat gpt-4o"]
    auth_span = by_name["auth /chat/completions"]
    # Parented to the server root, NOT the auth span it was emitted inside.
    assert llm_span.parent.span_id == server.get_span_context().span_id
    assert llm_span.parent.span_id != auth_span.get_span_context().span_id


def test_live_llm_span_anchors_to_root_with_no_active_span():
    """Bug 2 (pass-through), live path: even with no span active at ``pre_call``,
    the anchor is a recordable parent, so the span opens live under the server root
    instead of orphaning — and the detached close just ends it, in the right
    trace."""
    logger, exporter = _logger()
    server = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    set_request_root_span(server)
    kwargs = _kwargs()
    logger.log_pre_api_call(model="gpt-4o", messages=[], kwargs=kwargs)
    assert logger._open_llm_calls["call_1"].spans  # live, via anchor
    asyncio.run(logger.async_log_success_event(kwargs, None, None, None))
    server.end()
    by_name = {s.name: s for s in exporter.get_finished_spans()}
    llm_span = by_name["chat gpt-4o"]
    assert llm_span.parent.span_id == server.get_span_context().span_id
    assert llm_span.context.trace_id == server.get_span_context().trace_id


def test_deferred_llm_span_reads_anchor_at_close():
    """Bug 2, deferred path: when the anchor isn't visible at ``pre_call`` (a
    sync-only provider's thread-pool call) the span defers; the close — back on the
    request task, anchor visible — must parent it to the root, not orphan it."""
    logger, exporter = _logger()
    server = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    kwargs = _kwargs()
    # pre_call with NO anchor and no active span → deferred.
    logger.log_pre_api_call(model="gpt-4o", messages=[], kwargs=kwargs)
    assert logger._open_llm_calls["call_1"].spans == ()  # deferred
    # Anchor becomes visible at close (worker copied the request task's context).
    set_request_root_span(server)
    asyncio.run(logger.async_log_success_event(kwargs, None, None, None))
    server.end()
    by_name = {s.name: s for s in exporter.get_finished_spans()}
    llm_span = by_name["chat gpt-4o"]
    assert llm_span.parent.span_id == server.get_span_context().span_id
    assert llm_span.context.trace_id == server.get_span_context().trace_id


def test_synthetic_error_log_produces_no_llm_span():
    """Bug 1 root cause: a proxy-gate error log (auth/rate-limit) fires ``pre_call``
    for a request that never reached a provider. Tagged with
    ``LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL``, it must open no carrier and emit no
    LLM-call span — even though the failure callback also fires."""
    from litellm.constants import LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL

    logger, exporter = _logger()
    server = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    set_request_root_span(server)
    payload = _payload(
        status="failure",
        error_information={"error_class": "ProxyException", "error_code": "401"},
    )
    kwargs = _kwargs(payload=payload)
    kwargs[LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL] = True
    with trace.use_span(server, end_on_exit=False):
        with logger.start_phase_span("auth /chat/completions"):
            logger.log_pre_api_call(model="gpt-4o", messages=[], kwargs=kwargs)
            assert "call_1" not in logger._open_llm_calls  # no carrier opened
            asyncio.run(logger.async_log_failure_event(kwargs, None, None, None))
    server.end()
    names = {s.name for s in exporter.get_finished_spans()}
    assert "chat gpt-4o" not in names  # no phantom LLM span
    assert "auth /chat/completions" in names  # auth span itself still recorded


def test_lazy_activation_emits_llm_span_when_destination_resolves(monkeypatch):
    """LIT-3850 lazy-activation seam: a v2 instance born inside the success path
    (because the destination resolver appended its backend to ``success_callback``
    on this request) was not in the callback list when ``pre_call`` iterated, so
    no carrier was opened. The close must still emit the gen-ai span when the
    payload is present and the admin-resolved destinations name this backend, so
    the per-tenant exporter ships it. Without the fallthrough, this test sees
    zero spans."""
    logger, exporter = _logger()
    monkeypatch.setattr(logger, "callback_name", "in_memory")
    server = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    set_request_root_span(server)
    # Make ``tracers_for`` return the logger's default tracer regardless of the
    # destinations passed: the test asserts the close-path emitted the span, not
    # that the per-destination provider clone wired up an OTLP exporter (the
    # routing cache's job, covered separately). The default tracer is bound to
    # the in-memory exporter so the test can read the result.
    tracer_for_calls: list[tuple] = []

    def _fake_tracers_for(default, destinations):
        tracer_for_calls.append(destinations)
        return (default,)

    monkeypatch.setattr(logger._tenant_tracers, "tracers_for", _fake_tracers_for)
    kwargs = _kwargs()
    _anchor([
            {
                "callback_name": "in_memory",
                "endpoint": "https://otlp.example.com/v1",
                "headers": {"api_key": "k"},
            }
        ])
    assert "call_1" not in logger._open_llm_calls  # no carrier opened
    asyncio.run(logger.async_log_success_event(kwargs, None, None, None))
    server.end()
    names = [s.name for s in exporter.get_finished_spans()]
    assert "chat gpt-4o" in names
    # ``tracers_for`` was invoked with exactly the resolved destination, proving
    # the deferred path used per-tenant routing rather than the default tracer
    # blindly.
    assert len(tracer_for_calls) == 1
    (dests,) = tracer_for_calls
    assert len(dests) == 1 and dests[0].endpoint == "https://otlp.example.com/v1"


def test_second_close_after_opened_call_does_not_emit_duplicate(monkeypatch):
    logger, exporter = _logger()
    monkeypatch.setattr(logger, "callback_name", "in_memory")
    monkeypatch.setattr(
        logger._tenant_tracers,
        "tracers_for",
        lambda default, destinations: (default,),
    )
    server = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    set_request_root_span(server)
    kwargs = _kwargs()
    _anchor([
            {
                "callback_name": "in_memory",
                "endpoint": "https://otlp.example.com/v1",
                "headers": {"api_key": "k"},
            }
        ])

    logger.log_pre_api_call(model="gpt-4o", messages=[], kwargs=kwargs)
    asyncio.run(logger.async_log_success_event(kwargs, None, None, None))
    asyncio.run(logger.async_log_failure_event(kwargs, None, None, None))
    server.end()

    names = [s.name for s in exporter.get_finished_spans()]
    assert names.count("chat gpt-4o") == 1


def test_close_without_carrier_and_without_destination_drops_silently():
    """The pre-existing early-return semantics (auth gate / pre-call guardrail
    rejection with no destination resolving to this backend) must be preserved:
    no phantom span. The fix only widens emit-on-close when the admin-resolved
    destinations name this backend AND the payload exists."""
    logger, exporter = _logger()
    server = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    set_request_root_span(server)
    asyncio.run(logger.async_log_success_event(_kwargs(), None, None, None))
    server.end()
    names = [s.name for s in exporter.get_finished_spans()]
    assert "chat gpt-4o" not in names


def test_close_without_carrier_drops_when_payload_missing(monkeypatch):
    """No carrier + no payload = the auth-gate rejection case (no upstream call
    happened). Must drop even when destinations resolve, so a phantom span is
    never emitted for a request the gate refused."""
    logger, exporter = _logger()
    monkeypatch.setattr(logger, "callback_name", "in_memory")
    kwargs = {"litellm_params": {"metadata": {}}}
    _anchor(
        [
            {
                "callback_name": "in_memory",
                "endpoint": "https://otlp.example.com/v1",
                "headers": {},
            }
        ]
    )
    asyncio.run(logger.async_log_success_event(kwargs, None, None, None))
    assert exporter.get_finished_spans() == ()


def test_close_dedupes_duplicate_callbacks(monkeypatch):
    """Cursor BugBot regression: a normal close pops the carrier and emits
    the LLM span; a second callback for the same call_id must not emit a
    duplicate. Before the dedup guard, the second close hit the
    carrier-is-None branch and fired _emit_deferred_llm_call again whenever
    payload + destinations remained on the kwargs, double-exporting the
    span (e.g. success + failure callbacks both firing, or a custom callback
    fanning out)."""
    logger, exporter = _logger()
    monkeypatch.setattr(logger, "callback_name", "in_memory")
    server = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    set_request_root_span(server)
    monkeypatch.setattr(
        logger._tenant_tracers, "tracers_for", lambda default, dests: (default,)
    )
    kwargs = _kwargs()
    _anchor([
            {
                "callback_name": "in_memory",
                "endpoint": "https://otlp.example.com/v1",
                "headers": {},
            }
        ])
    logger.log_pre_api_call(model="gpt-4o", messages=[], kwargs=kwargs)
    asyncio.run(logger.async_log_success_event(kwargs, None, None, None))
    # Second callback for the same call_id: payload + destinations still on
    # kwargs, but no second span must emit.
    asyncio.run(logger.async_log_success_event(kwargs, None, None, None))
    asyncio.run(logger.async_log_failure_event(kwargs, None, None, None))
    server.end()
    llm_spans = [s for s in exporter.get_finished_spans() if s.name == "chat gpt-4o"]
    assert len(llm_spans) == 1


def test_create_request_started_span_captures_anchor():
    """``create_litellm_proxy_request_started_span`` doubles as the anchor capture
    point: the active server span becomes the request root for later spans."""
    from litellm.integrations.otel.plumbing.context import request_root_span

    logger, _ = _logger()
    server = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    with trace.use_span(server, end_on_exit=False):
        returned = logger.create_litellm_proxy_request_started_span(
            start_time=datetime.now(), headers=None
        )
    server.end()
    assert returned.get_span_context().span_id == server.get_span_context().span_id
    assert (
        request_root_span().get_span_context().span_id
        == server.get_span_context().span_id
    )


def test_guardrail_span_anchors_to_root_inside_active_phase_span():
    """A guardrail emitted from a failure hook that runs inside the live ``auth``
    span must still be a sibling of the LLM call under the request root, not a
    child of auth."""
    logger, exporter = _logger()
    server = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    set_request_root_span(server)
    entry = {"guardrail_name": "my_guard", "guardrail_status": "success"}
    with trace.use_span(server, end_on_exit=False):
        with logger.start_phase_span("auth /chat/completions"):
            logger.emit_guardrail_span(entry)
    server.end()
    by_name = {s.name: s for s in exporter.get_finished_spans()}
    guard = by_name["execute_guardrail my_guard"]
    auth_span = by_name["auth /chat/completions"]
    assert guard.parent.span_id == server.get_span_context().span_id
    assert guard.parent.span_id != auth_span.get_span_context().span_id


# --------------------------------------------------------------------------- #
#  LIT-4179 — proxy-level failures that never reach an LLM call must still stamp
#  the structured error.* attributes onto the request's spans, restoring the v1
#  behavior v2 dropped when it stopped subclassing ``OpenTelemetry``.
# --------------------------------------------------------------------------- #


def _proxy_exc(message, code):
    from litellm.proxy._types import ProxyException

    return ProxyException(message=message, type="bad_request_error", param=None, code=code)


def test_async_post_call_failure_hook_stamps_error_on_root_span():
    """PATH B: an endpoint-level failure (empty body rejected before dispatch)
    reaches ``async_post_call_failure_hook``; it must stamp error.* + an exception
    event on the anchored request root span."""
    from litellm.proxy._types import UserAPIKeyAuth

    logger, exporter = _logger()
    server = logger._emitter.start_span(SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME)
    set_request_root_span(server)
    exc = _proxy_exc("litellm.BadRequestError: messages is required", 400)
    result = asyncio.run(
        logger.async_post_call_failure_hook(
            request_data={}, original_exception=exc, user_api_key_dict=UserAPIKeyAuth()
        )
    )
    server.end()
    assert result is None
    (span,) = exporter.get_finished_spans()
    assert span.attributes["error.type"] == "ProxyException"
    assert "messages is required" in span.attributes["error.message"]
    assert span.attributes["litellm.provider.error.code"] == "400"
    assert span.status.status_code is StatusCode.ERROR
    assert any(e.name == "exception" for e in span.events)


def test_async_post_call_failure_hook_falls_back_to_user_api_key_parent_span():
    """With no anchor set (a path that never captured the root), the hook must fall
    back to ``user_api_key_dict.parent_otel_span`` rather than dropping the error."""
    from litellm.proxy._types import UserAPIKeyAuth

    logger, exporter = _logger()
    server = logger._emitter.start_span(SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME)
    asyncio.run(
        logger.async_post_call_failure_hook(
            request_data={},
            original_exception=_proxy_exc("boom", 401),
            user_api_key_dict=UserAPIKeyAuth(parent_otel_span=server),
        )
    )
    server.end()
    (span,) = exporter.get_finished_spans()
    assert span.attributes["error.type"] == "ProxyException"
    assert span.attributes["litellm.provider.error.code"] == "401"


def test_record_error_attributes_on_span_decorates_without_ending():
    """PATH A: a failure that dies before any LLM-call span (malformed body,
    validation) is stamped onto the instrumentor-owned SERVER span. The method must
    not end the span or emit a duplicate exception event, and must pin error.code
    to the real response status (not the exception's own code)."""
    logger, exporter = _logger()
    server = logger._emitter.start_span(SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME)
    logger.record_error_attributes_on_span(server, _proxy_exc("Invalid JSON body", 400), 422)
    assert server.is_recording()
    server.end()
    (span,) = exporter.get_finished_spans()
    assert span.attributes["error.type"] == "ProxyException"
    assert span.attributes["error.message"] == "Invalid JSON body"
    assert span.attributes["litellm.provider.error.code"] == "422"
    assert all(e.name != "exception" for e in span.events)


def test_record_error_attributes_on_span_ignores_below_400_and_missing_span():
    logger, _ = _logger()
    server = logger._emitter.start_span(SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME)
    logger.record_error_attributes_on_span(None, _proxy_exc("boom", 400), 400)  # no span → no-op
    logger.record_error_attributes_on_span(server, None, 400)  # no exception → no-op
    server.end()
    assert "error.type" not in (server.attributes or {})


def test_start_phase_span_stamps_error_attributes_on_failure():
    """An ``auth`` phase span that dies (expired key) must carry the structured
    error.* attributes, not only the exception event ``use_span`` records."""
    logger, exporter = _logger()
    server = logger._emitter.start_span(SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME)
    set_request_root_span(server)
    exc = _proxy_exc("Authentication Error, ExpiredToken", 401)
    with trace.use_span(server, end_on_exit=False):
        with contextlib.suppress(Exception):
            with logger.start_phase_span("auth /chat/completions"):
                raise exc
    server.end()
    by_name = {s.name: s for s in exporter.get_finished_spans()}
    auth = by_name["auth /chat/completions"]
    assert auth.attributes["error.type"] == "ProxyException"
    assert "ExpiredToken" in auth.attributes["error.message"]
    assert auth.attributes["litellm.provider.error.code"] == "401"
    assert auth.status.status_code is StatusCode.ERROR
    assert any(e.name == "exception" for e in auth.events)


def test_start_phase_span_success_carries_no_error():
    logger, exporter = _logger()
    with logger.start_phase_span("auth /chat/completions"):
        pass
    (span,) = exporter.get_finished_spans()
    assert "error.type" not in span.attributes
    assert span.status.status_code is not StatusCode.ERROR


def test_real_logging_pre_call_opens_span_end_to_end():
    """Regression guard: a real ``LiteLLMLoggingObj.pre_call`` must fire
    ``log_pre_api_call`` on the V2 logger (via ``litellm.input_callback``), so the
    boundary span is opened and then closed by the success callback. If the logger
    is not wired into ``input_callback``, no span is produced at all."""
    import litellm
    from litellm.litellm_core_utils.litellm_logging import Logging

    logger, exporter = _logger()
    # Register exactly this logger as the (only) input callback pre_call iterates.
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(litellm, "input_callback", [logger], raising=False)
    try:
        logging_obj = Logging(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
            call_type="acompletion",
            start_time=datetime.now(),
            litellm_call_id="call_e2e",
            function_id="fn",
        )
        # The wrapper always runs this before pre_call — it's what seeds
        # ``litellm_params`` and ``litellm_call_id`` into ``model_call_details``
        # (the call id is how the close callback correlates back to this span).
        logging_obj.update_environment_variables(
            litellm_params={"metadata": {}},
            optional_params={},
            model="gpt-4o",
        )
        # pre_call fires log_pre_api_call → opens the boundary span on the obj.
        logging_obj.pre_call(input="hi", api_key="sk-test")
        # The success callback closes it, reading the typed payload.
        logging_obj.model_call_details["standard_logging_object"] = _payload(
            litellm_call_id="call_e2e"
        )
        asyncio.run(
            logger.async_log_success_event(
                logging_obj.model_call_details, None, None, None
            )
        )
    finally:
        monkeypatch.undo()
    (span,) = exporter.get_finished_spans()
    assert span.name == "chat gpt-4o"


def test_deferred_span_parents_to_ambient_at_close():
    """When ``pre_call`` runs off the request task (a sync-only provider driven
    through a thread pool, where contextvars don't follow), no ambient parent is
    visible there, so span creation is deferred. The async callback — whose worker
    context was copied from the request task and so still carries the server span —
    then creates it parented to that server span, not as an orphan root."""
    logger, exporter = _logger()
    kwargs = _kwargs()
    # pre_call with NO ambient span (the thread-pool case) → deferred.
    logger.log_pre_api_call(model="gpt-4o", messages=[], kwargs=kwargs)
    server = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    # The close callback runs with the (worker-copied) server span ambient.
    with trace.use_span(server, end_on_exit=False):
        asyncio.run(logger.async_log_success_event(kwargs, None, None, None))
    server.end()
    by_name = {s.name: s for s in exporter.get_finished_spans()}
    llm_span = by_name["chat gpt-4o"]
    assert llm_span.parent.span_id == server.get_span_context().span_id


# Inbound ``traceparent`` propagation is now the FastAPI instrumentor's job
# (see proxy_server's startup mount + ``test_otel_v2_mount``), not the logger's.


# --------------------------------------------------------------------------- #
#  Baggage promotion (LLM call writes identity into baggage so child spans
#  inherit team/key/model attrs).
# --------------------------------------------------------------------------- #


def test_baggage_identity_promoted_onto_llm_call():
    """On the deferred (SDK / no-proxy) path the callback seeds identity Baggage
    from the payload so the span is still labeled with team/key. (On the proxy
    boundary path identity rides in from auth-seeded ambient Baggage instead.)"""
    logger, exporter = _logger()
    _emit_llm(logger)
    (span,) = exporter.get_finished_spans()
    assert span.attributes[LiteLLM.TEAM_ID] == "t1"
    assert span.attributes[LiteLLM.TEAM_ALIAS] == "team one"
    assert span.attributes[GenAI.REQUEST_MODEL] == "gpt-4o"


class _Auth:
    """Stub matching the ``UserAPIKeyAuth`` fields the logger reads."""

    team_id = "t1"
    team_alias = "team one"
    team_metadata = {"tier": "gold", "cost_center": "42"}
    api_key = "hash1"
    user_id = "u1"
    org_id = None
    key_alias = "k1"
    end_user_id = None


def test_provider_model_and_team_metadata_on_real_boundary_flow():
    """End-to-end on the proxy boundary path (the gap a pure-emitter test misses):

    - ``litellm.team.metadata`` (filtered to the allowlisted sub-keys) is known
      at auth, so it rides identity Baggage seeded there onto EVERY span
      (server + LLM call).
    - ``litellm.provider.model`` is only known once routing picks a deployment
      (in the payload at close), AFTER the auth seed and AFTER the boundary span
      starts — so it can't ride Baggage. It's stamped directly on the LLM-call
      span by the mapper, and is absent from the server span (which starts first).
    """
    import json

    logger, exporter = _logger(team_metadata_keys=["tier", "cost_center"])
    server = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    payload = _payload(
        hidden_params={"litellm_model_name": "azure/my-deployment"},
        metadata={
            "user_api_key_team_id": "t1",
            "user_api_key_team_alias": "team one",
            "user_api_key_hash": "hash1",
            "user_api_key_team_metadata": {"tier": "gold", "cost_center": "42"},
        },
    )
    kwargs = _kwargs(payload=payload)
    with trace.use_span(server, end_on_exit=False):
        # auth boundary: seed identity (provider model unknown here)
        logger.seed_request_identity(_Auth(), model="gpt-4o")
        # pre_call boundary opens the LLM span; success closes it from the payload
        logger.log_pre_api_call(model="gpt-4o", messages=[], kwargs=kwargs)
        asyncio.run(logger.async_log_success_event(kwargs, None, None, None))
    server.end()

    spans = {s.name: s for s in exporter.get_finished_spans()}
    llm = spans["chat gpt-4o"]
    srv = spans[LITELLM_PROXY_REQUEST_SPAN_NAME]
    # provider model: on the LLM call span, NOT the server span
    assert llm.attributes[LiteLLM.PROVIDER_MODEL] == "azure/my-deployment"
    assert LiteLLM.PROVIDER_MODEL not in srv.attributes
    # team metadata: on every span, JSON-serialized
    expected = {"tier": "gold", "cost_center": "42"}
    assert json.loads(llm.attributes[LiteLLM.TEAM_METADATA]) == expected
    assert json.loads(srv.attributes[LiteLLM.TEAM_METADATA]) == expected


def test_pre_call_hook_seeds_baggage_onto_server_and_child_spans():
    """The pre-call hook seeds identity Baggage in the request context so the
    server span (stamped directly) AND later child spans (service here, via the
    Baggage processor) carry identity — not just the LLM-call span."""
    logger, exporter = _logger()
    server = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )

    async def _flow():
        # pre-call seeds baggage + stamps the active server span
        await logger.async_pre_call_hook(
            _Auth(), None, {"model": "gpt-4o"}, "completion"
        )
        # a later service call (same task) must inherit the identity
        await logger.async_service_success_hook(
            payload=_ServicePayload("redis", "set"), parent_otel_span=server
        )

    with trace.use_span(server, end_on_exit=False):
        asyncio.run(_flow())
    server.end()

    spans = {s.name: s for s in exporter.get_finished_spans()}
    redis = spans["redis set"]
    assert redis.attributes[LiteLLM.TEAM_ID] == "t1"
    assert redis.attributes[LiteLLM.KEY_HASH] == "hash1"
    assert redis.attributes[f"{LiteLLM.METADATA_PREFIX}user_api_key_user_id"] == "u1"
    srv = spans[LITELLM_PROXY_REQUEST_SPAN_NAME]
    assert (
        srv.attributes[LiteLLM.TEAM_ID] == "t1"
    )  # stamped directly on the server span
    assert srv.attributes[f"{LiteLLM.METADATA_PREFIX}user_api_key_user_id"] == "u1"


# --------------------------------------------------------------------------- #
#  Service hooks (Phase 3)
# --------------------------------------------------------------------------- #


class _Service:
    """Stub matching ``ServiceTypes(str, Enum)``."""

    def __init__(self, value):
        self.value = value


class _ServicePayload:
    def __init__(self, service="redis", call_type="set", error=None):
        self.service = _Service(service)
        self.call_type = call_type
        self.error = error


def _service_parent(logger):
    """Helper: a live PROXY_REQUEST span to parent service spans under."""
    return logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )


def test_async_service_success_hook_emits_service_span():
    logger, exporter = _logger()
    parent = _service_parent(logger)
    try:
        asyncio.run(
            logger.async_service_success_hook(
                payload=_ServicePayload("redis", "set"),
                parent_otel_span=parent,
                event_metadata={"key1": "val1"},
            )
        )
    finally:
        parent.end()
    by_name = {s.name: s for s in exporter.get_finished_spans()}
    # Name disambiguates calls to the same service; redis is an outbound
    # datastore call, so it's a CLIENT span with db.* semconv.
    span = by_name["redis set"]
    assert span.kind is SpanKind.CLIENT
    assert span.attributes["db.system.name"] == "redis"
    assert span.attributes["db.operation.name"] == "set"
    assert span.attributes[LiteLLM.SERVICE_NAME] == "redis"
    assert span.attributes[LiteLLM.SERVICE_CALL_TYPE] == "set"
    # Canonical (V2) namespaced metadata key
    assert span.attributes[f"{LiteLLM.METADATA_PREFIX}key1"] == "val1"
    # V1 bare key (legacy dual-emit)
    assert span.attributes["key1"] == "val1"
    assert span.attributes["service"] == "redis"  # V1 bare key
    assert span.attributes["call_type"] == "set"  # V1 bare key
    # Success leaves status UNSET (semconv default), not forced OK.
    assert span.status.status_code is StatusCode.UNSET


def test_async_service_failure_hook_marks_error_status():
    logger, exporter = _logger()
    parent = _service_parent(logger)
    try:
        asyncio.run(
            logger.async_service_failure_hook(
                payload=_ServicePayload("postgres", "query"),
                error="boom",
                parent_otel_span=parent,
            )
        )
    finally:
        parent.end()
    by_name = {s.name: s for s in exporter.get_finished_spans()}
    span = by_name["postgres query"]
    assert span.kind is SpanKind.CLIENT
    assert span.attributes["db.system.name"] == "postgresql"
    assert span.status.status_code is StatusCode.ERROR
    # Without an explicit error_type from the payload, V2 stamps the fallback.
    assert span.attributes["error.type"] == "error"
    assert span.attributes[LiteLLM.SERVICE_NAME] == "postgres"


def test_async_service_failure_hook_preserves_payload_error_over_override():
    """When the payload itself carries an error, that takes precedence over the override."""
    logger, exporter = _logger()
    parent = _service_parent(logger)
    try:
        asyncio.run(
            logger.async_service_failure_hook(
                payload=_ServicePayload("postgres", "query", error="db-down"),
                error="override-only-used-when-payload-clean",
                parent_otel_span=parent,
            )
        )
    finally:
        parent.end()
    by_name = {s.name: s for s in exporter.get_finished_spans()}
    span = by_name["postgres query"]
    assert span.status.status_code is StatusCode.ERROR
    assert "db-down" in (span.status.description or "")


def test_metrics_only_ping_without_timing_or_parent_is_noop():
    """A success with no timing and no parent is a prometheus-only ping (the
    per-request ``self`` latency hook, in-memory queue gauges) — not a traceable
    operation, so no span is emitted."""
    logger, exporter = _logger()
    asyncio.run(
        logger.async_service_success_hook(
            payload=_ServicePayload(), parent_otel_span=None
        )
    )
    assert exporter.get_finished_spans() == ()


def test_background_service_call_with_timing_emits_root_span():
    """A background datastore call (no request → no parent) but with real timing
    still emits — as its own root trace — instead of being dropped."""
    logger, exporter = _logger()
    asyncio.run(
        logger.async_service_success_hook(
            payload=_ServicePayload("postgres", "query"),
            parent_otel_span=None,
            start_time=1.0,
            end_time=2.0,
        )
    )
    spans = exporter.get_finished_spans()
    assert [s.name for s in spans] == ["postgres query"]
    # No parent → it's a root span of its own trace.
    assert spans[0].parent is None
    assert spans[0].kind is SpanKind.CLIENT


def test_internal_service_call_is_internal_kind_without_db_attrs():
    """A genuine internal service (background job) is an INTERNAL span, no db.*."""
    logger, exporter = _logger()
    asyncio.run(
        logger.async_service_success_hook(
            payload=_ServicePayload("reset_budget_job", "reset_budget"),
            parent_otel_span=None,
            start_time=1.0,
            end_time=2.0,
        )
    )
    span = exporter.get_finished_spans()[0]
    assert span.name == "reset_budget_job reset_budget"
    assert span.kind is SpanKind.INTERNAL
    assert "db.system.name" not in span.attributes
    assert span.attributes[LiteLLM.SERVICE_NAME] == "reset_budget_job"


def test_metrics_only_services_emit_no_span():
    """self / router / proxy_pre_call / auth duplicate gen-AI spans or get a live
    phase span — they are metrics-only and must not produce a service span."""
    for service in ("self", "router", "proxy_pre_call", "auth"):
        logger, exporter = _logger()
        asyncio.run(
            logger.async_service_success_hook(
                payload=_ServicePayload(service, "x"),
                parent_otel_span=None,
                start_time=1.0,
                end_time=2.0,
            )
        )
        assert exporter.get_finished_spans() == (), f"{service} should emit no span"


def test_service_span_inherits_parent_when_provided():
    logger, exporter = _logger()
    parent = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    try:
        asyncio.run(
            logger.async_service_success_hook(
                payload=_ServicePayload(), parent_otel_span=parent
            )
        )
    finally:
        parent.end()
    by_name = {s.name: s for s in exporter.get_finished_spans()}
    assert (
        by_name["redis set"].parent.span_id
        == by_name[LITELLM_PROXY_REQUEST_SPAN_NAME].get_span_context().span_id
    )


def test_service_span_prefers_ambient_context_over_threaded_parent():
    """Service/DB spans parent to the active (ambient) span when there is one, so
    they nest under whatever phase is active (e.g. a DB lookup under the live
    ``auth`` span). The threaded ``parent_otel_span`` is only a fallback for when
    ambient has no live span (a background service call)."""
    logger, exporter = _logger()
    ambient = logger._emitter.start_span(SpanRole.LLM_CALL, "chat gpt-4o")
    threaded = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    try:
        with trace.use_span(ambient, end_on_exit=False):
            asyncio.run(
                logger.async_service_success_hook(
                    payload=_ServicePayload("redis", "get"),
                    parent_otel_span=threaded,
                )
            )
    finally:
        ambient.end()
        threaded.end()
    by_name = {s.name: s for s in exporter.get_finished_spans()}
    assert by_name["redis get"].parent.span_id == ambient.get_span_context().span_id


# --------------------------------------------------------------------------- #
#  Proxy SERVER span lifecycle
# --------------------------------------------------------------------------- #


def test_create_proxy_request_started_span_returns_ambient_span():
    """V2 doesn't create a server span (the instrumentor does), but it returns
    the active server span so the proxy can thread it as the service-span parent
    — service logging only fires the OTel hook when that parent is non-None."""
    logger, exporter = _logger()
    # No ambient recordable span → None (and creates nothing).
    assert (
        logger.create_litellm_proxy_request_started_span(
            start_time=datetime.now(timezone.utc), headers={"traceparent": "x"}
        )
        is None
    )
    assert exporter.get_finished_spans() == ()
    # With an active server span, return it (do NOT create a new one).
    server = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    with trace.use_span(server, end_on_exit=False):
        got = logger.create_litellm_proxy_request_started_span(
            start_time=datetime.now(timezone.utc), headers=None
        )
    server.end()
    assert got is server


# --------------------------------------------------------------------------- #
#  Constructor / proxy global guard
# --------------------------------------------------------------------------- #


def test_constructor_accepts_v1_compatible_kwargs():
    """Mirrors V1's positional shape — config / callback_name / providers / **kwargs."""
    cfg = OpenTelemetryV2Config(exporter="in_memory")
    tp = providers.build_tracer_provider(cfg)
    logger = OpenTelemetryV2(
        config=cfg,
        callback_name="otel",
        tracer_provider=tp,
        logger_provider=None,
        meter_provider=None,
        turn_off_message_logging=True,
    )
    assert logger.callback_name == "otel"
    assert logger.turn_off_message_logging is True
    assert logger.tracer is not None


def test_default_config_reads_env(monkeypatch):
    """No explicit config → reads env (exporter=console by default)."""
    monkeypatch.delenv("OTEL_EXPORTER", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_PROTOCOL", raising=False)
    logger = OpenTelemetryV2(
        tracer_provider=providers.build_tracer_provider(
            OpenTelemetryV2Config(exporter="in_memory")
        )
    )
    assert logger.config.exporter == "console"


def test_proxy_global_first_registered_wins(monkeypatch):
    """``_init_otel_logger_on_litellm_proxy`` claims the global only when empty."""
    proxy_server = pytest.importorskip("litellm.proxy.proxy_server")
    monkeypatch.setattr(proxy_server, "open_telemetry_logger", None, raising=False)
    cfg = OpenTelemetryV2Config(exporter="in_memory")
    tp = providers.build_tracer_provider(cfg)

    first = OpenTelemetryV2(config=cfg, tracer_provider=tp)
    assert proxy_server.open_telemetry_logger is first

    second = OpenTelemetryV2(config=cfg, tracer_provider=tp)
    # Global still points at the first registration.
    assert proxy_server.open_telemetry_logger is first
    assert second is not first


def test_select_global_otel_v2_logger_reuses_existing_preset_logger():
    """The global-provider selection must reuse the logger the callback factory
    already built (e.g. an arize preset logger that folds the OTEL_* base exporter
    and its own exporter into one logger), not mint a second generic one.

    Regression for the orphan span: the startup publish used to search
    ``service_callback`` (which a preset logger does not always reach), miss the
    existing logger, and build a second generic ``OpenTelemetryV2`` whose provider
    became the OTel global. The server span then exported through that generic
    provider while the preset logger's gen-ai spans exported to the preset backend,
    so on that backend the LLM span had no parent. Selecting from the loggers the
    factory registered keeps one logger, one provider, one connected trace.
    """
    from litellm.integrations.otel.logger import select_global_otel_v2_logger

    cfg = OpenTelemetryV2Config(exporter="in_memory")
    tp = providers.build_tracer_provider(cfg)
    preset_logger = OpenTelemetryV2(
        config=cfg, callback_name="arize", tracer_provider=tp
    )

    chosen = select_global_otel_v2_logger([object(), preset_logger, object()])
    assert chosen is preset_logger


def test_select_global_otel_v2_logger_prefers_registered_owner_over_list_scan():
    """Selection reuses the canonical owner the factory registered, not whatever
    the ``in_memory_loggers`` scan happens to reach first.

    The factory designates one logger as ``proxy_server.open_telemetry_logger`` the
    moment it builds the first one, and every other v2 path (guardrail, seed,
    phase spans) routes through that owner. With two presets configured, the list
    scan's "first ``OpenTelemetryV2``" is order-dependent and could disagree with
    that owner, publishing one backend's provider as the global while the rest of
    the v2 code emits through another. Passing the registered owner pins the global
    provider to the same logger the rest of the code already uses.
    """
    from litellm.integrations.otel.logger import select_global_otel_v2_logger

    cfg = OpenTelemetryV2Config(exporter="in_memory")
    owner = OpenTelemetryV2(
        config=cfg,
        callback_name="arize",
        tracer_provider=providers.build_tracer_provider(cfg),
    )
    other = OpenTelemetryV2(
        config=cfg,
        callback_name="langfuse_otel",
        tracer_provider=providers.build_tracer_provider(cfg),
    )

    chosen = select_global_otel_v2_logger([other, owner], registered=owner)
    assert chosen is owner


def test_select_global_otel_v2_logger_builds_one_when_none_registered():
    """With no logger registered, selection builds exactly one generic logger so
    the proxy still publishes a provider; it must not return ``None``."""
    from litellm.integrations.otel.logger import select_global_otel_v2_logger

    chosen = select_global_otel_v2_logger([])
    assert isinstance(chosen, OpenTelemetryV2)


def test_publish_global_otel_v2_provider_sets_selected_logger_provider():
    """The startup publish must set the OTel global provider to the *selected*
    logger's provider (the preset logger that owns every exporter), so the FastAPI
    server span and the gen-ai spans share one provider and one trace.

    Drives the publish step the proxy runs at startup, with the global-setter
    injected so no real global OTel state is mutated. Guards the wiring that a unit
    test would otherwise miss: that the published provider is the selected logger's,
    not some other.
    """
    from litellm.integrations.otel.logger import publish_global_otel_v2_provider

    cfg = OpenTelemetryV2Config(exporter="in_memory")
    tp = providers.build_tracer_provider(cfg)
    preset_logger = OpenTelemetryV2(
        config=cfg, callback_name="arize", tracer_provider=tp
    )

    published = []
    chosen = publish_global_otel_v2_provider(
        [object(), preset_logger], published.append
    )

    assert chosen is preset_logger
    assert published == [preset_logger._tracer_provider]


def test_registers_into_litellm_service_callback(monkeypatch):
    """The logger must mutate ``litellm.service_callback`` in place. An empty
    list is falsy, so a ``getattr(..) or []`` would append to a throwaway local
    and service spans (Redis, …) would silently never fire on this logger.
    """
    import litellm

    pytest.importorskip("litellm.proxy.proxy_server")
    monkeypatch.setattr(litellm, "service_callback", [], raising=False)
    cfg = OpenTelemetryV2Config(exporter="in_memory")
    tp = providers.build_tracer_provider(cfg)

    first = OpenTelemetryV2(config=cfg, tracer_provider=tp)
    assert first in litellm.service_callback

    # A second OTel logger sees one is already registered and does not duplicate.
    OpenTelemetryV2(config=cfg, tracer_provider=tp)
    otel_registrations = [
        cb
        for cb in litellm.service_callback
        if cb.__class__.__module__.startswith("litellm.integrations.otel")
    ]
    assert len(otel_registrations) == 1


def test_registers_into_litellm_input_callback(monkeypatch):
    """The logger must land in ``litellm.input_callback`` — the list
    ``Logging.pre_call`` iterates to fire ``log_pre_api_call``. Without this the
    boundary hook never runs and the gen-AI span is never opened (the span goes
    completely missing). Deduped like ``service_callback``.
    """
    import litellm

    pytest.importorskip("litellm.proxy.proxy_server")
    monkeypatch.setattr(litellm, "input_callback", [], raising=False)
    cfg = OpenTelemetryV2Config(exporter="in_memory")
    tp = providers.build_tracer_provider(cfg)

    first = OpenTelemetryV2(config=cfg, tracer_provider=tp)
    assert first in litellm.input_callback

    OpenTelemetryV2(config=cfg, tracer_provider=tp)
    otel_registrations = [
        cb
        for cb in litellm.input_callback
        if cb.__class__.__module__.startswith("litellm.integrations.otel")
    ]
    assert len(otel_registrations) == 1


def test_registers_into_async_success_and_failure_callbacks(monkeypatch):
    """The logger must self-register into ``litellm._async_success_callback`` and
    ``litellm._async_failure_callback`` — the lists ``Logging.async_success_handler``
    / ``async_failure_handler`` iterate to fire ``async_log_success_event`` /
    ``async_log_failure_event``, where the boundary span is *closed*.

    ``input_callback`` opens the span; these lists close it. Relying only on the
    proxy's ``litellm.callbacks`` fan-out to populate them is not enough: a logger
    that reached litellm via ``service_callback`` / ``success_callback`` (or was
    created after the fan-out ran) is absent from ``litellm.callbacks``, so on a
    pass-through request (which never runs ``function_setup``) the span opens and is
    never ended — the gen-AI span leaks and never exports, while DB/service spans
    still show up. Self-registration here guarantees every open has a close.
    """
    import litellm

    pytest.importorskip("litellm.proxy.proxy_server")
    monkeypatch.setattr(litellm, "_async_success_callback", [], raising=False)
    monkeypatch.setattr(litellm, "_async_failure_callback", [], raising=False)
    cfg = OpenTelemetryV2Config(exporter="in_memory")
    tp = providers.build_tracer_provider(cfg)

    first = OpenTelemetryV2(config=cfg, tracer_provider=tp)
    assert first in litellm._async_success_callback
    assert first in litellm._async_failure_callback

    # Deduped — a second otel logger doesn't double up the close hook.
    OpenTelemetryV2(config=cfg, tracer_provider=tp)
    for callback_list in (
        litellm._async_success_callback,
        litellm._async_failure_callback,
    ):
        otel_registrations = [
            cb
            for cb in callback_list
            if cb.__class__.__module__.startswith("litellm.integrations.otel")
        ]
        assert len(otel_registrations) == 1


def test_boundary_span_closes_without_proxy_fanout(monkeypatch):
    """A span opened at ``pre_call`` is still closed and exported when the logger is
    registered ONLY via its own ``__init__`` (no ``litellm.callbacks`` fan-out, as
    happens for a logger configured through ``service_callback``) and the close runs
    through the real ``async_success_handler``.

    Self-registration must wire both ends: the open hook (``input_callback``) and the
    close hook (``_async_success_callback``). If only the open end were wired the span
    would leak — opened but never closed, never exported.
    """
    import litellm
    from litellm.litellm_core_utils.litellm_logging import Logging

    pytest.importorskip("litellm.proxy.proxy_server")
    monkeypatch.setattr(litellm, "input_callback", [], raising=False)
    monkeypatch.setattr(litellm, "_async_success_callback", [], raising=False)
    monkeypatch.setattr(litellm, "_async_failure_callback", [], raising=False)
    # Crucially: the logger is NOT in litellm.callbacks, so the proxy fan-out would
    # never reach it. Only __init__ self-registration wires the open + close hooks.
    monkeypatch.setattr(litellm, "callbacks", [], raising=False)

    logger, exporter = _logger()
    logging_obj = Logging(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="pass_through_endpoint",
        start_time=datetime.now(),
        litellm_call_id="pt_leak",
        function_id="fn",
    )
    logging_obj.update_environment_variables(
        litellm_params={"metadata": {}},
        optional_params={},
        model="gpt-4o",
    )
    logging_obj.model_call_details["litellm_call_id"] = "pt_leak"
    # pre_call opens the boundary span (logger is in input_callback).
    logging_obj.pre_call(input="hi", api_key="")
    assert "pt_leak" in logger._open_llm_calls
    # The close runs through the real async_success_handler, which iterates
    # _async_success_callback — where the logger self-registered.
    logging_obj.model_call_details["standard_logging_object"] = _payload(
        litellm_call_id="pt_leak"
    )
    asyncio.run(
        logging_obj.async_success_handler(
            result=None, start_time=datetime.now(), end_time=datetime.now()
        )
    )
    assert "pt_leak" not in logger._open_llm_calls  # carrier closed, not leaked
    (span,) = exporter.get_finished_spans()
    assert span.name == "chat gpt-4o"


# --------------------------------------------------------------------------- #
#  Guardrail span placement: request-level parent + real execution timestamps
# --------------------------------------------------------------------------- #


def _guardrail_entry(*, start, end):
    return {
        "guardrail_name": "openai-moderation",
        "guardrail_mode": "pre_call",
        "guardrail_status": "success",
        "start_time": start,
        "end_time": end,
        "duration": end - start,
    }


def test_guardrail_span_parents_to_ambient_server_span():
    """``emit_guardrail_span`` runs in the request task with the server span
    ambient, so with no explicit anchor set the guardrail span parents to it.
    (Auth already finished, so no phase span is active.)"""
    logger, exporter = _logger()
    server = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    entry = _guardrail_entry(start=1000.0, end=1000.5)
    try:
        with trace.use_span(server, end_on_exit=False):
            logger.emit_guardrail_span(entry)
    finally:
        server.end()
    g = {s.name: s for s in exporter.get_finished_spans()}[
        "execute_guardrail openai-moderation"
    ]
    assert g.parent.span_id == server.get_span_context().span_id


def test_guardrail_span_uses_actual_execution_timestamps():
    """A pre_call guardrail's span carries its real start/end (from the logging
    entry), so it sorts before the LLM call instead of at emission time."""
    logger, exporter = _logger()
    server = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    entry = _guardrail_entry(start=1700.0, end=1700.25)
    try:
        with trace.use_span(server, end_on_exit=False):
            logger.emit_guardrail_span(entry)
    finally:
        server.end()
    g = {s.name: s for s in exporter.get_finished_spans()}[
        "execute_guardrail openai-moderation"
    ]
    assert g.start_time == to_ns(1700.0)
    assert g.end_time == to_ns(1700.25)


def test_emit_guardrail_span_anchors_to_root_not_ambient_phase_span():
    """With an explicit request-root anchor set, the guardrail span parents to it
    even while a phase span is the active OTel context — the anchor wins over
    ambient, so a guardrail emitted mid-``auth`` is a sibling of the LLM call, not
    a child of ``auth``."""
    logger, exporter = _logger()
    server = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    set_request_root_span(server)
    entry = _guardrail_entry(start=2000.0, end=2000.1)
    with logger.start_phase_span("auth /chat/completions"):
        logger.emit_guardrail_span(entry)
    server.end()
    by_name = {s.name: s for s in exporter.get_finished_spans()}
    guard = by_name["execute_guardrail openai-moderation"]
    auth_span = by_name["auth /chat/completions"]
    assert guard.parent.span_id == server.get_span_context().span_id
    assert guard.parent.span_id != auth_span.get_span_context().span_id


def test_module_level_emit_guardrail_span_routes_to_registered_logger(monkeypatch):
    """The module-level entry point custom_guardrail calls routes the entry to the
    single registered v2 logger and emits exactly one span."""
    import litellm.integrations.otel.logger as otel_logger

    logger, exporter = _logger()
    monkeypatch.setattr(otel_logger, "_registered_v2_logger", lambda: logger)

    otel_logger.emit_guardrail_span(_guardrail_entry(start=3000.0, end=3000.2))

    names = [s.name for s in exporter.get_finished_spans()]
    assert names.count("execute_guardrail openai-moderation") == 1


def test_module_level_emit_guardrail_span_noop_without_registered_logger(monkeypatch):
    """No registered v2 logger (SDK path / OTel not configured) → emitting is a
    no-op rather than an error."""
    import litellm.integrations.otel.logger as otel_logger

    monkeypatch.setattr(otel_logger, "_registered_v2_logger", lambda: None)
    otel_logger.emit_guardrail_span(_guardrail_entry(start=1.0, end=2.0))


def test_module_level_emit_guardrail_span_swallows_emit_errors(monkeypatch):
    """Span emission is best-effort: a logger that raises must never propagate out
    of the guardrail-recording path and break guardrail evaluation."""
    import litellm.integrations.otel.logger as otel_logger

    class _Boom:
        def emit_guardrail_span(self, entry):
            raise RuntimeError("emit blew up")

    monkeypatch.setattr(otel_logger, "_registered_v2_logger", lambda: _Boom())
    otel_logger.emit_guardrail_span(_guardrail_entry(start=1.0, end=2.0))


# --------------------------------------------------------------------------- #
#  Metrics: invalid attribute-filter config is visible, not a silent no-op
# --------------------------------------------------------------------------- #


def _emitted_metric_names(reader) -> set:
    data = reader.get_metrics_data()
    if data is None:
        return set()
    return {
        m.name
        for rm in data.resource_metrics
        for sm in rm.scope_metrics
        for m in sm.metrics
        if any(m.data.data_points)
    }


def _metric_success_kwargs() -> dict:
    return {
        "model": "gpt-4o-mini",
        "call_type": "acompletion",
        "litellm_params": {"custom_llm_provider": "openai"},
        "optional_params": {},
        "response_cost": 0.001,
        "standard_logging_object": {"metadata": {}, "hidden_params": {}},
    }


def test_invalid_metric_filter_logged_once_records_nothing(caplog, monkeypatch):
    """An invalid ``callback_settings.otel.attributes`` (include_list + exclude_list
    both set) must make the operator-fixable config error visible once at ERROR and
    record no metrics — without raising out of the success path and without
    per-request log spam. Mirrors the v1 fix against the silent-no-op failure mode.
    """
    import logging

    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    import litellm

    monkeypatch.setattr(
        litellm,
        "callback_settings",
        {
            "otel": {
                "attributes": {
                    "include_list": ["gen_ai.system"],
                    "exclude_list": ["hidden_params"],
                }
            }
        },
        raising=False,
    )

    cfg = OpenTelemetryV2Config(exporter="in_memory", enable_metrics=True)
    reader = InMemoryMetricReader()
    logger = OpenTelemetryV2(
        config=cfg,
        callback_name="otel",
        tracer_provider=providers.build_tracer_provider(cfg),
        meter_provider=MeterProvider(metric_readers=[reader]),
    )

    start = datetime.now(timezone.utc)
    end = start + timedelta(seconds=1)
    response_obj = {"usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    with caplog.at_level(logging.ERROR, logger="LiteLLM"):
        # Neither call may raise; the bad filter is caught in the logger.
        asyncio.run(
            logger.async_log_success_event(
                _metric_success_kwargs(), response_obj, start, end
            )
        )
        asyncio.run(
            logger.async_log_success_event(
                _metric_success_kwargs(), response_obj, start, end
            )
        )

    assert _emitted_metric_names(reader) == set()  # nothing recorded
    errors = [
        r
        for r in caplog.records
        if r.levelno == logging.ERROR and "metric filter" in r.getMessage()
    ]
    assert len(errors) == 1  # logged once, second bad record does not re-log


def test_valid_metric_filter_records_six_metrics(monkeypatch):
    """The happy path: with no attribute filter, a successful LLM call records all
    six GenAI histograms, and the token metric keeps its input/output split."""
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    import litellm

    monkeypatch.setattr(litellm, "callback_settings", {}, raising=False)

    cfg = OpenTelemetryV2Config(exporter="in_memory", enable_metrics=True)
    reader = InMemoryMetricReader()
    logger = OpenTelemetryV2(
        config=cfg,
        callback_name="otel",
        tracer_provider=providers.build_tracer_provider(cfg),
        meter_provider=MeterProvider(metric_readers=[reader]),
    )

    start = datetime.now(timezone.utc)
    end = start + timedelta(seconds=2)
    kwargs = _metric_success_kwargs()
    kwargs["api_call_start_time"] = start.timestamp()
    kwargs["completion_start_time"] = (start + timedelta(seconds=0.5)).timestamp()
    kwargs["end_time"] = end.timestamp()
    kwargs["optional_params"] = {"stream": True}
    response_obj = {"usage": {"prompt_tokens": 5, "completion_tokens": 7}}

    asyncio.run(logger.async_log_success_event(kwargs, response_obj, start, end))

    assert _emitted_metric_names(reader) == {
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
        "gen_ai.client.token.cost",
        "gen_ai.client.response.time_to_first_token",
        "gen_ai.client.response.time_per_output_token",
        "gen_ai.client.response.duration",
    }

    data = reader.get_metrics_data()
    token_types = {
        dp.attributes.get("gen_ai.token.type")
        for rm in data.resource_metrics
        for sm in rm.scope_metrics
        for m in sm.metrics
        if m.name == "gen_ai.client.token.usage"
        for dp in m.data.data_points
    }
    assert token_types == {"input", "output"}


def test_metrics_disabled_by_default_records_nothing(monkeypatch):
    """With ``enable_metrics`` off (the default), no meter is built and a success
    event records nothing — the default behavior must stay unchanged."""
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    import litellm

    monkeypatch.setattr(litellm, "callback_settings", {}, raising=False)

    cfg = OpenTelemetryV2Config(exporter="in_memory")  # enable_metrics defaults False
    reader = InMemoryMetricReader()
    logger = OpenTelemetryV2(
        config=cfg,
        callback_name="otel",
        tracer_provider=providers.build_tracer_provider(cfg),
        meter_provider=MeterProvider(metric_readers=[reader]),
    )
    assert logger._metrics_recorder is None

    start = datetime.now(timezone.utc)
    end = start + timedelta(seconds=1)
    asyncio.run(
        logger.async_log_success_event(
            _metric_success_kwargs(),
            {"usage": {"prompt_tokens": 1, "completion_tokens": 1}},
            start,
            end,
        )
    )
    assert _emitted_metric_names(reader) == set()


def _second_group_tracer(logger):
    """A second independent in-memory provider standing in for a second Resource group
    (e.g. a second Arize project); returns (tracer, exporter)."""
    exporter = InMemorySpanExporter()
    provider = providers.build_tracer_provider(logger.config, exporter=exporter)
    return provider.get_tracer("litellm"), exporter


def test_genai_span_emitted_to_every_group_live(monkeypatch):
    """Multi-destination fix (live path): when ``tracers_for`` returns two tracers (two
    Resource groups, e.g. two Arize projects), the gen-AI span opened at ``pre_call``
    must be opened+finished on BOTH -- the bug was only one project receiving it."""
    logger, exporter_a = _logger()
    monkeypatch.setattr(logger, "callback_name", "in_memory")
    tracer_b, exporter_b = _second_group_tracer(logger)
    monkeypatch.setattr(
        logger._tenant_tracers,
        "tracers_for",
        lambda default, dests: (default, tracer_b),
    )
    server = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    set_request_root_span(server)
    kwargs = _kwargs()
    _anchor([
            {"callback_name": "in_memory", "endpoint": "https://x/v1", "headers": {}}
        ])
    _emit_llm(logger, kwargs, ambient=server)
    server.end()
    assert [s.name for s in exporter_a.get_finished_spans()].count("chat gpt-4o") == 1
    assert [s.name for s in exporter_b.get_finished_spans()].count("chat gpt-4o") == 1


def test_genai_span_emitted_to_every_group_deferred(monkeypatch):
    """Same fix, deferred path (no carrier at ``pre_call``): ``emit_fanout`` dedups once
    on the call id then emits the span on every group's tracer, so both projects get
    exactly one."""
    logger, exporter_a = _logger()
    monkeypatch.setattr(logger, "callback_name", "in_memory")
    tracer_b, exporter_b = _second_group_tracer(logger)
    monkeypatch.setattr(
        logger._tenant_tracers,
        "tracers_for",
        lambda default, dests: (default, tracer_b),
    )
    server = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    set_request_root_span(server)
    kwargs = _kwargs()
    _anchor([
            {"callback_name": "in_memory", "endpoint": "https://x/v1", "headers": {}}
        ])
    # no pre_call -> no carrier -> deferred close path
    assert "call_1" not in logger._open_llm_calls
    asyncio.run(logger.async_log_success_event(kwargs, None, None, None))
    server.end()
    assert [s.name for s in exporter_a.get_finished_spans()].count("chat gpt-4o") == 1
    assert [s.name for s in exporter_b.get_finished_spans()].count("chat gpt-4o") == 1


def _generic_dest():
    return {
        "callback_name": "generic",
        "endpoint": "http://collector:4318",
        "headers": {},
    }


def test_generic_destination_emits_genai_span(monkeypatch):
    """Acceptance #1: a request whose only admin destination is a Generic OTLP
    destination emits the chat <model> gen-AI span (regression: 'generic' had no preset,
    so the gen-AI span was dropped and only proxy-internal spans reached the endpoint).
    The destination's callback_name='generic' is matched by the generic logger, and the
    span is routed through the per-destination tracer."""
    logger, exporter = _logger()
    monkeypatch.setattr(logger, "callback_name", "generic")
    routed: list = []
    monkeypatch.setattr(
        logger._tenant_tracers,
        "tracers_for",
        lambda default, dests: (routed.append(dests) or (default,)),
    )
    server = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    set_request_root_span(server)
    kwargs = _kwargs()
    _anchor([_generic_dest()])
    _emit_llm(logger, kwargs, ambient=server)
    server.end()
    assert "chat gpt-4o" in [s.name for s in exporter.get_finished_spans()]
    # the gen-AI span was routed for the generic destination (not an empty/global set)
    assert any(
        len(d) == 1 and d[0].endpoint == "http://collector:4318" for d in routed
    ), "generic destination was not routed to the generic logger's tracer"


def test_generic_destination_emits_error_span_on_failure(monkeypatch):
    """Acceptance #3: a FAILED call to a Generic OTLP destination still emits the
    chat <model> span, with OTEL status ERROR and the exception type."""
    logger, exporter = _logger()
    monkeypatch.setattr(logger, "callback_name", "generic")
    monkeypatch.setattr(
        logger._tenant_tracers, "tracers_for", lambda default, dests: (default,)
    )
    server = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    set_request_root_span(server)
    payload = _payload(
        status="failure",
        error_information={
            "error_class": "AuthenticationError",
            "error_message": "Incorrect API key",
        },
    )
    kwargs = _kwargs(payload=payload)
    _anchor([_generic_dest()])
    _emit_llm(logger, kwargs, ambient=server, fail=True)
    server.end()
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert "chat gpt-4o" in spans
    assert spans["chat gpt-4o"].status.status_code is StatusCode.ERROR
    assert spans["chat gpt-4o"].attributes["error.type"] == "AuthenticationError"
