"""
Tests for litellm/_service_logger.py

Regression test for KeyError: 'call_type' when async_log_success_event
is called without call_type in kwargs (e.g. from batch polling callbacks).
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch

import litellm
from litellm._service_logger import ServiceLogging
from litellm.types.services import ServiceTypes


@pytest.mark.asyncio
async def test_async_log_success_event_should_not_raise_when_call_type_missing():
    """
    When async_log_success_event is called with kwargs that omit 'call_type',
    it should not raise a KeyError. This happens in the batch polling flow
    where check_batch_cost.py creates a Logging object whose model_call_details
    don't include call_type.
    """
    service_logger = ServiceLogging(mock_testing=True)

    start_time = datetime(2026, 2, 13, 22, 35, 0)
    end_time = datetime(2026, 2, 13, 22, 35, 1)
    kwargs_without_call_type = {"model": "gpt-4", "stream": False}

    with patch.object(
        service_logger, "async_service_success_hook", new_callable=AsyncMock
    ) as mock_hook:
        await service_logger.async_log_success_event(
            kwargs=kwargs_without_call_type,
            response_obj=None,
            start_time=start_time,
            end_time=end_time,
        )

        mock_hook.assert_called_once()
        call_kwargs = mock_hook.call_args
        assert call_kwargs.kwargs["call_type"] == "unknown"


@pytest.mark.asyncio
async def test_async_log_success_event_should_pass_call_type_when_present():
    """
    When call_type IS present in kwargs, it should be forwarded correctly.
    """
    service_logger = ServiceLogging(mock_testing=True)

    start_time = datetime(2026, 2, 13, 22, 35, 0)
    end_time = datetime(2026, 2, 13, 22, 35, 1)
    kwargs_with_call_type = {
        "model": "gpt-4",
        "stream": False,
        "call_type": "aretrieve_batch",
    }

    with patch.object(
        service_logger, "async_service_success_hook", new_callable=AsyncMock
    ) as mock_hook:
        await service_logger.async_log_success_event(
            kwargs=kwargs_with_call_type,
            response_obj=None,
            start_time=start_time,
            end_time=end_time,
        )

        mock_hook.assert_called_once()
        call_kwargs = mock_hook.call_args
        assert call_kwargs.kwargs["call_type"] == "aretrieve_batch"


@pytest.mark.asyncio
async def test_async_log_success_event_should_handle_float_duration():
    """
    When start_time and end_time produce a float duration (not timedelta),
    it should still work correctly.
    """
    service_logger = ServiceLogging(mock_testing=True)

    start_time = 1000.0
    end_time = 1001.5

    with patch.object(
        service_logger, "async_service_success_hook", new_callable=AsyncMock
    ) as mock_hook:
        await service_logger.async_log_success_event(
            kwargs={"call_type": "completion"},
            response_obj=None,
            start_time=start_time,
            end_time=end_time,
        )

        mock_hook.assert_called_once()
        call_kwargs = mock_hook.call_args
        assert call_kwargs.kwargs["duration"] == 1.5


@pytest.mark.asyncio
async def test_async_log_success_event_forwards_start_and_end_time():
    """The LITELLM service span must carry its real execution window, so
    ``async_log_success_event`` forwards ``start_time``/``end_time`` to the service
    hook. Without forwarding, the span emits with a synthetic now() boundary
    instead of the call's actual timing."""
    service_logger = ServiceLogging(mock_testing=True)

    start_time = datetime(2026, 2, 13, 22, 35, 0)
    end_time = datetime(2026, 2, 13, 22, 35, 1)

    with patch.object(
        service_logger, "async_service_success_hook", new_callable=AsyncMock
    ) as mock_hook:
        await service_logger.async_log_success_event(
            kwargs={"call_type": "completion"},
            response_obj=None,
            start_time=start_time,
            end_time=end_time,
        )

    mock_hook.assert_called_once()
    forwarded = mock_hook.call_args.kwargs
    assert forwarded["start_time"] == start_time
    assert forwarded["end_time"] == end_time


# --------------------------------------------------------------------------- #
#  V2 OpenTelemetry service-span dispatch (regression: service spans were always
#  dropped because the dispatch only recognized the legacy OpenTelemetry class).
# --------------------------------------------------------------------------- #


def _make_otel_v2_logger():
    pytest.importorskip("opentelemetry")
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from litellm.integrations.otel import OpenTelemetryV2Config
    from litellm.integrations.otel.plumbing import providers
    from litellm.integrations.otel.logger import OpenTelemetryV2

    cfg = OpenTelemetryV2Config(exporter="in_memory")
    exporter = InMemorySpanExporter()
    tracer_provider = providers.build_tracer_provider(cfg, exporter=exporter)
    return OpenTelemetryV2(config=cfg, tracer_provider=tracer_provider), exporter


def test_resolve_otel_service_logger_recognizes_v2_instance():
    """The V2 logger is a plain CustomLogger, not a subclass of the legacy
    OpenTelemetry. The resolver must still recognize it (else service spans are
    silently dropped)."""
    service_logger = ServiceLogging()
    v2_logger, _ = _make_otel_v2_logger()
    assert service_logger._resolve_otel_service_logger(v2_logger) is v2_logger


def test_resolve_otel_service_logger_recognizes_otel_string(monkeypatch):
    # The "otel" string path resolves through the proxy's registered logger, so
    # it needs the proxy server module importable.
    try:
        import litellm.proxy.proxy_server as proxy_server
    except ImportError:
        pytest.skip("proxy server dependencies not installed")
    service_logger = ServiceLogging()
    v2_logger, _ = _make_otel_v2_logger()

    monkeypatch.setattr(proxy_server, "open_telemetry_logger", v2_logger, raising=False)
    assert service_logger._resolve_otel_service_logger("otel") is v2_logger


def test_resolve_otel_service_logger_ignores_unrelated_callback():
    service_logger = ServiceLogging()
    assert service_logger._resolve_otel_service_logger("prometheus_system") is None
    assert service_logger._resolve_otel_service_logger(object()) is None


@pytest.mark.asyncio
async def test_service_span_emitted_for_v2_logger_in_service_callback(monkeypatch):
    """End-to-end: a V2 logger registered in ``litellm.service_callback`` produces
    a service span when ``async_service_success_hook`` fires with a parent span."""
    from litellm.integrations.otel.model.spans import SpanRole

    v2_logger, exporter = _make_otel_v2_logger()
    parent = v2_logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, "POST /chat/completions"
    )

    monkeypatch.setattr(litellm, "service_callback", [v2_logger])
    service_logger = ServiceLogging()

    await service_logger.async_service_success_hook(
        service=ServiceTypes.REDIS,
        call_type="async_set_cache",
        duration=0.01,
        parent_otel_span=parent,
    )
    parent.end()

    names = [s.name for s in exporter.get_finished_spans()]
    # Span name is "{service} {call_type}" so repeated calls stay distinguishable.
    assert "redis async_set_cache" in names


@pytest.mark.asyncio
async def test_service_span_not_duplicated_for_string_and_instance(monkeypatch):
    """``service_callback`` can hold the ``"otel"`` string AND the registered
    logger instance — the V2 logger self-registers its instance even when the
    string is present. Both references resolve to the same logger, so the dispatch
    loop must emit only ONE span per service event, not one per reference. Before
    the dedup guard this produced duplicate ``postgres ...`` / ``redis ...`` spans.
    """
    try:
        import litellm.proxy.proxy_server as proxy_server
    except ImportError:
        pytest.skip("proxy server dependencies not installed")
    from litellm.integrations.otel.model.spans import SpanRole

    v2_logger, exporter = _make_otel_v2_logger()
    parent = v2_logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, "POST /chat/completions"
    )

    # The "otel" string resolves to the proxy's registered logger (the same
    # instance), so the list holds two references to one logger.
    monkeypatch.setattr(proxy_server, "open_telemetry_logger", v2_logger, raising=False)
    monkeypatch.setattr(litellm, "service_callback", ["otel", v2_logger])
    service_logger = ServiceLogging()

    await service_logger.async_service_success_hook(
        service=ServiceTypes.DB,
        call_type="get_user_object",
        duration=0.01,
        parent_otel_span=parent,
    )
    parent.end()

    db_spans = [
        s for s in exporter.get_finished_spans() if s.name == "postgres get_user_object"
    ]
    assert len(db_spans) == 1


@pytest.mark.asyncio
async def test_service_failure_span_not_duplicated_for_string_and_instance(
    monkeypatch,
):
    """Failure path mirror of the dedup guard — one span per failed service event,
    even with both the ``"otel"`` string and the instance in ``service_callback``."""
    try:
        import litellm.proxy.proxy_server as proxy_server
    except ImportError:
        pytest.skip("proxy server dependencies not installed")
    from litellm.integrations.otel.model.spans import SpanRole

    v2_logger, exporter = _make_otel_v2_logger()
    parent = v2_logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, "POST /chat/completions"
    )

    monkeypatch.setattr(proxy_server, "open_telemetry_logger", v2_logger, raising=False)
    monkeypatch.setattr(litellm, "service_callback", ["otel", v2_logger])
    service_logger = ServiceLogging()

    await service_logger.async_service_failure_hook(
        service=ServiceTypes.DB,
        call_type="get_user_object",
        duration=0.01,
        error="boom",
        parent_otel_span=parent,
    )
    parent.end()

    db_spans = [
        s for s in exporter.get_finished_spans() if s.name == "postgres get_user_object"
    ]
    assert len(db_spans) == 1
