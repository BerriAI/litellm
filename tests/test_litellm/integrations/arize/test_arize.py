import json
from typing import Optional
from unittest.mock import MagicMock, Mock, patch

# Adds the grandparent directory to sys.path to allow importing project modules

import asyncio
import datetime
from collections.abc import Callable

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import litellm
from litellm.integrations.arize.arize import ArizeLogger
from litellm.integrations.opentelemetry import OpenTelemetryConfig


@pytest.mark.asyncio
async def test_arize_dynamic_params():
    """Test that the OpenTelemetry logger uses the correct dynamic headers for each Arize request."""

    # Create ArizeLogger instance
    arize_logger = ArizeLogger()

    # Capture the get_tracer_to_use_for_request calls
    tracer_calls = []
    original_get_tracer = arize_logger.get_tracer_to_use_for_request

    def mock_get_tracer_to_use_for_request(kwargs):
        # Capture the kwargs to see what dynamic headers are being used
        tracer_calls.append(kwargs)
        # Return the default tracer
        return arize_logger.tracer

    # Mock the get_tracer_to_use_for_request method
    arize_logger.get_tracer_to_use_for_request = mock_get_tracer_to_use_for_request

    # Set up callbacks
    litellm.callbacks = [arize_logger]

    # First request with team1 credentials
    await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi test from arize dynamic config"}],
        temperature=0.1,
        mock_response="test_response",
        arize_api_key="team1_key",
        arize_space_id="team1_space_id",
    )

    # Second request with team2 credentials
    await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi test from arize dynamic config"}],
        temperature=0.1,
        mock_response="test_response",
        arize_api_key="team2_key",
        arize_space_id="team2_space_id",
    )

    # Allow some time for async processing
    await asyncio.sleep(5)

    # Assertions
    print(f"Tracer calls: {len(tracer_calls)}")

    # We should have captured calls for both requests
    assert len(tracer_calls) >= 2, f"Expected at least 2 tracer calls, got {len(tracer_calls)}"

    # Check that we have the expected dynamic params in the kwargs
    team1_found = False
    team2_found = False

    print("args to tracer calls", tracer_calls)

    for call_kwargs in tracer_calls:
        dynamic_params = call_kwargs.get("standard_callback_dynamic_params", {})
        if dynamic_params.get("arize_api_key") == "team1_key":
            team1_found = True
            assert dynamic_params.get("arize_space_id") == "team1_space_id"
        elif dynamic_params.get("arize_api_key") == "team2_key":
            team2_found = True
            assert dynamic_params.get("arize_space_id") == "team2_space_id"

    # Verify both teams were found
    assert team1_found, "team1 dynamic params not found"
    assert team2_found, "team2 dynamic params not found"

    print("✅ All assertions passed - OpenTelemetry logger correctly received dynamic params")


@pytest.mark.asyncio
async def test_arize_dynamic_headers_in_grpc_requests():
    """Test that dynamic Arize params are passed as headers to the gRPC/HTTP exporter."""

    # Track all exporter calls and their headers
    exporter_headers = []

    def mock_otlp_http_exporter(*args, **kwargs):
        # Capture the headers passed to the HTTP exporter
        headers = kwargs.get("headers", {})
        exporter_headers.append(headers)

        # Return a mock exporter
        mock_exporter = MagicMock()
        mock_exporter.export = MagicMock(return_value=None)
        return mock_exporter

    # Patch the HTTP exporter (Arize uses HTTP by default)
    with patch(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter",
        mock_otlp_http_exporter,
    ):
        # Create ArizeLogger with HTTP configuration
        config = OpenTelemetryConfig(exporter="otlp_http", endpoint="https://otlp.arize.com/v1")
        arize_logger = ArizeLogger(config=config)
        litellm.callbacks = [arize_logger]

        # Request 1: team1 dynamic params
        await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hi from team1"}],
            mock_response="response1",
            arize_api_key="team1_api_key",
            arize_space_id="team1_space_id",
        )

        # Request 2: team2 dynamic params
        await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hi from team2"}],
            mock_response="response2",
            arize_api_key="team2_api_key",
            arize_space_id="team2_space_id",
        )

        # Allow time for async processing
        await asyncio.sleep(3)

        # Assertions
        print(f"Captured exporter headers: {exporter_headers}")

        # Should have multiple exporter calls (default + dynamic)
        assert len(exporter_headers) >= 2, f"Expected at least 2 exporter calls, got {len(exporter_headers)}"

        # Find team1 and team2 headers
        team1_found = False
        team2_found = False

        for headers in exporter_headers:
            if headers.get("api_key") == "team1_api_key" and headers.get("arize-space-id") == "team1_space_id":
                team1_found = True
                print(f"✅ Found team1 headers: {headers}")
            elif headers.get("api_key") == "team2_api_key" and headers.get("arize-space-id") == "team2_space_id":
                team2_found = True
                print(f"✅ Found team2 headers: {headers}")

        # Verify both dynamic header sets were used
        assert team1_found, "team1 dynamic headers not found in exporter calls"
        assert team2_found, "team2 dynamic headers not found in exporter calls"

        print("✅ Test passed - Dynamic Arize params correctly passed to gRPC/HTTP exporter")


_START = datetime.datetime.now()
_END = datetime.datetime.now()


def _sampled_arize_logger(
    random_draw: Callable[[], float] | None = None,
) -> tuple[ArizeLogger, InMemorySpanExporter]:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    logger = ArizeLogger(tracer_provider=provider, random_draw=random_draw)
    return logger, exporter


def _request_spans(exporter: InMemorySpanExporter) -> int:
    return sum(1 for span in exporter.get_finished_spans() if span.name == "litellm_request")


def _arize_kwargs(callback_vars: dict[str, str] | None = None) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "model": "gpt-4",
        "litellm_params": {"metadata": {}},
        "standard_logging_object": {
            "id": "call-1",
            "call_type": "completion",
            "model": "gpt-4",
            "metadata": {},
            "messages": [{"role": "user", "content": "hi"}],
        },
    }
    if callback_vars is not None:
        kwargs["standard_callback_dynamic_params"] = callback_vars
    return kwargs


@pytest.mark.asyncio
async def test_success_sampling_rate_zero_exports_no_spans():
    logger, exporter = _sampled_arize_logger(random_draw=lambda: 0.0)
    await logger.async_log_success_event(_arize_kwargs({"arize_success_sampling_rate": "0.0"}), None, _START, _END)
    assert _request_spans(exporter) == 0


@pytest.mark.asyncio
async def test_success_sampling_rate_one_exports_a_span():
    logger, exporter = _sampled_arize_logger()
    await logger.async_log_success_event(_arize_kwargs({"arize_success_sampling_rate": "1.0"}), None, _START, _END)
    assert _request_spans(exporter) == 1


@pytest.mark.asyncio
async def test_unset_sampling_rate_exports_everything():
    logger, exporter = _sampled_arize_logger()
    await logger.async_log_success_event(_arize_kwargs(), None, _START, _END)
    assert _request_spans(exporter) == 1


@pytest.mark.asyncio
async def test_draw_above_rate_is_dropped_draw_at_rate_is_exported():
    dropped, dropped_exporter = _sampled_arize_logger(random_draw=lambda: 0.3)
    await dropped.async_log_success_event(_arize_kwargs({"arize_success_sampling_rate": "0.2"}), None, _START, _END)
    assert _request_spans(dropped_exporter) == 0

    kept, kept_exporter = _sampled_arize_logger(random_draw=lambda: 0.2)
    await kept.async_log_success_event(_arize_kwargs({"arize_success_sampling_rate": "0.2"}), None, _START, _END)
    assert _request_spans(kept_exporter) == 1


@pytest.mark.asyncio
async def test_success_and_error_rates_are_independent():
    logger, exporter = _sampled_arize_logger()
    kwargs = _arize_kwargs({"arize_success_sampling_rate": "0.0", "arize_error_sampling_rate": "1.0"})
    await logger.async_log_success_event(kwargs, None, _START, _END)
    await logger.async_log_failure_event(kwargs, ValueError("boom"), _START, _END)
    assert _request_spans(exporter) == 1

    logger2, exporter2 = _sampled_arize_logger()
    kwargs2 = _arize_kwargs({"arize_success_sampling_rate": "1.0", "arize_error_sampling_rate": "0.0"})
    await logger2.async_log_success_event(kwargs2, None, _START, _END)
    await logger2.async_log_failure_event(kwargs2, ValueError("boom"), _START, _END)
    assert _request_spans(exporter2) == 1


@pytest.mark.asyncio
async def test_one_draw_per_request_across_sync_and_async_handlers():
    draws: list[int] = []

    def counting_draw() -> float:
        draws.append(1)
        return 0.5

    logger, _ = _sampled_arize_logger(random_draw=counting_draw)
    kwargs = _arize_kwargs({"arize_success_sampling_rate": "1.0"})
    logger.log_success_event(kwargs, None, _START, _END)
    await logger.async_log_success_event(kwargs, None, _START, _END)
    assert len(draws) == 1


@pytest.mark.asyncio
async def test_unparsable_sampling_rate_exports_rather_than_dropping():
    logger, exporter = _sampled_arize_logger(random_draw=lambda: 0.99)
    await logger.async_log_success_event(_arize_kwargs({"arize_success_sampling_rate": "abc"}), None, _START, _END)
    assert _request_spans(exporter) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "1.5", "-0.1"])
async def test_out_of_range_sampling_rate_exports_rather_than_dropping(bad: str):
    logger, exporter = _sampled_arize_logger(random_draw=lambda: 0.99)
    await logger.async_log_success_event(_arize_kwargs({"arize_success_sampling_rate": bad}), None, _START, _END)
    assert _request_spans(exporter) == 1
