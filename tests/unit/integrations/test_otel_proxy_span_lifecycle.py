import logging
from datetime import datetime, timezone

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from litellm.integrations.opentelemetry import (
    LITELLM_PROXY_REQUEST_SPAN_NAME,
    OpenTelemetry,
)


def test_success_stamps_team_attributes_before_proxy_span_end(caplog):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel = OpenTelemetry()
    otel.tracer = provider.get_tracer("otel-proxy-span-lifecycle-test")
    otel.message_logging = False

    server_span = otel.create_litellm_proxy_request_started_span(
        start_time=datetime.now(timezone.utc), headers={}
    )
    kwargs = {
        "model": "test-model",
        "litellm_call_id": "test-call",
        "call_type": "completion",
        "litellm_params": {
            "metadata": {
                "litellm_parent_otel_span": server_span,
                "user_api_key_team_id": "test-team-id",
                "user_api_key_team_alias": "test-team",
            }
        },
        "standard_logging_object": {
            "metadata": {
                "user_api_key_team_id": "test-team-id",
                "user_api_key_team_alias": "test-team",
            },
            "call_type": "completion",
        },
    }

    with caplog.at_level(logging.WARNING, logger="opentelemetry.sdk.trace"):
        now = datetime.now(timezone.utc)
        otel.log_success_event(kwargs, {"id": "test-response"}, now, now)

    warnings = [
        record
        for record in caplog.records
        if record.getMessage() == "Setting attribute on ended span."
    ]
    assert warnings == []

    spans = {span.name: span for span in exporter.get_finished_spans()}
    root = spans[LITELLM_PROXY_REQUEST_SPAN_NAME]
    assert root.attributes["metadata.user_api_key_team_id"] == "test-team-id"
    assert root.attributes["metadata.user_api_key_team_alias"] == "test-team"
