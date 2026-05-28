"""
LIT-3391: OTEL surface - assert that when StandardLoggingGuardrailInformation
carries ``provider_request_id`` (e.g. AWS Bedrock's x-amzn-RequestId), the
OpenTelemetry integration emits it as the queryable span attribute
``guardrail_provider_request_id`` on the per-guardrail span.

This is the "dashboard" half of LIT-3391 - once it's a span attribute, every
OTEL backend (Langfuse, Honeycomb, Datadog APM, Tempo, Grafana Cloud, etc.)
can filter / pivot on it without re-parsing the redacted guardrail_response.
"""
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def otel_handler():
    from litellm.integrations.opentelemetry import OpenTelemetry, OpenTelemetryConfig

    cfg = OpenTelemetryConfig(exporter="console")
    return OpenTelemetry(config=cfg)


def _make_kwargs(provider_request_id: Any) -> Dict[str, Any]:
    return {
        "standard_logging_object": {
            "trace_id": "trace-test",
            "metadata": {
                "user_api_key": "test-key",
                "user_api_key_team_id": None,
                "user_api_key_team_alias": None,
            },
            "guardrail_information": [
                {
                    "guardrail_name": "test_bedrock_guardrail",
                    "guardrail_mode": "pre_call",
                    "guardrail_status": "success",
                    "guardrail_provider": "bedrock",
                    "guardrail_response": {"action": "NONE"},
                    "guardrail_action": "NONE",
                    "provider_request_id": provider_request_id,
                    "start_time": 1.0,
                    "end_time": 2.0,
                }
            ],
        },
        "litellm_params": {"metadata": {}},
        "model": "gpt-4o",
    }


def _capture_attrs(otel_handler, kwargs):
    attrs: List[Any] = []

    class _SpanCapture:
        def set_attribute(self, key, value):
            attrs.append((key, value))

        def end(self, end_time=None):
            pass

    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = _SpanCapture()
    with patch.object(
        otel_handler, "get_tracer_to_use_for_request", return_value=fake_tracer
    ), patch.object(otel_handler, "_emit_once", return_value=True), patch.object(
        otel_handler, "safe_set_attribute"
    ) as safe_set:
        safe_set.side_effect = lambda span, key, value: span.set_attribute(key, value)
        otel_handler._create_guardrail_span(kwargs=kwargs, context=None)
    return attrs


def test_otel_emits_provider_request_id_when_present(otel_handler):
    attrs = _capture_attrs(
        otel_handler,
        _make_kwargs(provider_request_id="7c1f8a4d-4b8a-4f9a-9d1c-0c2e1a8b3f47"),
    )
    flat = dict(attrs)
    assert (
        flat.get("guardrail_provider_request_id")
        == "7c1f8a4d-4b8a-4f9a-9d1c-0c2e1a8b3f47"
    ), f"missing guardrail_provider_request_id. Got: {attrs}"


def test_otel_omits_provider_request_id_when_absent(otel_handler):
    attrs = _capture_attrs(otel_handler, _make_kwargs(provider_request_id=None))
    keys = {k for k, _ in attrs}
    assert "guardrail_provider_request_id" not in keys


def test_otel_omits_provider_request_id_when_empty_string(otel_handler):
    attrs = _capture_attrs(otel_handler, _make_kwargs(provider_request_id=""))
    keys = {k for k, _ in attrs}
    assert "guardrail_provider_request_id" not in keys
