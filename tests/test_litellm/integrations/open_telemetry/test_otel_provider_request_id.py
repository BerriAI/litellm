"""LIT-3091: provider request id surfaced as a dedicated OTEL span attribute.

The upstream provider's request identifier (the value of the response header
``x-request-id`` for OpenAI / Azure / Vertex AI, or ``request-id`` for
Anthropic) is captured by LiteLLM into
``standard_logging_payload.hidden_params.additional_headers`` as
``llm_provider-x-request-id`` / ``llm_provider-request-id``. Previously it was
only emitted as part of the dumped ``hidden_params`` JSON blob, which is not
queryable in OTEL backends. These tests pin that it is now emitted as a
dedicated, queryable attribute ``gen_ai.provider.request.id`` on the request
span, while still preserving the existing ``gen_ai.response.id`` (provider
response body id) and ``litellm.call_id`` (internal id) so all three
identifiers can be correlated.
"""
from datetime import datetime, timezone
from typing import Optional, Tuple

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from litellm.integrations.opentelemetry import OpenTelemetry


@pytest.fixture
def otel_with_exporter() -> Tuple[OpenTelemetry, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel = OpenTelemetry()
    otel.OTEL_TRACER_PROVIDER = provider
    otel.tracer = provider.get_tracer("lit-3091-tests")
    return otel, exporter


def _build_kwargs(
    additional_headers: Optional[dict],
    *,
    response_id: str = "chatcmpl-AbCdEf123456",
    litellm_call_id: str = "litellm-call-1234abcd",
):
    hidden_params = {"model_id": "deploy-id-1", "api_base": "https://api.openai.com/v1"}
    if additional_headers is not None:
        hidden_params["additional_headers"] = additional_headers

    slp = {
        "id": response_id,
        "call_type": "completion",
        "litellm_call_id": litellm_call_id,
        "trace_id": "trace-zzzz",
        "metadata": {
            "user_api_key_alias": "alice-key",
            "user_api_key_user_id": "user-7",
            "user_api_key_team_id": "team-42",
            "user_api_key_team_alias": "team-alpha",
            "user_api_key_user_email": "alice@example.com",
            "user_api_key_hash": "hashed-thingy",
        },
        "hidden_params": hidden_params,
    }
    kwargs = {
        "model": "openai/gpt-4o-mini",
        "litellm_params": {"custom_llm_provider": "openai", "metadata": {}},
        "optional_params": {"max_tokens": 50, "temperature": 0.0},
        "standard_logging_object": slp,
    }
    response_obj = {
        "id": response_id,
        "model": "gpt-4o-mini-2024-07-18",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    return kwargs, response_obj


# ---------------------------------------------------------------------------
# Helper: _get_provider_request_id_from_hidden_params
# ---------------------------------------------------------------------------
class TestGetProviderRequestIdHelper:
    """Exercises the static extraction helper in isolation."""

    def test_prefers_x_request_id(self):
        hp = {
            "additional_headers": {
                "llm_provider-x-request-id": "req_x",
                "llm_provider-request-id": "req_anthropic",
            }
        }
        assert (
            OpenTelemetry._get_provider_request_id_from_hidden_params(hp) == "req_x"
        )

    def test_falls_back_to_request_id(self):
        """Anthropic-style: only `request-id` header is returned."""
        hp = {"additional_headers": {"llm_provider-request-id": "msg_01F6"}}
        assert (
            OpenTelemetry._get_provider_request_id_from_hidden_params(hp) == "msg_01F6"
        )

    def test_returns_none_when_no_additional_headers(self):
        assert (
            OpenTelemetry._get_provider_request_id_from_hidden_params(
                {"model_id": "deploy-1"}
            )
            is None
        )

    def test_returns_none_when_additional_headers_empty(self):
        assert (
            OpenTelemetry._get_provider_request_id_from_hidden_params(
                {"additional_headers": {}}
            )
            is None
        )

    def test_returns_none_when_additional_headers_is_not_a_mapping(self):
        # Defensive: do not crash on malformed payloads from upstream callers.
        assert (
            OpenTelemetry._get_provider_request_id_from_hidden_params(
                {"additional_headers": ["x-request-id"]}
            )
            is None
        )

    @pytest.mark.parametrize("bad", [None, "string-not-dict", 42, []])
    def test_returns_none_when_hidden_params_is_not_a_mapping(self, bad):
        assert OpenTelemetry._get_provider_request_id_from_hidden_params(bad) is None

    def test_returns_none_when_value_is_empty_string(self):
        # Empty header value should not produce an attribute (falsy guard).
        hp = {"additional_headers": {"llm_provider-x-request-id": ""}}
        assert OpenTelemetry._get_provider_request_id_from_hidden_params(hp) is None

    def test_coerces_to_string(self):
        # Defensive: if a stray non-str value sneaks in, we coerce to str so
        # safe_set_attribute does not blow up on the otel side.
        hp = {"additional_headers": {"llm_provider-x-request-id": 12345}}
        assert (
            OpenTelemetry._get_provider_request_id_from_hidden_params(hp) == "12345"
        )


# ---------------------------------------------------------------------------
# End-to-end: span attribute emission
# ---------------------------------------------------------------------------
class TestSpanAttributeEmission:
    """Drives log_success_event end-to-end and inspects the emitted span."""

    @staticmethod
    def _collect_request_span(exporter: InMemorySpanExporter):
        spans = exporter.get_finished_spans()
        for s in spans:
            if s.name == "litellm_request":
                return s
        raise AssertionError(
            f"litellm_request span not found; got: {[s.name for s in spans]}"
        )

    def test_provider_request_id_surfaced_as_dedicated_attribute(
        self, otel_with_exporter
    ):
        otel, exporter = otel_with_exporter
        kwargs, response_obj = _build_kwargs(
            {
                "llm_provider-x-request-id": "req_85f49b546c7b4d3180755621f36631a1",
                "llm_provider-request-id": "msg_01F6CycZZPSHKRCCctcS1Vto",
                "x_ratelimit_remaining_requests": 9999,
            }
        )
        now = datetime.now(timezone.utc)
        otel.log_success_event(kwargs=kwargs, response_obj=response_obj, start_time=now, end_time=now)

        span = self._collect_request_span(exporter)
        # NEW attribute from LIT-3091:
        assert (
            span.attributes["gen_ai.provider.request.id"]
            == "req_85f49b546c7b4d3180755621f36631a1"
        )
        # Existing identifiers preserved so all three can be correlated:
        assert span.attributes["gen_ai.response.id"] == "chatcmpl-AbCdEf123456"
        assert span.attributes["litellm.call_id"] == "litellm-call-1234abcd"

    def test_anthropic_style_request_id_fallback(self, otel_with_exporter):
        otel, exporter = otel_with_exporter
        kwargs, response_obj = _build_kwargs(
            {"llm_provider-request-id": "msg_01F6CycZZPSHKRCCctcS1Vto"}
        )
        now = datetime.now(timezone.utc)
        otel.log_success_event(kwargs=kwargs, response_obj=response_obj, start_time=now, end_time=now)

        span = self._collect_request_span(exporter)
        assert (
            span.attributes["gen_ai.provider.request.id"]
            == "msg_01F6CycZZPSHKRCCctcS1Vto"
        )

    def test_attribute_omitted_when_no_additional_headers(self, otel_with_exporter):
        """When the upstream call did not return a request-id header (e.g.
        embedding routes or a custom provider that omits it), the attribute
        is omitted rather than emitted as None / empty string."""
        otel, exporter = otel_with_exporter
        kwargs, response_obj = _build_kwargs(None)  # no additional_headers at all
        now = datetime.now(timezone.utc)
        otel.log_success_event(kwargs=kwargs, response_obj=response_obj, start_time=now, end_time=now)

        span = self._collect_request_span(exporter)
        assert "gen_ai.provider.request.id" not in span.attributes
        # The existing identifiers are still emitted.
        assert span.attributes["gen_ai.response.id"] == "chatcmpl-AbCdEf123456"
        assert span.attributes["litellm.call_id"] == "litellm-call-1234abcd"

    def test_attribute_omitted_when_header_value_empty(self, otel_with_exporter):
        otel, exporter = otel_with_exporter
        kwargs, response_obj = _build_kwargs(
            {"llm_provider-x-request-id": "", "llm_provider-request-id": ""}
        )
        now = datetime.now(timezone.utc)
        otel.log_success_event(kwargs=kwargs, response_obj=response_obj, start_time=now, end_time=now)

        span = self._collect_request_span(exporter)
        assert "gen_ai.provider.request.id" not in span.attributes
