"""Tests for Baggage-based promotion of request-scoped attributes onto every span,
and the two antipattern boundaries: http.* is never promoted, and the full
metadata blob is never promoted (only the bounded allowlist)."""

import pytest

pytest.importorskip("opentelemetry")

from litellm.integrations.otel import (  # noqa: E402
    GenAI,
    HTTP,
    LiteLLM,
    OpenTelemetryV2Config,
    promoted_baggage,
)
from litellm.integrations.otel import context as ctx_mod  # noqa: E402
from litellm.integrations.otel import providers  # noqa: E402
from litellm.integrations.otel.emitter import SpanEmitter  # noqa: E402
from litellm.integrations.otel.payloads import (  # noqa: E402
    GuardrailSpanData,
    LLMCallSpanData,
    ServiceSpanData,
)
from litellm.integrations.otel.baggage import BAGGAGE_PROMOTED_KEYS  # noqa: E402
from litellm.integrations.otel.spans import SpanRole  # noqa: E402


def _payload():
    return {
        "call_type": "acompletion",
        "custom_llm_provider": "openai",
        "model": "gpt-4o",
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2,
        "metadata": {
            "team_id": "t1",
            "team_alias": "team one",
            "user_api_key_hash": "hsh",
            "user_api_key_org_id": "org1",
            "private_note": "do-not-promote",
        },
        "status": "success",
        "litellm_call_id": "call_1",
        "hidden_params": {},
    }


def _engine_and_exporter(config=None):
    cfg = config or OpenTelemetryV2Config(exporter="in_memory")
    provider, exporter = providers.in_memory_provider(cfg)
    tracer = providers.get_tracer(provider, "litellm-baggage-test")
    return SpanEmitter(tracer, cfg), exporter


def test_identity_promoted_onto_every_span():
    engine, exporter = _engine_and_exporter()
    data = LLMCallSpanData.from_standard_logging_payload(_payload())
    bag = promoted_baggage(data.identity, data.request_model, BAGGAGE_PROMOTED_KEYS)
    ctx = ctx_mod.set_request_baggage(bag)

    root = engine.start_span(SpanRole.PROXY_REQUEST, "POST /chat/completions", ctx)
    root_ctx = ctx_mod.context_from_span(root, ctx)
    engine.emit(SpanRole.LLM_CALL, data, parent_context=root_ctx)
    engine.emit(
        SpanRole.GUARDRAIL, GuardrailSpanData("presidio", status="success"), root_ctx
    )
    engine.emit(SpanRole.SERVICE, ServiceSpanData("redis", call_type="set"), root_ctx)
    root.end()

    spans = exporter.get_finished_spans()
    assert len(spans) == 4
    for span in spans:
        assert span.attributes.get(LiteLLM.TEAM_ID) == "t1"
        assert span.attributes.get(LiteLLM.TEAM_ALIAS) == "team one"
        assert span.attributes.get(GenAI.REQUEST_MODEL) == "gpt-4o"


def test_allowlisted_metadata_subkey_promoted_blob_excluded():
    engine, exporter = _engine_and_exporter()
    data = LLMCallSpanData.from_standard_logging_payload(_payload())
    bag = promoted_baggage(data.identity, data.request_model, BAGGAGE_PROMOTED_KEYS)
    ctx = ctx_mod.set_request_baggage(bag)
    engine.emit(SpanRole.SERVICE, ServiceSpanData("redis", call_type="set"), ctx)
    (span,) = exporter.get_finished_spans()
    # allowlisted metadata sub-key is promoted
    assert (
        span.attributes.get(f"{LiteLLM.METADATA_PREFIX}user_api_key_org_id") == "org1"
    )
    # non-allowlisted metadata is NOT promoted (no full-blob dumping)
    assert all("private_note" not in k for k in span.attributes)


def test_http_attributes_never_promoted():
    """Even if http.* is present in baggage, the processor must not stamp it on
    child spans (it belongs on the SERVER span only)."""
    engine, exporter = _engine_and_exporter()
    ctx = ctx_mod.set_request_baggage(
        {
            LiteLLM.TEAM_ID: "t1",
            HTTP.ROUTE: "/chat/completions",
            HTTP.REQUEST_METHOD: "POST",
        }
    )
    engine.emit(SpanRole.SERVICE, ServiceSpanData("redis", call_type="set"), ctx)
    (span,) = exporter.get_finished_spans()
    assert span.attributes.get(LiteLLM.TEAM_ID) == "t1"
    assert HTTP.ROUTE not in span.attributes
    assert HTTP.REQUEST_METHOD not in span.attributes


def test_arbitrary_upstream_baggage_not_promoted():
    engine, exporter = _engine_and_exporter()
    ctx = ctx_mod.set_request_baggage(
        {LiteLLM.TEAM_ID: "t1", "some.upstream.key": "leak"}
    )
    engine.emit(SpanRole.SERVICE, ServiceSpanData("redis", call_type="set"), ctx)
    (span,) = exporter.get_finished_spans()
    assert span.attributes.get(LiteLLM.TEAM_ID) == "t1"
    assert "some.upstream.key" not in span.attributes


def test_baggage_processor_allowlist_can_be_widened():
    cfg = OpenTelemetryV2Config(
        exporter="in_memory",
        baggage_promoted_keys=[LiteLLM.TEAM_ID, "custom.key"],
    )
    engine, exporter = _engine_and_exporter(cfg)
    ctx = ctx_mod.set_request_baggage({"custom.key": "v", LiteLLM.TEAM_ALIAS: "ta"})
    engine.emit(SpanRole.SERVICE, ServiceSpanData("redis"), ctx)
    (span,) = exporter.get_finished_spans()
    assert span.attributes.get("custom.key") == "v"
    # team_alias not in this config's allowlist -> not promoted
    assert LiteLLM.TEAM_ALIAS not in span.attributes
