"""Behavior of three V2 OTel instrumentation areas:

1. Baggage allowlists are configurable via env vars and config.yaml
   (``callback_settings.otel.*``), not just hard-coded.
2. Pass-through LLM-call spans nest under the proxy server span because they are
   opened at the ``pre_call`` boundary in the request task (where the server span
   is ambient) — no span threaded through metadata.
3. Guardrail span data is built from the typed
   ``StandardLoggingGuardrailInformation`` shape (provider-agnostic), not from
   one provider's assumed field names.
"""

import asyncio

import pytest

pytest.importorskip("opentelemetry")

from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

from litellm.integrations.otel import LiteLLM, OpenTelemetryV2Config  # noqa: E402
from litellm.integrations.otel.plumbing import providers  # noqa: E402
from litellm.integrations.otel.model.baggage import (  # noqa: E402
    BAGGAGE_PROMOTED_KEYS,
    DEFAULT_BAGGAGE_METADATA_KEYS,
)
from litellm.integrations.otel.logger import OpenTelemetryV2  # noqa: E402
from litellm.integrations.otel.model.payloads import GuardrailSpanData  # noqa: E402
from litellm.integrations.otel.model.spans import (  # noqa: E402
    LITELLM_PROXY_REQUEST_SPAN_NAME,
    SpanRole,
)

# --------------------------------------------------------------------------- #
#  Area 1 — baggage allowlists configurable
# --------------------------------------------------------------------------- #


def test_baggage_keys_default_when_unset():
    cfg = OpenTelemetryV2Config()
    assert cfg.baggage_promoted_keys == list(BAGGAGE_PROMOTED_KEYS)
    assert cfg.baggage_metadata_keys == list(DEFAULT_BAGGAGE_METADATA_KEYS)


def test_baggage_promoted_keys_from_env_csv(monkeypatch):
    monkeypatch.setenv(
        "LITELLM_OTEL_BAGGAGE_PROMOTED_KEYS",
        f"{LiteLLM.TEAM_ID}, {LiteLLM.KEY_HASH}",
    )
    monkeypatch.setenv(
        "LITELLM_OTEL_BAGGAGE_METADATA_KEYS",
        "user_api_key_user_id,requester_ip_address",
    )
    cfg = OpenTelemetryV2Config()
    # Whitespace around comma-separated entries is trimmed.
    assert cfg.baggage_promoted_keys == [LiteLLM.TEAM_ID, LiteLLM.KEY_HASH]
    assert cfg.baggage_metadata_keys == [
        "user_api_key_user_id",
        "requester_ip_address",
    ]


def test_baggage_keys_from_config_yaml_kwargs():
    """``callback_settings.otel.*`` reaches the config through the logger kwargs."""
    logger = OpenTelemetryV2(
        baggage_promoted_keys=[LiteLLM.TEAM_ALIAS],
        baggage_metadata_keys=["user_api_key_alias"],
    )
    assert logger.config.baggage_promoted_keys == [LiteLLM.TEAM_ALIAS]
    assert logger.config.baggage_metadata_keys == ["user_api_key_alias"]


def test_baggage_processor_allowlist_uses_config_keys():
    cfg = OpenTelemetryV2Config(
        exporter="in_memory", baggage_promoted_keys=[LiteLLM.TEAM_ID]
    )
    provider, exporter = providers.in_memory_provider(cfg)
    from litellm.integrations.otel.plumbing import context as ctx_mod
    from litellm.integrations.otel.emitter import SpanEmitter
    from litellm.integrations.otel.model.payloads import ServiceSpanData

    engine = SpanEmitter(providers.get_tracer(provider, "t"), cfg)
    ctx = ctx_mod.set_request_baggage({LiteLLM.TEAM_ID: "t1", LiteLLM.TEAM_ALIAS: "ta"})
    engine.emit(SpanRole.SERVICE, ServiceSpanData("redis"), ctx)
    (span,) = exporter.get_finished_spans()
    assert span.attributes.get(LiteLLM.TEAM_ID) == "t1"
    assert LiteLLM.TEAM_ALIAS not in span.attributes  # not in this allowlist


# --------------------------------------------------------------------------- #
#  Area 2 — pass-through LLM span parents to the ambient server span
# --------------------------------------------------------------------------- #


def _logger():
    cfg = OpenTelemetryV2Config(exporter="in_memory")
    exporter = InMemorySpanExporter()
    tracer_provider = providers.build_tracer_provider(cfg, exporter=exporter)
    return OpenTelemetryV2(config=cfg, tracer_provider=tracer_provider), exporter


def _payload():
    return {
        "call_type": "pass_through_endpoint",
        "custom_llm_provider": "openai",
        "model": "gpt-4o",
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2,
        "status": "success",
        "litellm_call_id": "call_pt",
        "metadata": {},
        "hidden_params": {},
    }


def test_passthrough_llm_span_parents_to_ambient_server_span():
    """Pass-through calls ``logging_obj.pre_call`` in the request task, where the
    server span is the ambient context — so the LLM-call span is opened there and
    parents to it natively, with no ``litellm_parent_otel_span`` threading. The
    later (possibly detached) success callback only closes the already-parented
    span, so it never becomes a separate root trace."""
    logger, exporter = _logger()
    server = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    kwargs = {
        "standard_logging_object": _payload(),
        "litellm_params": {"metadata": {}},
    }
    # pre_call runs in the request task (server span ambient); success closes it.
    with trace.use_span(server, end_on_exit=False):
        logger.log_pre_api_call(model="gpt-4o", messages=[], kwargs=kwargs)
    asyncio.run(logger.async_log_success_event(kwargs, None, None, None))
    server.end()

    by_name = {s.name: s for s in exporter.get_finished_spans()}
    llm_span = by_name["chat gpt-4o"]
    assert llm_span.parent is not None
    assert llm_span.parent.span_id == server.get_span_context().span_id


def test_llm_span_unaffected_by_phase_span_active_at_close():
    """The LLM-call span's parent is captured at the ``pre_call`` boundary (under
    the server span), so a phase span (e.g. ``auth``) that happens to be ambient
    when the *close* callback fires can't re-parent it. This is the structural
    successor to the old auth-failure-401 case where the LLM log nested under
    ``auth``: the span is now born after auth, parented to the request root."""
    logger, exporter = _logger()
    server = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    kwargs = {
        "standard_logging_object": _payload(),
        "litellm_params": {"metadata": {}},
    }
    with trace.use_span(server, end_on_exit=False):
        logger.log_pre_api_call(model="gpt-4o", messages=[], kwargs=kwargs)
    # A phase span is ambient when the close callback fires — must not re-parent.
    phase = logger._emitter.start_span(SpanRole.SERVICE, "auth /v1/chat/completions")
    with trace.use_span(phase, end_on_exit=False):
        asyncio.run(logger.async_log_success_event(kwargs, None, None, None))
    phase.end()
    server.end()
    by_name = {s.name: s for s in exporter.get_finished_spans()}
    llm_span = by_name["chat gpt-4o"]
    assert llm_span.parent.span_id == server.get_span_context().span_id


# --------------------------------------------------------------------------- #
#  Area 3 — typed, provider-agnostic guardrail span data
# --------------------------------------------------------------------------- #


def test_guardrail_mode_enum_normalized_to_value():
    from litellm.types.guardrails import GuardrailEventHooks

    d = GuardrailSpanData.from_logging_entry(
        {
            "guardrail_name": "bedrock-guardrail",
            "guardrail_mode": GuardrailEventHooks.pre_call,
            "guardrail_status": "success",
        }
    )
    # The enum *value* ("pre_call"), not "GuardrailEventHooks.pre_call".
    assert d.mode == "pre_call"


def test_guardrail_mode_list_of_enums_joined():
    from litellm.types.guardrails import GuardrailEventHooks

    d = GuardrailSpanData.from_logging_entry(
        {
            "guardrail_name": "g",
            "guardrail_mode": [
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.post_call,
            ],
            "guardrail_status": "success",
        }
    )
    assert d.mode == "pre_call,post_call"


def test_guardrail_typed_metadata_fields_mapped_to_span():
    from litellm.integrations.otel.mappers.genai import GenAIMapper

    d = GuardrailSpanData.from_logging_entry(
        {
            "guardrail_name": "eu-pii",
            "guardrail_status": "success",
            "guardrail_id": "gd-eu-pii-001",
            "policy_template": "EU AI Act Article 5",
            "detection_method": "presidio",
        }
    )
    assert d.guardrail_id == "gd-eu-pii-001"
    assert d.policy_template == "EU AI Act Article 5"
    assert d.detection_method == "presidio"
    attrs = GenAIMapper().map(d)
    assert attrs[LiteLLM.GUARDRAIL_ID] == "gd-eu-pii-001"
    assert attrs[LiteLLM.GUARDRAIL_POLICY_TEMPLATE] == "EU AI Act Article 5"
    assert attrs[LiteLLM.GUARDRAIL_DETECTION_METHOD] == "presidio"


def test_guardrail_ignores_non_canonical_provider_keys():
    """Only canonical ``StandardLoggingGuardrailInformation`` keys are read; a
    provider's ad-hoc bare ``name``/``status``/``mode`` keys are not assumed."""
    d = GuardrailSpanData.from_logging_entry(
        {"name": "bare", "status": "blocked", "mode": "pre"}  # type: ignore[typeddict-unknown-key]
    )
    assert d.guardrail_name == "guardrail"  # fell back to the default
    assert d.status is None
    assert d.mode is None
