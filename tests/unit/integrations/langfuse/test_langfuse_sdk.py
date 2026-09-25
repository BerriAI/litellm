"""Covers litellm's own Langfuse export channel: plain OTel spans carrying the v2 contracts.

The timestamp assertions are the regression guard for the migration: the v4 SDK's public
API has no observation start time, so a callback running after the model call would
otherwise record its own duration instead of the call's.
"""

import json
import logging
import threading
import uuid
from base64 import b64encode
from datetime import datetime, timedelta, timezone
from time import monotonic, sleep
from types import MappingProxyType
from typing import Final

import httpx
import opentelemetry.trace as otel_trace
import pytest
from langfuse import LangfuseOtelSpanAttributes as A
from langfuse.api.core.api_error import ApiError
from langfuse.api.core.request_options import RequestOptions
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from litellm.integrations.langfuse.langfuse import (
    MINIMUM_LANGFUSE_VERSION,
    installed_langfuse_version,
    raise_if_unsupported_langfuse_version,
)
from litellm.integrations.langfuse.langfuse_sdk import (
    DiscardingSpanExporter,
    LangfuseApiClient,
    LangfusePromptError,
    LangfuseSpanExporter,
    LangfuseTracing,
    _build_span_exporter,
    _encode,
    acquire_langfuse_tracing,
    build_langfuse_client,
    build_langfuse_tracing,
    configured_flush_at,
    configured_prompt_cache_ttl,
    configured_sample_rate,
    enable_langfuse_debug_logging,
    flush_langfuse_tracing,
    observation_attributes,
    release_langfuse_tracing,
    resolve_observation_id,
    resolve_trace_id,
    start_child_span,
    start_generation,
    to_unix_nanos,
    trace_attributes,
)
from litellm.llms.custom_httpx.http_handler import HTTPHandler

CALL_START = datetime(2024, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
FIRST_TOKEN = CALL_START + timedelta(seconds=5)
CALL_END = CALL_START + timedelta(seconds=20)
TRACE_A = "a" * 32
PARENT_C = "c" * 16


@pytest.fixture(autouse=True)
def _own_channel_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Channels leaked by other test modules would otherwise take part in every process-wide flush here."""
    monkeypatch.setattr("litellm.integrations.langfuse.langfuse_sdk._TRACING", {})


@pytest.fixture(name="channel")
def _channel() -> tuple[LangfuseTracing, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    return (
        build_langfuse_tracing(
            exporter=exporter, environment=None, release=None, sample_rate=1.0, flush_interval_millis=10
        ),
        exporter,
    )


def _only_span(exporter, name):
    return next(s for s in exporter.get_finished_spans() if s.name == name)


def _generation(
    tracing,
    *,
    name="gen",
    trace_id=TRACE_A,
    parent=None,
    existing=False,
    observation_id=None,
    public=None,
    attributes=None,
):
    return start_generation(
        tracing=tracing,
        trace_id=trace_id,
        parent_observation_id=parent,
        existing_trace=existing,
        observation_id=observation_id,
        name=name,
        start_time=CALL_START,
        public=public,
        attributes=attributes if attributes is not None else {},
    )


def test_generation_records_the_model_call_window_not_the_callback(channel):
    tracing, exporter = channel
    attributes = observation_attributes(observation_type="generation", completion_start_time=FIRST_TOKEN)
    _generation(tracing, attributes=attributes).end(CALL_END)
    tracing.flush()

    span = _only_span(exporter, "gen")
    assert span.start_time == to_unix_nanos(CALL_START)
    assert span.end_time == to_unix_nanos(CALL_END)
    assert (span.end_time - span.start_time) == 20 * 1_000_000_000
    assert datetime.fromisoformat(json.loads(span.attributes[A.OBSERVATION_COMPLETION_START_TIME])) == FIRST_TOKEN


@pytest.mark.parametrize(
    "supplied",
    [1709294400.5, datetime(2024, 3, 1, 12, 0, 0, 500000, tzinfo=timezone.utc)],
    ids=["unix-seconds-float", "datetime"],
)
def test_timestamps_accept_both_shapes_guardrails_and_callbacks_use(supplied):
    """Guardrail entries carry unix seconds as floats, the callback carries datetimes."""
    assert to_unix_nanos(supplied) == 1709294400500000000


def test_guardrail_span_with_float_timestamps_keeps_its_own_window_under_the_generation(channel):
    tracing, exporter = channel
    guardrail_start = 1709294400.0
    generation = _generation(tracing)
    start_child_span(
        tracing=tracing, parent=generation, name="guardrail", start_time=guardrail_start, attributes={}
    ).end(guardrail_start + 2)
    generation.end(CALL_END)
    tracing.flush()

    guardrail = _only_span(exporter, "guardrail")
    exported_generation = _only_span(exporter, "gen")
    assert (guardrail.end_time - guardrail.start_time) == 2 * 1_000_000_000
    assert guardrail.context.trace_id == exported_generation.context.trace_id
    assert guardrail.parent.span_id == exported_generation.context.span_id


def test_requested_trace_id_is_the_exported_trace_id_and_the_generation_is_its_root(channel):
    """v2 ``trace(id=...)``: the caller's id is the trace and the generation has no parent."""
    tracing, exporter = channel
    generation = _generation(tracing)
    generation.end(CALL_END)
    tracing.flush()

    span = _only_span(exporter, "gen")
    assert generation.trace_id == TRACE_A
    assert format(span.context.trace_id, "032x") == TRACE_A
    assert span.parent is None


def test_parent_observation_id_nests_the_generation_under_the_callers_observation(channel):
    tracing, exporter = channel
    _generation(tracing, name="child-gen", parent=PARENT_C).end(CALL_END)
    tracing.flush()

    span = _only_span(exporter, "child-gen")
    assert format(span.context.trace_id, "032x") == TRACE_A
    assert format(span.parent.span_id, "016x") == PARENT_C
    assert span.parent.is_remote


def test_existing_trace_is_appended_to_rather_than_rewritten(channel):
    """v2 ``existing_trace_id``: the generation joins the trace without becoming its root."""
    tracing, exporter = channel
    _generation(tracing, name="continued", existing=True).end(CALL_END)
    tracing.flush()

    span = _only_span(exporter, "continued")
    assert format(span.context.trace_id, "032x") == TRACE_A
    assert span.parent is not None
    assert span.parent.span_id != 0


def test_generation_does_not_hang_under_the_callers_active_span(channel):
    """The caller's own OTel span must stay untouched and must not become the generation's parent."""
    tracing, exporter = channel
    app_tracer = TracerProvider().get_tracer("app")
    with app_tracer.start_as_current_span("app-span") as app_span:
        _generation(tracing, trace_id="b" * 32).end(CALL_END)
        attributes_after = dict(app_span.attributes or {})
    tracing.flush()

    span = _only_span(exporter, "gen")
    assert span.parent is None
    assert format(span.context.trace_id, "032x") == "b" * 32
    assert attributes_after == {}


def test_requested_observation_id_becomes_the_exported_span_id(channel):
    """v2 ``generation(id=...)``: the caller's id is what the export carries and what ``.id`` returns."""
    tracing, exporter = channel
    requested = resolve_observation_id("chatcmpl-123")

    generation = _generation(tracing, observation_id=requested)
    generation.end(CALL_END)
    tracing.flush()

    assert generation.id == requested
    assert format(_only_span(exporter, "gen").context.span_id, "016x") == requested


def test_requested_ids_do_not_leak_into_the_next_span(channel):
    tracing, exporter = channel
    requested = resolve_observation_id("chatcmpl-123")

    _generation(tracing, name="first", observation_id=requested).end(CALL_END)
    second = _generation(tracing, name="second", trace_id=resolve_trace_id(None))
    second.end(CALL_END)
    child = start_child_span(tracing=tracing, parent=second, name="child", start_time=CALL_END, attributes={})
    child.end(CALL_END)
    tracing.flush()

    assert second.id != requested
    assert child.id not in (requested, second.id)
    assert len({span.context.span_id for span in exporter.get_finished_spans()}) == 3


@pytest.mark.parametrize("public", [True, False], ids=["public", "private"])
def test_trace_public_flag_lands_on_the_root_observation(channel, public):
    tracing, exporter = channel
    _generation(tracing, public=public, attributes=trace_attributes(public=public)).end(CALL_END)
    tracing.flush()
    assert _only_span(exporter, "gen").attributes[A.TRACE_PUBLIC] is public


def test_trace_public_flag_is_absent_when_not_requested(channel):
    tracing, exporter = channel
    _generation(tracing, attributes=trace_attributes(public=None)).end(CALL_END)
    tracing.flush()
    assert A.TRACE_PUBLIC not in _only_span(exporter, "gen").attributes


@pytest.mark.parametrize("public", [True, False, None], ids=["public", "private", "unset"])
def test_child_span_repeats_the_generation_public_flag(channel, public):
    """The server folds ``public`` across observations and reads a missing value as False.

    A guardrail span without the flag turned a ``trace_public: true`` request private on Langfuse Cloud.
    """
    tracing, exporter = channel
    generation = _generation(tracing, public=public)
    start_child_span(tracing=tracing, parent=generation, name="guardrail", start_time=CALL_END, attributes={}).end()
    generation.end(CALL_END)
    tracing.flush()

    assert _only_span(exporter, "guardrail").attributes.get(A.TRACE_PUBLIC) is public


def test_trace_attributes_carry_the_v2_trace_fields():
    attributes = trace_attributes(
        name="trace-name",
        user_id="user-1",
        session_id="session-1",
        version="v2",
        release="rel-1",
        tags=("a", "b"),
        metadata={"tenant": "t1", "nested": {"k": 1}},
        input={"messages": []},
        output="answer",
    )
    assert attributes[A.TRACE_NAME] == "trace-name"
    assert attributes[A.TRACE_USER_ID] == "user-1"
    assert attributes[A.TRACE_SESSION_ID] == "session-1"
    assert attributes[A.VERSION] == "v2"
    assert attributes[A.RELEASE] == "rel-1"
    assert attributes[A.TRACE_TAGS] == ("a", "b")
    assert attributes[f"{A.TRACE_METADATA}.tenant"] == "t1"
    assert json.loads(attributes[f"{A.TRACE_METADATA}.nested"]) == {"k": 1}
    assert json.loads(attributes[A.TRACE_INPUT]) == {"messages": []}
    assert attributes[A.TRACE_OUTPUT] == "answer"


def test_trace_attributes_skip_what_the_request_did_not_supply():
    assert dict(trace_attributes()) == {}


def test_non_mapping_metadata_is_carried_whole_instead_of_raising():
    """A truthy non-dict ``trace_metadata`` used to blow up the callback on ``**`` unpacking."""
    attributes = trace_attributes(metadata=("not", "a", "dict"))
    assert json.loads(attributes[A.TRACE_METADATA]) == ["not", "a", "dict"]


def test_observation_attributes_serialize_the_generation_fields():
    attributes = observation_attributes(
        observation_type="generation",
        input=[{"role": "user", "content": "hi"}],
        output={"role": "assistant", "content": "hello"},
        metadata=MappingProxyType({"litellm_call_id": "call-1", "cache_hit": False}),
        level="ERROR",
        status_message="boom",
        model="gpt-4o",
        model_parameters={"temperature": 0.1},
        usage_details={"input": 1, "output": 2},
        cost_details={"total": 0.01},
        prompt="not-a-prompt-client",
    )
    assert attributes[A.OBSERVATION_TYPE] == "generation"
    assert attributes[A.OBSERVATION_LEVEL] == "ERROR"
    assert attributes[A.OBSERVATION_STATUS_MESSAGE] == "boom"
    assert attributes[A.OBSERVATION_MODEL] == "gpt-4o"
    assert json.loads(attributes[A.OBSERVATION_INPUT]) == [{"role": "user", "content": "hi"}]
    assert json.loads(attributes[A.OBSERVATION_OUTPUT]) == {"role": "assistant", "content": "hello"}
    assert json.loads(attributes[A.OBSERVATION_MODEL_PARAMETERS]) == {"temperature": 0.1}
    assert json.loads(attributes[A.OBSERVATION_USAGE_DETAILS]) == {"input": 1, "output": 2}
    assert json.loads(attributes[A.OBSERVATION_COST_DETAILS]) == {"total": 0.01}
    assert attributes[f"{A.OBSERVATION_METADATA}.litellm_call_id"] == "call-1"
    assert attributes[f"{A.OBSERVATION_METADATA}.cache_hit"] is False
    assert A.OBSERVATION_PROMPT_NAME not in attributes


@pytest.mark.parametrize(
    "supplied, expected",
    [
        ("0123456789abcdef0123456789abcdef", "0123456789abcdef0123456789abcdef"),
        ("0123456789ABCDEF0123456789ABCDEF", "0123456789abcdef0123456789abcdef"),
        ("3fe0c940-b69a-de3b-a77c-06102505349a", "3fe0c940b69ade3ba77c06102505349a"),
    ],
    ids=["already-hex", "uppercase-hex", "uuid-with-dashes"],
)
def test_trace_id_passes_through_when_it_is_already_usable(supplied, expected):
    assert resolve_trace_id(supplied) == expected


def test_arbitrary_trace_id_is_hashed_deterministically():
    first = resolve_trace_id("order-4471")
    assert first == resolve_trace_id("order-4471")
    assert len(first) == 32 and first == first.lower()
    assert first != resolve_trace_id("order-4472")


@pytest.mark.parametrize("supplied", [12345, 12.5, True, False], ids=["int", "float", "true", "false"])
def test_non_string_trace_id_is_normalized(supplied):
    resolved = resolve_trace_id(supplied)

    assert len(resolved) == 32
    assert resolved == resolve_trace_id(supplied)


@pytest.mark.parametrize("supplied", [12345, 12.5, True, False], ids=["int", "float", "true", "false"])
def test_non_string_observation_id_is_normalized(supplied):
    resolved = resolve_observation_id(supplied)

    assert len(resolved) == 16
    assert resolved == resolve_observation_id(supplied)


def test_all_zero_ids_are_hashed_instead_of_passed_through():
    zero_trace = "0" * 32
    zero_span = "0" * 16

    assert resolve_trace_id(zero_trace) != zero_trace
    assert resolve_trace_id(zero_trace) == resolve_trace_id(zero_trace)
    assert int(resolve_trace_id(zero_trace), 16) != 0
    assert resolve_observation_id(zero_span) != zero_span
    assert int(resolve_observation_id(zero_span), 16) != 0


def test_hyphen_only_trace_ids_are_deterministic():
    assert resolve_trace_id("---") == resolve_trace_id("---")


def test_trace_id_with_trailing_newline_is_hashed():
    supplied = "a" * 32 + "\n"

    resolved = resolve_trace_id(supplied)

    assert resolved != supplied
    assert len(resolved) == 32


def test_missing_trace_id_still_yields_a_valid_trace_id():
    generated = resolve_trace_id(None)
    assert len(generated) == 32
    assert int(generated, 16) >= 0


@pytest.mark.parametrize(
    "supplied, expected",
    [
        ("0123456789abcdef", "0123456789abcdef"),
        (None, None),
        ("", None),
    ],
    ids=["already-hex", "none", "empty"],
)
def test_observation_id_normalisation(supplied, expected):
    assert resolve_observation_id(supplied) == expected


def test_arbitrary_observation_id_is_hashed_to_a_span_id():
    resolved = resolve_observation_id("my-parent-observation")
    assert len(resolved) == 16
    assert resolved == resolve_observation_id("my-parent-observation")


@pytest.mark.parametrize("unsupported", ["2.59.7", "3.15.0", "5.0.0"], ids=["v2", "v3", "v5"])
def test_unsupported_sdk_fails_loudly_rather_than_dropping_every_event(unsupported):
    with pytest.raises(ImportError) as raised:
        raise_if_unsupported_langfuse_version(unsupported)
    assert unsupported in str(raised.value)
    assert MINIMUM_LANGFUSE_VERSION in str(raised.value)


def test_supported_sdk_is_accepted():
    assert raise_if_unsupported_langfuse_version(installed_langfuse_version()) is None


def test_channel_carries_environment_and_release_on_the_resource():
    tracing = build_langfuse_tracing(
        exporter=DiscardingSpanExporter(),
        environment="staging",
        release="v9",
        sample_rate=1.0,
        flush_interval_millis=10,
    )
    attributes = tracing.provider.resource.attributes
    assert attributes[A.ENVIRONMENT] == "staging"
    assert attributes[A.RELEASE] == "v9"


def _generations_exported_at(sample_rate: float, trace_ids: tuple[str, ...]) -> frozenset[str]:
    exporter: Final = InMemorySpanExporter()
    tracing: Final = build_langfuse_tracing(
        exporter=exporter, environment=None, release=None, sample_rate=sample_rate, flush_interval_millis=10
    )
    for trace_id in trace_ids:
        _generation(tracing, name="sampled", trace_id=trace_id).end(CALL_END)
    tracing.flush()
    return frozenset(format(span.context.trace_id, "032x") for span in exporter.get_finished_spans())


def test_sample_rate_zero_drops_and_one_keeps_every_trace():
    trace_ids: Final = tuple(resolve_trace_id(uuid.uuid4()) for _ in range(20))
    assert _generations_exported_at(0, trace_ids) == frozenset()
    assert _generations_exported_at(1, trace_ids) == frozenset(trace_ids)


def test_fractional_sample_rate_keeps_a_deterministic_share_of_uuid_trace_ids():
    trace_ids: Final = tuple(resolve_trace_id(uuid.uuid4()) for _ in range(400))
    kept: Final = _generations_exported_at(0.5, trace_ids)
    assert 140 <= len(kept) <= 260
    assert _generations_exported_at(0.5, trace_ids) == kept
    assert kept < _generations_exported_at(0.9, trace_ids)


@pytest.mark.parametrize("raw", ["1.5", "-0.5", "abc"])
def test_unusable_sample_rate_warns_and_exports_everything(
    monkeypatch: pytest.MonkeyPatch, raw: str, caplog: pytest.LogCaptureFixture
):
    monkeypatch.setenv("LANGFUSE_SAMPLE_RATE", raw)
    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        assert configured_sample_rate() == 1.0
    assert "LANGFUSE_SAMPLE_RATE" in caplog.text


def test_configured_sample_rate_reads_the_env_var(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LANGFUSE_SAMPLE_RATE", raising=False)
    assert configured_sample_rate() == 1.0
    monkeypatch.setenv("LANGFUSE_SAMPLE_RATE", "0.25")
    assert configured_sample_rate() == 0.25


def _exported_generations(exporter: InMemorySpanExporter, tracing: LangfuseTracing, count: int) -> tuple:
    for _ in range(count):
        _generation(tracing, trace_id=resolve_trace_id(uuid.uuid4())).end(CALL_END)
    tracing.flush()
    return exporter.get_finished_spans()


def test_full_sample_rate_exports_every_trace_even_when_the_host_turned_otel_sampling_off(
    monkeypatch: pytest.MonkeyPatch,
):
    """A provider built without a sampler reads ``OTEL_TRACES_SAMPLER``, which belongs to the host's tracing."""
    monkeypatch.setenv("OTEL_TRACES_SAMPLER", "always_off")
    exporter = InMemorySpanExporter()
    tracing = build_langfuse_tracing(
        exporter=exporter, environment=None, release=None, sample_rate=1.0, flush_interval_millis=10
    )
    assert len(_exported_generations(exporter, tracing, 5)) == 5


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT", "4"),
        ("OTEL_ATTRIBUTE_COUNT_LIMIT", "4"),
        ("OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT", "8"),
        ("OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT", "8"),
    ],
)
def test_host_otel_span_limits_do_not_truncate_langfuse_observations(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str
):
    monkeypatch.setenv(variable, value)
    exporter = InMemorySpanExporter()
    tracing = build_langfuse_tracing(
        exporter=exporter, environment=None, release=None, sample_rate=1.0, flush_interval_millis=10
    )
    attributes = {f"langfuse.observation.metadata.k{i}": "v" * 32 for i in range(40)}
    _generation(tracing, attributes=attributes).end(CALL_END)
    tracing.flush()

    span = _only_span(exporter, "gen")
    assert span.dropped_attributes == 0
    assert all(span.attributes[key] == "v" * 32 for key in attributes)


def test_host_otel_resource_env_does_not_reach_the_langfuse_resource(monkeypatch: pytest.MonkeyPatch):
    """``OTEL_RESOURCE_ATTRIBUTES`` and ``OTEL_SERVICE_NAME`` belong to the host's tracing; Langfuse files a
    trace under any ``deployment.environment`` it finds on the resource, and v2 shipped no resource at all."""
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "team.secret.note=internal-only,deployment.environment=hijack")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "the-hosts-own-service")
    tracing = build_langfuse_tracing(
        exporter=InMemorySpanExporter(), environment="prod", release="r1", sample_rate=1.0, flush_interval_millis=10
    )

    assert dict(tracing.provider.resource.attributes) == {A.ENVIRONMENT: "prod", A.RELEASE: "r1"}


@pytest.mark.parametrize(
    ("value", "encoded"),
    [
        (2**53 - 1, ("int_value", 2**53 - 1)),
        (-(2**53) + 1, ("int_value", -(2**53) + 1)),
        (2**53, ("string_value", str(2**53))),
        (2**63 - 1, ("string_value", str(2**63 - 1))),
        (2**63, ("string_value", str(2**63))),
        (10**20, ("string_value", str(10**20))),
        (-(2**63) - 1, ("string_value", str(-(2**63) - 1))),
        (True, ("bool_value", True)),
    ],
    ids=[
        "json-safe-max",
        "json-safe-min",
        "json-safe-plus-one",
        "int64-max",
        "int64-max-plus-one",
        "huge",
        "int64-min-minus-one",
        "bool",
    ],
)
def test_metadata_ints_past_the_json_safe_range_reach_the_wire_as_strings(value, encoded):
    """OTLP carries int64 only and its encoder silently drops any attribute it cannot fit, while the export
    still succeeds, and Langfuse's reader rounds ints past 2**53 (int64 max read back as 9223372036854776000
    on 2026-09-21, where the v2 leg showed the exact digits as a string), so both ranges go as strings."""
    from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans

    exporter = InMemorySpanExporter()
    tracing = build_langfuse_tracing(
        exporter=exporter, environment=None, release=None, sample_rate=1.0, flush_interval_millis=10
    )
    attributes = observation_attributes(observation_type="generation", metadata={"order_id": value, "sibling": "kept"})
    _generation(tracing, attributes=attributes).end(CALL_END)
    tracing.flush()

    (encoded_span,) = encode_spans(exporter.get_finished_spans()).resource_spans[0].scope_spans[0].spans
    wire = {kv.key: kv.value for kv in encoded_span.attributes}
    order_id = wire[f"{A.OBSERVATION_METADATA}.order_id"]
    carried = {
        "int_value": order_id.int_value,
        "string_value": order_id.string_value,
        "bool_value": order_id.bool_value,
    }
    assert (order_id.WhichOneof("value"), carried[order_id.WhichOneof("value")]) == encoded
    assert wire[f"{A.OBSERVATION_METADATA}.sibling"].string_value == "kept"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, 60.0), ("5", 5.0), ("0", 0.0), (" -1 ", 60.0), ("2.5", 60.0), ("abc", 60.0)],
    ids=["unset", "whole", "zero", "negative", "fraction", "text"],
)
def test_prompt_cache_ttl_env_falls_back_instead_of_raising(monkeypatch: pytest.MonkeyPatch, raw, expected, caplog):
    """The SDK reads this knob as whole seconds; a negative one passes its import but must not cache forever,
    and anything else falls back rather than raising out of logger construction."""
    if raw is None:
        monkeypatch.delenv("LANGFUSE_PROMPT_CACHE_DEFAULT_TTL_SECONDS", raising=False)
    else:
        monkeypatch.setenv("LANGFUSE_PROMPT_CACHE_DEFAULT_TTL_SECONDS", raw)
    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        assert configured_prompt_cache_ttl() == expected
    assert ("LANGFUSE_PROMPT_CACHE_DEFAULT_TTL_SECONDS" in caplog.text) is (expected == 60.0 and raw is not None)


def test_many_metadata_keys_never_evict_the_generation_input_and_output():
    """OTel's default 128-attribute cap drops the earliest attributes, and v2 never capped metadata."""
    exporter = InMemorySpanExporter()
    tracing = build_langfuse_tracing(
        exporter=exporter, environment=None, release=None, sample_rate=1.0, flush_interval_millis=10
    )
    attributes = {
        A.OBSERVATION_INPUT: "the-prompt",
        A.OBSERVATION_OUTPUT: "the-completion",
        **{f"langfuse.observation.metadata.k{i}": str(i) for i in range(300)},
    }
    _generation(tracing, attributes=attributes).end(CALL_END)
    tracing.flush()

    span = _only_span(exporter, "gen")
    assert span.dropped_attributes == 0
    assert span.attributes[A.OBSERVATION_INPUT] == "the-prompt"
    assert span.attributes[A.OBSERVATION_OUTPUT] == "the-completion"
    assert span.attributes["langfuse.observation.metadata.k299"] == "299"


def test_otel_sdk_disabled_still_wins_but_is_called_out(monkeypatch: pytest.MonkeyPatch, caplog):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    exporter = InMemorySpanExporter()
    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        tracing = build_langfuse_tracing(
            exporter=exporter, environment=None, release=None, sample_rate=1.0, flush_interval_millis=10
        )
    assert "OTEL_SDK_DISABLED" in caplog.text
    assert _exported_generations(exporter, tracing, 3) == ()


def test_spans_carry_the_langfuse_sdk_scope_name(channel):
    """Langfuse keys on the SDK's instrumentation scope (langfuse 4.15.2, ``langfuse/_client/constants.py``,
    read 2026-09-17); any other scope is foreign OTel traffic whose raw attributes get echoed into metadata."""
    tracing, exporter = channel
    _generation(tracing).end(CALL_END)
    tracing.flush()
    assert _only_span(exporter, "gen").instrumentation_scope.name == "langfuse-sdk"


class _GatedExporter(SpanExporter):
    """Hold the export thread until released, so spans pile up in the processor queue."""

    def __init__(self) -> None:
        self.gate = threading.Event()
        self.batches: list[int] = []

    def export(self, spans) -> SpanExportResult:
        self.gate.wait(timeout=30)
        self.batches.append(len(spans))
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


def test_export_queue_holds_a_v2_sized_burst_while_the_destination_stalls():
    """v2 queued 100k events; OTel's default 2048 dropped most of a burst during a destination stall."""
    exporter = _GatedExporter()
    tracing = build_langfuse_tracing(
        exporter=exporter, environment=None, release=None, sample_rate=1.0, flush_interval_millis=10
    )
    for _ in range(6000):
        _generation(tracing, trace_id=resolve_trace_id(uuid.uuid4())).end(CALL_END)
    exporter.gate.set()
    assert tracing.flush(timeout_millis=30_000) is True
    assert sum(exporter.batches) == 6000


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, 512), ("64", 64), ("0", 512), ("-5", 512), ("abc", 512), ("100001", 512), ("100000", 100_000)],
    ids=["unset", "valid", "zero", "negative", "text", "over-queue", "at-queue"],
)
def test_langfuse_flush_at_is_parsed_like_the_sdk_did(monkeypatch: pytest.MonkeyPatch, raw, expected, caplog):
    if raw is None:
        monkeypatch.delenv("LANGFUSE_FLUSH_AT", raising=False)
    else:
        monkeypatch.setenv("LANGFUSE_FLUSH_AT", raw)
    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        assert configured_flush_at() == expected
    assert ("LANGFUSE_FLUSH_AT" in caplog.text) is (raw is not None and str(expected) != raw)


def test_langfuse_flush_at_sizes_the_export_batches(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LANGFUSE_FLUSH_AT", "64")
    exporter = _GatedExporter()
    exporter.gate.set()
    tracing = build_langfuse_tracing(
        exporter=exporter,
        environment=None,
        release=None,
        sample_rate=1.0,
        flush_interval_millis=60_000,
        flush_at=configured_flush_at(),
    )
    for _ in range(200):
        _generation(tracing, trace_id=resolve_trace_id(uuid.uuid4())).end(CALL_END)
    tracing.flush()
    assert sum(exporter.batches) == 200
    assert max(exporter.batches) == 64


def test_acquired_channel_reads_langfuse_flush_at(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LANGFUSE_FLUSH_AT", "7")
    small = _acquire(public_key="pk-flush-at-test")
    monkeypatch.setenv("LANGFUSE_FLUSH_AT", "9")
    assert _acquire(public_key="pk-flush-at-test") is not small


def test_channel_does_not_take_over_the_process_tracer_provider():
    provider_before = otel_trace.get_tracer_provider()

    tracing = acquire_langfuse_tracing(
        public_key="pk-global-test",
        secret_key="sk",
        base_url="http://127.0.0.1:1",
        environment=None,
        release=None,
        flush_interval=1.0,
        mock_mode=True,
    )

    assert otel_trace.get_tracer_provider() is provider_before
    assert tracing.provider is not provider_before


def _acquire(**overrides):
    parameters = {
        "public_key": "pk-cache-test",
        "secret_key": "sk-cache",
        "base_url": "http://127.0.0.1:1",
        "environment": None,
        "release": None,
        "flush_interval": 1.0,
        "mock_mode": True,
    }
    return acquire_langfuse_tracing(**{**parameters, **overrides})


def test_same_credentials_share_one_channel():
    assert _acquire() is _acquire()


@pytest.mark.parametrize(
    "override",
    [
        {"secret_key": "sk-rotated"},
        {"base_url": "http://127.0.0.1:2"},
        {"environment": "staging"},
        {"mock_mode": False},
    ],
    ids=["secret", "host", "environment", "mock-to-live"],
)
def test_changed_credentials_or_settings_get_their_own_channel(override):
    assert _acquire() is not _acquire(**override)


class _RecordsShutdown(InMemorySpanExporter):
    def __init__(self) -> None:
        super().__init__()
        self.shutdowns = 0

    def shutdown(self) -> None:
        self.shutdowns += 1
        super().shutdown()


def _acquire_recorded(monkeypatch: pytest.MonkeyPatch, public_key: str) -> tuple[LangfuseTracing, _RecordsShutdown]:
    exporter = _RecordsShutdown()
    monkeypatch.setattr("litellm.integrations.langfuse.langfuse_sdk._build_span_exporter", lambda **_: exporter)
    return _acquire(public_key=public_key, mock_mode=False, flush_interval=600.0), exporter


def test_channel_is_retired_only_after_its_last_holder_releases_it(monkeypatch: pytest.MonkeyPatch):
    """Two loggers on one credential set share the channel: the first release must leave it
    exporting for the second, and the last release must shut the batch thread down and drop
    the registry entry so the next logger gets a fresh channel instead of a dead one."""
    first, exporter = _acquire_recorded(monkeypatch, "pk-lease-test")
    second = _acquire(public_key="pk-lease-test", mock_mode=False, flush_interval=600.0)
    assert second is first

    release_langfuse_tracing(first, grace_seconds=0.0)
    second.tracer.start_span("generation").end()
    assert exporter.shutdowns == 0
    assert flush_langfuse_tracing() is True
    assert len(exporter.get_finished_spans()) == 1

    release_langfuse_tracing(second, grace_seconds=0.0)
    assert exporter.shutdowns == 1
    assert _acquire(public_key="pk-lease-test", mock_mode=False, flush_interval=600.0) is not first


def test_release_flushes_the_queued_spans_before_the_channel_goes_away(monkeypatch: pytest.MonkeyPatch):
    tracing, exporter = _acquire_recorded(monkeypatch, "pk-lease-flush-test")
    tracing.tracer.start_span("generation").end()

    release_langfuse_tracing(tracing, grace_seconds=0.0)

    assert len(exporter.get_finished_spans()) == 1


def test_channel_reacquired_within_the_grace_is_kept(monkeypatch: pytest.MonkeyPatch):
    """A logger rebuilt for the same credentials right after the old one expired, and a callback
    that fetched the old logger just before expiry, both keep exporting through the same channel."""
    tracing, exporter = _acquire_recorded(monkeypatch, "pk-lease-grace-test")

    release_langfuse_tracing(tracing, grace_seconds=0.2)
    assert _acquire(public_key="pk-lease-grace-test", mock_mode=False, flush_interval=600.0) is tracing

    threading.Event().wait(0.5)
    tracing.tracer.start_span("generation").end()
    assert exporter.shutdowns == 0
    assert flush_langfuse_tracing() is True
    assert len(exporter.get_finished_spans()) == 1


def test_retire_timer_of_an_earlier_release_cannot_kill_a_reacquired_channel(monkeypatch: pytest.MonkeyPatch):
    """release, re-acquire, release: the first timer used to fire into a channel that a later holder still
    counted on for its own grace period, shutting the batch thread down while spans were still queued."""
    tracing, exporter = _acquire_recorded(monkeypatch, "pk-lease-race-test")

    release_langfuse_tracing(tracing, grace_seconds=0.2)
    assert _acquire(public_key="pk-lease-race-test", mock_mode=False, flush_interval=600.0) is tracing
    release_langfuse_tracing(tracing, grace_seconds=600.0)

    threading.Event().wait(0.5)
    assert exporter.shutdowns == 0
    assert _acquire(public_key="pk-lease-race-test", mock_mode=False, flush_interval=600.0) is tracing
    tracing.tracer.start_span("generation").end()
    assert tracing.flush() is True
    assert len(exporter.get_finished_spans()) == 1


class _RejectsEverything(SpanExporter):
    def export(self, spans) -> SpanExportResult:
        return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


def test_flush_is_false_when_the_destination_rejected_a_batch(monkeypatch: pytest.MonkeyPatch):
    """The shutdown hook logs "channels flushed" off this value; a drained queue whose batches all
    failed at the destination is a loss, not a flush."""
    monkeypatch.setattr(
        "litellm.integrations.langfuse.langfuse_sdk._build_span_exporter", lambda **_: _RejectsEverything()
    )
    tracing = _acquire(public_key="pk-flush-truth-test", mock_mode=False, flush_interval=600.0)
    tracing.tracer.start_span("generation").end()

    assert tracing.flush() is False
    assert flush_langfuse_tracing() is True, "an empty queue after the loss has nothing left to fail"


def test_release_of_a_channel_the_registry_never_handed_out_is_a_no_op():
    exporter = InMemorySpanExporter()
    tracing = build_langfuse_tracing(
        exporter=exporter, environment=None, release=None, sample_rate=1.0, flush_interval_millis=10
    )

    release_langfuse_tracing(tracing, grace_seconds=0.0)
    tracing.tracer.start_span("generation").end()

    assert tracing.flush() is True
    assert len(exporter.get_finished_spans()) == 1


def test_flush_langfuse_tracing_exports_the_queued_spans_of_every_channel(monkeypatch: pytest.MonkeyPatch):
    """The proxy shutdown hook flushes through this, so a span finished just before a
    graceful restart must reach the exporter without waiting for the batch interval."""
    exporters: Final[
        list[InMemorySpanExporter]
    ] = []  # mutable-ok: collects the exporters the patched builder hands out

    def build_in_memory(*, public_key: str, secret_key: str, base_url: str) -> InMemorySpanExporter:
        exporters.append(InMemorySpanExporter())
        return exporters[-1]

    monkeypatch.setattr("litellm.integrations.langfuse.langfuse_sdk._build_span_exporter", build_in_memory)
    for public_key in ("pk-flush-test-a", "pk-flush-test-b"):
        _acquire(public_key=public_key, mock_mode=False, flush_interval=600.0).tracer.start_span("generation").end()

    assert [len(exporter.get_finished_spans()) for exporter in exporters] == [0, 0]
    assert flush_langfuse_tracing() is True
    assert [len(exporter.get_finished_spans()) for exporter in exporters] == [1, 1]


def test_flush_langfuse_tracing_flushes_channels_concurrently_under_one_deadline(monkeypatch: pytest.MonkeyPatch):
    """A channel stuck on an unreachable host must not spend the whole deadline before the
    next channel gets its turn; the first exporter here only returns once the second exported."""
    second_exported = threading.Event()

    class WaitsForTheOther(SpanExporter):
        def export(self, spans):
            return SpanExportResult.SUCCESS if second_exported.wait(timeout=5.0) else SpanExportResult.FAILURE

        def shutdown(self) -> None:
            return None

    class Unblocks(SpanExporter):
        def export(self, spans):
            second_exported.set()
            return SpanExportResult.SUCCESS

        def shutdown(self) -> None:
            return None

    exporters = iter((WaitsForTheOther(), Unblocks()))

    def build_next(*, public_key: str, secret_key: str, base_url: str) -> SpanExporter:
        return next(exporters)

    monkeypatch.setattr("litellm.integrations.langfuse.langfuse_sdk._build_span_exporter", build_next)
    for public_key in ("pk-concurrent-flush-a", "pk-concurrent-flush-b"):
        _acquire(public_key=public_key, mock_mode=False, flush_interval=600.0).tracer.start_span("generation").end()

    assert flush_langfuse_tracing(timeout_millis=2_000) is True
    assert second_exported.is_set()


def test_flush_langfuse_tracing_leaves_an_overrunning_channel_on_a_daemon_thread():
    """A channel whose flush outlives the deadline is reported as failed and must not be able to
    hold up interpreter exit, so the thread still flushing it has to be a daemon."""
    release = threading.Event()

    class BlocksUntilReleased(SpanProcessor):
        def force_flush(self, timeout_millis: int = 30_000) -> bool:
            return release.wait(timeout=10.0)

    _acquire(public_key="pk-overrunning-flush", mock_mode=True, flush_interval=600.0).provider.add_span_processor(
        BlocksUntilReleased()
    )
    try:
        assert flush_langfuse_tracing(timeout_millis=200) is False
        stuck = [thread for thread in threading.enumerate() if thread.name.startswith("langfuse-flush")]
        assert stuck and all(thread.daemon for thread in stuck)
    finally:
        release.set()


def test_a_changed_sample_rate_rebuilds_the_channel(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LANGFUSE_SAMPLE_RATE", "0.25")
    quarter = _acquire(public_key="pk-resample-test")
    monkeypatch.setenv("LANGFUSE_SAMPLE_RATE", "1")
    full = _acquire(public_key="pk-resample-test")

    assert full is not quarter
    assert quarter.provider.sampler.get_description() == "TraceIdHashSampler{0.25}"
    assert "TraceIdHashSampler" not in full.provider.sampler.get_description()


def _recording_transport(requests: list[httpx.Request], status: int = 401) -> httpx.Client:
    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status, json=_PROJECTS_BODY if status == 200 else {"message": "unauthorized"})

    return httpx.Client(transport=httpx.MockTransport(record))


_PROJECTS_BODY: Final = {
    "data": [{"id": "proj-under-test", "name": "p", "metadata": {}, "organization": {"id": "o", "name": "o"}}]
}


def test_rest_client_authenticates_with_the_credentials_it_was_built_with():
    """Two loggers for one public key but different secrets or hosts each talk to their own project."""
    requests: list[httpx.Request] = []
    build_langfuse_client(
        public_key="pk-rest-test",
        secret_key="sk-first",
        base_url="http://127.0.0.1:1",
        httpx_client=_recording_transport(requests),
    )
    rotated = build_langfuse_client(
        public_key="pk-rest-test",
        secret_key="sk-second",
        base_url="http://127.0.0.1:2",
        httpx_client=_recording_transport(requests),
    )

    assert rotated.auth_check() is not None
    assert requests[-1].url.host == "127.0.0.1" and requests[-1].url.port == 2
    assert requests[-1].headers["authorization"] == "Basic " + b64encode(b"pk-rest-test:sk-second").decode()


def test_rest_client_without_keys_fails_auth_check_instead_of_raising(monkeypatch):
    """``/health/services?service=langfuse`` with no credentials must report a failed check, not crash."""
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    client = build_langfuse_client(public_key=None, secret_key=None, base_url="http://127.0.0.1:1", httpx_client=None)
    assert client.auth_check() is not None


def test_auth_check_names_the_servers_rejection(caplog):
    """``/health/services`` used to print the 401 verbatim; a generic credentials message hides a 403 or a 500."""
    client = build_langfuse_client(
        public_key="pk",
        secret_key="sk",
        base_url="http://127.0.0.1:1",
        httpx_client=_recording_transport([], status=401),
    )
    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        failure = client.auth_check()
    assert failure is not None
    assert failure.reason == "status_code: 401, body: {'message': 'unauthorized'}"
    assert failure.reason in caplog.text


def test_auth_check_names_an_unreachable_destination_rather_than_the_keys():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused by lf.internal.example", request=request)

    client = build_langfuse_client(
        public_key="pk",
        secret_key="sk",
        base_url="http://lf.internal.example",
        httpx_client=httpx.Client(transport=httpx.MockTransport(refuse)),
    )
    failure = client.auth_check()
    assert failure is not None
    assert "connection refused by lf.internal.example" in failure.reason


def test_auth_check_fails_when_the_keys_reach_no_project():
    """A 200 with an empty project list is what the SDK's own ``auth_check`` raises on; it is not a pass."""
    client = build_langfuse_client(
        public_key="pk",
        secret_key="sk",
        base_url="http://127.0.0.1:1",
        httpx_client=httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"data": []}))),
    )
    failure = client.auth_check()
    assert failure is not None
    assert "no project" in failure.reason


@pytest.mark.parametrize("status", [500, 503, 429], ids=["http-500", "http-503", "http-429"])
def test_auth_check_and_project_id_make_one_round_trip_when_langfuse_is_down(status):
    """Both run on the event loop; the generated client's default retries sleep for seconds, or for Retry-After."""
    requests: list[httpx.Request] = []

    def fail(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status, request=request, headers={"retry-after": "20"}, json={"message": "down"})

    client = build_langfuse_client(
        public_key="pk",
        secret_key="sk",
        base_url="http://127.0.0.1:1",
        httpx_client=httpx.Client(transport=httpx.MockTransport(fail)),
    )

    started = monotonic()
    failure = client.auth_check()
    with pytest.raises(ApiError):
        client.project_id()
    assert failure is not None and f"status_code: {status}" in failure.reason
    assert len(requests) == 2
    assert monotonic() - started < 0.5


@pytest.mark.parametrize(
    ("status", "round_trips"),
    [(500, 2), (503, 2), (429, 1), (404, 1)],
    ids=["http-500", "http-503", "http-429", "http-404"],
)
def test_cold_prompt_miss_never_sleeps_when_langfuse_is_down(status: int, round_trips: int):
    """A cold ``get_prompt`` fetches inline on the event loop; with the generated client's default retries a
    429 carrying ``Retry-After: 30`` used to hold the loop for a minute. A 5xx gets the v2 client's one
    quick retry, a 429 or 4xx none."""
    requests: list[httpx.Request] = []

    def fail(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status, request=request, headers={"retry-after": "30"}, json={"message": "down"})

    client = build_langfuse_client(
        public_key="pk",
        secret_key="sk",
        base_url="http://127.0.0.1:1",
        httpx_client=httpx.Client(transport=httpx.MockTransport(fail)),
    )

    started = monotonic()
    with pytest.raises(LangfusePromptError) as caught:
        client.get_prompt("greeting")
    assert len(requests) == round_trips
    assert monotonic() - started < 0.5
    assert caught.value.status_code == status


@pytest.mark.parametrize("first_failure", [503, "connect-error"], ids=["http-503", "connect-error"])
def test_one_transient_failure_on_a_cold_prompt_miss_does_not_fail_the_call(first_failure: int | str):
    """The v2 client retried a cold fetch once; a single Langfuse blip must not fail the LLM call."""
    requests: list[httpx.Request] = []

    def flaky(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) > 1:
            return httpx.Response(200, request=request, json=_TEXT_PROMPT_BODY)
        if isinstance(first_failure, int):
            return httpx.Response(first_failure, request=request, json={"message": "down"})
        raise httpx.ConnectError("refused", request=request)

    client = build_langfuse_client(
        public_key="pk",
        secret_key="sk",
        base_url="http://127.0.0.1:1",
        httpx_client=httpx.Client(transport=httpx.MockTransport(flaky)),
    )

    started = monotonic()
    assert client.get_prompt("greeting").compile() == "hello"
    assert len(requests) == 2
    assert monotonic() - started < 0.5
    assert client.get_prompt("greeting").compile() == "hello", "the retried prompt is cached like any other"
    assert len(requests) == 2


def test_prompt_fetch_error_carries_status_and_body_but_no_upstream_headers():
    """The proxy forwards an exception's ``headers`` to its client and prints ``str(e)``; the generated
    ``ApiError`` carries Langfuse's response headers in both."""
    upstream_headers = {"server": "langfuse-edge", "set-cookie": "session=abc; HttpOnly", "x-upstream-internal": "1"}

    def not_found(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, request=request, headers=upstream_headers, json={"message": "Prompt not found"})

    client = build_langfuse_client(
        public_key="pk",
        secret_key="sk",
        base_url="http://127.0.0.1:1",
        httpx_client=httpx.Client(transport=httpx.MockTransport(not_found)),
    )

    with pytest.raises(Exception, match="Prompt not found") as caught:
        client.get_prompt("missing")

    error = caught.value
    assert getattr(error, "headers", None) is None
    assert getattr(error, "status_code", None) == 404
    assert not any(header in str(error) for header in upstream_headers)
    assert error.__cause__ is None and error.__suppress_context__, "the header-bearing ApiError must not ride along"


_TEXT_PROMPT_BODY: Final[dict[str, object]] = {
    "type": "text",
    "name": "n",
    "version": 1,
    "config": {},
    "labels": ["production"],
    "tags": [],
    "prompt": "hello",
}


@pytest.mark.parametrize(
    ("name", "encoded"),
    [
        ("what?", "what%3F"),
        ("folder/greeting", "folder%2Fgreeting"),
        ("my-prompt?label=staging", "my-prompt%3Flabel%3Dstaging"),
        ("100% sure#1", "100%25%20sure%231"),
    ],
    ids=["question-mark", "folder-slash", "query-injection", "percent-space-hash"],
)
def test_prompt_name_is_url_encoded_into_the_request_path(name: str, encoded: str):
    """The v2 client quoted the name before building the path and the v4 SDK's ``get_prompt`` does too; the
    generated client alone puts the raw name into the URL, so ``what?`` fetched prompt ``what`` and
    ``a/b`` left the prompts route."""
    requests: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, request=request, json=_TEXT_PROMPT_BODY)

    client = build_langfuse_client(
        public_key="pk",
        secret_key="sk",
        base_url="http://127.0.0.1:1",
        httpx_client=httpx.Client(transport=httpx.MockTransport(record)),
    )

    client.get_prompt(name, label="staging")

    (request,) = requests
    assert request.url.raw_path == f"/api/public/v2/prompts/{encoded}?label=staging".encode()


def test_rest_client_reports_the_project_id_and_a_passing_auth_check():
    requests: list[httpx.Request] = []
    client = build_langfuse_client(
        public_key="pk-project-test",
        secret_key="sk",
        base_url="http://127.0.0.1:1",
        httpx_client=_recording_transport(requests, status=200),
    )
    assert client.project_id() == "proj-under-test"
    assert client.auth_check() is None


def test_rest_client_leaves_a_host_applications_langfuse_client_alone():
    """The SDK hands every ``Langfuse()`` built for one public key the same resource bundle, so a
    litellm-built SDK client used to make a host application's client fetch through litellm's
    host, secret and httpx client. litellm now speaks REST directly and registers nothing."""
    from langfuse import Langfuse

    requests: list[httpx.Request] = []
    litellm_client = build_langfuse_client(
        public_key="pk-shared-with-host",
        secret_key="sk-litellm",
        base_url="http://litellm.example",
        httpx_client=_recording_transport(requests, status=200),
    )
    assert litellm_client.project_id() == "proj-under-test"

    host_requests: list[httpx.Request] = []
    host = Langfuse(
        public_key="pk-shared-with-host",
        secret_key="sk-host",
        base_url="http://host.example",
        httpx_client=_recording_transport(host_requests, status=200),
        tracing_enabled=False,
    )
    try:
        assert host.auth_check() is True
    finally:
        host.shutdown()

    assert [request.url.host for request in requests] == ["litellm.example"]
    assert host_requests[-1].url.host == "host.example"
    assert host_requests[-1].headers["authorization"] == "Basic " + b64encode(b"pk-shared-with-host:sk-host").decode()


def test_rest_client_does_not_take_over_the_process_tracer_provider():
    provider_before = otel_trace.get_tracer_provider()
    build_langfuse_client(
        public_key="pk-sdk-global-test", secret_key="sk", base_url="http://127.0.0.1:1", httpx_client=None
    )
    assert otel_trace.get_tracer_provider() is provider_before


def _finished_span():
    provider = TracerProvider()
    span = provider.get_tracer("t").start_span("generation")
    span.end()
    return span


def _exporter_over(responses, *, delays=(0.5, 1.5), timeout=5.0):
    """A LangfuseSpanExporter whose litellm HTTPHandler talks to a scripted transport instead of the network."""
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    seen = []
    script = list(responses)

    def transport(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        step = script.pop(0)
        if isinstance(step, Exception):
            raise step
        return httpx.Response(step, request=request)

    handler = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(transport)))
    exporter = LangfuseSpanExporter(
        handler=handler,
        endpoint="https://lf.internal.example/api/public/otel/v1/traces",
        headers=MappingProxyType({"Authorization": "Basic cGs6c2s=", "Content-Type": "application/x-protobuf"}),
        timeout=timeout,
        delays=delays,
    )
    return exporter, seen


def test_exporter_posts_the_otlp_batch_through_litellm_http_handler(monkeypatch):
    """Traces travel through litellm's own handler, so litellm's TLS and proxy settings apply to them."""
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    slept = []
    monkeypatch.setattr("litellm.integrations.langfuse.langfuse_sdk.sleep", slept.append)
    exporter, seen = _exporter_over([200])
    span = _finished_span()

    assert exporter.export((span,)) is SpanExportResult.SUCCESS

    (request,) = seen
    assert request.method == "POST"
    assert str(request.url) == "https://lf.internal.example/api/public/otel/v1/traces"
    assert request.headers["Authorization"] == "Basic cGs6c2s="
    assert request.headers["Content-Type"] == "application/x-protobuf"
    decoded = ExportTraceServiceRequest()
    decoded.ParseFromString(request.content)
    exported = decoded.resource_spans[0].scope_spans[0].spans[0]
    assert exported.name == "generation"
    assert exported.span_id == span.context.span_id.to_bytes(8, "big")
    assert slept == []


@pytest.mark.parametrize(
    "failure",
    [httpx.ReadTimeout("stalled"), httpx.ConnectError("refused"), 503, 429, 408, 501, 507, 599],
    ids=["read-timeout", "connect-error", "http-503", "http-429", "http-408", "http-501", "http-507", "http-599"],
)
def test_exporter_retries_a_failed_round_trip_and_then_succeeds(monkeypatch, failure):
    """A stalled or restarting destination used to drop the batch outright; v2 backed off and re-sent every 5xx."""
    slept = []
    monkeypatch.setattr("litellm.integrations.langfuse.langfuse_sdk.sleep", slept.append)
    exporter, seen = _exporter_over([failure, failure, 200], delays=(0.5, 1.5, 2.5))

    assert exporter.export((_finished_span(),)) is SpanExportResult.SUCCESS
    assert len(seen) == 3
    assert len({request.content for request in seen}) == 1
    assert slept == [0.5, 1.5]


def test_exporter_gives_up_after_the_last_delay(monkeypatch):
    slept = []
    monkeypatch.setattr("litellm.integrations.langfuse.langfuse_sdk.sleep", slept.append)
    exporter, seen = _exporter_over([httpx.ConnectError("refused")] * 3, delays=(1.0, 2.0))

    assert exporter.export((_finished_span(),)) is SpanExportResult.FAILURE
    assert len(seen) == 3
    assert slept == [1.0, 2.0]


def _exporter_with_body_cap(max_bytes: int, *, deliveries: list[int]):
    """A destination that answers 413 to any body over ``max_bytes``, the way an ingress with a body limit does."""
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    def transport(request: httpx.Request) -> httpx.Response:
        if len(request.content) > max_bytes:
            return httpx.Response(413, request=request)
        deliveries.append(len(request.content))
        return httpx.Response(200, request=request)

    return LangfuseSpanExporter(
        handler=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(transport))),
        endpoint="https://lf.internal.example/api/public/otel/v1/traces",
        headers=MappingProxyType({}),
        timeout=5.0,
        delays=(),
    )


def test_exporter_splits_a_batch_the_destination_finds_too_large(monkeypatch):
    """One 413 used to drop every span in the batch; the v2 consumer sized its batches by bytes before posting."""
    monkeypatch.setattr("litellm.integrations.langfuse.langfuse_sdk.sleep", lambda _: None)
    spans = tuple(_finished_span() for _ in range(8))
    whole = _encode(spans)
    assert whole is not None
    deliveries: list[int] = []
    exporter = _exporter_with_body_cap(len(whole) // 2, deliveries=deliveries)

    assert exporter.export(spans) is SpanExportResult.SUCCESS
    assert len(deliveries) >= 2
    assert all(size <= len(whole) // 2 for size in deliveries)
    assert (
        sum(deliveries) >= len(whole) - 8 * 8
    )  # each half repeats the resource and scope envelope, spans are not lost


def test_exporter_drops_only_the_single_span_that_alone_exceeds_the_cap(monkeypatch, caplog):
    monkeypatch.setattr("litellm.integrations.langfuse.langfuse_sdk.sleep", lambda _: None)
    provider = TracerProvider()
    huge = provider.get_tracer("t").start_span("generation", attributes={"body": "x" * 4000})
    huge.end()
    small = tuple(_finished_span() for _ in range(3))
    single_small = _encode(small[:1])
    assert single_small is not None
    deliveries: list[int] = []
    exporter = _exporter_with_body_cap(len(single_small) * 3, deliveries=deliveries)

    with caplog.at_level(logging.ERROR, logger="LiteLLM"):
        result = exporter.export((*small, huge))

    assert result is SpanExportResult.FAILURE
    assert len(deliveries) >= 1 and all(size <= len(single_small) * 3 for size in deliveries)
    assert "single" in caplog.text and "too large" in caplog.text


def _decoded_attributes(body: bytes) -> dict[str, str]:
    decoded = ExportTraceServiceRequest()
    decoded.ParseFromString(body)
    return {
        attribute.key: attribute.value.string_value
        for attribute in decoded.resource_spans[0].scope_spans[0].spans[0].attributes
    }


def _generation_span(**attributes: str):
    provider = TracerProvider()
    span = provider.get_tracer("t").start_span("generation", attributes=attributes)
    span.end()
    return span


def test_exporter_truncates_a_single_oversized_span_the_way_v2_did_instead_of_dropping_it(monkeypatch, caplog):
    """v2 replaced the largest of input, output and metadata with a marker and still delivered the observation; a
    vision request over a self-hosted ingress cap used to lose the whole generation, model and usage included."""
    monkeypatch.setattr("litellm.integrations.langfuse.langfuse_sdk.sleep", lambda _: None)
    span = _generation_span(
        **{
            "langfuse.observation.input": "data:image/png;base64," + "A" * 6000,
            "langfuse.trace.input": "data:image/png;base64," + "A" * 200,
            "langfuse.observation.output": "o" * 1000,
            "langfuse.observation.metadata.team": "m" * 100,
            "langfuse.observation.model.name": "gpt-4o",
        }
    )
    bodies: list[bytes] = []

    def transport(request: httpx.Request) -> httpx.Response:
        if len(request.content) > 2000:
            return httpx.Response(413, request=request)
        bodies.append(request.content)
        return httpx.Response(200, request=request)

    exporter = LangfuseSpanExporter(
        handler=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(transport))),
        endpoint="https://lf.internal.example/api/public/otel/v1/traces",
        headers=MappingProxyType({}),
        timeout=5.0,
        delays=(),
    )
    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        result = exporter.export((span,))

    assert result is SpanExportResult.SUCCESS
    delivered = _decoded_attributes(bodies[-1])
    assert delivered["langfuse.observation.input"] == "<truncated due to size exceeding limit>"
    assert delivered["langfuse.trace.input"] == "<truncated due to size exceeding limit>"
    assert delivered["langfuse.observation.output"] == "o" * 1000
    assert delivered["langfuse.observation.metadata.team"] == "m" * 100
    assert delivered["langfuse.observation.model.name"] == "gpt-4o"
    assert "dropping it" not in caplog.text and "truncated" in caplog.text


def test_exporter_truncates_largest_first_and_drops_only_when_nothing_is_left(monkeypatch, caplog):
    """Langfuse stores a bare ``langfuse.observation.metadata`` string as nothing, so the metadata marker travels
    under a flattened key the way every other metadata value does."""
    monkeypatch.setattr("litellm.integrations.langfuse.langfuse_sdk.sleep", lambda _: None)
    span = _generation_span(
        **{
            "langfuse.observation.input": "i" * 3000,
            "langfuse.observation.output": "o" * 2000,
            "langfuse.observation.metadata.a": "m" * 500,
            "langfuse.trace.metadata.b": "m" * 500,
        }
    )
    posted: list[dict[str, str]] = []

    def always_too_large(request: httpx.Request) -> httpx.Response:
        posted.append(_decoded_attributes(request.content))
        return httpx.Response(413, request=request)

    exporter = LangfuseSpanExporter(
        handler=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(always_too_large))),
        endpoint="https://lf.internal.example/api/public/otel/v1/traces",
        headers=MappingProxyType({}),
        timeout=5.0,
        delays=(),
    )
    with caplog.at_level(logging.ERROR, logger="LiteLLM"):
        assert exporter.export((span,)) is SpanExportResult.FAILURE

    marker = "<truncated due to size exceeding limit>"
    assert [sorted(key for key, value in body.items() if value == marker) for body in posted] == [
        [],
        ["langfuse.observation.input"],
        ["langfuse.observation.input", "langfuse.observation.output"],
        [
            "langfuse.observation.input",
            "langfuse.observation.metadata.truncated",
            "langfuse.observation.output",
            "langfuse.trace.metadata.truncated",
        ],
    ]
    assert "langfuse.observation.metadata.a" not in posted[-1] and "langfuse.trace.metadata.b" not in posted[-1]
    assert "dropping it" in caplog.text


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422, 499])
def test_exporter_does_not_retry_a_rejected_batch(monkeypatch, status):
    """Bad credentials or a bad payload will not get better on the next attempt, so retrying only delays the flush."""
    slept = []
    monkeypatch.setattr("litellm.integrations.langfuse.langfuse_sdk.sleep", slept.append)
    exporter, seen = _exporter_over([status, 200])

    assert exporter.export((_finished_span(),)) is SpanExportResult.FAILURE
    assert len(seen) == 1
    assert slept == []


def test_exporter_names_the_server_floor_when_the_otlp_route_is_missing(monkeypatch, caplog):
    """A Langfuse server too old to serve the OTLP route answers 404; a bare status leaves the operator guessing."""
    monkeypatch.setattr("litellm.integrations.langfuse.langfuse_sdk.sleep", lambda _: None)
    exporter, _ = _exporter_over([404])

    with caplog.at_level(logging.ERROR, logger="LiteLLM"):
        assert exporter.export((_finished_span(),)) is SpanExportResult.FAILURE

    assert "HTTP 404" in caplog.text and "3.63.0" in caplog.text


def _finished_span_named(name: object):
    provider = TracerProvider()
    span = provider.get_tracer("t").start_span("placeholder")
    span._name = name  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]  # the SDK only stores str
    span.end()
    return span


def test_exporter_drops_a_span_the_encoder_rejects_and_still_posts_the_rest(monkeypatch, caplog):
    """One span the OTLP encoder cannot serialize used to raise out of ``export`` and lose every span in the batch."""
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    monkeypatch.setattr("litellm.integrations.langfuse.langfuse_sdk.sleep", lambda _: None)
    exporter, seen = _exporter_over([200])

    with caplog.at_level(logging.ERROR, logger="LiteLLM"):
        result = exporter.export((_finished_span(), _finished_span_named(12345), _finished_span()))

    assert result is SpanExportResult.SUCCESS
    (request,) = seen
    decoded = ExportTraceServiceRequest()
    decoded.ParseFromString(request.content)
    assert [span.name for span in decoded.resource_spans[0].scope_spans[0].spans] == ["generation", "generation"]
    assert "dropped 1 span(s)" in caplog.text


def test_exporter_reports_failure_when_no_span_of_the_batch_can_be_encoded(monkeypatch):
    exporter, seen = _exporter_over([200])

    assert exporter.export((_finished_span_named(12345),)) is SpanExportResult.FAILURE
    assert seen == []


def test_built_exporter_uses_the_shared_litellm_handler_and_langfuse_headers(monkeypatch):
    """No private requests session or TLS adapter: the channel is the same handler the rest of litellm uses."""
    from litellm.llms.custom_httpx.http_handler import _get_httpx_client

    monkeypatch.delenv("LANGFUSE_TIMEOUT", raising=False)
    monkeypatch.delenv("LANGFUSE_MAX_RETRIES", raising=False)
    default = _build_span_exporter(public_key="pk", secret_key="sk", base_url="https://lf.internal.example")
    assert default.handler is _get_httpx_client()
    assert default.timeout == 20
    assert len(default.delays) == 3

    monkeypatch.setenv("LANGFUSE_TIMEOUT", "7.5")
    monkeypatch.setenv("LANGFUSE_MAX_RETRIES", "1")
    exporter = _build_span_exporter(public_key="pk", secret_key="sk", base_url="https://lf.internal.example")
    assert exporter.endpoint == "https://lf.internal.example/api/public/otel/v1/traces"
    assert exporter.timeout == 7.5
    assert exporter.delays == (1.0,)
    assert exporter.headers["Authorization"] == "Basic " + b64encode(b"pk:sk").decode()
    assert exporter.headers["x-langfuse-public-key"] == "pk"
    assert exporter.headers["x-langfuse-sdk-version"] == installed_langfuse_version()
    assert exporter.headers["x-langfuse-ingestion-version"] == "4"


def test_large_retry_count_builds_an_exporter_with_capped_backoff(monkeypatch):
    """``LANGFUSE_MAX_RETRIES=1025`` constructed a v2 client; here ``2.0**1024`` would raise ``OverflowError``
    and take the whole callback down at init."""
    monkeypatch.setenv("LANGFUSE_MAX_RETRIES", "1025")
    exporter = _build_span_exporter(public_key="pk", secret_key="sk", base_url="https://lf.internal.example")
    assert 3 < len(exporter.delays) <= 1025
    assert exporter.delays[:4] == (1.0, 2.0, 4.0, 8.0)
    assert max(exporter.delays) == exporter.delays[-1] <= 64.0


def test_absurd_retry_count_is_clamped_instead_of_allocating_one_delay_per_retry(monkeypatch, caplog):
    """A retry count with twelve digits must not turn callback init into a multi-gigabyte tuple allocation."""
    monkeypatch.setenv("LANGFUSE_MAX_RETRIES", "999999999999")
    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        exporter = _build_span_exporter(public_key="pk", secret_key="sk", base_url="https://lf.internal.example")
    assert 3 < len(exporter.delays) <= 1025
    assert exporter.delays[-1] <= 64.0
    assert any("LANGFUSE_MAX_RETRIES=999999999999" in record.getMessage() for record in caplog.records)

    caplog.clear()
    monkeypatch.setenv("LANGFUSE_MAX_RETRIES", "5")
    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        modest = _build_span_exporter(public_key="pk", secret_key="sk", base_url="https://lf.internal.example")
    assert len(modest.delays) == 5
    assert not any("LANGFUSE_MAX_RETRIES" in record.getMessage() for record in caplog.records)


def test_enable_langfuse_debug_logging_makes_deliveries_visible_on_the_langfuse_logger(caplog):
    """``LANGFUSE_DEBUG`` turned on the v2 SDK's own logger; it has to do the same for litellm's export channel."""
    exporter, _ = _exporter_over([200])
    langfuse_logger = logging.getLogger("langfuse")
    level_before = langfuse_logger.level
    try:
        with caplog.at_level(logging.INFO, logger="langfuse"):
            assert exporter.export((_finished_span(),)) is SpanExportResult.SUCCESS
        assert "Exported" not in caplog.text
        enable_langfuse_debug_logging()
        assert langfuse_logger.level == logging.DEBUG
        exporter_after, _ = _exporter_over([200])
        assert exporter_after.export((_finished_span(),)) is SpanExportResult.SUCCESS
        assert "Exported" in caplog.text and "lf.internal.example" in caplog.text
    finally:
        langfuse_logger.setLevel(level_before)


@pytest.mark.parametrize(
    ("base_url", "export_path", "expected"),
    [
        ("https://lf.internal.example/", None, "https://lf.internal.example/api/public/otel/v1/traces"),
        ("https://lf.internal.example", "/otel/traces", "https://lf.internal.example/otel/traces"),
        ("https://lf.internal.example/", "/otel/traces", "https://lf.internal.example/otel/traces"),
        ("https://lf.internal.example", "otel/traces", "https://lf.internal.example/otel/traces"),
        (
            "https://lf.internal.example",
            "//elsewhere.example/otel",
            "https://lf.internal.example/elsewhere.example/otel",
        ),
        (
            "https://lf.internal.example",
            "https://elsewhere.example/otel",
            "https://lf.internal.example/https://elsewhere.example/otel",
        ),
    ],
    ids=[
        "default",
        "leading-slash",
        "both-slashes",
        "no-slash",
        "scheme-relative-stays-on-host",
        "absolute-stays-on-host",
    ],
)
def test_export_endpoint_never_doubles_the_slash_or_leaves_the_configured_host(
    monkeypatch, base_url, export_path, expected
):
    if export_path is None:
        monkeypatch.delenv("LANGFUSE_OTEL_TRACES_EXPORT_PATH", raising=False)
    else:
        monkeypatch.setenv("LANGFUSE_OTEL_TRACES_EXPORT_PATH", export_path)

    exporter = _build_span_exporter(public_key="pk", secret_key="sk", base_url=base_url)

    assert exporter.endpoint == expected


class _RecordingPromptsApi:
    """Answers ``prompts.get`` with a text prompt that names the label it was asked for."""

    def __init__(self) -> None:
        self.prompts = self
        self.requests: list[tuple[str, int | None, str | None]] = []  # mutable-ok: test-side call log

    def get(self, name: str, *, version: int | None, label: str | None, request_options: RequestOptions):
        from langfuse.api import Prompt_Text

        assert request_options.get("max_retries") == 0, "a prompt fetch must not sleep through the client's retries"
        self.requests.append((name, version, label))
        return Prompt_Text(
            name=name,
            version=version or 1,
            config={},
            labels=[label or "production"],
            tags=[],
            prompt=f"label={label!r}",
        )


class _BlockingPromptsApi(_RecordingPromptsApi):
    """Every fetch after the first blocks until the test releases it, and may be told to fail."""

    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()
        self.fail_refresh = False

    def get(self, name: str, *, version: int | None, label: str | None, request_options: RequestOptions):
        is_refresh = bool(self.requests)
        prompt = super().get(name, version=version, label=label, request_options=request_options)
        if is_refresh:
            assert self.release.wait(5), "refresh was never released"
            if self.fail_refresh:
                raise RuntimeError("langfuse is down")
        return prompt


def _wait_until(predicate, timeout: float = 5.0) -> None:
    for _ in range(int(timeout / 0.01)):
        if predicate():
            return
        sleep(0.01)
    raise AssertionError("condition not met in time")


def test_stale_prompt_is_served_at_once_while_the_refresh_runs_elsewhere():
    """``get_prompt`` runs on the proxy's event loop; a stale entry used to refetch inline and block every
    request on the REST round trip. The stale prompt is returned immediately and refreshed off-thread."""
    api = _BlockingPromptsApi()
    client = LangfuseApiClient(api, prompt_cache_ttl_seconds=0)  # pyright: ignore[reportArgumentType]  # duck-typed prompts API

    first = client.get_prompt("greeting")
    started = monotonic()
    stale = client.get_prompt("greeting")

    assert stale is first, "the stale prompt must come back without waiting on the refresh"
    assert monotonic() - started < 1.0, "the stale read waited on the blocked refresh"
    _wait_until(lambda: len(api.requests) == 2)
    api.release.set()
    _wait_until(lambda: client.get_prompt("greeting") is not first)


def test_a_failed_background_refresh_keeps_the_stale_prompt_in_service(caplog):
    api = _BlockingPromptsApi()
    api.fail_refresh = True
    client = LangfuseApiClient(api, prompt_cache_ttl_seconds=0)  # pyright: ignore[reportArgumentType]  # duck-typed prompts API

    first = client.get_prompt("greeting")
    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        started = monotonic()
        assert client.get_prompt("greeting") is first
        assert monotonic() - started < 1.0, "the stale read waited on the blocked refresh"
        api.release.set()
        _wait_until(lambda: "refresh failed" in caplog.text)
    assert client.get_prompt("greeting") is first


def test_only_one_refresh_runs_for_a_stale_prompt_under_concurrent_reads():
    api = _BlockingPromptsApi()
    client = LangfuseApiClient(api, prompt_cache_ttl_seconds=0.3)  # pyright: ignore[reportArgumentType]  # duck-typed prompts API

    first = client.get_prompt("greeting")
    sleep(0.3)
    for _ in range(20):
        assert client.get_prompt("greeting") is first
    _wait_until(lambda: len(api.requests) == 2)
    api.release.set()
    _wait_until(lambda: client.get_prompt("greeting") is not first)
    assert len(api.requests) == 2


def test_prompt_cache_keeps_a_missing_label_apart_from_the_label_named_none():
    """A prompt labelled ``"None"`` and the unlabelled default are different prompts in Langfuse
    and must not answer each other's requests from the cache."""
    api = _RecordingPromptsApi()
    client = LangfuseApiClient(api, prompt_cache_ttl_seconds=60)  # pyright: ignore[reportArgumentType]  # duck-typed prompts API

    unlabelled = client.get_prompt("greeting")
    named_none = client.get_prompt("greeting", label="None")
    cached_unlabelled = client.get_prompt("greeting")

    assert unlabelled.prompt == "label=None"
    assert named_none.prompt == "label='None'"
    assert cached_unlabelled is unlabelled
    assert api.requests == [("greeting", None, None), ("greeting", None, "None")]
