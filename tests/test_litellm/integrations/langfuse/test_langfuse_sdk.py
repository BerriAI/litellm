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
from types import MappingProxyType
from typing import Final

import httpx
import opentelemetry.trace as otel_trace
import pytest
from langfuse import LangfuseOtelSpanAttributes as A
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
    LangfuseSpanExporter,
    LangfuseTracing,
    _build_span_exporter,
    acquire_langfuse_tracing,
    build_langfuse_client,
    build_langfuse_tracing,
    configured_flush_at,
    configured_sample_rate,
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

CALL_START = datetime(2024, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
FIRST_TOKEN = CALL_START + timedelta(seconds=5)
CALL_END = CALL_START + timedelta(seconds=20)
TRACE_A = "a" * 32
PARENT_C = "c" * 16


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

    assert rotated.auth_check() is False
    assert requests[-1].url.host == "127.0.0.1" and requests[-1].url.port == 2
    assert requests[-1].headers["authorization"] == "Basic " + b64encode(b"pk-rest-test:sk-second").decode()


def test_rest_client_without_keys_fails_auth_check_instead_of_raising(monkeypatch):
    """``/health/services?service=langfuse`` with no credentials must report a failed check, not crash."""
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    client = build_langfuse_client(public_key=None, secret_key=None, base_url="http://127.0.0.1:1", httpx_client=None)
    assert client.auth_check() is False


def test_rest_client_reports_the_project_id_and_a_passing_auth_check():
    requests: list[httpx.Request] = []
    client = build_langfuse_client(
        public_key="pk-project-test",
        secret_key="sk",
        base_url="http://127.0.0.1:1",
        httpx_client=_recording_transport(requests, status=200),
    )
    assert client.project_id() == "proj-under-test"
    assert client.auth_check() is True


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
    [httpx.ReadTimeout("stalled"), httpx.ConnectError("refused"), 503, 429],
    ids=["read-timeout", "connect-error", "http-503", "http-429"],
)
def test_exporter_retries_a_failed_round_trip_and_then_succeeds(monkeypatch, failure):
    """A stalled or restarting destination used to drop the batch outright; v2 backed off and re-sent it."""
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


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_exporter_does_not_retry_a_rejected_batch(monkeypatch, status):
    """Bad credentials or a bad payload will not get better on the next attempt, so retrying only delays the flush."""
    slept = []
    monkeypatch.setattr("litellm.integrations.langfuse.langfuse_sdk.sleep", slept.append)
    exporter, seen = _exporter_over([status, 200])

    assert exporter.export((_finished_span(),)) is SpanExportResult.FAILURE
    assert len(seen) == 1
    assert slept == []


def test_built_exporter_uses_the_shared_litellm_handler_and_langfuse_headers(monkeypatch):
    """No private requests session or TLS adapter: the channel is the same handler the rest of litellm uses."""
    from litellm.llms.custom_httpx.http_handler import _get_httpx_client

    monkeypatch.delenv("LANGFUSE_TIMEOUT", raising=False)
    default = _build_span_exporter(public_key="pk", secret_key="sk", base_url="https://lf.internal.example")
    assert default.handler is _get_httpx_client()
    assert default.timeout == 5

    monkeypatch.setenv("LANGFUSE_TIMEOUT", "20")
    exporter = _build_span_exporter(public_key="pk", secret_key="sk", base_url="https://lf.internal.example")
    assert exporter.endpoint == "https://lf.internal.example/api/public/otel/v1/traces"
    assert exporter.timeout == 20
    assert exporter.headers["Authorization"] == "Basic " + b64encode(b"pk:sk").decode()
    assert exporter.headers["x-langfuse-public-key"] == "pk"
    assert exporter.headers["x-langfuse-sdk-version"] == installed_langfuse_version()


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

    def get(self, name: str, *, version: int | None, label: str | None):
        from langfuse.api import Prompt_Text

        self.requests.append((name, version, label))
        return Prompt_Text(
            name=name,
            version=version or 1,
            config={},
            labels=[label or "production"],
            tags=[],
            prompt=f"label={label!r}",
        )


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
