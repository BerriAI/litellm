"""Covers the v4 observation plumbing: historical timestamps and id normalisation.

The timestamp assertions are the regression guard for the migration: v4 has no
public API for an observation start time, so a callback running after the model
call would otherwise record its own duration instead of the call's.
"""

import json
import logging
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Final

import opentelemetry.trace as otel_trace
import pytest
from langfuse import Langfuse
from langfuse._client.resource_manager import LangfuseResourceManager
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from litellm.integrations.langfuse.langfuse import (
    MINIMUM_LANGFUSE_VERSION,
    installed_langfuse_version,
    raise_if_unsupported_langfuse_version,
)
from litellm.integrations.langfuse.langfuse_sdk import (
    AS_ROOT_ATTRIBUTE,
    PUBLIC_ATTRIBUTE,
    RELEASE_ATTRIBUTE,
    _lifecycle_state,
    _litellm_built_providers,
    _teardown_langfuse_client,
    acquire_langfuse_client,
    build_isolated_tracer_provider,
    configured_sample_rate,
    evict_stale_langfuse_resources,
    lease_langfuse_client,
    open_trace_context,
    register_langfuse_client,
    resolve_observation_id,
    resolve_trace_id,
    shutdown_langfuse_client,
    start_child_span,
    start_generation,
    to_unix_nanos,
)

CALL_START = datetime(2024, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
FIRST_TOKEN = CALL_START + timedelta(seconds=5)
CALL_END = CALL_START + timedelta(seconds=20)


@pytest.fixture(name="client")
def _client():
    LangfuseResourceManager._instances.pop("pk-obs-test", None)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    client = Langfuse(
        public_key="pk-obs-test",
        secret_key="sk-obs-test",
        host="http://127.0.0.1:1",
        tracer_provider=provider,
        span_exporter=exporter,
    )
    yield client, exporter
    LangfuseResourceManager._instances.pop("pk-obs-test", None)


def _only_span(exporter, name):
    return next(s for s in exporter.get_finished_spans() if s.name == name)


def test_generation_records_the_model_call_window_not_the_callback(client):
    lf, exporter = client
    context, claim_root = open_trace_context(client=lf, trace_id="a" * 32, parent_observation_id=None)
    start_generation(
        client=lf,
        context=context,
        name="gen",
        start_time=CALL_START,
        claim_trace_root=claim_root,
        attributes={"completion_start_time": FIRST_TOKEN},
    ).end(end_time=to_unix_nanos(CALL_END))
    lf.flush()

    span = _only_span(exporter, "gen")
    assert span.start_time == to_unix_nanos(CALL_START)
    assert span.end_time == to_unix_nanos(CALL_END)
    assert (span.end_time - span.start_time) == 20 * 1_000_000_000
    assert json.loads(span.attributes["langfuse.observation.completion_start_time"]) == FIRST_TOKEN.isoformat().replace(
        "+00:00", "Z"
    )


@pytest.mark.parametrize(
    "supplied",
    [1709294400.5, datetime(2024, 3, 1, 12, 0, 0, 500000, tzinfo=timezone.utc)],
    ids=["unix-seconds-float", "datetime"],
)
def test_timestamps_accept_both_shapes_guardrails_and_callbacks_use(supplied):
    """Guardrail entries carry unix seconds as floats, the callback carries datetimes."""
    assert to_unix_nanos(supplied) == 1709294400500000000


def test_guardrail_span_with_float_timestamps_does_not_break_the_generation(client):
    """A guardrail entry must not take the whole event down with it."""
    lf, exporter = client
    context, claim_root = open_trace_context(client=lf, trace_id="9" * 32, parent_observation_id=None)
    guardrail_start = 1709294400.0
    generation = start_generation(
        client=lf, context=context, name="gen", start_time=CALL_START, claim_trace_root=claim_root, attributes={}
    )
    start_child_span(
        client=lf,
        parent=generation,
        name="guardrail",
        start_time=guardrail_start,
        attributes={},
    ).end(end_time=to_unix_nanos(guardrail_start + 2))
    generation.end(end_time=to_unix_nanos(CALL_END))
    lf.flush()

    guardrail = _only_span(exporter, "guardrail")
    assert (guardrail.end_time - guardrail.start_time) == 2 * 1_000_000_000
    assert _only_span(exporter, "gen") is not None


def test_generation_claims_trace_root_only_without_a_real_parent(client):
    lf, exporter = client
    context, claim_root = open_trace_context(client=lf, trace_id="b" * 32, parent_observation_id=None)
    assert claim_root is True
    start_generation(
        client=lf, context=context, name="root-gen", start_time=CALL_START, claim_trace_root=claim_root, attributes={}
    ).end()

    parented_context, parented_claim = open_trace_context(client=lf, trace_id="b" * 32, parent_observation_id="c" * 16)
    assert parented_claim is False
    start_generation(
        client=lf,
        context=parented_context,
        name="child-gen",
        start_time=CALL_START,
        claim_trace_root=parented_claim,
        attributes={},
    ).end()
    lf.flush()

    assert _only_span(exporter, "root-gen").attributes.get(AS_ROOT_ATTRIBUTE) is True
    assert _only_span(exporter, "child-gen").attributes.get(AS_ROOT_ATTRIBUTE) is None


def test_child_span_keeps_its_own_window_and_only_the_generation_claims_root(client):
    """The server takes trace name and I/O from every root observation, latest start wins.

    A post-call guardrail starts after the model call, so if it also claimed root the
    trace would show the guardrail's I/O instead of the model's.
    """
    lf, exporter = client
    context, claim_root = open_trace_context(client=lf, trace_id="d" * 32, parent_observation_id=None)
    generation = start_generation(
        client=lf, context=context, name="gen", start_time=CALL_START, claim_trace_root=claim_root, attributes={}
    )
    guardrail_start = CALL_END + timedelta(seconds=1)
    start_child_span(
        client=lf,
        parent=generation,
        name="guardrail",
        start_time=guardrail_start,
        attributes={},
    ).end(end_time=to_unix_nanos(guardrail_start + timedelta(seconds=2)))
    generation.end(end_time=to_unix_nanos(CALL_END))
    lf.flush()

    guardrail = _only_span(exporter, "guardrail")
    exported_generation = _only_span(exporter, "gen")
    assert (guardrail.end_time - guardrail.start_time) == 2 * 1_000_000_000
    assert guardrail.context.trace_id == exported_generation.context.trace_id
    assert guardrail.parent.span_id == exported_generation.context.span_id
    assert AS_ROOT_ATTRIBUTE not in guardrail.attributes
    assert exported_generation.attributes.get(AS_ROOT_ATTRIBUTE) is True


def test_release_is_carried_on_the_root_observation(client):
    lf, exporter = client
    context, claim_root = open_trace_context(client=lf, trace_id="e" * 32, parent_observation_id=None)
    start_generation(
        client=lf,
        context=context,
        name="gen",
        start_time=CALL_START,
        claim_trace_root=claim_root,
        release="v1.2.3",
        attributes={},
    ).end()
    lf.flush()
    assert _only_span(exporter, "gen").attributes[RELEASE_ATTRIBUTE] == "v1.2.3"


@pytest.mark.parametrize("public", [True, False], ids=["public", "private"])
def test_trace_public_flag_lands_on_the_root_observation(client, public):
    """v2 took ``public`` on ``trace()``; v4 reads ``langfuse.trace.public`` off the root observation."""
    lf, exporter = client
    context, claim_root = open_trace_context(client=lf, trace_id="a" * 32, parent_observation_id=None)
    start_generation(
        client=lf,
        context=context,
        name="gen",
        start_time=CALL_START,
        claim_trace_root=claim_root,
        public=public,
        attributes={},
    ).end()
    lf.flush()
    assert _only_span(exporter, "gen").attributes[PUBLIC_ATTRIBUTE] is public


def test_trace_public_flag_is_absent_when_not_requested(client):
    lf, exporter = client
    context, claim_root = open_trace_context(client=lf, trace_id="b" * 32, parent_observation_id=None)
    start_generation(
        client=lf, context=context, name="gen", start_time=CALL_START, claim_trace_root=claim_root, attributes={}
    ).end()
    lf.flush()
    assert PUBLIC_ATTRIBUTE not in _only_span(exporter, "gen").attributes


@pytest.mark.parametrize("public", [True, False, None], ids=["public", "private", "unset"])
def test_child_span_repeats_the_generation_public_flag(client, public):
    """The server folds ``public`` across observations and reads a missing value as False.

    A guardrail span without the flag turned a ``trace_public: true`` request private on Langfuse Cloud.
    """
    lf, exporter = client
    context, claim_root = open_trace_context(client=lf, trace_id="c" * 32, parent_observation_id=None)
    generation = start_generation(
        client=lf,
        context=context,
        name="gen",
        start_time=CALL_START,
        claim_trace_root=claim_root,
        public=public,
        attributes={},
    )
    start_child_span(client=lf, parent=generation, name="guardrail", start_time=CALL_END, attributes={}).end()
    generation.end(end_time=to_unix_nanos(CALL_END))
    lf.flush()

    assert _only_span(exporter, "guardrail").attributes.get(PUBLIC_ATTRIBUTE) is public


def test_request_release_beats_the_client_wide_release(monkeypatch):
    """A client configured with its own release must not overwrite trace_release."""
    monkeypatch.setenv("LANGFUSE_RELEASE", "client-wide-release")
    LangfuseResourceManager._instances.pop("pk-release-test", None)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    lf = Langfuse(
        public_key="pk-release-test",
        secret_key="sk-release-test",
        host="http://127.0.0.1:1",
        release="client-wide-release",
        tracer_provider=provider,
        span_exporter=exporter,
    )
    context, claim_root = open_trace_context(client=lf, trace_id="f" * 32, parent_observation_id=None)
    start_generation(
        client=lf,
        context=context,
        name="gen",
        start_time=CALL_START,
        claim_trace_root=claim_root,
        release="per-request-release",
        attributes={},
    ).end()
    lf.flush()
    LangfuseResourceManager._instances.pop("pk-release-test", None)

    assert _only_span(exporter, "gen").attributes[RELEASE_ATTRIBUTE] == "per-request-release"


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


PUBLIC_KEY = "pk-lifecycle-test"


@pytest.fixture(autouse=True)
def _clean_registry():
    LangfuseResourceManager._instances.pop(PUBLIC_KEY, None)
    yield
    LangfuseResourceManager._instances.pop(PUBLIC_KEY, None)


def _lifecycle_client(secret_key="sk-original", host="http://127.0.0.1:1"):
    return Langfuse(
        public_key=PUBLIC_KEY,
        secret_key=secret_key,
        host=host,
        tracer_provider=build_isolated_tracer_provider(environment=None, release=None),
    )


@pytest.mark.parametrize("unsupported", ["2.59.7", "3.15.0", "5.0.0"], ids=["v2", "v3", "v5"])
def test_unsupported_sdk_fails_loudly_rather_than_dropping_every_event(unsupported):
    with pytest.raises(ImportError) as raised:
        raise_if_unsupported_langfuse_version(unsupported)
    assert unsupported in str(raised.value)
    assert MINIMUM_LANGFUSE_VERSION in str(raised.value)


def test_supported_sdk_is_accepted():
    assert raise_if_unsupported_langfuse_version(installed_langfuse_version()) is None


def test_isolated_provider_carries_environment_and_release():
    provider = build_isolated_tracer_provider(environment="staging", release="v9")
    attributes = provider.resource.attributes
    assert attributes["langfuse.environment"] == "staging"
    assert attributes["langfuse.release"] == "v9"


def _generations_exported_at(sample_rate: float, trace_ids: tuple[str, ...]) -> frozenset[str]:
    exporter: Final = InMemorySpanExporter()
    provider: Final = build_isolated_tracer_provider(environment=None, release=None, sample_rate=sample_rate)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    pk: Final = f"pk-sample-{sample_rate}"
    LangfuseResourceManager._instances.pop(pk, None)
    client: Final = Langfuse(
        public_key=pk,
        secret_key="sk-sample",
        host="http://127.0.0.1:1",
        tracer_provider=provider,
        span_exporter=exporter,
    )
    try:
        for trace_id in trace_ids:
            context, claim_root = open_trace_context(client=client, trace_id=trace_id, parent_observation_id=None)
            start_generation(
                client=client,
                context=context,
                name="sampled",
                start_time=CALL_START,
                claim_trace_root=claim_root,
                attributes={},
            ).end()
    finally:
        LangfuseResourceManager._instances.pop(pk, None)
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

    pk: Final = f"pk-unusable-rate-{raw}"
    LangfuseResourceManager._instances.pop(pk, None)
    try:
        client: Final = acquire_langfuse_client(
            parameters={"public_key": pk, "secret_key": "sk", "base_url": "http://127.0.0.1:1"},
            environment=None,
            release=None,
            mock_mode=True,
        )
        assert client._resources.sample_rate == 1.0
    finally:
        LangfuseResourceManager._instances.pop(pk, None)


def test_configured_sample_rate_reads_the_env_var(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LANGFUSE_SAMPLE_RATE", raising=False)
    assert configured_sample_rate() == 1.0
    monkeypatch.setenv("LANGFUSE_SAMPLE_RATE", "0.25")
    assert configured_sample_rate() == 0.25


def _isolated_client_with_exporter():
    exporter = InMemorySpanExporter()
    provider = build_isolated_tracer_provider(environment=None, release=None)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    client = Langfuse(
        public_key=PUBLIC_KEY,
        secret_key="sk-observation-id",
        host="http://127.0.0.1:1",
        tracer_provider=provider,
        span_exporter=exporter,
    )
    return client, exporter


def _start_generation(client, name, observation_id):
    context, claim_root = open_trace_context(client=client, trace_id="b" * 32, parent_observation_id=None)
    return start_generation(
        client=client,
        context=context,
        name=name,
        start_time=CALL_START,
        claim_trace_root=claim_root,
        observation_id=observation_id,
        attributes={},
    )


def test_requested_observation_id_becomes_the_exported_span_id():
    """v2 ``generation(id=...)``: the caller's id is what the export carries and what ``.id`` returns."""
    lf, exporter = _isolated_client_with_exporter()
    requested = resolve_observation_id("chatcmpl-123")

    generation = _start_generation(lf, "requested", requested)
    generation.end(end_time=to_unix_nanos(CALL_END))
    lf.flush()

    assert generation.id == requested
    assert format(_only_span(exporter, "requested").context.span_id, "016x") == requested


def test_requested_observation_id_does_not_leak_into_the_next_span():
    lf, exporter = _isolated_client_with_exporter()
    requested = resolve_observation_id("chatcmpl-123")

    _start_generation(lf, "first", requested).end(end_time=to_unix_nanos(CALL_END))
    second = _start_generation(lf, "second", None)
    second.end(end_time=to_unix_nanos(CALL_END))
    third = _start_generation(lf, "third", None)
    third.end(end_time=to_unix_nanos(CALL_END))
    lf.flush()

    assert second.id != requested
    assert third.id != second.id
    assert len({span.context.span_id for span in exporter.get_finished_spans()}) == 3


def test_requested_observation_id_is_ignored_on_a_provider_litellm_did_not_build(client):
    lf, _ = client
    requested = resolve_observation_id("chatcmpl-123")

    generation = _start_generation(lf, "adopted", requested)
    generation.end(end_time=to_unix_nanos(CALL_END))

    assert generation.id != requested


def test_environment_override_lands_per_span_despite_shared_resources():
    """The SDK registry is keyed on public key alone, so a second client for the
    same key adopts the first client's provider; the observation wrapper stamps
    each span with its own client's environment, which the server prefers over
    the resource-level value."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    first = Langfuse(
        public_key=PUBLIC_KEY,
        secret_key="sk-original",
        host="http://127.0.0.1:1",
        environment="prod",
        tracer_provider=provider,
        span_exporter=exporter,
    )
    second = Langfuse(
        public_key=PUBLIC_KEY,
        secret_key="sk-original",
        host="http://127.0.0.1:1",
        environment="staging",
    )
    assert second._resources is first._resources

    for client, environment in ((first, "prod"), (second, "staging")):
        context, claim_trace_root = open_trace_context(client=client, trace_id="a" * 32, parent_observation_id=None)
        start_generation(
            client=client,
            context=context,
            name=f"generation-{environment}",
            start_time=CALL_START,
            claim_trace_root=claim_trace_root,
            attributes={},
        ).end()
    first.flush()

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert spans["generation-prod"].attributes["langfuse.environment"] == "prod"
    assert spans["generation-staging"].attributes["langfuse.environment"] == "staging"


def test_client_does_not_take_over_the_process_tracer_provider():
    # the global provider can only be set once per process, so assert it is left
    # alone rather than assuming this test is the one that installed it
    provider_before = otel_trace.get_tracer_provider()

    client = _lifecycle_client()

    assert otel_trace.get_tracer_provider() is provider_before
    assert client._resources.tracer_provider is not provider_before
    active = getattr(provider_before, "_active_span_processor", None)
    if active is not None:
        assert not any("Langfuse" in type(processor).__name__ for processor in active._span_processors)


def test_rotated_credentials_replace_the_cached_client():
    original = _lifecycle_client(secret_key="sk-original", host="http://127.0.0.1:1")
    original_resources = original._resources

    evict_stale_langfuse_resources(public_key=PUBLIC_KEY, secret_key="sk-rotated", base_url="http://127.0.0.1:2")
    rotated = _lifecycle_client(secret_key="sk-rotated", host="http://127.0.0.1:2")

    assert rotated._resources is not original_resources
    assert rotated._resources.secret_key == "sk-rotated"
    assert rotated._resources.base_url == "http://127.0.0.1:2"


def test_unchanged_credentials_keep_the_cached_client():
    original = _lifecycle_client()
    evict_stale_langfuse_resources(public_key=PUBLIC_KEY, secret_key="sk-original", base_url="http://127.0.0.1:1")
    assert LangfuseResourceManager._instances.get(PUBLIC_KEY) is original._resources


def test_eviction_flushes_queued_observations_before_tearing_down():
    """An observation already ended when the cache evicts must still reach langfuse."""
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from litellm.integrations.langfuse.langfuse_sdk import (
        open_trace_context,
        start_generation,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    # a long delay keeps the span queued, so only the shutdown can flush it
    provider.add_span_processor(BatchSpanProcessor(exporter, schedule_delay_millis=600000))
    client = Langfuse(
        public_key=PUBLIC_KEY,
        secret_key="sk-original",
        host="http://127.0.0.1:1",
        tracer_provider=provider,
        span_exporter=exporter,
    )
    context, claim_root = open_trace_context(client=client, trace_id="a" * 32, parent_observation_id=None)
    start_generation(
        client=client, context=context, name="in-flight", start_time=None, claim_trace_root=claim_root, attributes={}
    ).end()
    assert exporter.get_finished_spans() == ()

    shutdown_langfuse_client(client)

    assert any(span.name == "in-flight" for span in exporter.get_finished_spans())


def test_shutdown_deregisters_so_a_later_client_is_not_a_corpse():
    client = _lifecycle_client()
    resources = client._resources

    shutdown_langfuse_client(client)

    assert LangfuseResourceManager._instances.get(PUBLIC_KEY) is not resources


def _shared_resources_pair():
    """The SDK hands a second client on the same public key the first client's resources."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    _litellm_built_providers.add(provider)
    first = Langfuse(
        public_key=PUBLIC_KEY,
        secret_key="sk-original",
        host="http://127.0.0.1:1",
        tracer_provider=provider,
        span_exporter=exporter,
    )
    second = Langfuse(
        public_key=PUBLIC_KEY,
        secret_key="sk-original",
        host="http://127.0.0.1:1",
        tracer_provider=build_isolated_tracer_provider(environment="per-key-override", release=None),
    )
    assert second._resources is first._resources
    register_langfuse_client(first)
    register_langfuse_client(second)
    return first, second, exporter


def _exports(client, exporter, name):
    client.start_observation(name=name).end()
    client.flush()
    return any(span.name == name for span in exporter.get_finished_spans())


def _never_renew():
    raise AssertionError("the lease renewed a client eviction never reached")


def test_garbage_collected_throwaway_clients_do_not_hold_shared_resources_open():
    """A health probe or alerting lookup builds a client it never shuts down.

    Once such a client is garbage collected it must stop counting, or the last
    managed client's shutdown would skip the teardown forever.
    """
    import gc

    first, second, exporter = _shared_resources_pair()
    throwaway = Langfuse(public_key=PUBLIC_KEY, secret_key="sk-original", host="http://127.0.0.1:1")
    register_langfuse_client(throwaway)
    shutdown_langfuse_client(second)
    del throwaway
    gc.collect()

    shutdown_langfuse_client(first)

    assert not _exports(first, exporter, "after-managed-teardown")
    assert LangfuseResourceManager._instances.get(PUBLIC_KEY) is not first._resources


def test_evicting_a_client_that_shares_resources_keeps_the_other_exporting():
    """A per-key ``langfuse_environment`` override is a second client on the global key.

    When the cache evicts it, the global logger must keep exporting.
    """
    first, second, exporter = _shared_resources_pair()

    shutdown_langfuse_client(second)

    assert _exports(first, exporter, "after-sibling-eviction")
    assert LangfuseResourceManager._instances.get(PUBLIC_KEY) is first._resources


def test_eviction_defers_teardown_until_active_callback_finishes():
    """A cached client must keep exporting while its callback lease is active."""
    exporter = InMemorySpanExporter()
    provider = build_isolated_tracer_provider(environment=None, release=None)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    client = Langfuse(
        public_key=PUBLIC_KEY,
        secret_key="sk-original",
        host="http://127.0.0.1:1",
        tracer_provider=provider,
        span_exporter=exporter,
    )
    register_langfuse_client(client)

    def evict() -> None:
        shutdown_langfuse_client(client)

    with lease_langfuse_client(client, _never_renew):
        evictor = threading.Thread(target=evict)
        evictor.start()
        evictor.join(timeout=5)
        assert not evictor.is_alive()
        assert not exporter._stopped

        context, claim_root = open_trace_context(client=client, trace_id="a" * 32, parent_observation_id=None)
        start_generation(
            client=client,
            context=context,
            name="active-callback",
            start_time=None,
            claim_trace_root=claim_root,
            attributes={},
        ).end()
        client.flush()
        assert any(span.name == "active-callback" for span in exporter.get_finished_spans())

    evictor.join(timeout=5)
    assert not evictor.is_alive()
    assert exporter._stopped
    assert LangfuseResourceManager._instances.get(PUBLIC_KEY) is not client._resources


def test_teardown_failure_does_not_strand_queued_clients(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    first = Langfuse(
        public_key=PUBLIC_KEY,
        secret_key="sk-original",
        host="http://127.0.0.1:1",
        tracer_provider=provider,
        span_exporter=exporter,
    )
    clients = (
        first,
        Langfuse(public_key=PUBLIC_KEY, secret_key="sk-original", host="http://127.0.0.1:1"),
        Langfuse(public_key=PUBLIC_KEY, secret_key="sk-original", host="http://127.0.0.1:1"),
    )
    assert len({client._resources for client in clients}) == 1
    for client in clients:
        register_langfuse_client(client)
    state = _lifecycle_state(clients[0])
    original_teardown = _teardown_langfuse_client
    calls = []

    def teardown(client):
        calls.append(client)
        original_teardown(client)
        if len(calls) == 1:
            raise RuntimeError("teardown failed")

    monkeypatch.setattr("litellm.integrations.langfuse.langfuse_sdk._teardown_langfuse_client", teardown)
    with lease_langfuse_client(clients[0], _never_renew):
        for client in clients:
            shutdown_langfuse_client(client)

    assert len(calls) == 3
    assert not state.pending_clients
    assert not state.teardown_in_progress


def test_interrupt_during_deferred_teardown_propagates_and_requeues_the_client(monkeypatch):
    """A Ctrl-C landing in the lease exit's teardown must reach the caller, not be swallowed."""
    client = Langfuse(public_key=PUBLIC_KEY, secret_key="sk-original", host="http://127.0.0.1:1")
    register_langfuse_client(client)
    state = _lifecycle_state(client)
    original_teardown = _teardown_langfuse_client
    calls = []

    def teardown(target):
        calls.append(target)
        if len(calls) == 1:
            raise KeyboardInterrupt
        original_teardown(target)

    monkeypatch.setattr("litellm.integrations.langfuse.langfuse_sdk._teardown_langfuse_client", teardown)
    with pytest.raises(KeyboardInterrupt):
        with lease_langfuse_client(client, _never_renew):
            shutdown_langfuse_client(client)

    assert state.pending_clients == {client}
    assert not state.teardown_in_progress

    replacement = Langfuse(public_key=PUBLIC_KEY, secret_key="sk-original", host="http://127.0.0.1:1")
    with lease_langfuse_client(client, lambda: replacement) as leased:
        assert leased is replacement

    assert calls == [client, client]
    assert not state.pending_clients


def test_lease_on_a_client_evicted_after_the_cache_lookup_exports_through_a_renewed_one():
    """Eviction can land between the cache handing out the logger and the callback taking its lease.

    That callback must not export into a shut-down provider; the lease has to hand it a live client.
    """
    exporter = InMemorySpanExporter()
    provider = build_isolated_tracer_provider(environment=None, release=None)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    evicted = Langfuse(
        public_key=PUBLIC_KEY,
        secret_key="sk-original",
        host="http://127.0.0.1:1",
        tracer_provider=provider,
        span_exporter=exporter,
    )
    register_langfuse_client(evicted)
    shutdown_langfuse_client(evicted)
    assert exporter._stopped

    renewed_exporter = InMemorySpanExporter()
    renewed_provider = build_isolated_tracer_provider(environment=None, release=None)
    renewed_provider.add_span_processor(SimpleSpanProcessor(renewed_exporter))

    def renew():
        renewed = Langfuse(
            public_key=PUBLIC_KEY,
            secret_key="sk-original",
            host="http://127.0.0.1:1",
            tracer_provider=renewed_provider,
            span_exporter=renewed_exporter,
        )
        register_langfuse_client(renewed)
        return renewed

    with lease_langfuse_client(evicted, renew) as leased:
        assert leased is not evicted
        assert _exports(leased, renewed_exporter, "after-lookup-eviction")
        shutdown_langfuse_client(leased)
        assert not renewed_exporter._stopped

    assert renewed_exporter._stopped


def test_queued_eviction_waits_for_the_last_of_two_overlapping_leases():
    exporter = InMemorySpanExporter()
    provider = build_isolated_tracer_provider(environment=None, release=None)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    client = Langfuse(
        public_key=PUBLIC_KEY,
        secret_key="sk-original",
        host="http://127.0.0.1:1",
        tracer_provider=provider,
        span_exporter=exporter,
    )
    register_langfuse_client(client)

    with lease_langfuse_client(client, _never_renew):
        with lease_langfuse_client(client, _never_renew):
            shutdown_langfuse_client(client)
        assert not exporter._stopped

    assert exporter._stopped


def test_a_client_adopted_during_deferred_teardown_keeps_exporting():
    """The registry hands the same bundle back out while its teardown is queued behind a lease;
    the holder count must degrade that teardown to a flush."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    _litellm_built_providers.add(provider)
    evicted = Langfuse(
        public_key=PUBLIC_KEY,
        secret_key="sk-original",
        host="http://127.0.0.1:1",
        tracer_provider=provider,
        span_exporter=exporter,
    )
    register_langfuse_client(evicted)

    with lease_langfuse_client(evicted, _never_renew):
        shutdown_langfuse_client(evicted)
        adopter = Langfuse(public_key=PUBLIC_KEY, secret_key="sk-original", host="http://127.0.0.1:1")
        assert adopter._resources is evicted._resources
        register_langfuse_client(adopter)

    assert _exports(adopter, exporter, "after-deferred-teardown")
    assert LangfuseResourceManager._instances.get(PUBLIC_KEY) is adopter._resources


def test_leases_on_one_client_do_not_serialise_callbacks():
    """Every langfuse callback in the process shares one client, so leases must overlap."""
    client = _lifecycle_client()
    both_inside = threading.Barrier(2, timeout=5)

    def hold_lease() -> None:
        with lease_langfuse_client(client, _never_renew):
            both_inside.wait()

    holders = tuple(threading.Thread(target=hold_lease) for _ in range(2))
    for holder in holders:
        holder.start()
    for holder in holders:
        holder.join(timeout=5)

    assert not any(holder.is_alive() for holder in holders)
    assert not both_inside.broken


def test_last_client_on_shared_resources_tears_them_down():
    first, second, exporter = _shared_resources_pair()
    shutdown_langfuse_client(second)

    shutdown_langfuse_client(first)

    assert not _exports(first, exporter, "after-last-eviction")
    assert LangfuseResourceManager._instances.get(PUBLIC_KEY) is not first._resources


def test_shutdown_of_a_stale_client_does_not_deregister_the_live_one():
    stale = _lifecycle_client(secret_key="sk-original", host="http://127.0.0.1:1")
    stale_resources = stale._resources
    evict_stale_langfuse_resources(public_key=PUBLIC_KEY, secret_key="sk-rotated", base_url="http://127.0.0.1:2")
    live = _lifecycle_client(secret_key="sk-rotated", host="http://127.0.0.1:2")

    shutdown_langfuse_client(stale)

    assert stale_resources is not live._resources
    assert LangfuseResourceManager._instances.get(PUBLIC_KEY) is live._resources


def _rotation_provider():
    """A litellm-built provider on the lifecycle public key, exporting in memory."""
    exporter = InMemorySpanExporter()
    provider = build_isolated_tracer_provider(environment=None, release=None)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    client = Langfuse(
        public_key=PUBLIC_KEY,
        secret_key="sk-original",
        host="http://127.0.0.1:1",
        tracer_provider=provider,
        span_exporter=exporter,
    )
    register_langfuse_client(client)
    assert _exports(client, exporter, "before-rotation")
    return client, exporter


def test_rotation_retires_the_provider_no_client_is_left_on():
    """Prompt management builds throwaway clients from request credentials.

    Alternating the secret for one public key evicts a bundle nobody holds any
    more, and its export thread has to go with it or every rotation leaks one.
    """
    import gc

    client, exporter = _rotation_provider()
    del client
    gc.collect()

    evict_stale_langfuse_resources(public_key=PUBLIC_KEY, secret_key="sk-rotated", base_url="http://127.0.0.1:2")

    assert exporter._stopped


def test_rotation_keeps_a_still_live_client_exporting():
    """The evicted bundle is only retired when nothing is on it; a live logger must survive."""
    client, exporter = _rotation_provider()

    evict_stale_langfuse_resources(public_key=PUBLIC_KEY, secret_key="sk-rotated", base_url="http://127.0.0.1:2")

    assert _exports(client, exporter, "after-rotation")


def test_a_client_dropped_without_shutdown_gets_its_provider_retired():
    """The prompt-management LRU drops rotated-out clients without shutting them down.

    Nothing ever calls ``shutdown_langfuse_client`` on such a client, so the next
    lifecycle call has to reap the bundle instead of leaking its export thread.
    """
    import gc

    client, exporter = _rotation_provider()
    evict_stale_langfuse_resources(public_key=PUBLIC_KEY, secret_key="sk-rotated", base_url="http://127.0.0.1:2")
    assert _exports(client, exporter, "still-held")

    del client
    gc.collect()
    evict_stale_langfuse_resources(public_key="pk-unrelated", secret_key="sk", base_url="http://127.0.0.1:3")

    assert exporter._stopped


def test_the_registrys_current_bundle_is_not_reaped_when_its_clients_die():
    """The registry hands its bundle to the next client on the same key, so a bundle
    that is still current keeps its provider even after every client is collected."""
    import gc

    client, exporter = _rotation_provider()
    del client
    gc.collect()

    evict_stale_langfuse_resources(public_key="pk-unrelated", secret_key="sk", base_url="http://127.0.0.1:3")

    successor = Langfuse(public_key=PUBLIC_KEY, secret_key="sk-original", host="http://127.0.0.1:1")
    assert _exports(successor, exporter, "after-collection")


def test_a_sweep_overlapping_registration_and_rotation_keeps_the_live_provider():
    """A sweep can snapshot providers before a client registers, then wait on the registry
    lock while that client registers and a rotation evicts its fresh bundle. Holders are
    re-read after the registry snapshot, so the stale first look must not win."""
    import threading

    from litellm.integrations.langfuse.langfuse_sdk import _retire_orphaned_providers

    exporter = InMemorySpanExporter()
    provider = build_isolated_tracer_provider(environment=None, release=None)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    client = Langfuse(
        public_key=PUBLIC_KEY,
        secret_key="sk-original",
        host="http://127.0.0.1:1",
        tracer_provider=provider,
        span_exporter=exporter,
    )
    registry_lock = LangfuseResourceManager._lock
    registry_lock.acquire()
    try:
        sweeper = threading.Thread(target=_retire_orphaned_providers)
        sweeper.start()
        sweeper.join(timeout=0.5)  # parks on the registry lock once its provider snapshot is taken
        register_langfuse_client(client)
        LangfuseResourceManager._instances.pop(PUBLIC_KEY, None)  # the rotation that evicts the fresh bundle
    finally:
        registry_lock.release()
    sweeper.join(timeout=5)
    assert not sweeper.is_alive()

    assert _exports(client, exporter, "after-racing-sweep")


def test_ssl_exporter_carries_litellm_tls_material(monkeypatch, tmp_path):
    """v4 exports over its own OTLP channel, so litellm's CA bundle must be rebuilt onto it."""
    import litellm
    from litellm.integrations.langfuse.langfuse_sdk import _build_span_exporter

    for name in ("SSL_CERTIFICATE", "SSL_VERIFY", "SSL_CERT_FILE", "LANGFUSE_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(litellm, "ssl_verify", True)
    monkeypatch.setattr(litellm, "ssl_certificate", None)
    default = _build_span_exporter(public_key="pk", secret_key="sk", base_url="https://lf.internal.example").exporter
    assert default._certificate_file is True
    assert default._timeout == 5

    ca_path = tmp_path / "private-ca.pem"
    ca_path.write_text("dummy")
    monkeypatch.setattr(litellm, "ssl_verify", str(ca_path))
    monkeypatch.setenv("LANGFUSE_TIMEOUT", "20")
    exporter = _build_span_exporter(public_key="pk", secret_key="sk", base_url="https://lf.internal.example").exporter
    assert exporter._endpoint == "https://lf.internal.example/api/public/otel/v1/traces"
    assert exporter._certificate_file == str(ca_path)
    assert exporter._timeout == 20
    assert exporter._headers["x-langfuse-public-key"] == "pk"
    assert exporter._headers["x-langfuse-sdk-version"] == installed_langfuse_version()


@pytest.mark.parametrize(
    ("base_url", "export_path", "expected"),
    [
        ("https://lf.internal.example/", None, "https://lf.internal.example/api/public/otel/v1/traces"),
        ("https://lf.internal.example", "/otel/traces", "https://lf.internal.example/otel/traces"),
        ("https://lf.internal.example/", "/otel/traces", "https://lf.internal.example/otel/traces"),
        ("https://lf.internal.example", "otel/traces", "https://lf.internal.example/otel/traces"),
    ],
)
def test_export_endpoint_never_doubles_the_slash(monkeypatch, base_url, export_path, expected):
    """A trailing host slash or a leading export path slash must not produce `//` in the OTLP route."""
    from litellm.integrations.langfuse.langfuse_sdk import _build_span_exporter

    if export_path is None:
        monkeypatch.delenv("LANGFUSE_OTEL_TRACES_EXPORT_PATH", raising=False)
    else:
        monkeypatch.setenv("LANGFUSE_OTEL_TRACES_EXPORT_PATH", export_path)

    exporter = _build_span_exporter(public_key="pk", secret_key="sk", base_url=base_url).exporter

    assert exporter._endpoint == expected


def test_retrying_exporter_retries_a_raised_export_and_then_succeeds(monkeypatch):
    """A read timeout used to drop the batch outright; v2 backed off and re-sent it."""
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
    from requests import ReadTimeout

    from litellm.integrations.langfuse.langfuse_sdk import RetryingSpanExporter

    attempts = []
    slept = []

    class Flaky(SpanExporter):
        def export(self, spans):
            attempts.append(spans)
            if len(attempts) < 3:
                raise ReadTimeout("destination stalled")
            return SpanExportResult.SUCCESS

    monkeypatch.setattr("litellm.integrations.langfuse.langfuse_sdk.sleep", slept.append)
    result = RetryingSpanExporter(Flaky(), delays=(0.5, 1.5, 2.5)).export(("span",))

    assert result is SpanExportResult.SUCCESS
    assert attempts == [("span",)] * 3
    assert slept == [0.5, 1.5]


def test_retrying_exporter_gives_up_after_the_last_delay(monkeypatch):
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
    from requests import ConnectionError as RequestsConnectionError

    from litellm.integrations.langfuse.langfuse_sdk import RetryingSpanExporter

    attempts = []
    slept = []

    class Down(SpanExporter):
        def export(self, spans):
            attempts.append(spans)
            raise RequestsConnectionError("refused")

    monkeypatch.setattr("litellm.integrations.langfuse.langfuse_sdk.sleep", slept.append)
    result = RetryingSpanExporter(Down(), delays=(1.0, 2.0)).export(("span",))

    assert result is SpanExportResult.FAILURE
    assert len(attempts) == 3
    assert slept == [1.0, 2.0]


@pytest.mark.parametrize("switch", ["attribute", "env"])
def test_ssl_exporter_disables_verification_when_litellm_does(monkeypatch, switch):
    """v2 exported through the httpx client, so ``ssl_verify=False`` reached ingestion; v4's exporter must match."""
    import litellm
    from litellm.integrations.langfuse.langfuse_sdk import _build_span_exporter

    for name in ("SSL_CERTIFICATE", "SSL_VERIFY", "SSL_CERT_FILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(litellm, "ssl_certificate", None)
    if switch == "attribute":
        monkeypatch.setattr(litellm, "ssl_verify", False)
    else:
        monkeypatch.setattr(litellm, "ssl_verify", True)
        monkeypatch.setenv("SSL_VERIFY", "False")

    exporter = _build_span_exporter(public_key="pk", secret_key="sk", base_url="https://lf.internal.example").exporter
    assert exporter._certificate_file is False
    assert exporter._client_cert is None

    posted = []

    def post(self, url, **kwargs):
        posted.append((url, kwargs["verify"]))
        raise ConnectionError("stop before the network")

    monkeypatch.setattr("requests.Session.post", post)
    with pytest.raises(ConnectionError):
        exporter._export(b"payload")
    assert posted[0] == ("https://lf.internal.example/api/public/otel/v1/traces", False)


@pytest.mark.parametrize("with_client_certificate", [False, True])
def test_ssl_exporter_falls_back_to_default_ca_when_the_bundle_path_is_missing(
    monkeypatch, tmp_path, with_client_certificate
):
    """The httpx client ignores a CA path that does not exist; handing it to requests would fail every export."""
    import litellm
    from litellm.integrations.langfuse.langfuse_sdk import _build_span_exporter

    for name in ("SSL_CERTIFICATE", "SSL_VERIFY", "SSL_CERT_FILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(litellm, "ssl_verify", True)
    monkeypatch.setenv("SSL_VERIFY", str(tmp_path / "missing-ca.pem"))
    client_cert = tmp_path / "client.pem"
    client_cert.write_text("dummy")
    monkeypatch.setattr(litellm, "ssl_certificate", str(client_cert) if with_client_certificate else None)

    exporter = _build_span_exporter(public_key="pk", secret_key="sk", base_url="https://lf.internal.example").exporter
    assert exporter._certificate_file is True
    assert exporter._client_cert == (str(client_cert) if with_client_certificate else None)


def test_second_client_on_the_same_key_does_not_build_another_provider():
    """A discarded TracerProvider is pinned forever by its atexit hook."""
    import gc

    from litellm.integrations.langfuse.langfuse_sdk import (
        _retire_orphaned_providers,
        acquire_langfuse_client,
    )

    # reap earlier tests' orphans first, so the count below only moves if a provider is built
    gc.collect()
    _retire_orphaned_providers()

    pk = "pk-provider-reuse-test"
    LangfuseResourceManager._instances.pop(pk, None)
    parameters = {"public_key": pk, "secret_key": "sk-reuse", "base_url": "http://127.0.0.1:1"}
    try:
        first = acquire_langfuse_client(parameters=parameters, environment=None, release=None, mock_mode=True)
        providers_after_first = len(_litellm_built_providers)
        second = acquire_langfuse_client(parameters=parameters, environment=None, release=None, mock_mode=True)

        assert second._resources is first._resources
        assert len(_litellm_built_providers) == providers_after_first
    finally:
        LangfuseResourceManager._instances.pop(pk, None)


def test_leaving_mock_mode_on_the_same_key_stops_using_the_discarding_exporter():
    from litellm.integrations.langfuse.langfuse_sdk import DiscardingSpanExporter

    pk = "pk-mock-to-live-test"
    LangfuseResourceManager._instances.pop(pk, None)
    parameters = {"public_key": pk, "secret_key": "sk-live", "base_url": "http://127.0.0.1:1"}
    try:
        mocked = acquire_langfuse_client(parameters=parameters, environment=None, release=None, mock_mode=True)
        assert isinstance(mocked._resources.span_exporter, DiscardingSpanExporter)

        live = acquire_langfuse_client(parameters=parameters, environment=None, release=None, mock_mode=False)
        assert live._resources is not mocked._resources
        assert not isinstance(live._resources.span_exporter, DiscardingSpanExporter)
        assert LangfuseResourceManager._instances.get(pk) is live._resources
    finally:
        LangfuseResourceManager._instances.pop(pk, None)


def test_a_changed_sample_rate_on_the_same_key_rebuilds_the_bundle(monkeypatch: pytest.MonkeyPatch):
    pk = "pk-resample-test"
    LangfuseResourceManager._instances.pop(pk, None)
    parameters = {"public_key": pk, "secret_key": "sk-resample", "base_url": "http://127.0.0.1:1"}
    try:
        monkeypatch.setenv("LANGFUSE_SAMPLE_RATE", "0.25")
        quarter = acquire_langfuse_client(parameters=parameters, environment=None, release=None, mock_mode=True)
        monkeypatch.setenv("LANGFUSE_SAMPLE_RATE", "1")
        full = acquire_langfuse_client(parameters=parameters, environment=None, release=None, mock_mode=True)

        assert quarter._resources.sample_rate == 0.25
        assert full._resources is not quarter._resources
        assert full._resources.sample_rate == 1.0
    finally:
        LangfuseResourceManager._instances.pop(pk, None)
