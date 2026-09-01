"""Covers the v4 observation plumbing: historical timestamps and id normalisation.

The timestamp assertions are the regression guard for the migration: v4 has no
public API for an observation start time, so a callback running after the model
call would otherwise record its own duration instead of the call's.
"""

import json
from datetime import datetime, timedelta, timezone

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
    _litellm_built_providers,
    build_isolated_tracer_provider,
    evict_stale_langfuse_resources,
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
    start_child_span(
        client=lf, context=context, name="guardrail", start_time=guardrail_start, claim_trace_root=claim_root, attributes={}
    ).end(end_time=to_unix_nanos(guardrail_start + 2))
    start_generation(
        client=lf, context=context, name="gen", start_time=CALL_START, claim_trace_root=claim_root, attributes={}
    ).end(end_time=to_unix_nanos(CALL_END))
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


def test_child_span_keeps_its_own_window_and_stays_a_sibling(client):
    lf, exporter = client
    context, claim_root = open_trace_context(client=lf, trace_id="d" * 32, parent_observation_id=None)
    guardrail_start = CALL_START + timedelta(seconds=1)
    start_child_span(
        client=lf, context=context, name="guardrail", start_time=guardrail_start, claim_trace_root=claim_root, attributes={}
    ).end(end_time=to_unix_nanos(guardrail_start + timedelta(seconds=2)))
    start_generation(
        client=lf, context=context, name="gen", start_time=CALL_START, claim_trace_root=claim_root, attributes={}
    ).end(end_time=to_unix_nanos(CALL_END))
    lf.flush()

    guardrail = _only_span(exporter, "guardrail")
    generation = _only_span(exporter, "gen")
    assert (guardrail.end_time - guardrail.start_time) == 2 * 1_000_000_000
    assert guardrail.context.trace_id == generation.context.trace_id
    # the shared remote parent is fabricated and never exported, so both must claim trace root
    assert guardrail.attributes.get(AS_ROOT_ATTRIBUTE) is True
    assert generation.attributes.get(AS_ROOT_ATTRIBUTE) is True


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


def test_langfuse_sample_rate_drops_spans_on_the_isolated_provider(monkeypatch):
    """The SDK only installs its sampler on providers it builds itself; v2 sampled via the same env var."""
    monkeypatch.setenv("LANGFUSE_SAMPLE_RATE", "0")
    dropped_exporter = InMemorySpanExporter()
    dropping_provider = build_isolated_tracer_provider(environment=None, release=None)
    dropping_provider.add_span_processor(SimpleSpanProcessor(dropped_exporter))
    dropping_provider.get_tracer("test").start_span("dropped").end()
    assert not dropped_exporter.get_finished_spans()

    monkeypatch.delenv("LANGFUSE_SAMPLE_RATE")
    kept_exporter = InMemorySpanExporter()
    keeping_provider = build_isolated_tracer_provider(environment=None, release=None)
    keeping_provider.add_span_processor(SimpleSpanProcessor(kept_exporter))
    keeping_provider.get_tracer("test").start_span("kept").end()
    assert [span.name for span in kept_exporter.get_finished_spans()] == ["kept"]


def test_invalid_sample_rate_fails_at_construction_like_the_sdk(monkeypatch):
    monkeypatch.setenv("LANGFUSE_SAMPLE_RATE", "1.5")
    with pytest.raises(ValueError, match=r"between 0\.0 and 1\.0"):
        build_isolated_tracer_provider(environment=None, release=None)


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


def test_ssl_exporter_is_only_built_with_custom_tls_material(monkeypatch, tmp_path):
    """v4 exports over its own OTLP channel, so litellm's CA bundle must be rebuilt onto it."""
    import litellm
    from litellm.integrations.langfuse.langfuse_sdk import _build_verified_span_exporter

    monkeypatch.delenv("SSL_CERTIFICATE", raising=False)
    monkeypatch.setattr(litellm, "ssl_verify", True)
    monkeypatch.setattr(litellm, "ssl_certificate", None)
    assert (
        _build_verified_span_exporter(public_key="pk", secret_key="sk", base_url="https://lf.internal.example") is None
    )

    ca_path = tmp_path / "private-ca.pem"
    ca_path.write_text("dummy")
    monkeypatch.setattr(litellm, "ssl_verify", str(ca_path))
    exporter = _build_verified_span_exporter(public_key="pk", secret_key="sk", base_url="https://lf.internal.example")
    assert exporter is not None
    assert exporter._endpoint == "https://lf.internal.example/api/public/otel/v1/traces"
    assert exporter._certificate_file == str(ca_path)
    assert exporter._headers["x-langfuse-public-key"] == "pk"


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
