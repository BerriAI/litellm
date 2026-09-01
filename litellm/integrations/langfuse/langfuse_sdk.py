from __future__ import annotations

import os
import re
import threading
from base64 import b64encode
from collections.abc import Mapping
from datetime import datetime
from hashlib import sha256
from types import MappingProxyType
from typing import Final
from weakref import WeakKeyDictionary, WeakSet

import opentelemetry.trace as otel_trace
from langfuse import Langfuse, LangfuseGeneration, LangfuseSpan, propagate_attributes
from langfuse._client.resource_manager import LangfuseResourceManager
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

__all__ = (
    "AS_ROOT_ATTRIBUTE",
    "PUBLIC_ATTRIBUTE",
    "RELEASE_ATTRIBUTE",
    "DiscardingSpanExporter",
    "acquire_langfuse_client",
    "build_isolated_tracer_provider",
    "evict_stale_langfuse_resources",
    "open_trace_context",
    "propagate_attributes",
    "register_langfuse_client",
    "resolve_observation_id",
    "resolve_trace_id",
    "shutdown_langfuse_client",
    "start_child_span",
    "start_generation",
    "to_unix_nanos",
)

AS_ROOT_ATTRIBUTE: Final = "langfuse.internal.as_root"
PUBLIC_ATTRIBUTE: Final = "langfuse.trace.public"
RELEASE_ATTRIBUTE: Final = "langfuse.release"
_TRACE_ID_PATTERN: Final = re.compile(r"^[0-9a-f]{32}$")
_OBSERVATION_ID_PATTERN: Final = re.compile(r"^[0-9a-f]{16}$")


def to_unix_nanos(value: datetime | float | None) -> int | None:
    """Langfuse v4 takes OTel timestamps, which are integer nanoseconds since the epoch.

    Guardrail entries carry unix seconds as floats rather than datetimes, so both
    shapes have to convert; the v2 SDK accepted either through a pydantic model.
    """
    if value is None:
        return None
    seconds: Final = value.timestamp() if isinstance(value, datetime) else float(value)
    return int(seconds * 1_000_000_000)


def resolve_trace_id(trace_id: object | None) -> str:
    """Map a caller's trace id onto the 32 lowercase hex characters v4 requires."""
    serialized: Final = "" if trace_id is None else str(trace_id)
    normalized: Final = serialized.lower().replace("-", "")
    if _TRACE_ID_PATTERN.fullmatch(normalized):
        return normalized
    return Langfuse.create_trace_id(seed=serialized) if serialized else Langfuse.create_trace_id()


def resolve_observation_id(observation_id: object | None) -> str | None:
    """Map a caller's parent observation id onto v4's 16 lowercase hex characters."""
    serialized: Final = "" if observation_id is None else str(observation_id)
    normalized: Final = serialized.lower().replace("-", "")
    if _OBSERVATION_ID_PATTERN.fullmatch(normalized):
        return normalized
    if not serialized:
        return None
    return sha256(serialized.encode("utf-8")).digest()[:8].hex()


def open_trace_context(
    *,
    client: Langfuse,
    trace_id: str,
    parent_observation_id: str | None,
) -> tuple[Context, bool]:
    """Build the OTel context that places new observations inside ``trace_id``.

    Returns the context plus whether the caller must claim trace root. Langfuse
    fabricates a random parent span id when no real parent is supplied, so the
    observation is a child of something that will never be exported; the public
    SDK path compensates by marking the span as root and this path must do the
    same.
    """
    remote_parent: Final = client._create_remote_parent_span(  # pyright: ignore[reportPrivateUsage]  # no public equivalent in v4
        trace_id=trace_id, parent_span_id=parent_observation_id
    )
    return otel_trace.set_span_in_context(remote_parent), parent_observation_id is None


def start_generation(
    *,
    client: Langfuse,
    context: Context,
    name: str,
    start_time: datetime | float | None,
    claim_trace_root: bool,
    release: str | None = None,
    public: bool | None = None,
    attributes: Mapping[str, object],
) -> LangfuseGeneration:
    """Create a generation whose start time is when the model call began.

    No public v4 API accepts a historical start time, so this drives the SDK's
    own OTel tracer, which does. Langfuse documents this route for backdated
    ingestion.

    ``public`` is the v2 ``trace(public=...)`` flag; v4 reads it off the root
    observation's ``langfuse.trace.public`` attribute instead.
    """
    otel_span: Final = client._otel_tracer.start_span(  # pyright: ignore[reportPrivateUsage]  # only route to a historical start time
        name=name, context=context, start_time=to_unix_nanos(start_time)
    )
    if claim_trace_root:
        otel_span.set_attribute(AS_ROOT_ATTRIBUTE, True)
    if public is not None:
        otel_span.set_attribute(PUBLIC_ATTRIBUTE, public)
    generation: Final = LangfuseGeneration(otel_span=otel_span, langfuse_client=client, **attributes)  # pyright: ignore[reportArgumentType]  # kwargs-ok: callback-built params, v2 accepted the same shapes
    if release is not None:
        # after the wrapper, which stamps the client-wide release and would otherwise
        # overwrite the release this request asked for
        otel_span.set_attribute(RELEASE_ATTRIBUTE, release)
    return generation


def start_child_span(
    *,
    client: Langfuse,
    context: Context,
    name: str,
    start_time: datetime | float | None,
    claim_trace_root: bool,
    attributes: Mapping[str, object],
) -> LangfuseSpan:
    """Create a sibling observation inside the same trace, keeping its own window.

    When the shared parent is the fabricated remote span, every observation must
    claim trace root itself — the SDK's own remote-parent paths stamp each span —
    or it exports with a parent id that is never exported.
    """
    otel_span: Final = client._otel_tracer.start_span(  # pyright: ignore[reportPrivateUsage]  # only route to a historical start time
        name=name, context=context, start_time=to_unix_nanos(start_time)
    )
    if claim_trace_root:
        otel_span.set_attribute(AS_ROOT_ATTRIBUTE, True)
    return LangfuseSpan(otel_span=otel_span, langfuse_client=client, **attributes)  # pyright: ignore[reportArgumentType]  # kwargs-ok: callback-built params, v2 accepted the same shapes


_ENVIRONMENT_ATTRIBUTE: Final = "langfuse.environment"

# providers litellm itself constructed; a bundle adopted from user code may hold the
# process-global provider, which litellm must never shut down.
_litellm_built_providers: Final[WeakSet] = WeakSet()


def build_isolated_tracer_provider(*, environment: str | None, release: str | None) -> TracerProvider:
    """Give the langfuse client a provider of its own instead of the process-wide one.

    v4 is built on OpenTelemetry and otherwise either claims the global tracer
    provider, which silently disables litellm's own exporters, or attaches its
    processor to litellm's, which sends litellm spans to every langfuse project
    and langfuse spans to every other litellm destination.

    The resource is rebuilt here because langfuse only applies ``environment``
    and ``release`` when it constructs the provider itself, and the sampler is
    rebuilt for the same reason: ``LANGFUSE_SAMPLE_RATE`` is otherwise silently
    ignored and every trace exports.
    """
    raw_sample_rate: Final = os.environ.get("LANGFUSE_SAMPLE_RATE")
    sample_rate: Final = float(raw_sample_rate) if raw_sample_rate is not None else 1.0
    if not 0.0 <= sample_rate <= 1.0:
        raise ValueError(f"Sample rate must be between 0.0 and 1.0, got {sample_rate}")
    attributes: Final = MappingProxyType(
        {
            key: value
            for key, value in ((_ENVIRONMENT_ATTRIBUTE, environment), (RELEASE_ATTRIBUTE, release))
            if value is not None
        }
    )
    provider: Final = TracerProvider(
        resource=Resource.create(dict(attributes)),
        sampler=TraceIdRatioBased(sample_rate) if sample_rate < 1 else None,
    )
    _litellm_built_providers.add(provider)
    return provider


class DiscardingSpanExporter(SpanExporter):
    """Accept and drop every span, for mock mode.

    The mock intercepts the httpx client langfuse used to take, but v4 ships
    observations through its own OTLP exporter, so without this the "no network
    calls" contract silently sends real traces to the configured host.
    """

    def export(self, spans: object) -> SpanExportResult:
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


_LIVE_CLIENTS_LOCK: Final = threading.Lock()
# litellm clients still using each SDK resource bundle; the bundle is torn down with the last one.
# Both sides are weak so a throwaway client (a health probe, an alerting lookup) that is simply
# garbage-collected stops holding the bundle open rather than inflating a counter forever.
_live_clients: Final[WeakKeyDictionary[LangfuseResourceManager, WeakSet]] = WeakKeyDictionary()


def _evict_if_stale_locked(
    *, public_key: object, secret_key: object, base_url: object
) -> tuple[LangfuseResourceManager | None, LangfuseResourceManager | None]:
    """Assumes ``LangfuseResourceManager._lock`` is held; returns the still-valid bundle and the evicted one."""
    if not public_key:
        return None, None
    cached: Final = LangfuseResourceManager._instances.get(public_key)  # pyright: ignore[reportPrivateUsage]  # registry has no public accessor
    if cached is None:
        return None, None
    if getattr(cached, "secret_key", None) == secret_key and getattr(cached, "base_url", None) == base_url:
        return cached, None
    return None, LangfuseResourceManager._instances.pop(public_key, None)  # pyright: ignore[reportPrivateUsage]  # registry has no public accessor


def _shutdown_abandoned_provider(resources: LangfuseResourceManager | None) -> None:
    """Retire the provider litellm built for an evicted bundle, once no live client is still on it.

    Runs after the registry lock is released: provider shutdown flushes and joins the export
    thread, which must never happen under the lock every other client init contends on. A
    client still holding these resources tears them down itself in ``shutdown_langfuse_client``.
    """
    if resources is None:
        return
    with _LIVE_CLIENTS_LOCK:
        holders: Final = _live_clients.get(resources)
        if holders is not None and len(holders) > 0:
            return
        _live_clients.pop(resources, None)
    provider: Final = getattr(resources, "tracer_provider", None)
    if provider is not None and provider in _litellm_built_providers:
        provider.shutdown()


def evict_stale_langfuse_resources(*, public_key: str | None, secret_key: str | None, base_url: str | None) -> None:
    """Drop a cached client whose credentials no longer match the ones being requested."""
    with LangfuseResourceManager._lock:  # pyright: ignore[reportPrivateUsage]  # registry has no public accessor
        _, abandoned = _evict_if_stale_locked(public_key=public_key, secret_key=secret_key, base_url=base_url)
    _shutdown_abandoned_provider(abandoned)


def _build_verified_span_exporter(*, public_key: object, secret_key: object, base_url: object) -> SpanExporter | None:
    """Rebuild litellm's TLS material onto the export channel.

    v2 ingested through the injected httpx client, which carried litellm's CA
    bundle and client certificate; v4 ships every observation through its own
    OTLP exporter, so a private-CA deployment would fail TLS on every export in
    a background thread while ``auth_check`` (still on the httpx client) stays
    green. Only built when custom TLS material is configured; endpoint and
    headers mirror ``langfuse._client.span_processor``.
    """
    import litellm

    ca_bundle: Final = litellm.ssl_verify if isinstance(litellm.ssl_verify, str) else None
    configured_certificate: Final = os.getenv("SSL_CERTIFICATE") or litellm.ssl_certificate
    client_certificate: Final = configured_certificate if isinstance(configured_certificate, str) else None
    if ca_bundle is None and client_certificate is None:
        return None
    import langfuse as langfuse_package
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    langfuse_version: Final = getattr(langfuse_package, "__version__", "unknown")

    export_path: Final = os.getenv("LANGFUSE_OTEL_TRACES_EXPORT_PATH")
    endpoint: Final = f"{base_url}/{export_path}" if export_path else f"{base_url}/api/public/otel/v1/traces"
    encoded_auth: Final = b64encode(f"{public_key}:{secret_key}".encode()).decode("ascii")
    return OTLPSpanExporter(
        endpoint=endpoint,
        headers={  # mutable-ok: the exporter copies these into its session headers
            "Authorization": "Basic " + encoded_auth,
            "x-langfuse-sdk-name": "python",
            "x-langfuse-sdk-version": langfuse_version,
            "x-langfuse-public-key": str(public_key),
        },
        certificate_file=ca_bundle,
        client_certificate_file=client_certificate,
    )


def acquire_langfuse_client(
    *,
    parameters: Mapping[str, object],
    environment: str | None,
    release: str | None,
    mock_mode: bool,
) -> Langfuse:
    """Evict-check, construct, and register a client as one atomic step.

    The SDK registry lock is held across the whole sequence: released between
    eviction and construction, two concurrent inits for the same public key
    with different secrets can bind one tenant's logger to the other tenant's
    exporter. The isolated provider is only built when the registry does not
    already hold the key — a discarded ``TracerProvider`` stays pinned forever
    by its atexit hook, so building one per health probe or alerting lookup
    would leak a provider each time.
    """
    public_key: Final = parameters.get("public_key")
    span_exporter: Final = (
        DiscardingSpanExporter()
        if mock_mode
        else _build_verified_span_exporter(
            public_key=public_key,
            secret_key=parameters.get("secret_key"),
            base_url=parameters.get("base_url"),
        )
    )
    with LangfuseResourceManager._lock:  # pyright: ignore[reportPrivateUsage]  # registry has no public accessor
        cached, abandoned = _evict_if_stale_locked(
            public_key=public_key,
            secret_key=parameters.get("secret_key"),
            base_url=parameters.get("base_url"),
        )
        client: Final = Langfuse(
            **parameters,  # pyright: ignore[reportArgumentType]  # kwargs-ok: dict mirrors the typed ctor, values resolved by the callers
            tracer_provider=None
            if cached is not None
            else build_isolated_tracer_provider(environment=environment, release=release),
            span_exporter=span_exporter,
        )
        register_langfuse_client(client)
    _shutdown_abandoned_provider(abandoned)
    return client


def register_langfuse_client(client: Langfuse) -> None:
    """Track the client against the SDK resources it ended up with.

    langfuse keys its resources on the public key alone, so a second client for
    the same key (a per-key ``langfuse_environment`` override, a team whose
    callback_vars repeat the global credentials) is handed the first client's
    tracer provider and export thread rather than its own. Only the last live
    client may shut those down; see ``shutdown_langfuse_client``.
    """
    resources: Final = getattr(client, "_resources", None)
    if resources is None:
        return
    with _LIVE_CLIENTS_LOCK:
        holders = _live_clients.get(resources)
        if holders is None:
            holders = WeakSet()
            _live_clients[resources] = holders
        holders.add(client)


def _release_langfuse_resources(resources: LangfuseResourceManager, client: Langfuse) -> bool:
    """Drop the client's claim; True when no other live client still uses ``resources``."""
    with _LIVE_CLIENTS_LOCK:
        holders: Final = _live_clients.get(resources)
        if holders is None:
            return True
        holders.discard(client)
        if len(holders) > 0:
            return False
        _live_clients.pop(resources, None)
        return True


def shutdown_langfuse_client(client: Langfuse) -> None:
    """Release everything the client owns, which the SDK's own shutdown does not.

    ``Langfuse.shutdown`` joins the score and media consumers but leaves the
    tracer provider's export thread running and leaves the client in the
    registry, so a later request for the same key gets a dead client back.

    A client that shares its resources with another live client only flushes:
    shutting the shared provider down here would silence the other client for
    the rest of its life, as it did before the reference count existed.

    The registry entry is removed before the blocking shutdown so a concurrent
    construct builds a fresh bundle instead of adopting a dying one, and the
    provider is only shut down when litellm built it: a bundle adopted from
    user code may share the process-global provider.
    """
    resources: Final = getattr(client, "_resources", None)
    client.flush()
    if resources is None:
        client.shutdown()
        return
    public_key: Final = getattr(resources, "public_key", None)
    with LangfuseResourceManager._lock:  # pyright: ignore[reportPrivateUsage]  # registry has no public accessor
        if not _release_langfuse_resources(resources, client):
            return
        if public_key is not None and LangfuseResourceManager._instances.get(public_key) is resources:  # pyright: ignore[reportPrivateUsage]  # registry has no public accessor
            LangfuseResourceManager._instances.pop(public_key, None)  # pyright: ignore[reportPrivateUsage]  # registry has no public accessor
    client.shutdown()
    provider: Final = getattr(resources, "tracer_provider", None)
    if provider is not None and provider in _litellm_built_providers:
        provider.shutdown()
