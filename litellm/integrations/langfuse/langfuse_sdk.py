from __future__ import annotations

import re
import threading
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

__all__ = (
    "AS_ROOT_ATTRIBUTE",
    "PUBLIC_ATTRIBUTE",
    "RELEASE_ATTRIBUTE",
    "DiscardingSpanExporter",
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
    normalized: Final = "" if trace_id is None else str(trace_id).lower().replace("-", "")
    if _TRACE_ID_PATTERN.fullmatch(normalized):
        return normalized
    return Langfuse.create_trace_id(seed=str(trace_id)) if normalized else Langfuse.create_trace_id()


def resolve_observation_id(observation_id: object | None) -> str | None:
    """Map a caller's parent observation id onto v4's 16 lowercase hex characters."""
    normalized: Final = "" if observation_id is None else str(observation_id).lower().replace("-", "")
    if not normalized:
        return None
    if _OBSERVATION_ID_PATTERN.fullmatch(normalized):
        return normalized
    return sha256(normalized.encode("utf-8")).digest()[:8].hex()


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
    attributes: Mapping[str, object],
) -> LangfuseSpan:
    """Create a sibling observation inside the same trace, keeping its own window."""
    otel_span: Final = client._otel_tracer.start_span(  # pyright: ignore[reportPrivateUsage]  # only route to a historical start time
        name=name, context=context, start_time=to_unix_nanos(start_time)
    )
    return LangfuseSpan(otel_span=otel_span, langfuse_client=client, **attributes)  # pyright: ignore[reportArgumentType]  # kwargs-ok: callback-built params, v2 accepted the same shapes


_ENVIRONMENT_ATTRIBUTE: Final = "langfuse.environment"
_RELEASE_ATTRIBUTE: Final = "langfuse.release"


def build_isolated_tracer_provider(*, environment: str | None, release: str | None) -> TracerProvider:
    """Give the langfuse client a provider of its own instead of the process-wide one.

    v4 is built on OpenTelemetry and otherwise either claims the global tracer
    provider, which silently disables litellm's own exporters, or attaches its
    processor to litellm's, which sends litellm spans to every langfuse project
    and langfuse spans to every other litellm destination.

    The resource is rebuilt here because langfuse only applies ``environment``
    and ``release`` when it constructs the provider itself.
    """
    attributes: Final = MappingProxyType(
        {
            key: value
            for key, value in ((_ENVIRONMENT_ATTRIBUTE, environment), (_RELEASE_ATTRIBUTE, release))
            if value is not None
        }
    )
    return TracerProvider(resource=Resource.create(dict(attributes)))


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


def evict_stale_langfuse_resources(*, public_key: str | None, secret_key: str | None, base_url: str | None) -> None:
    """Drop a cached client whose credentials no longer match the ones being requested.

    langfuse keys its client registry on the public key alone, so a rotated
    secret or a moved host silently keeps exporting with the original values.
    Only the one stale entry is removed; the SDK's own reset would shut down
    every other tenant in the process.
    """
    if not public_key:
        return
    with LangfuseResourceManager._lock:  # pyright: ignore[reportPrivateUsage]  # registry has no public accessor
        cached: Final = LangfuseResourceManager._instances.get(public_key)  # pyright: ignore[reportPrivateUsage]  # registry has no public accessor
        if cached is None:
            return
        if getattr(cached, "secret_key", None) == secret_key and getattr(cached, "base_url", None) == base_url:
            return
        LangfuseResourceManager._instances.pop(public_key, None)  # pyright: ignore[reportPrivateUsage]  # registry has no public accessor


_LIVE_CLIENTS_LOCK: Final = threading.Lock()
# litellm clients still using each SDK resource bundle; the bundle is torn down with the last one.
# Both sides are weak so a throwaway client (a health probe, an alerting lookup) that is simply
# garbage-collected stops holding the bundle open rather than inflating a counter forever.
_live_clients: Final[WeakKeyDictionary[LangfuseResourceManager, WeakSet]] = WeakKeyDictionary()


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
    """
    resources: Final = getattr(client, "_resources", None)
    client.flush()
    if resources is None:
        client.shutdown()
        return
    if not _release_langfuse_resources(resources, client):
        return
    client.shutdown()
    provider: Final = getattr(resources, "tracer_provider", None)
    if provider is not None and not isinstance(provider, otel_trace.ProxyTracerProvider):
        provider.shutdown()
    public_key: Final = getattr(resources, "public_key", None)
    if public_key is None:
        return
    with LangfuseResourceManager._lock:  # pyright: ignore[reportPrivateUsage]  # registry has no public accessor
        if LangfuseResourceManager._instances.get(public_key) is resources:  # pyright: ignore[reportPrivateUsage]  # registry has no public accessor
            LangfuseResourceManager._instances.pop(public_key, None)  # pyright: ignore[reportPrivateUsage]  # registry has no public accessor
