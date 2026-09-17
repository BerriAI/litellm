from __future__ import annotations

import os
import re
import threading
from base64 import b64encode
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from importlib.metadata import version
from itertools import chain
from time import sleep
from types import MappingProxyType
from typing import Final
from weakref import WeakKeyDictionary, WeakSet

import opentelemetry.trace as otel_trace
from langfuse import Langfuse, LangfuseGeneration, LangfuseSpan, propagate_attributes
from langfuse._client.resource_manager import LangfuseResourceManager
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
from opentelemetry.sdk.trace.sampling import Decision, Sampler, SamplingResult
from opentelemetry.trace import Link, SpanKind, TraceState
from opentelemetry.util.types import Attributes
from requests import RequestException

from litellm._logging import verbose_logger

__all__ = (
    "AS_ROOT_ATTRIBUTE",
    "PUBLIC_ATTRIBUTE",
    "RELEASE_ATTRIBUTE",
    "DiscardingSpanExporter",
    "RetryingSpanExporter",
    "TraceIdHashSampler",
    "acquire_langfuse_client",
    "build_isolated_tracer_provider",
    "configured_sample_rate",
    "evict_stale_langfuse_resources",
    "lease_langfuse_client",
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
_TRACE_ID_PATTERN: Final = re.compile(r"^(?=.*[1-9a-f])[0-9a-f]{32}$")
_OBSERVATION_ID_PATTERN: Final = re.compile(r"^(?=.*[1-9a-f])[0-9a-f]{16}$")


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
    existing_trace: bool = False,
) -> tuple[Context, bool]:
    """Build the OTel context that places new observations inside ``trace_id``.

    Returns the context plus whether the caller must claim trace root. Langfuse
    fabricates a random parent span id when no real parent is supplied, so the
    observation is a child of something that will never be exported; the public
    SDK path compensates by marking the span as root and this path must do the
    same.

    ``existing_trace`` is the v2 ``existing_trace_id`` contract: the trace is
    appended to, never rewritten. The server takes a root observation's name and
    I/O as the trace's, so a continuation must not claim root; trace fields it
    does want changed travel as explicit ``langfuse.trace.*`` attributes.
    """
    remote_parent: Final = client._create_remote_parent_span(  # pyright: ignore[reportPrivateUsage]  # no public equivalent in v4
        trace_id=trace_id, parent_span_id=parent_observation_id
    )
    return otel_trace.set_span_in_context(remote_parent), parent_observation_id is None and not existing_trace


def start_generation(
    *,
    client: Langfuse,
    context: Context,
    name: str,
    start_time: datetime | float | None,
    claim_trace_root: bool,
    release: str | None = None,
    public: bool | None = None,
    observation_id: str | None = None,
    attributes: Mapping[str, object],
) -> LangfuseGeneration:
    """Create a generation whose start time is when the model call began.

    No public v4 API accepts a historical start time, so this drives the SDK's
    own OTel tracer, which does. Langfuse documents this route for backdated
    ingestion.

    ``public`` is the v2 ``trace(public=...)`` flag; v4 reads it off the root
    observation's ``langfuse.trace.public`` attribute instead.

    ``observation_id`` is the v2 ``generation(id=...)`` argument. v4 derives the
    observation id from the OTel span id, so it is honoured through the
    isolated provider's id generator; a provider adopted from user code keeps
    its own generator and the returned generation's ``id`` is the truth.
    """
    requested: Final = _requested_span_id.set(int(observation_id, 16) if observation_id is not None else None)
    try:
        otel_span: Final = client._otel_tracer.start_span(  # pyright: ignore[reportPrivateUsage]  # only route to a historical start time
            name=name, context=context, start_time=to_unix_nanos(start_time)
        )
    finally:
        _requested_span_id.reset(requested)
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
_requested_span_id: Final[ContextVar[int | None]] = ContextVar("litellm_langfuse_requested_span_id", default=None)


class _RequestedSpanIdGenerator(RandomIdGenerator):
    """Hand out the span id the calling context asked for, random otherwise."""

    def generate_span_id(self) -> int:
        requested: Final = _requested_span_id.get()
        return super().generate_span_id() if requested is None else requested


# providers litellm itself constructed; a bundle adopted from user code may hold the
# process-global provider, which litellm must never shut down.
_litellm_built_providers: Final[WeakSet] = WeakSet()


@dataclass(frozen=True, slots=True)
class TraceIdHashSampler(Sampler):
    """Sample on a SHA-256 of the trace id rather than its low 64 bits.

    litellm trace ids are UUIDs, whose variant bits pin the top of that low word, so
    ``TraceIdRatioBased`` drops every trace at rates up to 0.5 and skews above it.
    """

    rate: float

    def should_sample(
        self,
        parent_context: Context | None,
        trace_id: int,
        name: str,
        kind: SpanKind | None = None,
        attributes: Attributes = None,
        links: Sequence[Link] | None = None,
        trace_state: TraceState | None = None,
    ) -> SamplingResult:
        digest: Final = sha256(trace_id.to_bytes(16, "big")).digest()
        sampled: Final = int.from_bytes(digest[:8], "big") < round(self.rate * 2**64)
        parent: Final = otel_trace.get_current_span(parent_context).get_span_context()
        return SamplingResult(
            Decision.RECORD_AND_SAMPLE if sampled else Decision.DROP,
            attributes if sampled else None,
            parent.trace_state if parent.is_valid else None,
        )

    def get_description(self) -> str:
        return f"TraceIdHashSampler{{{self.rate}}}"


def _parse_sample_rate(raw: str) -> float | None:
    try:
        rate: Final = float(raw)
    except ValueError:
        return None
    return rate if 0.0 <= rate <= 1.0 else None


def configured_sample_rate() -> float:
    """``LANGFUSE_SAMPLE_RATE`` as a fraction, exporting everything when it is unset or unusable."""
    raw: Final = os.environ.get("LANGFUSE_SAMPLE_RATE")
    if raw is None:
        return 1.0
    parsed: Final = _parse_sample_rate(raw)
    if parsed is None:
        verbose_logger.warning(
            "LANGFUSE_SAMPLE_RATE=%r is not a number between 0.0 and 1.0; ignoring it and exporting every trace", raw
        )
        return 1.0
    return parsed


def build_isolated_tracer_provider(
    *, environment: str | None, release: str | None, sample_rate: float = 1.0
) -> TracerProvider:
    """Give the langfuse client a provider of its own instead of the process-wide one.

    v4 is built on OpenTelemetry and otherwise either claims the global tracer
    provider, which silently disables litellm's own exporters, or attaches its
    processor to litellm's, which sends litellm spans to every langfuse project
    and langfuse spans to every other litellm destination.

    The resource is rebuilt here because langfuse only applies ``environment``
    and ``release`` when it constructs the provider itself, and the sampler is
    installed for the same reason: ``sample_rate`` is otherwise silently
    ignored and every trace exports.
    """
    attributes: Final = MappingProxyType(
        {
            key: value
            for key, value in ((_ENVIRONMENT_ATTRIBUTE, environment), (RELEASE_ATTRIBUTE, release))
            if value is not None
        }
    )
    provider: Final = TracerProvider(
        resource=Resource.create(attributes),
        sampler=TraceIdHashSampler(sample_rate) if sample_rate < 1 else None,
        id_generator=_RequestedSpanIdGenerator(),
    )
    with _LIVE_CLIENTS_LOCK:
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


@dataclass(frozen=True, slots=True)
class RetryingSpanExporter(SpanExporter):
    """Retry a batch whose HTTP round trip raised, as the v2 ingestion consumer did.

    The OTLP http exporter only retries 429 and 5xx responses; a connect or
    read timeout propagates, and ``BatchSpanProcessor`` drops the whole batch
    on any exception. A destination that stalls for a few seconds therefore
    lost every observation in flight, where v2 backed off three times first.
    """

    exporter: SpanExporter
    delays: Sequence[float] = (1.0, 2.0, 4.0)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for delay in chain(self.delays, (None,)):
            try:
                return self.exporter.export(spans)
            except RequestException as error:
                if delay is None:
                    verbose_logger.error("Langfuse export failed after %d retries: %s", len(self.delays), error)
                    return SpanExportResult.FAILURE
                verbose_logger.warning("Langfuse export raised %s, retrying in %ss", error, delay)
                sleep(delay)
        return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        self.exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self.exporter.force_flush(timeout_millis)


_LIVE_CLIENTS_LOCK: Final = threading.Lock()
# litellm clients still using each SDK resource bundle; the bundle is torn down with the last one.
# Both sides are weak so a throwaway client (a health probe, an alerting lookup) that is simply
# garbage-collected stops holding the bundle open rather than inflating a counter forever.
_live_clients: Final[WeakKeyDictionary[LangfuseResourceManager, WeakSet]] = WeakKeyDictionary()


class _LangfuseLifecycleState:
    """How many callbacks are leasing one SDK resource bundle, and what eviction has queued behind them.

    ``lock`` is never held across a teardown, which takes the SDK's own registry lock.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active_leases = 0
        self.teardown_in_progress = False
        self.teardown_owner: int | None = None
        self.pending_clients: set[Langfuse] = set()  # mutable-ok: eviction and callback threads queue into it
        self.retired: WeakSet[Langfuse] = WeakSet()  # mutable-ok: eviction marks its clients here from its own thread

    def open_lease(self, client: Langfuse) -> bool:
        """Take a lease on ``client``; False when eviction already reached it, so a lease would guard a dead client."""
        with self.lock:
            if client in self.retired:
                return False
            self.active_leases += 1
            return True

    def claim_for_teardown(self, client: Langfuse) -> bool:
        """Whether this thread owns ``client``'s teardown; a lease or another teardown in flight queues it instead."""
        with self.lock:
            self.retired.add(client)
            if self.active_leases > 0 or self.teardown_in_progress:
                self.pending_clients.add(client)
                return False
            self.teardown_in_progress = True
            self.teardown_owner = threading.get_ident()
            return True

    def release_lease(self) -> tuple[Langfuse, ...]:
        """Drop this lease and take ownership of the teardowns it was holding up, if it was the last one."""
        with self.lock:
            self.active_leases -= 1
            if self.active_leases > 0 or self.teardown_in_progress or not self.pending_clients:
                return ()
            claimed: Final = tuple(self.pending_clients)
            self.pending_clients.clear()
            self.teardown_in_progress = True
            self.teardown_owner = threading.get_ident()
            return claimed

    def next_teardown_batch(self) -> tuple[Langfuse, ...]:
        """Whatever eviction queued while the last batch was draining, handing the ownership flag back when empty."""
        with self.lock:
            if self.active_leases == 0 and self.pending_clients:
                claimed: Final = tuple(self.pending_clients)
                self.pending_clients.clear()
                return claimed
            self.teardown_in_progress = False
            self.teardown_owner = None
            return ()

    def requeue(self, clients: tuple[Langfuse, ...]) -> None:
        with self.lock:
            self.pending_clients.update(clients)

    def end_teardown(self) -> None:
        with self.lock:
            if self.teardown_owner == threading.get_ident():
                self.teardown_in_progress = False
                self.teardown_owner = None


_LIFECYCLE_STATES_LOCK: Final = threading.Lock()
_LIFECYCLE_STATES: Final[WeakKeyDictionary[object, _LangfuseLifecycleState]] = WeakKeyDictionary()


def _lifecycle_state(client: Langfuse) -> _LangfuseLifecycleState:
    """One state per resource bundle, since teardown closes the provider every client on that bundle exports through."""
    resources: Final = getattr(client, "_resources", None)
    key: Final = client if resources is None else resources
    with _LIFECYCLE_STATES_LOCK:
        existing: Final = _LIFECYCLE_STATES.get(key)
        if existing is not None:
            return existing
        created: Final = _LangfuseLifecycleState()
        _LIFECYCLE_STATES[key] = created
        return created


@contextmanager
def lease_langfuse_client(client: Langfuse, renew: Callable[[], Langfuse]) -> Generator[Langfuse]:
    """Hold off cache eviction's teardown of ``client`` while the export inside is in flight.

    Eviction reaches a client the cache handed a callback moments earlier, so closing the SDK client
    and its tracer provider there drops the spans that callback is still writing. The lease protects
    exactly the window it wraps: an eviction arriving inside it is deferred to the last lease exit.
    Taking a lease never blocks. When eviction already claimed ``client`` between the cache lookup
    and this call, the lease is taken on ``renew()``'s fresh client instead and that client is what
    the caller must export through: the registry hands it the live bundle when one remains, where
    it registers as a holder and the reference count degrades the queued teardown to a flush, or a
    fresh bundle once the old one is gone.
    """
    leased, state = _open_lease(client, renew)
    try:
        yield leased
    finally:
        _run_teardowns(state, state.release_lease())


def _open_lease(client: Langfuse, renew: Callable[[], Langfuse]) -> tuple[Langfuse, _LangfuseLifecycleState]:
    """The first of ``client`` then ``renew()``'s clients that is not already retired, with its lease taken."""
    return next(
        (candidate, state)
        for candidate in chain((client,), iter(renew, None))
        if (state := _lifecycle_state(candidate)).open_lease(candidate)
    )


def _run_teardowns(state: _LangfuseLifecycleState, clients: tuple[Langfuse, ...]) -> None:
    """Tear down ``clients``, then whatever eviction queued meanwhile, and hand the flag back.

    A failing ordinary teardown is logged and skipped rather than raised: the thread here is usually a
    request callback that merely held the last lease, and its request must not fail on eviction's behalf.
    An interrupt requeues the unfinished batch for the next eviction or lease exit and propagates.
    """
    batch = clients  # rebind-ok: drains each batch queued while the previous one was being torn down
    try:
        from litellm._logging import verbose_logger

        while batch:
            for index, client in enumerate(batch):
                try:
                    _teardown_langfuse_client(client)
                except Exception:  # noqa: BLE001  # SDK shutdown can raise anything; the request holding the lease must survive it
                    verbose_logger.exception("Langfuse client teardown failed during cache eviction")
                except BaseException:
                    state.requeue(batch[index:])
                    raise
            batch = state.next_teardown_batch()
    finally:
        state.end_teardown()


def _evict_if_stale_locked(
    *, public_key: object, secret_key: object, base_url: object
) -> LangfuseResourceManager | None:
    """Assumes ``LangfuseResourceManager._lock`` is held; returns the still-valid bundle, evicting a stale one."""
    if not public_key:
        return None
    cached: Final = LangfuseResourceManager._instances.get(public_key)  # pyright: ignore[reportPrivateUsage]  # registry has no public accessor
    if cached is None:
        return None
    if getattr(cached, "secret_key", None) == secret_key and getattr(cached, "base_url", None) == base_url:
        return cached
    LangfuseResourceManager._instances.pop(public_key, None)  # pyright: ignore[reportPrivateUsage]  # registry has no public accessor
    return None


def _retire_orphaned_providers() -> None:
    """Shut down every provider litellm built whose bundle nothing uses any more.

    A rotated-out bundle whose last client is simply garbage collected, which is how the
    prompt-management LRU drops clients, never reaches ``shutdown_langfuse_client``, and the
    provider's own atexit hook would keep its export thread alive for the rest of the process.

    Holders are snapshotted last: a client is registered in the same registry-locked block
    that builds its provider, so once the registry snapshot's lock has been acquired, the
    client of any provider from the first snapshot is visible to the final one even when a
    concurrent rotation already evicted its bundle again. Runs outside both locks because
    provider shutdown flushes and joins the export thread.
    """
    with _LIVE_CLIENTS_LOCK:
        candidates: Final = tuple(_litellm_built_providers)
    with LangfuseResourceManager._lock:  # pyright: ignore[reportPrivateUsage]  # registry has no public accessor
        registered: Final = tuple(
            getattr(resources, "tracer_provider", None)
            for resources in LangfuseResourceManager._instances.values()  # pyright: ignore[reportPrivateUsage]  # registry has no public accessor
        )
    with _LIVE_CLIENTS_LOCK:
        held: Final = tuple(
            getattr(resources, "tracer_provider", None)
            for resources, holders in _live_clients.items()
            if len(holders) > 0
        )
    orphaned: Final = tuple(provider for provider in candidates if provider not in registered and provider not in held)
    for provider in orphaned:
        _litellm_built_providers.discard(provider)
        provider.shutdown()


def evict_stale_langfuse_resources(*, public_key: str | None, secret_key: str | None, base_url: str | None) -> None:
    """Drop a cached client whose credentials no longer match the ones being requested."""
    with LangfuseResourceManager._lock:  # pyright: ignore[reportPrivateUsage]  # registry has no public accessor
        _evict_if_stale_locked(public_key=public_key, secret_key=secret_key, base_url=base_url)
    _retire_orphaned_providers()


def _build_span_exporter(*, public_key: object, secret_key: object, base_url: object) -> RetryingSpanExporter:
    """Build the OTLP export channel with litellm's TLS material and v2's retry behaviour.

    v2 ingested through the injected httpx client, which carried litellm's CA
    bundle and client certificate; v4 ships every observation through its own
    OTLP exporter, so a private-CA deployment would fail TLS on every export in
    a background thread while ``auth_check`` (still on the httpx client) stays
    green. Endpoint, headers and timeout mirror ``langfuse._client.span_processor``.
    """
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    import litellm
    from litellm.llms.custom_httpx.http_handler import get_ssl_verify

    ssl_verify: Final = get_ssl_verify()
    ca_bundle: Final = ssl_verify if isinstance(ssl_verify, str) and os.path.exists(ssl_verify) else None
    configured_certificate: Final = os.getenv("SSL_CERTIFICATE") or litellm.ssl_certificate
    client_certificate: Final = configured_certificate if isinstance(configured_certificate, str) else None
    export_path: Final = os.getenv("LANGFUSE_OTEL_TRACES_EXPORT_PATH") or "/api/public/otel/v1/traces"
    endpoint: Final = f"{str(base_url).rstrip('/')}/{export_path.lstrip('/')}"
    encoded_auth: Final = b64encode(f"{public_key}:{secret_key}".encode()).decode("ascii")
    exporter: Final = OTLPSpanExporter(
        endpoint=endpoint,
        headers={  # mutable-ok: the exporter copies these into its session headers
            "Authorization": "Basic " + encoded_auth,
            "x-langfuse-sdk-name": "python",
            "x-langfuse-sdk-version": version("langfuse"),
            "x-langfuse-public-key": str(public_key),
        },
        timeout=int(os.getenv("LANGFUSE_TIMEOUT", "5")),
        certificate_file=ca_bundle,
        client_certificate_file=client_certificate,
    )
    if ssl_verify is False:
        exporter._certificate_file = False  # pyright: ignore[reportPrivateUsage]  # the ctor coerces a False certificate_file back to True
    return RetryingSpanExporter(exporter)


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
        else _build_span_exporter(
            public_key=public_key,
            secret_key=parameters.get("secret_key"),
            base_url=parameters.get("base_url"),
        )
    )
    sample_rate: Final = configured_sample_rate()
    with LangfuseResourceManager._lock:  # pyright: ignore[reportPrivateUsage]  # registry has no public accessor
        cached: Final = _evict_if_stale_locked(
            public_key=public_key,
            secret_key=parameters.get("secret_key"),
            base_url=parameters.get("base_url"),
        )
        client: Final = Langfuse(
            **parameters,  # pyright: ignore[reportArgumentType]  # kwargs-ok: dict mirrors the typed ctor, values resolved by the callers
            sample_rate=sample_rate,
            tracer_provider=None
            if cached is not None
            else build_isolated_tracer_provider(environment=environment, release=release, sample_rate=sample_rate),
            span_exporter=span_exporter,
        )
        register_langfuse_client(client)
    _retire_orphaned_providers()
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

    A callback holding a lease on the client's bundle postpones all of this to
    the moment that lease ends, so eviction cannot close the provider out from
    under an export the lease is wrapping. See ``lease_langfuse_client``.
    """
    state: Final = _lifecycle_state(client)
    if not state.claim_for_teardown(client):
        return
    _run_teardowns(state, (client,))


def _teardown_langfuse_client(client: Langfuse) -> None:
    """The blocking teardown behind ``shutdown_langfuse_client``.

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
        _litellm_built_providers.discard(provider)
        provider.shutdown()
    _retire_orphaned_providers()
