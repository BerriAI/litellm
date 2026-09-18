from __future__ import annotations

import os
import re
import threading
from base64 import b64encode
from collections.abc import Iterable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from importlib.metadata import version
from itertools import chain
from time import monotonic, sleep
from types import MappingProxyType
from typing import Final, Literal

import httpx
import opentelemetry.trace as otel_trace
from langfuse import LangfuseOtelSpanAttributes
from langfuse.api import LangfuseAPI, Prompt, Prompt_Chat
from langfuse.model import BasePromptClient, ChatPromptClient, PromptClient, TextPromptClient
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
from opentelemetry.sdk.trace.sampling import ALWAYS_ON, Decision, Sampler, SamplingResult
from opentelemetry.trace import Link, NonRecordingSpan, Span, SpanContext, SpanKind, TraceFlags, Tracer, TraceState
from opentelemetry.util.types import Attributes, AttributeValue
from requests import PreparedRequest, RequestException, Response, Session
from requests.adapters import HTTPAdapter

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

__all__ = (
    "DiscardingSpanExporter",
    "LangfuseApiClient",
    "LangfuseObservation",
    "LangfuseTracing",
    "RetryingSpanExporter",
    "TraceIdHashSampler",
    "acquire_langfuse_tracing",
    "build_langfuse_client",
    "build_langfuse_tracing",
    "configured_flush_at",
    "configured_sample_rate",
    "flush_langfuse_tracing",
    "observation_attributes",
    "resolve_observation_id",
    "resolve_trace_id",
    "start_child_span",
    "start_generation",
    "to_unix_nanos",
    "trace_attributes",
)

_TRACE_ID_PATTERN: Final = re.compile(r"^(?=.*[1-9a-f])[0-9a-f]{32}$")
_OBSERVATION_ID_PATTERN: Final = re.compile(r"^(?=.*[1-9a-f])[0-9a-f]{16}$")
_TRACER_NAME: Final = "langfuse-sdk"
_MAX_QUEUE_SIZE: Final = 100_000
_DEFAULT_FLUSH_AT: Final = 512
_SPAN_LIMITS: Final = SpanLimits(
    max_attributes=SpanLimits.UNSET,
    max_events=128,
    max_links=128,
    max_span_attributes=SpanLimits.UNSET,
    max_event_attributes=128,
    max_link_attributes=128,
    max_attribute_length=SpanLimits.UNSET,
    max_span_attribute_length=SpanLimits.UNSET,
)


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
    if not serialized:
        return format(RandomIdGenerator().generate_trace_id(), "032x")
    return sha256(serialized.encode("utf-8")).digest()[:16].hex()


def resolve_observation_id(observation_id: object | None) -> str | None:
    """Map a caller's parent observation id onto v4's 16 lowercase hex characters."""
    serialized: Final = "" if observation_id is None else str(observation_id)
    normalized: Final = serialized.lower().replace("-", "")
    if _OBSERVATION_ID_PATTERN.fullmatch(normalized):
        return normalized
    if not serialized:
        return None
    return sha256(serialized.encode("utf-8")).digest()[:8].hex()


def _serialize(value: object) -> str | None:
    return value if value is None or isinstance(value, str) else safe_dumps(value)


def _string_or_none(value: object) -> str | None:
    return None if value is None else str(value)


def _serialize_datetime(value: object) -> str | None:
    """A datetime the way the SDK's ``EventSerializer`` sends one: a JSON string, naive values read as local time."""
    if isinstance(value, datetime):
        return safe_dumps(value.astimezone().isoformat())
    return _serialize(value)


def _strings(items: Iterable[object]) -> tuple[str, ...]:
    return tuple(str(item) for item in items)


def _string_sequence(value: object) -> Sequence[str] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set, frozenset)):
        return _strings(value) or None
    return (str(value),)


def _present(entries: Iterable[tuple[str, AttributeValue | None]]) -> Mapping[str, AttributeValue]:
    return MappingProxyType({key: value for key, value in entries if value is not None})


def _flattened_metadata(prefix: str, metadata: object) -> Mapping[str, AttributeValue]:
    """Mirror the SDK's wire shape: one ``<prefix>.<key>`` attribute per key, or ``<prefix>`` for a non-dict."""
    if metadata is None:
        return _present(())
    if not isinstance(metadata, Mapping):
        return _present(((prefix, _serialize(metadata)),))
    return _present(
        (f"{prefix}.{key}", value if isinstance(value, (str, int)) else _serialize(value))
        for key, value in metadata.items()
    )


def trace_attributes(
    *,
    name: object = None,
    user_id: object = None,
    session_id: object = None,
    version: object = None,
    release: object = None,
    tags: object = None,
    metadata: object = None,
    public: bool | None = None,
    input: object = None,
    output: object = None,
) -> Mapping[str, AttributeValue]:
    """Trace-level fields ride on an observation's span as ``langfuse.trace.*`` style attributes in v4.

    On the root observation they define the trace; on a continuation they update it, which is
    how v2's ``trace(...)`` and ``update_trace_keys`` contracts map onto the OTLP ingestion.
    """
    scalar: Final[tuple[tuple[str, str | bool | None], ...]] = (
        (LangfuseOtelSpanAttributes.TRACE_NAME, _string_or_none(name)),
        (LangfuseOtelSpanAttributes.TRACE_USER_ID, _string_or_none(user_id)),
        (LangfuseOtelSpanAttributes.TRACE_SESSION_ID, _string_or_none(session_id)),
        (LangfuseOtelSpanAttributes.VERSION, _string_or_none(version)),
        (LangfuseOtelSpanAttributes.RELEASE, _string_or_none(release)),
        (LangfuseOtelSpanAttributes.TRACE_PUBLIC, public),
        (LangfuseOtelSpanAttributes.TRACE_INPUT, _serialize(input)),
        (LangfuseOtelSpanAttributes.TRACE_OUTPUT, _serialize(output)),
    )
    tags_entry: Final[tuple[str, Sequence[str] | None]] = (
        LangfuseOtelSpanAttributes.TRACE_TAGS,
        _string_sequence(tags),
    )
    return _present(
        chain(scalar, (tags_entry,), _flattened_metadata(LangfuseOtelSpanAttributes.TRACE_METADATA, metadata).items())
    )


def observation_attributes(
    *,
    observation_type: Literal["generation", "span"],
    input: object = None,
    output: object = None,
    metadata: object = None,
    level: object = None,
    status_message: object = None,
    version: object = None,
    model: object = None,
    model_parameters: object = None,
    usage_details: object = None,
    cost_details: object = None,
    completion_start_time: object = None,
    prompt: object = None,
) -> Mapping[str, AttributeValue]:
    """The observation's own fields, serialized the way the SDK's ``create_generation_attributes`` does.

    ``prompt`` links the generation to a managed prompt only when it is a real prompt client;
    v2 dropped anything else, and a fallback prompt has no server-side version to link.
    """
    linked_prompt: Final = prompt if isinstance(prompt, BasePromptClient) and not prompt.is_fallback else None
    scalar: Final[tuple[tuple[str, str | int | None], ...]] = (
        (LangfuseOtelSpanAttributes.OBSERVATION_TYPE, observation_type),
        (LangfuseOtelSpanAttributes.OBSERVATION_LEVEL, _string_or_none(level)),
        (LangfuseOtelSpanAttributes.OBSERVATION_STATUS_MESSAGE, _string_or_none(status_message)),
        (LangfuseOtelSpanAttributes.VERSION, _string_or_none(version)),
        (LangfuseOtelSpanAttributes.OBSERVATION_INPUT, _serialize(input)),
        (LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT, _serialize(output)),
        (LangfuseOtelSpanAttributes.OBSERVATION_MODEL, _string_or_none(model)),
        (LangfuseOtelSpanAttributes.OBSERVATION_MODEL_PARAMETERS, _serialize(model_parameters)),
        (LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS, _serialize(usage_details)),
        (LangfuseOtelSpanAttributes.OBSERVATION_COST_DETAILS, _serialize(cost_details)),
        (LangfuseOtelSpanAttributes.OBSERVATION_COMPLETION_START_TIME, _serialize_datetime(completion_start_time)),
        (LangfuseOtelSpanAttributes.OBSERVATION_PROMPT_NAME, linked_prompt.name if linked_prompt else None),
        (LangfuseOtelSpanAttributes.OBSERVATION_PROMPT_VERSION, linked_prompt.version if linked_prompt else None),
    )
    return _present(
        chain(scalar, _flattened_metadata(LangfuseOtelSpanAttributes.OBSERVATION_METADATA, metadata).items())
    )


@dataclass(frozen=True, slots=True)
class LangfuseObservation:
    """A Langfuse observation as the OTel span litellm exports for it."""

    span: Span
    public: bool | None

    @property
    def id(self) -> str:
        return format(self.span.get_span_context().span_id, "016x")

    @property
    def trace_id(self) -> str:
        return format(self.span.get_span_context().trace_id, "032x")

    def end(self, end_time: datetime | float | None = None) -> None:
        self.span.end(end_time=to_unix_nanos(end_time))


_requested_trace_id: Final[ContextVar[int | None]] = ContextVar("litellm_langfuse_requested_trace_id", default=None)
_requested_span_id: Final[ContextVar[int | None]] = ContextVar("litellm_langfuse_requested_span_id", default=None)


class _RequestedIdGenerator(RandomIdGenerator):
    """Hand out the ids the calling context asked for, random otherwise.

    v2 took caller trace and generation ids as plain fields; OTel derives both from
    the tracer's id generator, so the request rides on a context variable instead.
    """

    def generate_trace_id(self) -> int:
        requested: Final = _requested_trace_id.get()
        return super().generate_trace_id() if requested is None else requested

    def generate_span_id(self) -> int:
        requested: Final = _requested_span_id.get()
        return super().generate_span_id() if requested is None else requested


def _parent_context(*, trace_id: str, parent_observation_id: str | None, existing_trace: bool) -> Context:
    """Where a new observation hangs: nowhere for a fresh trace, under a remote parent when continuing one.

    ``existing_trace`` is the v2 ``existing_trace_id`` contract: the trace is appended to, never
    rewritten. The server takes a root observation's name and I/O as the trace's, so a continuation
    without a known parent hangs under a parent id that is never exported instead of claiming root.
    An explicitly empty context also keeps the caller's own active span out of the picture.
    """
    if parent_observation_id is None and not existing_trace:
        return Context()
    parent_span_id: Final = (
        int(parent_observation_id, 16) if parent_observation_id is not None else RandomIdGenerator().generate_span_id()
    )
    remote_parent: Final = NonRecordingSpan(
        SpanContext(
            trace_id=int(trace_id, 16),
            span_id=parent_span_id,
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
    )
    return otel_trace.set_span_in_context(remote_parent)


def _start_span(
    tracer: Tracer,
    *,
    name: str,
    context: Context,
    start_time: datetime | float | None,
    trace_id: str | None,
    observation_id: str | None,
    attributes: Mapping[str, AttributeValue],
) -> Span:
    trace_token: Final = _requested_trace_id.set(int(trace_id, 16) if trace_id is not None else None)
    span_token: Final = _requested_span_id.set(int(observation_id, 16) if observation_id is not None else None)
    try:
        return tracer.start_span(
            name=name, context=context, start_time=to_unix_nanos(start_time), attributes=attributes
        )
    finally:
        _requested_span_id.reset(span_token)
        _requested_trace_id.reset(trace_token)


def start_generation(
    *,
    tracing: LangfuseTracing,
    trace_id: str,
    parent_observation_id: str | None,
    existing_trace: bool,
    observation_id: str | None,
    name: str,
    start_time: datetime | float | None,
    public: bool | None,
    attributes: Mapping[str, AttributeValue],
) -> LangfuseObservation:
    """Create the generation for one model call, timed from when that call began.

    ``trace_id``, ``parent_observation_id`` and ``observation_id`` are the v2 ``trace(id=...)``,
    ``generation(parent_observation_id=...)`` and ``generation(id=...)`` arguments, already
    normalized by ``resolve_trace_id`` and ``resolve_observation_id``.
    """
    span: Final = _start_span(
        tracing.tracer,
        name=name,
        context=_parent_context(
            trace_id=trace_id, parent_observation_id=parent_observation_id, existing_trace=existing_trace
        ),
        start_time=start_time,
        trace_id=trace_id,
        observation_id=observation_id,
        attributes=attributes,
    )
    return LangfuseObservation(span=span, public=public)


def start_child_span(
    *,
    tracing: LangfuseTracing,
    parent: LangfuseObservation,
    name: str,
    start_time: datetime | float | None,
    attributes: Mapping[str, AttributeValue],
) -> LangfuseObservation:
    """Create an observation under the generation, keeping its own time window.

    The server folds the trace's ``public`` flag across every observation, with a missing
    attribute read as ``False``, so the child repeats the generation's value.
    """
    public_entry: Final[tuple[str, bool | None]] = (LangfuseOtelSpanAttributes.TRACE_PUBLIC, parent.public)
    span: Final = _start_span(
        tracing.tracer,
        name=name,
        context=otel_trace.set_span_in_context(parent.span),
        start_time=start_time,
        trace_id=None,
        observation_id=None,
        attributes=_present(chain((public_entry,), attributes.items())),
    )
    return LangfuseObservation(span=span, public=parent.public)


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


def configured_flush_at() -> int:
    """``LANGFUSE_FLUSH_AT`` as the export batch size, the SDK's own knob, with its default when unset or unusable."""
    raw: Final = os.environ.get("LANGFUSE_FLUSH_AT")
    if raw is None:
        return _DEFAULT_FLUSH_AT
    parsed: Final = int(raw) if raw.strip().isdigit() else None
    if parsed is None or not 0 < parsed <= _MAX_QUEUE_SIZE:
        verbose_logger.warning(
            "LANGFUSE_FLUSH_AT=%r is not a whole number between 1 and %d; exporting batches of %d",
            raw,
            _MAX_QUEUE_SIZE,
            _DEFAULT_FLUSH_AT,
        )
        return _DEFAULT_FLUSH_AT
    return parsed


class DiscardingSpanExporter(SpanExporter):
    """Accept and drop every span, for mock mode.

    The mock intercepts the httpx client behind the REST API, but observations
    travel over OTLP, so without this the "no network calls" contract silently sends
    real traces to the configured host.
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
        for delay in self.delays:
            try:
                return self.exporter.export(spans)
            except RequestException as error:
                verbose_logger.warning("Langfuse export raised %s, retrying in %ss", error, delay)
                sleep(delay)
        try:
            return self.exporter.export(spans)
        except RequestException as error:
            verbose_logger.error("Langfuse export failed after %d retries: %s", len(self.delays), error)
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        self.exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self.exporter.force_flush(timeout_millis)


class _UnverifiedTlsAdapter(HTTPAdapter):
    """Honour ``ssl_verify=False``: the exporter passes ``verify`` per request, which outranks ``Session.verify``."""

    def send(  # pyright: ignore[reportIncompatibleMethodOverride]  # the stub types verify as bool | str, the base accepts both
        self,
        request: PreparedRequest,
        stream: bool = False,
        timeout: float | tuple[float, float] | tuple[float, None] | None = None,
        verify: bool | str = True,
        cert: str | tuple[str, str] | None = None,
        proxies: Mapping[str, str] | None = None,
    ) -> Response:
        return super().send(request, stream=stream, timeout=timeout, verify=False, cert=cert, proxies=proxies)


def _build_span_exporter(*, public_key: str, secret_key: str, base_url: str) -> RetryingSpanExporter:
    """Build the OTLP export channel with litellm's TLS material and v2's retry behaviour.

    v2 ingested through the injected httpx client, which carried litellm's CA
    bundle and client certificate. Endpoint, headers and timeout mirror the SDK's
    own span processor so the server treats the spans as v4 SDK traffic.
    """
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    import litellm
    from litellm.llms.custom_httpx.http_handler import get_ssl_verify

    ssl_verify: Final = get_ssl_verify()
    ca_bundle: Final = ssl_verify if isinstance(ssl_verify, str) and os.path.exists(ssl_verify) else None
    configured_certificate: Final = os.getenv("SSL_CERTIFICATE") or litellm.ssl_certificate
    client_certificate: Final = configured_certificate if isinstance(configured_certificate, str) else None
    export_path: Final = os.getenv("LANGFUSE_OTEL_TRACES_EXPORT_PATH") or "/api/public/otel/v1/traces"
    endpoint: Final = f"{base_url.rstrip('/')}/{export_path.lstrip('/')}"
    encoded_auth: Final = b64encode(f"{public_key}:{secret_key}".encode()).decode("ascii")
    session: Final = Session()
    if ssl_verify is False:
        session.mount("https://", _UnverifiedTlsAdapter())
    exporter: Final = OTLPSpanExporter(
        session=session,
        endpoint=endpoint,
        headers={  # mutable-ok: the exporter copies these into its session headers
            "Authorization": "Basic " + encoded_auth,
            "x-langfuse-sdk-name": "python",
            "x-langfuse-sdk-version": version("langfuse"),
            "x-langfuse-public-key": public_key,
        },
        timeout=int(os.getenv("LANGFUSE_TIMEOUT", "5")),
        certificate_file=ca_bundle,
        client_certificate_file=client_certificate,
    )
    return RetryingSpanExporter(exporter)


def _resource(*, environment: str | None, release: str | None) -> Resource:
    return Resource.create(
        _present(
            (
                (LangfuseOtelSpanAttributes.ENVIRONMENT, environment),
                (LangfuseOtelSpanAttributes.RELEASE, release),
            )
        )
    )


@dataclass(frozen=True, slots=True)
class LangfuseTracing:
    """litellm's own export channel to one Langfuse project: a provider, its tracer and the exporter behind them.

    The channel is litellm's rather than the SDK's so that the process-global OTel provider stays
    untouched, historical timestamps and caller ids are honoured, and no SDK internals are needed.
    """

    provider: TracerProvider
    tracer: Tracer

    def flush(self, timeout_millis: int = 30_000) -> bool:
        return self.provider.force_flush(timeout_millis)


@dataclass(frozen=True, slots=True)
class _TracingKey:
    public_key: str
    secret_key: str
    base_url: str
    environment: str | None
    release: str | None
    sample_rate: float
    flush_at: int
    flush_interval_millis: int
    mock_mode: bool


_TRACING_LOCK: Final = threading.Lock()
_TRACING: Final[
    dict[_TracingKey, LangfuseTracing]
] = {}  # mutable-ok: process-wide channel cache, guarded by _TRACING_LOCK


def acquire_langfuse_tracing(
    *,
    public_key: str,
    secret_key: str,
    base_url: str,
    environment: str | None,
    release: str | None,
    flush_interval: float,
    mock_mode: bool,
) -> LangfuseTracing:
    """One export channel per credential set, shared by every logger built for it.

    Channels live for the process: a provider owns a batch export thread, and tearing one down
    while another logger for the same credentials still exports through it would drop its spans.
    """
    key: Final = _TracingKey(
        public_key=public_key,
        secret_key=secret_key,
        base_url=base_url,
        environment=environment,
        release=release,
        sample_rate=configured_sample_rate(),
        flush_at=configured_flush_at(),
        flush_interval_millis=int(flush_interval * 1000),
        mock_mode=mock_mode,
    )
    with _TRACING_LOCK:
        cached: Final = _TRACING.get(key)
        if cached is not None:
            return cached
        created: Final = build_langfuse_tracing(
            exporter=DiscardingSpanExporter()
            if mock_mode
            else _build_span_exporter(public_key=public_key, secret_key=secret_key, base_url=base_url),
            environment=environment,
            release=release,
            sample_rate=key.sample_rate,
            flush_at=key.flush_at,
            flush_interval_millis=key.flush_interval_millis,
        )
        _TRACING[key] = created
        return created


class _FlushWorker(threading.Thread):
    """Daemon, so a channel still blocked at the deadline cannot hold up interpreter exit."""

    def __init__(self, channel: LangfuseTracing, timeout_millis: int) -> None:
        super().__init__(name="langfuse-flush", daemon=True)
        self.channel: Final = channel
        self.timeout_millis: Final = timeout_millis
        self.flushed = False

    def run(self) -> None:
        self.flushed = self.channel.flush(self.timeout_millis)


def flush_langfuse_tracing(timeout_millis: int = 30_000) -> bool:
    """Force-flush every export channel this process acquired, all within one ``timeout_millis`` deadline.

    ``True`` only when every channel flushed in time; a channel still blocked at the deadline is left to
    finish in the background rather than pushing the deadline out for the channels after it.
    """
    with _TRACING_LOCK:
        channels: Final = tuple(_TRACING.values())
    workers: Final = tuple(_FlushWorker(channel, timeout_millis) for channel in channels)
    deadline: Final = monotonic() + timeout_millis / 1000
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(max(0.0, deadline - monotonic()))
    return all(not worker.is_alive() and worker.flushed for worker in workers)


def build_langfuse_tracing(
    *,
    exporter: SpanExporter,
    environment: str | None,
    release: str | None,
    sample_rate: float,
    flush_interval_millis: int,
    flush_at: int = _DEFAULT_FLUSH_AT,
) -> LangfuseTracing:
    """Wire the provider from litellm's own settings so a host's ``OTEL_*`` variables do not steer it.

    An unset sampler or span limit falls back to ``OTEL_TRACES_SAMPLER`` and
    ``OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT`` style variables, which are meant for the
    host application's own tracing. ``OTEL_SDK_DISABLED`` still applies, as it does to the SDK.

    The tracer carries the SDK's scope name because Langfuse keys on it: spans from any other
    scope are treated as foreign OTel traffic and get their raw attributes echoed into metadata.
    """
    if os.environ.get("OTEL_SDK_DISABLED", "").strip().lower() == "true":
        verbose_logger.warning("OTEL_SDK_DISABLED=true also disables the langfuse callback's export channel")
    provider: Final = TracerProvider(
        resource=_resource(environment=environment, release=release),
        sampler=ALWAYS_ON if sample_rate >= 1 else TraceIdHashSampler(sample_rate),
        id_generator=_RequestedIdGenerator(),
        span_limits=_SPAN_LIMITS,
    )
    provider.add_span_processor(
        BatchSpanProcessor(
            exporter,
            max_queue_size=_MAX_QUEUE_SIZE,
            max_export_batch_size=flush_at,
            schedule_delay_millis=flush_interval_millis,
        )
    )
    return LangfuseTracing(provider=provider, tracer=provider.get_tracer(_TRACER_NAME))


@dataclass(frozen=True, slots=True)
class _CachedPrompt:
    prompt: PromptClient
    fetched_at: float


def _prompt_client(prompt: Prompt) -> PromptClient:
    return ChatPromptClient(prompt) if isinstance(prompt, Prompt_Chat) else TextPromptClient(prompt)


class LangfuseApiClient:
    """litellm's handle on one Langfuse project over its REST API: prompts, ``auth_check`` and the project id.

    The SDK's ``Langfuse`` client is deliberately not constructed. It keeps one tracing bundle per
    public key and hands it to every ``Langfuse()`` a host application builds for the same key, so
    litellm's exporter, host and masking would leak into that application. Observations travel
    over ``LangfuseTracing``; nothing here exports spans.

    Prompts are cached for ``LANGFUSE_PROMPT_CACHE_DEFAULT_TTL_SECONDS`` (60 by default) as the SDK
    does, refreshed on the request that finds them stale; a refresh that fails keeps serving the
    stale prompt rather than failing the request, again like the SDK.
    """

    def __init__(self, api: LangfuseAPI, *, prompt_cache_ttl_seconds: float) -> None:
        self.api: Final = api
        self.prompt_cache_ttl_seconds: Final = prompt_cache_ttl_seconds
        self._prompts: Final[dict[str, _CachedPrompt]] = {}  # mutable-ok: per-client prompt cache, guarded by _lock
        self._lock: Final = threading.Lock()

    def auth_check(self) -> bool:
        try:
            self.api.projects.get()
        except Exception:
            return False
        return True

    def project_id(self) -> str | None:
        projects: Final = self.api.projects.get().data
        return projects[0].id if projects else None

    def get_prompt(self, name: str, *, label: str | None = None, version: int | None = None) -> PromptClient:
        key: Final = f"{name}:version:{version}" if version is not None else f"{name}:label:{label}"
        with self._lock:
            cached: Final = self._prompts.get(key)
        if cached is not None and monotonic() - cached.fetched_at < self.prompt_cache_ttl_seconds:
            return cached.prompt
        try:
            fetched: Final = _prompt_client(self.api.prompts.get(name, version=version, label=label))
        except Exception as error:
            if cached is None:
                raise
            verbose_logger.warning("Langfuse prompt %r refresh failed, serving the cached version: %s", name, error)
            return cached.prompt
        with self._lock:
            self._prompts[key] = _CachedPrompt(prompt=fetched, fetched_at=monotonic())
        return fetched


def build_langfuse_client(
    *,
    public_key: str | None,
    secret_key: str | None,
    base_url: str,
    httpx_client: httpx.Client | None,
) -> LangfuseApiClient:
    """The REST client for prompt management, ``auth_check`` and the Slack project link.

    Missing keys are passed through as absent credentials: the server answers 401, which
    ``auth_check`` reports as ``False`` rather than raising at construction.
    """
    return LangfuseApiClient(
        LangfuseAPI(
            base_url=base_url,
            username=public_key,
            password=secret_key,
            x_langfuse_sdk_name="python",
            x_langfuse_sdk_version=version("langfuse"),
            x_langfuse_public_key=public_key,
            httpx_client=httpx_client,
            timeout=int(os.getenv("LANGFUSE_TIMEOUT", "5")),
        ),
        prompt_cache_ttl_seconds=float(os.getenv("LANGFUSE_PROMPT_CACHE_DEFAULT_TTL_SECONDS", "60")),
    )
