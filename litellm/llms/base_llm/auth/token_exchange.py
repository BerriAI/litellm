"""RFC 7523 JWT-bearer token exchange engine, shared across providers.

One sync state machine per process: bounded engine-owned entry map, two-tier
refresh (advisory background refresh + mandatory single-flight), HTTPS pinning,
response caps, and RFC 6749 5.2 redaction. Providers describe a grant profile as
a ``TokenExchangeSpec`` and map the typed ``ExchangeError`` union to their own
public exception contract.
"""

import asyncio
import hashlib
import json
import re
import threading
import time
from collections.abc import Callable, Coroutine, Mapping, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from math import inf
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol, TypeAlias
from urllib.parse import unquote, unquote_plus, urlencode, urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, SecretStr, TypeAdapter, ValidationError
from typing_extensions import assert_never

from litellm._logging import verbose_logger
from litellm.llms.base_llm.auth.types import (
    AssertionReader,
    AssertionSource,
    AssertionSourceError,
    ExchangeCallType,
    ExchangeError,
    ExchangeResult,
    InsecureTokenUrl,
    MalformedTokenResponse,
    MintedToken,
    SyncTokenPoster,
    TokenEndpointError,
    TokenExchangeMetricsSink,
    TokenExchangeSpec,
    TokenTransportError,
)
from litellm.types.services import ServiceTypes

if TYPE_CHECKING:
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

CALL_TYPE_COLD_MINT: Final[ExchangeCallType] = "cold_mint"
CALL_TYPE_MANDATORY_REFRESH: Final[ExchangeCallType] = "mandatory_refresh"
CALL_TYPE_ADVISORY_REFRESH: Final[ExchangeCallType] = "advisory_refresh"
CALL_TYPE_CACHE_HIT: Final = "cache_hit"

ADVISORY_REFRESH_SECONDS: Final = 120.0
MANDATORY_REFRESH_SECONDS: Final = 30.0
ADVISORY_REFRESH_LIFETIME_FRACTION: Final = 0.5
MANDATORY_REFRESH_LIFETIME_FRACTION: Final = 0.125
ADVISORY_REFRESH_BACKOFF_SECONDS: Final = 5.0
FALLBACK_TOKEN_TTL_SECONDS: Final = 60.0
# Metrics are best-effort, so the backlog is capped and further events are dropped. Request volume
# must not be able to grow this queue without bound when a telemetry backend stalls.
_METRICS_QUEUE_LIMIT: Final = 1000
MAX_ASSERTION_BYTES: Final = 16 * 1024
MAX_RESPONSE_BYTES: Final = 1024 * 1024

_REDACTION_CAP: Final = 256
_FOLLOWER_WAIT_GRACE_SECONDS: Final = 5.0
_LOCAL_HOSTS: Final = frozenset({"localhost", "127.0.0.1", "::1"})
_OAUTH_ERROR_FIELDS: Final = ("error", "error_description", "error_uri")
_NESTED_ERROR_FIELDS: Final = ("type", "message")
_CONTENT_TYPES: Final = MappingProxyType({"json": "application/json", "form": "application/x-www-form-urlencoded"})
_OVERSIZED_BODY_MESSAGE: Final = "oversized error response omitted"
_NON_OBJECT_BODY_MESSAGE: Final = "non-object error response omitted"
_NO_OAUTH_FIELDS_MESSAGE: Final = "error response carried no RFC 6749 fields"
_UNSTRUCTURED_BODY_MESSAGE: Final = "non-JSON error response omitted"
_REFLECTED_VALUE_MESSAGE: Final = "<redacted: response echoed the request>"
# A credential fragment shorter than this is not worth the false positives; longer, and a run
# shared with the assertion is reflection rather than coincidence.
_REFLECTION_MIN_RUN: Final = 8
# Everything a base64url credential is NOT made of, stripped so a fragment split by delimiters
# still lines up against the assertion.
_CREDENTIAL_CHARS: Final = re.compile(r"[^A-Za-z0-9._~+/=-]")
_SENTINEL_BODY_MESSAGES: Final = frozenset({_OVERSIZED_BODY_MESSAGE, _NON_OBJECT_BODY_MESSAGE})


class _TokenExchangeResponse(BaseModel):
    access_token: str
    expires_in: int | None = None
    token_type: str | None = None


_RedactableBody: TypeAlias = Mapping[str, object] | list[object] | str | int | float | bool | None
_REDACTABLE_BODY_ADAPTER: Final = TypeAdapter[_RedactableBody](_RedactableBody)


def endpoint_url_for_error_message(url: str) -> str:
    """``url`` reduced to scheme, host and path for operator-facing errors.

    A token endpoint is configuration, not a secret, and naming it is what makes these errors
    actionable. But nothing stops an operator writing a credential into it, as a query parameter
    or as userinfo, and these errors reach model callers, so neither part is echoed.
    """
    parsed: Final = urlsplit(url)
    host: Final = parsed.hostname or ""
    authority: Final = f"{host}:{parsed.port}" if parsed.port is not None else host
    return urlunsplit((parsed.scheme, authority, parsed.path, "", ""))


def validate_token_endpoint_url(url: str) -> str | InsecureTokenUrl:
    parsed: Final = urlsplit(url)
    if parsed.scheme == "https":
        return url
    if parsed.scheme == "http" and (parsed.hostname or "") in _LOCAL_HOSTS:
        return url
    return InsecureTokenUrl(host=parsed.hostname or "")


def redact_oauth_error_body(
    status_code: int,
    body_text: str,
    assertion: SecretStr | Sequence[SecretStr] | None = None,
) -> TokenEndpointError:
    """``assertion`` may be every form of the credential that went out on the wire.

    A grant that encodes its credential before sending it (``client_secret_basic`` base64s
    ``id:secret``) can have that encoded form echoed back, and it decodes straight to the secret,
    so checking only the raw value lets reversible material through.
    """
    rendered: Final = _redact_body_text(body_text)
    secrets: Final = () if assertion is None else (assertion,) if isinstance(assertion, SecretStr) else tuple(assertion)
    redacted: Final = next(
        (
            _REFLECTED_VALUE_MESSAGE
            for secret in secrets
            if _drop_reflected_assertion(rendered, secret) is _REFLECTED_VALUE_MESSAGE
        ),
        rendered,
    )
    return TokenEndpointError(status_code=status_code, redacted_body=redacted)


def _drop_reflected_assertion(rendered: str, assertion: SecretStr | None) -> str:
    """Catches an endpoint that echoes the submitted credential back, verbatim or in fragments,
    however it split or percent-encoded it.

    Both sides are reduced to the characters a credential is made of before comparison. Stripping
    only the rendered side would stop matching a secret that carries spaces or punctuation of its
    own, which is exactly the hand-set passphrase most at risk of being echoed.

    This stops an accidental or naive echo. It cannot stop an endpoint that deliberately re-encodes
    or interleaves the credential, and it is not what keeps the credential from the endpoint, which
    already holds it. What it protects is blast radius: keeping the value out of the caller's error
    and out of third-party log sinks.
    """
    if assertion is None:
        return rendered
    secret: Final = assertion.get_secret_value()
    if not secret:
        return rendered
    if secret in rendered:
        return _REFLECTED_VALUE_MESSAGE
    compacted_secret: Final = _CREDENTIAL_CHARS.sub("", secret)
    if not compacted_secret:
        return rendered
    return _REFLECTED_VALUE_MESSAGE if _shares_a_credential_run(rendered, compacted_secret) else rendered


def _shares_a_credential_run(rendered: str, compacted_secret: str) -> bool:
    """``unquote`` covers a credential sent form-encoded, without every caller enumerating that
    shape for itself: percent-escaping is reversible and applies to any field, query string
    included.

    A secret shorter than the probe run is compared whole: a window longer than the secret can
    never be found inside it, which would leave a short client secret unprotected in every shape
    but the verbatim one.
    """
    # unquote covers %XX; unquote_plus additionally covers the "+" a form-encoded body uses for a
    # space. Both are kept rather than only the wider one, because "+" is a base64 character and
    # decoding it away would lose a run that the undecoded candidate still matches on.
    run: Final = min(_REFLECTION_MIN_RUN, len(compacted_secret))
    compacted_candidates: Final = tuple(
        _CREDENTIAL_CHARS.sub("", candidate) for candidate in (rendered, unquote(rendered), unquote_plus(rendered))
    )
    return any(
        compacted[start : start + run] in compacted_secret
        for compacted in compacted_candidates
        for start in range(len(compacted) - run + 1)
    )


def _redact_body_text(body_text: str) -> str:
    if body_text in _SENTINEL_BODY_MESSAGES:
        return body_text
    if len(body_text) > MAX_RESPONSE_BYTES:
        return _OVERSIZED_BODY_MESSAGE
    try:
        parsed: Final = _REDACTABLE_BODY_ADAPTER.validate_json(body_text)
    except ValidationError:
        return _UNSTRUCTURED_BODY_MESSAGE
    match parsed:
        case Mapping():
            return _format_oauth_error_fields(parsed)
        case _:
            return _NON_OBJECT_BODY_MESSAGE


def _format_oauth_error_fields(body: Mapping[str, object]) -> str:
    fields: Final = tuple(
        f"{name}: {_format_oauth_error_value(value)}"
        for name in _OAUTH_ERROR_FIELDS
        for value in (body.get(name),)
        if value is not None
    )
    return "; ".join(fields) if fields else _NO_OAUTH_FIELDS_MESSAGE


def _format_oauth_error_value(value: object) -> str:
    """RFC 6749 types ``error`` as a string, but Anthropic (and other providers) nest their
    own ``{"type": ..., "message": ...}`` envelope there; render that rather than a dict repr."""
    if isinstance(value, Mapping):
        nested: Final = tuple(
            f"{str(part)[:_REDACTION_CAP]}"
            for key in _NESTED_ERROR_FIELDS
            for part in (value.get(key),)
            if part is not None
        )
        if nested:
            return " - ".join(nested)
    return str(value)[:_REDACTION_CAP]


def _error_summary(error: ExchangeError) -> str:
    match error:
        case AssertionSourceError():
            return f"AssertionSourceError: assertion {error.kind} from {error.source_ref}"
        case InsecureTokenUrl():
            return f"InsecureTokenUrl: insecure token endpoint host {error.host}"
        case TokenEndpointError():
            return f"TokenEndpointError: HTTP {error.status_code}: {error.redacted_body}"
        case TokenTransportError():
            return f"TokenTransportError: {error.detail}"
        case MalformedTokenResponse():
            return f"MalformedTokenResponse: {error.detail}"
        case _:
            assert_never(error)


class _MetricsFailure(Exception):
    """Never raised: typed carriers handed to the service failure hook so the prometheus
    ``error_class`` label names the ``ExchangeError`` variant; the message is the redacted
    ``_error_summary`` and carries no credential material."""


class TokenExchangeAssertionSourceFailure(_MetricsFailure): ...


class TokenExchangeInsecureUrlFailure(_MetricsFailure): ...


class TokenExchangeEndpointFailure(_MetricsFailure): ...


class TokenExchangeTransportFailure(_MetricsFailure): ...


class TokenExchangeMalformedResponseFailure(_MetricsFailure): ...


def _failure_exception(error: ExchangeError) -> _MetricsFailure:
    summary: Final = _error_summary(error)
    match error:
        case AssertionSourceError():
            return TokenExchangeAssertionSourceFailure(summary)
        case InsecureTokenUrl():
            return TokenExchangeInsecureUrlFailure(summary)
        case TokenEndpointError():
            return TokenExchangeEndpointFailure(summary)
        case TokenTransportError():
            return TokenExchangeTransportFailure(summary)
        case MalformedTokenResponse():
            return TokenExchangeMalformedResponseFailure(summary)
        case _:
            assert_never(error)


def _cache_key(spec: TokenExchangeSpec) -> str:
    return hashlib.sha256(
        "\x1f".join((spec.token_url, spec.assertion_ref, *spec.cache_key_identity)).encode()
    ).hexdigest()


def _assertion_fetch(reader: AssertionReader, spec: TokenExchangeSpec) -> AssertionSource:
    """``spec.assertion_source`` (an identity source's own fetch/mint closure) takes priority over
    the engine-level reader when set; either way, failures are reported against ``spec.assertion_ref``."""
    if spec.assertion_source is not None:
        return spec.assertion_source
    return lambda: reader(spec.assertion_ref)


def _read_assertion(fetch: AssertionSource, ref: str) -> SecretStr | AssertionSourceError:
    from litellm.secret_managers.main import OidcPathNotAllowedError

    try:
        raw: Final = fetch()
    except OidcPathNotAllowedError:
        return AssertionSourceError(kind="disallowed_path", source_ref=ref)
    except ValueError as e:
        return AssertionSourceError(kind="unreadable", source_ref=ref, detail=str(e)[:_REDACTION_CAP])
    except Exception:  # noqa: BLE001  # injected readers (secret managers) raise arbitrarily; all failures become values
        return AssertionSourceError(kind="unreadable", source_ref=ref)
    if raw is None:
        return AssertionSourceError(kind="missing", source_ref=ref)
    stripped: Final = raw.strip()
    if not stripped:
        return AssertionSourceError(kind="empty", source_ref=ref)
    if len(stripped.encode("utf-8")) > MAX_ASSERTION_BYTES:
        return AssertionSourceError(kind="oversized", source_ref=ref)
    return SecretStr(stripped)


def _serialize_body(spec: TokenExchangeSpec, assertion: SecretStr) -> bytes:
    if spec.body_encoding == "json":
        return json.dumps(
            {  # mutable-ok: transient body dict consumed inline by the serializer
                **spec.static_body,
                spec.assertion_field: assertion.get_secret_value(),
            }
        ).encode()
    return urlencode(
        {  # mutable-ok: transient body dict consumed inline by the serializer
            **spec.static_body,
            spec.assertion_field: assertion.get_secret_value(),
        }
    ).encode()


def _sanitize_expires_in(expires_in: int | None) -> float:
    if expires_in is None or expires_in <= 0:
        return FALLBACK_TOKEN_TTL_SECONDS
    return float(expires_in)


@dataclass(frozen=True, slots=True)
class _RefreshWindows:
    advisory: float
    mandatory: float


def _refresh_windows(lifetime_seconds: float | None) -> _RefreshWindows:
    """A token whose whole life is shorter than the flat windows sits inside them from the moment it
    is minted, so every request would arm another background exchange against the token endpoint.
    Scaling each window by a fraction of the observed lifetime makes a 60s token refresh around its
    half life instead; at a lifetime of 240s and above both fractions reach the flat windows, so
    ordinary long-lived tokens keep exactly the 120s/30s behaviour."""
    if lifetime_seconds is None or lifetime_seconds <= 0.0:
        return _RefreshWindows(advisory=ADVISORY_REFRESH_SECONDS, mandatory=MANDATORY_REFRESH_SECONDS)
    return _RefreshWindows(
        advisory=min(ADVISORY_REFRESH_SECONDS, lifetime_seconds * ADVISORY_REFRESH_LIFETIME_FRACTION),
        mandatory=min(MANDATORY_REFRESH_SECONDS, lifetime_seconds * MANDATORY_REFRESH_LIFETIME_FRACTION),
    )


def _capped_body_text(response: httpx.Response) -> str:
    if len(response.content) > MAX_RESPONSE_BYTES:
        return _OVERSIZED_BODY_MESSAGE
    return response.text


def _default_assertion_reader(ref: str) -> str | None:
    from litellm.secret_managers.main import get_secret_str

    return get_secret_str(ref)


def _new_exchange_handler() -> "HTTPHandler":
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    handler: Final = HTTPHandler(timeout=httpx.Timeout(timeout=30.0, connect=5.0))
    handler.client.follow_redirects = False
    return handler


class _HttpxSyncTokenPoster:
    """Default poster: a dedicated HTTPHandler (no logging_obj, so litellm's
    pre/post-call body logging never sees the exchange POST); returns the
    response for any status."""

    def __init__(self, handler_factory: Callable[[], "HTTPHandler"] = _new_exchange_handler) -> None:
        self._lock: Final = threading.Lock()
        self._handler_factory: Final = handler_factory
        self._handler: HTTPHandler | None = None

    def _handler_instance(self) -> "HTTPHandler":
        with self._lock:
            if self._handler is None:
                self._handler = self._handler_factory()
            return self._handler

    def post(self, url: str, *, content: bytes, headers: Mapping[str, str], timeout: float) -> httpx.Response:
        try:
            response: Final[httpx.Response | None] = self._handler_instance().post(  # pyright: ignore[reportUnknownMemberType]  # HTTPHandler.post is legacy-untyped; the result is validated below
                url,
                content=content,
                headers=dict(headers),  # mutable-ok: HTTPHandler.post requires a concrete dict
                timeout=timeout,
            )
        except httpx.HTTPStatusError as e:
            return e.response
        if response is None:
            raise httpx.TransportError("token endpoint returned no response")
        return response


class _ServiceLoggingHooks(Protocol):
    """The slice of ``litellm._service_logger.ServiceLogging`` the metrics sink calls; a protocol
    so tests inject a recorder instead of monkeypatching."""

    async def async_service_success_hook(self, service: ServiceTypes, call_type: str, duration: float) -> None: ...

    async def async_service_failure_hook(
        self, service: ServiceTypes, duration: float, error: str | Exception, call_type: str
    ) -> None: ...


_HooksCoroFactory: TypeAlias = Callable[
    [_ServiceLoggingHooks],  # mutable-ok: Callable param-list syntax, not a list
    Coroutine[object, object, None],
]


def _default_service_logging() -> _ServiceLoggingHooks:
    from litellm._service_logger import ServiceLogging

    return ServiceLogging()


class ServiceLoggingMetricsSink:
    """Default sink: bridges engine metrics onto litellm's ServiceTypes pattern
    (prometheus ``litellm_anthropic_wif_*`` via ``service_callback``). The engine's entry points
    are sync threads with no event loop, and the service hooks are async, so every emission is
    fire-and-forget on a dedicated single worker thread that owns its own short-lived loop --
    the mint path only ever pays for an executor queue put."""

    def __init__(
        self,
        service_logging_factory: Callable[[], _ServiceLoggingHooks] = _default_service_logging,
        executor: Executor | None = None,
    ) -> None:
        self._lock: Final = threading.Lock()
        self._service_logging_factory: Final = service_logging_factory
        self._service_logging: _ServiceLoggingHooks | None = None
        self._executor: Executor | None = executor
        self._queued: int = 0  # rebind-ok: backlog depth, guarded by _lock

    def _service_logging_instance(self) -> _ServiceLoggingHooks:
        with self._lock:
            if self._service_logging is None:
                self._service_logging = self._service_logging_factory()
            return self._service_logging

    def _executor_instance(self) -> Executor:
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="litellm-token-exchange-metrics")
            return self._executor

    def _emit(self, coro_factory: _HooksCoroFactory) -> None:
        try:
            asyncio.run(coro_factory(self._service_logging_instance()))
        except Exception as e:  # noqa: BLE001  # metrics are best-effort; emission failures must never surface
            verbose_logger.debug("token exchange metrics emission failed: %s", e)

    def _submit(self, coro_factory: _HooksCoroFactory) -> None:
        """Drop the event rather than queue it once the backlog is full. A stalled telemetry
        backend must not let request volume grow an unbounded queue in the proxy: losing a
        metric sample is always cheaper than losing the process."""
        with self._lock:
            if self._queued >= _METRICS_QUEUE_LIMIT:
                verbose_logger.debug("token exchange metrics queue full, dropping event")
                return
            self._queued += 1
        try:
            self._executor_instance().submit(self._emit_and_release, coro_factory)
        except Exception as e:  # noqa: BLE001  # a rejected submit must not surface to the mint
            with self._lock:
                self._queued -= 1
            verbose_logger.debug("token exchange metrics submit failed: %s", e)

    def _emit_and_release(self, coro_factory: _HooksCoroFactory) -> None:
        try:
            self._emit(coro_factory)
        finally:
            with self._lock:
                self._queued -= 1

    def exchange_success(self, *, call_type: ExchangeCallType, duration_seconds: float) -> None:
        def start(hooks: _ServiceLoggingHooks) -> Coroutine[object, object, None]:
            return hooks.async_service_success_hook(
                service=ServiceTypes.ANTHROPIC_WIF, call_type=call_type, duration=duration_seconds
            )

        self._submit(start)

    def exchange_failure(self, *, call_type: ExchangeCallType, duration_seconds: float, error: ExchangeError) -> None:
        failure: Final = _failure_exception(error)

        def start(hooks: _ServiceLoggingHooks) -> Coroutine[object, object, None]:
            return hooks.async_service_failure_hook(
                service=ServiceTypes.ANTHROPIC_WIF, duration=duration_seconds, error=failure, call_type=call_type
            )

        self._submit(start)

    def cache_hit(self) -> None:
        def start(hooks: _ServiceLoggingHooks) -> Coroutine[object, object, None]:
            return hooks.async_service_success_hook(
                service=ServiceTypes.ANTHROPIC_WIF_CACHE, call_type=CALL_TYPE_CACHE_HIT, duration=0.0
            )

        self._submit(start)


class _Entry:
    """Single-flight state for one cache key; mutable by design, confined to the
    engine, and only ever mutated under the engine lock."""

    __slots__ = ("backoff_until", "done", "force_refresh", "in_flight", "last_error", "lifetime_seconds", "token")

    def __init__(self, force_refresh: bool = False) -> None:
        self.token: MintedToken | None = None
        self.lifetime_seconds: float | None = None
        self.in_flight: bool = False
        self.done: Final = threading.Event()
        self.backoff_until: float = float("-inf")
        self.force_refresh: bool = force_refresh
        self.last_error: ExchangeError | None = None

    def arm(self) -> None:
        self.in_flight = True
        self.last_error = None
        self.done.clear()

    def _store(self, token: MintedToken, now: float) -> None:
        self.token = token
        self.lifetime_seconds = None if token.expires_at is None else max(token.expires_at - now, 0.0)
        self.last_error = None

    def publish(self, result: ExchangeResult, now: float) -> None:
        match result:
            case MintedToken():
                self._store(result, now)
            case _:
                self.last_error = result
                self.backoff_until = now + ADVISORY_REFRESH_BACKOFF_SECONDS
        self.force_refresh = False
        self.in_flight = False
        self.done.set()

    def publish_advisory(self, result: ExchangeResult, now: float) -> None:
        """A failed advisory refresh records only the backoff, never ``last_error``: a follower whose
        cached token expires while this runs must be free to re-lead a fresh mint and recover."""
        match result:
            case MintedToken():
                self._store(result, now)
            case _:
                self.backoff_until = now + ADVISORY_REFRESH_BACKOFF_SECONDS
        self.in_flight = False
        self.done.set()


@dataclass(frozen=True, slots=True)
class _Serve:
    token: MintedToken


@dataclass(frozen=True, slots=True)
class _ServeAndRefresh:
    token: MintedToken


@dataclass(frozen=True, slots=True)
class _Lead:
    call_type: ExchangeCallType


@dataclass(frozen=True, slots=True)
class _Follow:
    pass


@dataclass(frozen=True, slots=True)
class _Fail:
    error: ExchangeError


_Decision: TypeAlias = _Serve | _ServeAndRefresh | _Lead | _Follow | _Fail


@dataclass(frozen=True, slots=True)
class _Unauthorized:
    response: httpx.Response
    assertion: SecretStr


class JwtBearerTokenExchangeEngine:
    def __init__(
        self,
        poster: SyncTokenPoster | None = None,
        assertion_reader: AssertionReader | None = None,
        clock: Callable[[], float] = time.monotonic,
        refresh_executor: Executor | None = None,
        max_entries: int = 64,
        metrics_sink: TokenExchangeMetricsSink | None = None,
    ) -> None:
        self._poster: Final[SyncTokenPoster] = poster if poster is not None else _HttpxSyncTokenPoster()
        self._assertion_reader: Final[AssertionReader] = (
            assertion_reader if assertion_reader is not None else _default_assertion_reader
        )
        self._clock: Final = clock
        self._refresh_executor: Executor | None = refresh_executor
        self._max_entries: Final = max_entries
        self._metrics_sink: Final[TokenExchangeMetricsSink] = (
            metrics_sink if metrics_sink is not None else ServiceLoggingMetricsSink()
        )
        self._lock: Final = threading.Lock()
        self._entries: Final[dict[str, _Entry]] = {}  # mutable-ok: engine-owned map guarded by _lock

    def get_token(self, spec: TokenExchangeSpec) -> ExchangeResult:
        """A follower whose leader published nothing re-classifies rather than recursing, so a
        contended entry cannot grow the stack one frame per failed leader."""
        while True:
            with self._lock:
                entry = self._get_or_create_entry_locked(spec)  # rebind-ok: re-read per follower round
                decision = self._classify_and_arm_locked(entry)  # rebind-ok: re-read per follower round
            match decision:
                case _Serve(token=token):
                    self._report_cache_hit()
                    return token
                case _ServeAndRefresh(token=token):
                    self._report_cache_hit()
                    self._executor_instance().submit(self._advisory_refresh, spec, entry)
                    return token
                case _Fail(error=error):
                    return error
                case _Lead(call_type=call_type):
                    return self._lead(spec, entry, call_type)
                case _Follow():
                    followed = self._await_leader(spec, entry)  # rebind-ok: one leader wait per round
                    if followed is not None:
                        return followed
                case _:
                    assert_never(decision)

    async def aget_token(self, spec: TokenExchangeSpec) -> ExchangeResult:
        return await asyncio.to_thread(self.get_token, spec)

    def invalidate(self, spec: TokenExchangeSpec) -> None:
        key: Final = _cache_key(spec)
        with self._lock:
            if key in self._entries:
                self._entries[key] = _Entry(force_refresh=True)

    def _get_or_create_entry_locked(self, spec: TokenExchangeSpec) -> _Entry:
        key: Final = _cache_key(spec)
        existing: Final = self._entries.get(key)
        if existing is not None:
            return existing
        if len(self._entries) >= self._max_entries:
            self._evict_locked()
        created: Final = _Entry()
        self._entries[key] = created
        return created

    def _evict_locked(self) -> None:
        now: Final = self._clock()
        stale: Final = tuple(
            key
            for key, entry in self._entries.items()
            if not entry.in_flight
            and (entry.token is None or (entry.token.expires_at is not None and entry.token.expires_at <= now))
        )
        for key in stale:
            del self._entries[key]
        if len(self._entries) < self._max_entries:
            return
        # Evict soonest-to-expire first, and take as many as the overshoot needs rather than one, so a
        # burst of distinct identities does not leave the map permanently above max_entries. An entry
        # a leader owns or a follower waits on is never a candidate, so a moment where every entry is
        # in flight still over-inserts; that residue is bounded by the concurrent mints themselves.
        evictable: Final = sorted(
            (
                entry.token.expires_at if entry.token is not None and entry.token.expires_at is not None else -inf,
                key,
            )
            for key, entry in self._entries.items()
            if not entry.in_flight
        )
        for _, key in evictable[: len(self._entries) - self._max_entries + 1]:
            del self._entries[key]

    def _classify_and_arm_locked(self, entry: _Entry) -> _Decision:
        token: Final = entry.token
        if token is not None and not entry.force_refresh:
            if token.expires_at is None:
                return _Serve(token=token)
            windows: Final = _refresh_windows(entry.lifetime_seconds)
            remaining: Final = token.expires_at - self._clock()
            if remaining > windows.advisory:
                return _Serve(token=token)
            if remaining > windows.mandatory:
                if entry.in_flight or self._clock() < entry.backoff_until:
                    return _Serve(token=token)
                entry.arm()
                return _ServeAndRefresh(token=token)
        if entry.in_flight:
            return _Follow()
        if entry.last_error is not None and self._clock() < entry.backoff_until:
            return _Fail(error=entry.last_error)
        entry.arm()
        return _Lead(call_type=CALL_TYPE_COLD_MINT if token is None else CALL_TYPE_MANDATORY_REFRESH)

    def _executor_instance(self) -> Executor:
        with self._lock:
            if self._refresh_executor is None:
                self._refresh_executor = ThreadPoolExecutor(
                    max_workers=2, thread_name_prefix="litellm-token-exchange-refresh"
                )
            return self._refresh_executor

    def _lead(self, spec: TokenExchangeSpec, entry: _Entry, call_type: ExchangeCallType) -> ExchangeResult:
        started: Final = self._clock()
        result: Final = self._exchange_never_raises(spec)
        duration: Final = self._clock() - started
        with self._lock:
            entry.publish(result, now=self._clock())
        self._report_exchange(call_type, duration, result)
        return result

    def _await_leader(self, spec: TokenExchangeSpec, entry: _Entry) -> "ExchangeResult | None":
        """None means the finished round left neither a valid token nor an error
        (a failed advisory refresh); the caller re-enters and leads a fresh exchange."""
        leader_finished: Final = entry.done.wait(2 * spec.timeout_seconds + _FOLLOWER_WAIT_GRACE_SECONDS)
        with self._lock:
            token: Final = entry.token
            if token is not None and (token.expires_at is None or token.expires_at > self._clock()):
                return token
            if entry.last_error is not None:
                return entry.last_error
        if leader_finished:
            return None
        return TokenTransportError(detail="timed out waiting for the token exchange leader")

    def _advisory_refresh(self, spec: TokenExchangeSpec, entry: _Entry) -> None:
        started: Final = self._clock()
        result: Final = self._exchange_never_raises(spec)
        duration: Final = self._clock() - started
        with self._lock:
            now: Final = self._clock()
            entry.publish_advisory(result, now=now)
            stale_expires_at: Final = entry.token.expires_at if entry.token is not None else None
            stale_mandatory: Final = _refresh_windows(entry.lifetime_seconds).mandatory
        self._report_exchange(CALL_TYPE_ADVISORY_REFRESH, duration, result)
        if isinstance(result, MintedToken):
            return
        seconds_to_mandatory_wall: Final = (
            max(stale_expires_at - now - stale_mandatory, 0.0) if stale_expires_at is not None else 0.0
        )
        verbose_logger.warning(
            "Advisory token refresh against %s failed (%s); serving the cached token for up to "
            "%.0fs before the mandatory refresh wall; next attempt after %.0fs backoff",
            urlsplit(spec.token_url).hostname or "",
            _error_summary(result),
            seconds_to_mandatory_wall,
            ADVISORY_REFRESH_BACKOFF_SECONDS,
        )

    def _report_exchange(self, call_type: ExchangeCallType, duration_seconds: float, result: ExchangeResult) -> None:
        try:
            match result:
                case MintedToken():
                    self._metrics_sink.exchange_success(call_type=call_type, duration_seconds=duration_seconds)
                case _:
                    self._metrics_sink.exchange_failure(
                        call_type=call_type, duration_seconds=duration_seconds, error=result
                    )
        except Exception as e:  # noqa: BLE001  # metrics are best-effort; a sink failure must never fail a mint
            verbose_logger.debug("token exchange metrics emission failed: %s", e)

    def _report_cache_hit(self) -> None:
        try:
            self._metrics_sink.cache_hit()
        except Exception as e:  # noqa: BLE001  # metrics are best-effort; a sink failure must never fail a serve
            verbose_logger.debug("token exchange cache-hit metric emission failed: %s", e)

    def _exchange_never_raises(self, spec: TokenExchangeSpec) -> ExchangeResult:
        """The single-flight leader and the advisory refresher must always publish a result: an
        unhandled exception here would leave the entry armed (in_flight, cleared event) forever, so
        every subsequent caller for this key would follow a leader that never finishes."""
        try:
            return self._exchange(spec)
        except Exception as e:  # noqa: BLE001  # a leader must resolve its entry; any failure becomes a value
            return TokenTransportError(detail=f"{type(e).__name__}: {e}"[:_REDACTION_CAP])

    def _exchange(self, spec: TokenExchangeSpec) -> ExchangeResult:
        first: Final = self._attempt_exchange(spec)
        if not isinstance(first, _Unauthorized):
            return first
        second: Final = self._attempt_exchange(spec)
        if isinstance(second, _Unauthorized):
            return redact_oauth_error_body(
                second.response.status_code, _capped_body_text(second.response), second.assertion
            )
        return second

    def _attempt_exchange(self, spec: TokenExchangeSpec) -> "ExchangeResult | _Unauthorized":
        assertion: Final = _read_assertion(_assertion_fetch(self._assertion_reader, spec), spec.assertion_ref)
        if isinstance(assertion, AssertionSourceError):
            return assertion
        url_check: Final = validate_token_endpoint_url(spec.token_url)
        if isinstance(url_check, InsecureTokenUrl):
            return url_check
        try:
            response: Final = self._poster.post(
                spec.token_url,
                content=_serialize_body(spec, assertion),
                headers=MappingProxyType({"content-type": _CONTENT_TYPES[spec.body_encoding], **spec.request_headers}),
                timeout=spec.timeout_seconds,
            )
        except Exception as e:  # noqa: BLE001  # injected posters may raise beyond httpx; transport failures become values
            return TokenTransportError(detail=f"{type(e).__name__}: {e}"[:_REDACTION_CAP])
        if response.status_code == 401:
            return _Unauthorized(response=response, assertion=assertion)
        return self._parse_response(response, assertion)

    def _parse_response(self, response: httpx.Response, assertion: SecretStr | None = None) -> ExchangeResult:
        if not 200 <= response.status_code < 300:
            return redact_oauth_error_body(response.status_code, _capped_body_text(response), assertion)
        if len(response.content) > MAX_RESPONSE_BYTES:
            return MalformedTokenResponse(detail="token response body exceeds the 1 MiB cap")
        try:
            parsed: Final = _TokenExchangeResponse.model_validate_json(response.content)
        except ValidationError:
            return MalformedTokenResponse(detail="token response failed RFC 6749 5.1 schema validation")
        if parsed.token_type is not None and parsed.token_type.lower() != "bearer":
            return MalformedTokenResponse(detail="token response carried a non-bearer token_type")
        if not parsed.access_token.strip():
            return MalformedTokenResponse(detail="token response carried an empty access_token")
        return MintedToken(
            access_token=SecretStr(parsed.access_token),
            expires_at=self._clock() + _sanitize_expires_in(parsed.expires_in),
        )


default_token_exchange_engine: Final = JwtBearerTokenExchangeEngine()
