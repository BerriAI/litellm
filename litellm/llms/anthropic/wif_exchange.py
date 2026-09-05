"""Anthropic workload identity federation on the Anthropic SDK. ``WorkloadIdentityCredentials``
performs the RFC 7523 exchange and ``TokenCache`` owns refresh timing, single flight and the 401
retry. LiteLLM keeps what the SDK cannot know: where the identity token comes from (secret refs, the
credential-dir allowlist, identity sources), one bounded cache per deployment, service metrics, and
the redacted error values ``wif.py`` maps onto the public exception contract.

The SDK speaks ``httpx2``, a separate HTTP stack from the ``httpx`` LiteLLM's handlers use, so the
exchange client is built here with LiteLLM's SSL settings instead of being borrowed from a handler."""

import asyncio
import json
import os
import threading
import time
from collections.abc import Callable, Coroutine, Mapping
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol, TypeAlias

import httpx2
from anthropic.lib.credentials import AccessToken, TokenCache, WorkloadIdentityCredentials, WorkloadIdentityError
from pydantic import SecretStr, TypeAdapter
from typing_extensions import assert_never

import litellm
from litellm._logging import verbose_logger
from litellm.llms.base_llm.auth.oauth_endpoint import (
    drop_reflected_credential,
    redact_oauth_error_body,
    validate_token_endpoint_url,
)
from litellm.llms.base_llm.auth.types import (
    AssertionReader,
    AssertionSource,
    AssertionSourceError,
    ExchangeCallType,
    ExchangeError,
    ExchangeResult,
    InsecureTokenUrl,
    MalformedTokenResponse,
    TokenEndpointError,
    TokenExchangeMetricsSink,
    TokenTransportError,
)
from litellm.types.services import ServiceTypes

if TYPE_CHECKING:
    from litellm.llms.anthropic.wif import AnthropicWifParams

CALL_TYPE_COLD_MINT: Final[ExchangeCallType] = "cold_mint"
CALL_TYPE_REFRESH: Final[ExchangeCallType] = "refresh"
CALL_TYPE_CACHE_HIT: Final = "cache_hit"
MAX_ASSERTION_BYTES: Final = 16 * 1024
EXCHANGE_TIMEOUT_SECONDS: Final = 30.0
EXCHANGE_CONNECT_TIMEOUT_SECONDS: Final = 5.0
_DETAIL_CAP: Final = 256
_METRICS_QUEUE_LIMIT: Final = 1000


def _default_assertion_reader(ref: str) -> str | None:
    from litellm.secret_managers.main import get_secret_str

    return get_secret_str(ref)


def _read_assertion(fetch: AssertionSource, ref: str) -> SecretStr | AssertionSourceError:
    from litellm.secret_managers.main import OidcPathNotAllowedError

    try:
        raw: Final = fetch()
    except OidcPathNotAllowedError:
        return AssertionSourceError(kind="disallowed_path", source_ref=ref)
    except (ValueError, ImportError) as e:
        return AssertionSourceError(kind="unreadable", source_ref=ref, detail=str(e)[:_DETAIL_CAP])
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


class _AssertionUnavailable(WorkloadIdentityError):
    """Raised out of the identity token provider so ``TokenCache`` treats a source failure like a
    failed exchange: a still-valid cached token keeps serving through the advisory window."""

    def __init__(self, error: AssertionSourceError) -> None:
        super().__init__(f"identity token {error.kind} from {error.source_ref}")
        self.error: Final = error


class _IdentityTokenSource:
    """Reads the identity token fresh for every exchange, so a rotated file or a re-minted
    internal-issuer assertion is what the next exchange carries, and remembers the assertion on
    the wire so an endpoint that echoes it is scrubbed out of the error."""

    def __init__(self, fetch: AssertionSource, ref: str) -> None:
        self._fetch: Final = fetch
        self._ref: Final = ref
        self.last_sent: SecretStr | None = None

    def __call__(self) -> str:
        match _read_assertion(self._fetch, self._ref):
            case SecretStr() as assertion:
                self.last_sent = assertion
                return assertion.get_secret_value()
            case AssertionSourceError() as error:
                raise _AssertionUnavailable(error)


class _SdkErrorFields(Protocol):
    """``WorkloadIdentityError`` declares ``body: Any``; reading it through this protocol keeps
    ``Any`` out of the error mapping."""

    status_code: int | None
    body: object


_SdkBody: TypeAlias = Mapping[str, object] | str | None
_SDK_BODY_ADAPTER: Final = TypeAdapter[_SdkBody](_SdkBody)


def exchange_error(error: WorkloadIdentityError, assertion: SecretStr | None) -> ExchangeError:
    if isinstance(error, _AssertionUnavailable):
        return error.error
    return _endpoint_error(error, str(error).removesuffix("."), assertion)


def _endpoint_error(fields: _SdkErrorFields, message: str, assertion: SecretStr | None) -> ExchangeError:
    """The SDK folds every failure into one exception class; ``status_code`` and ``body`` tell
    them apart. A missing status is a transport failure, a 2xx is a response the SDK could not
    use (its message quotes the body, so a reflected assertion is scrubbed from it), and a 4xx/5xx
    without a body is one the SDK refused to read for size."""
    if fields.status_code is None:
        return TokenTransportError(detail=message)
    if fields.status_code < 400:
        return MalformedTokenResponse(detail=drop_reflected_credential(message, assertion)[:_DETAIL_CAP])
    body: Final = _SDK_BODY_ADAPTER.validate_python(fields.body)
    if body is None:
        return TokenEndpointError(status_code=fields.status_code, redacted_body=message)
    return redact_oauth_error_body(fields.status_code, body if isinstance(body, str) else json.dumps(body), assertion)


def _emit(event: Callable[[], None]) -> None:
    try:
        event()
    except Exception as e:  # noqa: BLE001  # metrics are best-effort; a sink failure must never surface to the mint
        verbose_logger.debug("token exchange metrics sink raised: %s", e)


class _MeteredExchange:
    """Times each SDK exchange for the metrics sink and counts them, which is how a ``get_token``
    that never reached the provider is recognised as a cache hit."""

    def __init__(
        self, credentials: WorkloadIdentityCredentials, source: _IdentityTokenSource, sink: TokenExchangeMetricsSink
    ) -> None:
        self._credentials: Final = credentials
        self._source: Final = source
        self._sink: Final = sink
        self.exchanges: int = 0
        self._minted: bool = False

    def __call__(self, *, force_refresh: bool = False) -> AccessToken:
        call_type: Final = CALL_TYPE_REFRESH if self._minted else CALL_TYPE_COLD_MINT
        self.exchanges += 1
        started: Final = time.monotonic()
        try:
            token: Final = self._credentials(force_refresh=force_refresh)
        except WorkloadIdentityError as e:
            error: Final = exchange_error(e, self._source.last_sent)
            _emit(
                lambda: self._sink.exchange_failure(
                    call_type=call_type, duration_seconds=time.monotonic() - started, error=error
                )
            )
            raise
        self._minted = True
        _emit(lambda: self._sink.exchange_success(call_type=call_type, duration_seconds=time.monotonic() - started))
        return token


@dataclass(frozen=True, slots=True)
class _CacheKey:
    exchange_base: str
    assertion_ref: str
    federation_rule_id: str
    organization_id: str
    service_account_id: str | None
    workspace_id: str | None


@dataclass(frozen=True, slots=True)
class _Deployment:
    cache: TokenCache
    exchange: _MeteredExchange
    source: _IdentityTokenSource


def new_exchange_client() -> httpx2.Client:
    """The client the SDK would build for itself ignores LiteLLM's SSL settings, so the exchange
    gets one built the way ``HTTPHandler`` builds its own: the same CA bundle, verification switch
    and client certificate. Redirects stay off: only the bound base URL passed the host allowlist,
    and a 3xx must not replay the assertion elsewhere."""
    from litellm.llms.custom_httpx.http_handler import get_ssl_configuration

    return httpx2.Client(
        verify=get_ssl_configuration(),
        cert=os.getenv("SSL_CERTIFICATE", litellm.ssl_certificate),
        timeout=httpx2.Timeout(EXCHANGE_TIMEOUT_SECONDS, connect=EXCHANGE_CONNECT_TIMEOUT_SECONDS),
        follow_redirects=False,
    )


class AnthropicWifTokenExchange:
    """One ``TokenCache`` per (endpoint, identity, federation target), bounded so a deployment
    churn cannot grow the process without limit. Everything credential-shaped stays inside the
    SDK objects; callers get a token or a typed, redacted ``ExchangeError``."""

    def __init__(
        self,
        http_client: httpx2.Client | None = None,
        assertion_reader: AssertionReader = _default_assertion_reader,
        max_entries: int = 64,
        metrics_sink: TokenExchangeMetricsSink | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._lock: Final = threading.Lock()
        self._http_client: httpx2.Client | None = http_client
        self._assertion_reader: Final = assertion_reader
        self._max_entries: Final = max_entries
        self._clock: Final = clock
        self._metrics_sink: Final = metrics_sink if metrics_sink is not None else ServiceLoggingMetricsSink()
        self._deployments: Final[dict[_CacheKey, _Deployment]] = {}  # mutable-ok: bounded cache, guarded by _lock

    def get_token(self, params: "AnthropicWifParams", exchange_base: str) -> ExchangeResult:
        match validate_token_endpoint_url(exchange_base):
            case InsecureTokenUrl() as insecure:
                _emit(
                    lambda: self._metrics_sink.exchange_failure(
                        call_type=CALL_TYPE_COLD_MINT, duration_seconds=0.0, error=insecure
                    )
                )
                return insecure
            case str():
                pass
        deployment: Final = self._deployment(params, exchange_base)
        exchanges_before: Final = deployment.exchange.exchanges
        try:
            token: Final = deployment.cache.get_token()
        except WorkloadIdentityError as e:
            return exchange_error(e, deployment.source.last_sent)
        if not token.strip():
            deployment.cache.invalidate()
            return MalformedTokenResponse(detail="empty access_token")
        if deployment.exchange.exchanges == exchanges_before:
            _emit(self._metrics_sink.cache_hit)
        return token

    async def aget_token(self, params: "AnthropicWifParams", exchange_base: str) -> ExchangeResult:
        return await asyncio.to_thread(self.get_token, params, exchange_base)

    def _deployment(self, params: "AnthropicWifParams", exchange_base: str) -> _Deployment:
        key: Final = _CacheKey(
            exchange_base=exchange_base,
            assertion_ref=params.assertion_ref,
            federation_rule_id=params.federation_rule_id,
            organization_id=params.organization_id,
            service_account_id=params.service_account_id,
            workspace_id=params.workspace_id,
        )
        with self._lock:
            existing: Final = self._deployments.get(key)
            if existing is not None:
                return existing
            if len(self._deployments) >= self._max_entries:
                del self._deployments[next(iter(self._deployments))]
            created: Final = self._new_deployment(params, exchange_base)
            self._deployments[key] = created
            return created

    def _new_deployment(self, params: "AnthropicWifParams", exchange_base: str) -> _Deployment:
        if self._http_client is None:
            self._http_client = new_exchange_client()
        fetch: Final[AssertionSource] = (
            params.assertion_source
            if params.assertion_source is not None
            else lambda: self._assertion_reader(params.assertion_ref)
        )
        source: Final = _IdentityTokenSource(fetch, params.assertion_ref)
        credentials: Final = WorkloadIdentityCredentials(
            identity_token_provider=source,
            federation_rule_id=params.federation_rule_id,
            organization_id=params.organization_id,
            service_account_id=params.service_account_id,
            workspace_id=params.workspace_id,
            http_client=self._http_client,
        )
        credentials.bind_base_url(exchange_base)
        exchange: Final = _MeteredExchange(credentials, source, self._metrics_sink)
        return _Deployment(cache=TokenCache(exchange, time_source=self._clock), exchange=exchange, source=source)


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
    """Default sink: bridges exchange metrics onto litellm's ServiceTypes pattern
    (prometheus ``litellm_anthropic_wif_*`` via ``service_callback``). Exchanges run on sync
    threads with no event loop, and the service hooks are async, so every emission is
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


default_anthropic_wif_exchange: Final = AnthropicWifTokenExchange()
