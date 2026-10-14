import os
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, Protocol
from urllib.parse import urlsplit

from opentelemetry.sdk._logs.export import LogExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv.attributes.http_attributes import HTTP_REQUEST_METHOD, HTTP_RESPONSE_STATUS_CODE
from opentelemetry.semconv.attributes.server_attributes import SERVER_ADDRESS, SERVER_PORT
from opentelemetry.semconv.attributes.url_attributes import URL_PATH, URL_SCHEME
from opentelemetry.trace import Span, SpanKind, StatusCode, Tracer, get_current_span, use_span
from pydantic import AliasChoices

from litellm.integrations.otel.model.config import OpenTelemetryV2Config
from litellm.integrations.otel.plumbing.context import inject_trace_context
from litellm.integrations.otel.plumbing.providers import build_log_exporter as _build_log_exporter
from litellm.integrations.otel.plumbing.providers import build_resource as _build_resource
from litellm.integrations.otel.plumbing.providers import build_tracer_provider
from litellm.proxy.control_plane.audit.log import redact_path
from litellm.proxy.control_plane.auth.rbac import Principal
from litellm.proxy.control_plane.settings import ControlPlaneSettings

if TYPE_CHECKING:
    from fastapi import FastAPI


def _setting_aliases(field_name: Literal["endpoint", "traces_endpoint"]) -> tuple[str, ...]:
    alias: Final = OpenTelemetryV2Config.model_fields[field_name].validation_alias
    return (
        tuple(choice for choice in alias.choices if isinstance(choice, str)) if isinstance(alias, AliasChoices) else ()
    )


TRACER_NAME: Final = "litellm.control_plane"
ENDUSER_ID: Final = "enduser.id"
ENDUSER_ROLE: Final = "enduser.role"
HTTP_REQUEST_RESEND_COUNT: Final = "http.request.resend_count"
PEER_SERVICE: Final = "peer.service"
TOKEN_KIND_ATTRIBUTE: Final = "litellm.control_plane.token_kind"
SCOPES_ATTRIBUTE: Final = "litellm.control_plane.scopes"
CREDENTIAL_ATTRIBUTE: Final = "litellm.control_plane.credential"
DATAPLANE_ATTRIBUTE: Final = "litellm.control_plane.dataplane"
UNTRACED_PATHS: Final = ("/healthz", "/readyz", "/docs", "/openapi.json")
OTEL_ENDPOINT_ENV_ALIASES: Final = _setting_aliases("endpoint") + _setting_aliases("traces_endpoint")
DEFAULT_PORTS: Final = {"https": 443, "http": 80}
PATH_ATTRIBUTES: Final = ("http.target", "url.path")
FULL_URL_ATTRIBUTES: Final = ("http.url", "url.full")
QUERY_ATTRIBUTES: Final = ("url.query",)


@dataclass(frozen=True, slots=True)
class DataplaneTarget:
    dataplane_id: str
    url: str
    method: str
    path: str
    credential: str

    @property
    def server(self) -> tuple[str, int | None]:
        parts: Final = urlsplit(self.url)
        return parts.hostname or self.url, parts.port or DEFAULT_PORTS.get(parts.scheme)


class SpanRecorder(Protocol):
    def client_span(self, target: DataplaneTarget, attempt: int) -> "AbstractSpan": ...

    def outgoing_headers(self, headers: Mapping[str, str]) -> dict[str, str]: ...  # mutable-ok: httpx takes dict


class AbstractSpan(Protocol):
    def __enter__(self) -> "AbstractSpan": ...

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None: ...

    def replied(self, status: int) -> None: ...

    def failed(self, error_type: str) -> None: ...


@dataclass(frozen=True, slots=True)
class NoSpan:
    def __enter__(self) -> "NoSpan":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def replied(self, status: int) -> None:
        return None

    def failed(self, error_type: str) -> None:
        return None


@dataclass(frozen=True, slots=True)
class Untraced:
    def client_span(self, target: DataplaneTarget, attempt: int) -> NoSpan:
        return NoSpan()

    def outgoing_headers(self, headers: Mapping[str, str]) -> dict[str, str]:  # mutable-ok: httpx takes dict
        return dict(headers)


@dataclass(frozen=True, slots=True)
class ClientSpan:
    span: Span
    scope: AbstractContextManager[Span]

    def __enter__(self) -> "ClientSpan":
        self.scope.__enter__()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.scope.__exit__(None, None, None)

    def replied(self, status: int) -> None:
        self.span.set_attribute(HTTP_RESPONSE_STATUS_CODE, status)
        if status >= 500:
            self.span.set_status(StatusCode.ERROR)
            self.span.set_attribute(ERROR_TYPE, str(status))

    def failed(self, error_type: str) -> None:
        self.span.set_status(StatusCode.ERROR, error_type)
        self.span.set_attribute(ERROR_TYPE, error_type)


def _client_attributes(target: DataplaneTarget, attempt: int) -> Mapping[str, str | int]:
    host, port = target.server
    base: Final[Mapping[str, str | int]] = {
        HTTP_REQUEST_METHOD: target.method,
        URL_PATH: target.path,
        URL_SCHEME: urlsplit(target.url).scheme,
        SERVER_ADDRESS: host,
        PEER_SERVICE: target.dataplane_id,
        DATAPLANE_ATTRIBUTE: target.dataplane_id,
        CREDENTIAL_ATTRIBUTE: target.credential,
    }
    with_port: Final = base if port is None else {**base, SERVER_PORT: port}
    return with_port if attempt == 0 else {**with_port, HTTP_REQUEST_RESEND_COUNT: attempt}


@dataclass(frozen=True, slots=True)
class Traced:
    tracer: Tracer

    def client_span(self, target: DataplaneTarget, attempt: int) -> ClientSpan:
        span: Final = self.tracer.start_span(
            f"{target.method} {target.path}", kind=SpanKind.CLIENT, attributes=_client_attributes(target, attempt)
        )
        return ClientSpan(span, use_span(span, end_on_exit=True))

    def outgoing_headers(self, headers: Mapping[str, str]) -> dict[str, str]:  # mutable-ok: httpx takes dict
        return inject_trace_context(headers)


def annotate_principal(principal: Principal, identity: bool) -> None:
    span: Final = get_current_span()
    if not span.is_recording():
        return
    span.set_attribute(TOKEN_KIND_ATTRIBUTE, principal.token_kind)
    span.set_attribute(SCOPES_ATTRIBUTE, tuple(sorted(scope.value for scope in principal.scopes)))
    if not identity:
        return
    span.set_attribute(ENDUSER_ID, principal.subject)
    span.set_attribute(ENDUSER_ROLE, tuple(sorted(principal.roles)))


class _ServerSpan(Protocol):
    def is_recording(self) -> bool: ...

    def set_attribute(self, key: str, value: str) -> None: ...


def _url_with_path(scope: Mapping[str, object], path: str) -> str:
    scheme: Final = scope.get("scheme")
    server: Final = scope.get("server")
    if not isinstance(scheme, str):
        return path
    match server:
        case (str() as host, int() as port):
            port_suffix: Final = "" if DEFAULT_PORTS.get(scheme) == port else f":{port}"
            return f"{scheme}://{host}{port_suffix}{path}"
        case _:
            return path


def redact_server_span(span: _ServerSpan | None, scope: Mapping[str, object]) -> None:
    """Redact secrets from the request target before the span is exported."""
    if span is None or not span.is_recording():
        return
    raw_path: Final = scope.get("path")
    path: Final = redact_path(raw_path) if isinstance(raw_path, str) else ""
    full_url: Final = _url_with_path(scope, path)
    for key in PATH_ATTRIBUTES:
        span.set_attribute(key, path)
    for key in FULL_URL_ATTRIBUTES:
        span.set_attribute(key, full_url)
    for key in QUERY_ATTRIBUTES:
        span.set_attribute(key, "")


@dataclass(frozen=True, slots=True)
class Telemetry:
    provider: TracerProvider | None
    recorder: SpanRecorder
    identity_attributes: bool

    def instrument(self, app: "FastAPI") -> None:
        if self.provider is None:
            return
        from opentelemetry.instrumentation import fastapi  # pyright: ignore[reportMissingTypeStubs]  # untyped contrib

        fastapi.FastAPIInstrumentor.instrument_app(  # pyright: ignore[reportUnknownMemberType]  # untyped contrib
            app,
            tracer_provider=self.provider,
            server_request_hook=redact_server_span,
            excluded_urls=",".join(UNTRACED_PATHS),
            exclude_spans=["receive", "send"],
        )

    def shutdown(self) -> None:
        if self.provider is None:
            return
        self.provider.shutdown()


UNTRACED: Final = Telemetry(provider=None, recorder=Untraced(), identity_attributes=False)


def tracing_enabled(settings: ControlPlaneSettings, environ: Mapping[str, str] = os.environ) -> bool:
    return (
        settings.otel_enabled
        if settings.otel_enabled is not None
        else any(environ.get(alias) for alias in OTEL_ENDPOINT_ENV_ALIASES)
    )


def _config(settings: ControlPlaneSettings) -> OpenTelemetryV2Config:
    return OpenTelemetryV2Config.from_env().model_copy(update={"service_name": settings.otel_service_name})


def build_provider(settings: ControlPlaneSettings, exporter: SpanExporter | None = None) -> TracerProvider:
    return build_tracer_provider(
        _config(settings), exporter=exporter, use_simple_processor=None if exporter is None else True
    )


def build_log_exporter(settings: ControlPlaneSettings) -> LogExporter:
    return _build_log_exporter(_config(settings))


def build_resource(settings: ControlPlaneSettings) -> Resource:
    return _build_resource(_config(settings))


def build_telemetry(settings: ControlPlaneSettings, provider: TracerProvider | None = None) -> Telemetry:
    chosen: Final = (
        provider if provider is not None else build_provider(settings) if tracing_enabled(settings) else None
    )
    if chosen is None:
        return UNTRACED
    return Telemetry(
        provider=chosen,
        recorder=Traced(chosen.get_tracer(TRACER_NAME)),
        identity_attributes=settings.otel_identity_attributes,
    )
