"""AgentOps preset — OTLP/HTTP to AgentOps' endpoint with a lazily-fetched JWT.

AgentOps authenticates with a short-lived JWT minted from the API key. Fetching
it is blocking network I/O, so it must never run on the event loop: callback
construction (where presets are built) can run inside the proxy's async startup
or, in the SDK, on the first request. Instead of fetching at config-build time,
this preset registers a custom exporter (``kind="agentops"``) that mints the JWT
**on its first export** — which the ``BatchSpanProcessor`` runs in its own
worker thread, off any event loop — and caches it for the process lifetime.
"""

from typing import Final

import httpx
import requests
from opentelemetry.sdk.trace.export import SpanExporter
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from requests.auth import AuthBase

from litellm._logging import verbose_logger
from litellm.integrations.otel.model.config import (
    ExporterOwner,
    ExporterSpec,
    OpenTelemetryV2Config,
)
from litellm.integrations.otel.plumbing.otlp_tls import OtlpHttpTls, resolve_otlp_http_tls
from litellm.integrations.otel.plumbing.providers import register_exporter_factory

_AGENTOPS_ENDPOINT: Final = "https://otlp.agentops.ai/v1/traces"
_AGENTOPS_AUTH_ENDPOINT: Final = "https://api.agentops.ai/v3/auth/token"
_AGENTOPS_EXPORTER_KIND: Final = "agentops"


class _AgentOpsSettings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    api_key: str | None = Field(default=None, validation_alias="AGENTOPS_API_KEY")
    service_name: str = Field(default="agentops", validation_alias="AGENTOPS_SERVICE_NAME")
    environment: str | None = Field(default=None, validation_alias="AGENTOPS_ENVIRONMENT")


def agentops_preset(
    *,
    config_overrides: OpenTelemetryV2Config | None = None,
    allow_missing_credentials: bool = False,
) -> OpenTelemetryV2Config:
    """Build the AgentOps config without any network I/O.

    The ``agentops`` exporter mints (and caches) the JWT lazily on its first
    export, so this stays non-blocking. ``project.id`` is therefore not a
    resource attribute — it is encoded in the JWT, which AgentOps uses to route
    the trace to the right project.
    """
    settings: Final = _AgentOpsSettings()
    base: Final = config_overrides or OpenTelemetryV2Config()
    return base.model_copy(
        update={
            "exporters": [
                *base.exporters,
                ExporterSpec(
                    kind=_AGENTOPS_EXPORTER_KIND,
                    endpoint=_AGENTOPS_ENDPOINT,
                    options=({"api_key": settings.api_key} if settings.api_key else None),
                    owner=ExporterOwner.AGENTOPS,
                ),
            ],
            "resource_attributes": {
                **base.resource_attributes,
                "service.name": settings.service_name,
                "telemetry.sdk.name": "agentops",
                **({"deployment.environment": settings.environment} if settings.environment else {}),
            },
        }
    )


class _AgentOpsAuth(AuthBase):
    def __init__(self, api_key: str | None) -> None:
        self._api_key = api_key
        self._resolved = False
        self._token: str | None = None

    def __call__(self, request: requests.PreparedRequest) -> requests.PreparedRequest:
        if not self._resolved:
            self._resolved = True
            if self._api_key:
                try:
                    token: Final = _fetch_agentops_jwt(self._api_key).get("token")
                    if isinstance(token, str) and token:
                        self._token = token
                except Exception as error:
                    verbose_logger.debug("AgentOps JWT fetch failed: %s", error)
        authenticated: Final = request.copy()
        if self._token:
            authenticated.headers["Authorization"] = f"Bearer {self._token}"
        return authenticated


def _build_agentops_exporter(spec: ExporterSpec) -> SpanExporter:
    """Authenticate lazily through requests' public session authentication API."""
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    from litellm.integrations.otel.plumbing.otlp_transport import build_otlp_http_transport

    options: Final = spec.options or {}
    tls: Final = resolve_otlp_http_tls("TRACES", isolate_headers=True)
    session: Final = tls.session or requests.Session()
    session.auth = _AgentOpsAuth(options.get("api_key"))
    return OTLPSpanExporter(
        endpoint=spec.endpoint,
        _transport=build_otlp_http_transport(
            "TRACES", OtlpHttpTls(certificate_file=tls.certificate_file, session=session)
        ),
    )


def _fetch_agentops_jwt(api_key: str) -> dict[str, object]:
    # Own a short-lived client rather than ``_get_httpx_client()``: that returns
    # a process-wide cached ``HTTPHandler`` whose connection pool is shared by
    # every caller, so closing it here would break concurrent/subsequent
    # requests. This one-shot auth call gets its own client to close.
    with httpx.Client(timeout=10) as client:
        response: Final = client.post(
            url=_AGENTOPS_AUTH_ENDPOINT,
            headers={"Content-Type": "application/json", "Connection": "keep-alive"},
            json={"api_key": api_key},
        )
        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch AgentOps token: {response.text}")
        return response.json()


register_exporter_factory(_AGENTOPS_EXPORTER_KIND, _build_agentops_exporter)
