"""AgentOps preset — OTLP/HTTP to AgentOps' endpoint with a lazily-fetched JWT.

AgentOps authenticates with a short-lived JWT minted from the API key. Fetching
it is blocking network I/O, so it must never run on the event loop: callback
construction (where presets are built) can run inside the proxy's async startup
or, in the SDK, on the first request. Instead of fetching at config-build time,
this preset registers a custom exporter (``kind="agentops"``) that mints the JWT
**on its first export** — which the ``BatchSpanProcessor`` runs in its own
worker thread, off any event loop — and caches it for the process lifetime.
"""

from typing import Any

import httpx
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from litellm._logging import verbose_logger
from litellm.integrations.otel.model.config import ExporterSpec, OpenTelemetryV2Config
from litellm.integrations.otel.plumbing.providers import register_exporter_factory

_AGENTOPS_ENDPOINT = "https://otlp.agentops.cloud/v1/traces"
_AGENTOPS_AUTH_ENDPOINT = "https://api.agentops.ai/v3/auth/token"
_AGENTOPS_EXPORTER_KIND = "agentops"


class _AgentOpsSettings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    api_key: str | None = Field(default=None, validation_alias="AGENTOPS_API_KEY")
    service_name: str = Field(
        default="agentops", validation_alias="AGENTOPS_SERVICE_NAME"
    )
    environment: str | None = Field(
        default=None, validation_alias="AGENTOPS_ENVIRONMENT"
    )


def agentops_preset(
    *,
    config_overrides: OpenTelemetryV2Config | None = None,
) -> OpenTelemetryV2Config:
    """Build the AgentOps config without any network I/O.

    The ``agentops`` exporter mints (and caches) the JWT lazily on its first
    export, so this stays non-blocking. ``project.id`` is therefore not a
    resource attribute — it is encoded in the JWT, which AgentOps uses to route
    the trace to the right project.
    """
    settings = _AgentOpsSettings()
    base = config_overrides or OpenTelemetryV2Config()
    return base.model_copy(
        update={
            "exporters": [
                *base.exporters,
                ExporterSpec(
                    kind=_AGENTOPS_EXPORTER_KIND,
                    endpoint=_AGENTOPS_ENDPOINT,
                    options=(
                        {"api_key": settings.api_key} if settings.api_key else None
                    ),
                ),
            ],
            "resource_attributes": {
                **base.resource_attributes,
                "service.name": settings.service_name,
                "telemetry.sdk.name": "agentops",
                **(
                    {"deployment.environment": settings.environment}
                    if settings.environment
                    else {}
                ),
            },
        }
    )


def _build_agentops_exporter(spec: ExporterSpec) -> Any:
    """Factory for the ``agentops`` exporter kind: a lazy-auth OTLP/HTTP exporter."""
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )

    class _LazyAuthAgentOpsExporter(OTLPSpanExporter):
        """OTLP/HTTP exporter that mints the AgentOps JWT on its first export.

        ``export`` runs in the ``BatchSpanProcessor`` worker thread, so the
        blocking token fetch never touches an event loop. The result is cached
        after the first attempt (success or failure) so it runs at most once.
        """

        def __init__(self, *, endpoint: str | None, api_key: str | None) -> None:
            super().__init__(endpoint=endpoint)
            self._agentops_api_key = api_key
            self._auth_resolved = False

        def _ensure_authenticated(self) -> None:
            if self._auth_resolved:
                return
            self._auth_resolved = True
            if not self._agentops_api_key:
                return
            try:
                token = _fetch_agentops_jwt(self._agentops_api_key).get("token")
                if token:
                    # ``_session`` is the requests.Session the base exporter
                    # POSTs through; updating its Authorization header is how the
                    # minted JWT reaches every subsequent export.
                    self._session.headers["Authorization"] = f"Bearer {token}"
            except Exception as e:
                verbose_logger.debug("AgentOps JWT fetch failed: %s", e)

        def export(self, spans: Any) -> Any:
            self._ensure_authenticated()
            return super().export(spans)

    options = spec.options or {}
    return _LazyAuthAgentOpsExporter(
        endpoint=spec.endpoint, api_key=options.get("api_key")
    )


def _fetch_agentops_jwt(api_key: str) -> dict[str, Any]:
    # Own a short-lived client rather than ``_get_httpx_client()``: that returns
    # a process-wide cached ``HTTPHandler`` whose connection pool is shared by
    # every caller, so closing it here would break concurrent/subsequent
    # requests. This one-shot auth call gets its own client to close.
    with httpx.Client(timeout=10) as client:
        response = client.post(
            url=_AGENTOPS_AUTH_ENDPOINT,
            headers={"Content-Type": "application/json", "Connection": "keep-alive"},
            json={"api_key": api_key},
        )
        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch AgentOps token: {response.text}")
        return response.json()


register_exporter_factory(_AGENTOPS_EXPORTER_KIND, _build_agentops_exporter)
