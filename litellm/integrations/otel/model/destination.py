"""The resolved OTLP destination a request's traces export to.

A destination is a backend-agnostic target: an endpoint plus the auth headers the
exporter sends. The proxy builds it from the named logging credential bound to the
request's identity chain, and the v2 logger exports through it. Every OTEL backend
-- Langfuse, Arize, Weave, a self-hosted collector -- reduces to this shape; the
per-backend field mapping lives in ``litellm.integrations.otel.presets.destinations``.
"""

from pydantic import BaseModel, ConfigDict, Field


class OtelDestination(BaseModel):
    model_config = ConfigDict(frozen=True)

    endpoint: str
    headers: dict[str, str] = Field(default_factory=dict)
    resource_attributes: dict[str, str] = Field(default_factory=dict)
    # The OTEL backend (callback_name) this destination belongs to, so a request
    # that fans out across backends routes each destination to the logger that
    # owns its attribute vocabulary. None for the legacy single-destination path.
    callback_name: str | None = None

    def header_string(self) -> str:
        """Render headers as the ``k=v,k2=v2`` form an ``ExporterSpec`` expects."""
        return ",".join(f"{key}={value}" for key, value in self.headers.items())
