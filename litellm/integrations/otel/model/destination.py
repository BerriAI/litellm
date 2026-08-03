"""The resolved OTLP destination a request's traces export to.

A destination is a backend-agnostic target: an endpoint plus the auth headers the
exporter sends. The proxy builds it from the named logging credential bound to the
request's identity chain, and the v2 logger exports through it. Every OTEL backend
-- Langfuse, Arize, Weave, a self-hosted collector -- reduces to this shape; the
per-backend field mapping lives in ``litellm.integrations.otel.presets.destinations``.
"""

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field


class OtelDestination(BaseModel):
    model_config = ConfigDict(frozen=True)

    endpoint: str
    headers: Mapping[str, str] = Field(default_factory=dict)
    resource_attributes: Mapping[str, str] = Field(default_factory=dict)
    callback_name: str | None = None
    protocol: str | None = Field(
        default=None,
        description=(
            "OTLP transport for this endpoint (``otlp_http`` / ``otlp_grpc``). The "
            "backend's intrinsic default is used when unset. A backend whose own cloud "
            "endpoint is gRPC can still be pointed at an HTTP collector, which the "
            "scheme alone cannot express: Arize's own ``https://otlp.arize.com/v1`` is gRPC."
        ),
    )

    def header_string(self) -> str:
        """Render headers as the ``k=v,k2=v2`` form an ``ExporterSpec`` expects."""
        return ",".join(f"{key}={value}" for key, value in self.headers.items())
