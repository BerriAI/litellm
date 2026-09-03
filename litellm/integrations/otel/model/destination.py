"""The resolved OTLP destination a request's traces export to.

A destination is a backend-agnostic target: an endpoint plus the auth headers the
exporter sends. The proxy builds one per backend from the key or team logging
config resolved at auth, and the fan-out span processor exports the request's
spans through it. Every OTEL backend reduces to this shape; the per-backend field
mapping lives in ``litellm.integrations.otel.presets.destinations``.
"""

from collections.abc import Mapping
from typing import Final
from urllib.parse import quote

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
        """Render headers as the ``k=v,k2=v2`` form an ``ExporterSpec`` expects.

        Values are percent-encoded because ``providers.parse_headers`` decodes them
        with the SDK's W3C-Baggage parser: a value carrying a ``,`` or ``=`` (a
        Langfuse project name, a base64 Authorization payload ending in ``==``)
        would otherwise be split into bogus pairs on the way back out.
        """
        return ",".join(f"{key}={quote(value, safe='')}" for key, value in self.headers.items())

    def cache_key(self) -> tuple[str, tuple[tuple[str, str], ...], tuple[tuple[str, str], ...], str | None]:
        """Identity for processor reuse: two requests naming the same destination
        must share one exporter rather than minting a connection pool each."""
        return (
            self.endpoint,
            tuple(sorted(self.headers.items())),
            tuple(sorted(self.resource_attributes.items())),
            self.protocol,
        )


NO_DESTINATIONS: Final[tuple[OtelDestination, ...]] = ()
