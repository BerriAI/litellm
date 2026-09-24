"""The resolved OTLP destination a request's traces export to.

Backend-agnostic on purpose: every OTEL backend reduces to an endpoint plus auth
headers. The per-backend field mapping lives in ``presets.destinations``.
"""

from collections.abc import Mapping
from typing import Final
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field

from litellm.types.utils import OtelSpanScope


class OtelDestination(BaseModel):
    model_config = ConfigDict(frozen=True)

    endpoint: str
    headers: Mapping[str, str] = Field(default_factory=dict)
    resource_attributes: Mapping[str, str] = Field(default_factory=dict)
    callback_name: str | None = None
    protocol: str | None = Field(
        default=None,
        description=(
            "OTLP transport, defaulting to the backend's own. Not derivable from the "
            "scheme: Arize's ``https://otlp.arize.com/v1`` is gRPC."
        ),
    )
    span_scope: OtelSpanScope = Field(
        default="full",
        description=(
            "``no_internal`` holds back the proxy's own SERVICE and DB_CALL spans (auth, Redis, "
            "Postgres, spend writes) while ``llm_only`` keeps just the model-call spans; the request "
            "root and the tenant's own model, guardrail and MCP spans still go under ``no_internal``."
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
        """Identity for processor reuse, so one destination means one exporter.

        ``span_scope`` is left out on purpose: it decides which spans reach the
        processor, not how the processor exports them, so a full and an ``llm_only``
        view of the same account share one exporter.
        """
        return (
            self.endpoint,
            tuple(sorted(self.headers.items())),
            tuple(sorted(self.resource_attributes.items())),
            self.protocol,
        )


NO_DESTINATIONS: Final[tuple[OtelDestination, ...]] = ()
