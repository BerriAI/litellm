"""The resolved OTLP destination a request's traces export to.

Backend-agnostic on purpose: every OTEL backend reduces to an endpoint plus auth
headers. The per-backend field mapping lives in ``presets.destinations``.
"""

from collections.abc import Mapping
from typing import Final
from urllib.parse import quote

from pydantic import ConfigDict, Field

from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.utils import OtelSpanScope, captures_span_content


class OtelDestination(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    endpoint: str
    headers: Mapping[str, str] = Field(default_factory=dict)
    resource_attributes: Mapping[str, str] = Field(default_factory=dict)
    resource_defaults: Mapping[str, str] = Field(
        default_factory=dict,
        description=(
            "Resource attributes the backend needs on every export, filled only where the "
            "span's own resource names none; ``resource_attributes`` override it."
        ),
    )
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
        description="``llm_only`` keeps just the model-call spans; the rest of the request tree is not forwarded.",
    )
    success_sampling_rate: float | None = Field(
        default=None,
        description=(
            "Share of the request trees forwarded, 0.0..1.0, drawn once per request when its root span ends; "
            "``None`` forwards every one."
        ),
    )
    error_sampling_rate: float | None = Field(
        default=None,
        description="Same, for the requests with a failed span in their tree; ``None`` forwards every one.",
    )
    capture_message_content: str | None = Field(
        default=None,
        description=(
            "An explicit mode overrides the global capture policy for this destination. "
            "Omitted follows the global setting."
        ),
    )

    def captures_content(self, default: str) -> bool:
        setting: Final = self.capture_message_content
        return captures_span_content(default if setting is None else setting)

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

        ``span_scope``, the sampling rates and ``capture_message_content`` are left out on
        purpose: they decide which spans reach the processor and what they carry, not how
        the processor exports them, so two views of the same account share one exporter.
        """
        return (
            self.endpoint,
            tuple(sorted(self.headers.items())),
            tuple(sorted(self.resource_attributes.items())),
            self.protocol,
        )


NO_DESTINATIONS: Final[tuple[OtelDestination, ...]] = ()
