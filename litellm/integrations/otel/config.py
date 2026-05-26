"""Typed configuration for the LiteLLM OpenTelemetry instrumentation.

Uses ``pydantic-settings`` so environment variables are read declaratively via
field aliases — there is no hand-rolled ``os.getenv`` parsing. Field defaults are
used when the corresponding env var is absent. This module has no
``opentelemetry`` import.
"""

from typing import List, Optional

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from litellm.integrations.otel.semconv import (
    BAGGAGE_PROMOTED_KEYS,
    DEFAULT_BAGGAGE_METADATA_KEYS,
)

#: Master feature-flag env var. The new engine is inert until this is truthy, so
#: the package can ship fully built without touching the existing OTel code path.
OTEL_V2_ENV = "LITELLM_OTEL_V2"


class CaptureMessageContent(str):
    """Values for ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT``."""

    NO_CONTENT = "no_content"
    SPAN_ONLY = "span_only"
    EVENT_ONLY = "event_only"
    SPAN_AND_EVENT = "span_and_event"


class _OTelV2Flag(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    enabled: bool = Field(default=False, validation_alias=AliasChoices(OTEL_V2_ENV))


def is_otel_v2_enabled() -> bool:
    """Whether the new instrumentation should be active. Off unless opted in."""
    return _OTelV2Flag().enabled


class OpenTelemetryV2Config(BaseSettings):
    """Fully typed OTel config, populated from env via field aliases.

    Construct with no arguments to read from the environment, or pass explicit
    values (field names work thanks to ``populate_by_name``) for tests / dynamic
    configuration.
    """

    model_config = SettingsConfigDict(populate_by_name=True, extra="ignore")

    exporter: str = Field(
        default="console",
        validation_alias=AliasChoices("OTEL_EXPORTER", "OTEL_EXPORTER_OTLP_PROTOCOL"),
    )
    endpoint: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("OTEL_ENDPOINT", "OTEL_EXPORTER_OTLP_ENDPOINT"),
    )
    headers: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("OTEL_HEADERS", "OTEL_EXPORTER_OTLP_HEADERS"),
    )
    service_name: str = Field(
        default="litellm", validation_alias=AliasChoices("OTEL_SERVICE_NAME")
    )
    deployment_environment: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("OTEL_ENVIRONMENT_NAME")
    )

    enable_metrics: bool = Field(
        default=False,
        validation_alias=AliasChoices("LITELLM_OTEL_INTEGRATION_ENABLE_METRICS"),
    )
    capture_message_content: str = Field(
        default=CaptureMessageContent.NO_CONTENT,
        validation_alias=AliasChoices(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
        ),
    )
    ignore_context_propagation: bool = Field(
        default=False,
        validation_alias=AliasChoices("OTEL_IGNORE_CONTEXT_PROPAGATION"),
    )

    #: Emit legacy attribute keys / span names alongside canonical ones during the
    #: deprecation window. Defaults on so existing dashboards keep working.
    legacy_compat: bool = Field(
        default=True, validation_alias=AliasChoices("LITELLM_OTEL_LEGACY_COMPAT")
    )

    #: Bounded allowlists for Baggage-based promotion of request-scoped identity
    #: onto every span (see ``providers.LiteLLMBaggageSpanProcessor``).
    baggage_promoted_keys: List[str] = Field(
        default_factory=lambda: list(BAGGAGE_PROMOTED_KEYS)
    )
    baggage_metadata_keys: List[str] = Field(
        default_factory=lambda: list(DEFAULT_BAGGAGE_METADATA_KEYS)
    )

    @model_validator(mode="after")
    def _endpoint_implies_otlp_http(self) -> "OpenTelemetryV2Config":
        # An endpoint with no explicit exporter implies OTLP/HTTP, matching the
        # behavior of the legacy integration.
        if self.endpoint and self.exporter == "console":
            self.exporter = "otlp_http"
        return self

    @classmethod
    def from_env(cls) -> "OpenTelemetryV2Config":
        """Read the configuration from the environment (alias for ``cls()``)."""
        return cls()
