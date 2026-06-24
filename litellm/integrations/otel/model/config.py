"""Typed configuration for the OpenTelemetry instrumentation."""

from enum import Enum
from functools import lru_cache
from typing import Any, List

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from typing_extensions import Annotated

from litellm.integrations.otel.model.baggage import (
    BAGGAGE_PROMOTED_KEYS,
    DEFAULT_BAGGAGE_METADATA_KEYS,
    DEFAULT_BAGGAGE_TEAM_METADATA_KEYS,
)

#: Master feature-flag env var. The logger is inert until this is truthy.
OTEL_V2_ENV = "LITELLM_OTEL_V2"


class CaptureMessageContent(str):
    NO_CONTENT = "no_content"
    SPAN_ONLY = "span_only"
    EVENT_ONLY = "event_only"
    SPAN_AND_EVENT = "span_and_event"


class ExporterOwner(str, Enum):
    """The preset that contributed an exporter. Values match the callback names
    in ``presets.PRESET_BY_CALLBACK`` so per-request dynamic-credential routing
    can match an exporter's owner against the credential source's callback name.
    A ``str`` enum so the value compares equal to the bare callback-name string."""

    # Arize AX (the hosted platform) and Arize Phoenix (the open-source / Phoenix
    # Cloud tracer) are distinct backends with separate config and auth, so they
    # are separate owners. The member value stays the public callback name.
    ARIZE_AX = "arize"
    ARIZE_PHOENIX = "arize_phoenix"
    LANGFUSE_OTEL = "langfuse_otel"
    WEAVE_OTEL = "weave_otel"
    LEVO = "levo"
    AGENTOPS = "agentops"


class _OTelV2Flag(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    enabled: bool = Field(default=False, validation_alias=AliasChoices(OTEL_V2_ENV))


@lru_cache(maxsize=1)
def is_otel_v2_enabled() -> bool:
    # Resolved once at startup and cached: constructing the pydantic-settings
    # model re-scans the environment and cost ~28us, which on the proxy hot path
    # (auth, logging-callback setup) compounded into a measurable throughput
    # regression. Tests that toggle the env must call ``is_otel_v2_enabled.cache_clear()``.
    return _OTelV2Flag().enabled


class ExporterSpec(BaseModel):
    """One span-export destination.

    The shared ``TracerProvider`` attaches one ``SpanProcessor`` per spec, so
    listing several specs sends every span to all of them at once (e.g. Arize +
    Phoenix + your own Honeycomb).
    """

    model_config = {"extra": "forbid"}

    kind: str = Field(
        default="console",
        description="console | in_memory | otlp_http | otlp_grpc | <factory kind>",
    )
    endpoint: str | None = None
    headers: str | None = None
    owner: ExporterOwner | None = Field(
        default=None,
        description=(
            "The preset that contributed this exporter. Per-request dynamic OTLP "
            "credentials are applied only to the exporter whose owner matches the "
            "credential source, so one tenant's vendor key never lands on a "
            "different backend's exporter."
        ),
    )
    options: dict[str, str] | None = Field(
        default=None,
        description=(
            "Factory-specific configuration for a custom exporter ``kind`` "
            "registered via ``providers.register_exporter_factory`` (e.g. an "
            "API key a lazy-auth exporter fetches a token with). Ignored by the "
            "built-in console/in_memory/otlp exporters."
        ),
    )
    use_simple_processor: bool | None = Field(
        default=None,
        description=(
            "Force SimpleSpanProcessor regardless of exporter kind. Default: "
            "auto (Simple for console/in_memory, Batch otherwise)."
        ),
    )


class OpenTelemetryV2Config(BaseSettings):
    model_config = SettingsConfigDict(populate_by_name=True, extra="ignore")

    # ----- single-destination shorthand, read from standard OTEL_* envs ----- #
    exporter: str = Field(
        default="console",
        validation_alias=AliasChoices("OTEL_EXPORTER", "OTEL_EXPORTER_OTLP_PROTOCOL"),
        description=(
            "Exporter kind for the single-destination shorthand. The model "
            "validator folds this (with ``endpoint`` / ``headers``) into a "
            "one-entry ``exporters`` list when ``exporters`` is empty; set "
            "``exporters`` directly for multiple destinations."
        ),
    )
    endpoint: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OTEL_ENDPOINT", "OTEL_EXPORTER_OTLP_ENDPOINT"),
    )
    headers: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OTEL_HEADERS", "OTEL_EXPORTER_OTLP_HEADERS"),
    )
    service_name: str = Field(
        default="litellm", validation_alias=AliasChoices("OTEL_SERVICE_NAME")
    )
    deployment_environment: str | None = Field(
        default=None, validation_alias=AliasChoices("OTEL_ENVIRONMENT_NAME")
    )

    enable_metrics: bool = Field(
        default=False,
        validation_alias=AliasChoices("LITELLM_OTEL_INTEGRATION_ENABLE_METRICS"),
    )
    enable_events: bool = Field(
        default=False,
        validation_alias=AliasChoices("LITELLM_OTEL_INTEGRATION_ENABLE_EVENTS"),
    )
    capture_message_content: str = Field(
        default=CaptureMessageContent.NO_CONTENT,
        validation_alias=AliasChoices(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
        ),
    )
    legacy_compat: bool = Field(
        default=True, validation_alias=AliasChoices("LITELLM_OTEL_LEGACY_COMPAT")
    )

    # ----- explicit multi-destination / vocabulary configuration ------------ #

    exporters: list[ExporterSpec] = Field(
        default_factory=list,
        description=(
            "One destination per spec. The shared TracerProvider attaches a "
            "SpanProcessor per entry. When empty, the model validator folds "
            "the ``exporter`` / ``endpoint`` / ``headers`` shorthand into a "
            "single spec so there is always at least one destination."
        ),
    )

    mapper_names: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: ["genai"],
        description=(
            "Ordered attribute vocabularies to emit. ``genai`` is the "
            "canonical OTel GenAI vocabulary and is always placed first. "
            "Vendor names: ``openinference`` (Arize + Phoenix), ``langfuse``, "
            "``weave``, ``langtrace``."
        ),
    )

    resource_attributes: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Extra Resource attributes beyond ``service.name`` and "
            "``deployment.environment`` (e.g. integration-specific markers)."
        ),
    )

    baggage_promoted_keys: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: list(BAGGAGE_PROMOTED_KEYS),
        validation_alias=AliasChoices(
            "baggage_promoted_keys", "LITELLM_OTEL_BAGGAGE_PROMOTED_KEYS"
        ),
        description=(
            "Identity attribute keys written into Baggage and stamped on every "
            "child span (e.g. ``litellm.team.id``). Configure via the "
            "``LITELLM_OTEL_BAGGAGE_PROMOTED_KEYS`` env var (comma-separated) or "
            "``callback_settings.otel.baggage_promoted_keys`` in config.yaml (a "
            "YAML list)."
        ),
    )
    baggage_metadata_keys: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: list(DEFAULT_BAGGAGE_METADATA_KEYS),
        validation_alias=AliasChoices(
            "baggage_metadata_keys", "LITELLM_OTEL_BAGGAGE_METADATA_KEYS"
        ),
        description=(
            "Metadata sub-keys promoted under the ``litellm.metadata.*`` "
            "namespace. Configure via the ``LITELLM_OTEL_BAGGAGE_METADATA_KEYS`` "
            "env var (comma-separated) or "
            "``callback_settings.otel.baggage_metadata_keys`` in config.yaml."
        ),
    )
    baggage_team_metadata_keys: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: list(DEFAULT_BAGGAGE_TEAM_METADATA_KEYS),
        validation_alias=AliasChoices(
            "baggage_team_metadata_keys", "LITELLM_OTEL_BAGGAGE_TEAM_METADATA_KEYS"
        ),
        description=(
            "Sub-keys of the team's free-form metadata promoted under "
            "``litellm.team.metadata``. Empty by default so none of a team's "
            "metadata leaves the process until explicitly allowlisted. Configure "
            "via the ``LITELLM_OTEL_BAGGAGE_TEAM_METADATA_KEYS`` env var "
            "(comma-separated) or "
            "``callback_settings.otel.baggage_team_metadata_keys`` in config.yaml."
        ),
    )

    @field_validator("capture_message_content", mode="before")
    @classmethod
    def _normalize_capture_message_content(cls, value: object) -> object:
        """Fold the capture mode to its canonical lower_snake_case form.

        V1 read this env var case-insensitively, so operators set the
        UPPER_SNAKE_CASE form (e.g. ``SPAN_AND_EVENT``). The canonical values
        here are lower_snake_case; normalizing at the boundary keeps both
        spellings working and lets every downstream comparison stay exact.
        """
        if isinstance(value, str):
            return value.lower()
        return value

    @field_validator(
        "baggage_promoted_keys",
        "baggage_metadata_keys",
        "baggage_team_metadata_keys",
        "mapper_names",
        mode="before",
    )
    @classmethod
    def _split_csv(cls, value: Any) -> Any:
        """Accept a comma-separated string for list fields.

        Env vars are strings, but these fields are lists. Pydantic-settings would
        otherwise require JSON for a list env var; splitting on commas here lets
        an operator write ``LITELLM_OTEL_BAGGAGE_PROMOTED_KEYS=litellm.team.id,litellm.api_key.hash``.
        YAML lists (from ``callback_settings.otel.*``) and real lists pass through
        unchanged.
        """
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @model_validator(mode="after")
    def _normalize(self) -> "OpenTelemetryV2Config":
        # An endpoint with the default exporter kind implies OTLP/HTTP.
        if self.endpoint and self.exporter == "console":
            self.exporter = "otlp_http"
        # When no explicit destinations are given, fold the single-destination
        # shorthand into one spec so the provider always has a destination.
        if not self.exporters:
            self.exporters = [
                ExporterSpec(
                    kind=self.exporter,
                    endpoint=self.endpoint,
                    headers=self.headers,
                )
            ]
        # Ensure ``genai`` is always present and first.
        names = list(self.mapper_names)
        if "genai" in names:
            names = ["genai"] + [n for n in names if n != "genai"]
        else:
            names = ["genai"] + names
        # When enabled, also emit attribute keys under their semconv-ai /
        # Traceloop names via the ``legacy`` mapper. Append it at the tail so
        # the canonical ``genai`` keys win on any conflict.
        if self.legacy_compat and "legacy" not in names:
            names.append("legacy")
        self.mapper_names = names
        return self

    @property
    def capture_span_content(self) -> bool:
        """Whether prompt/response content may be stamped as span attributes.

        Defaults off (``no_content``): an operator must opt in before message
        bodies leave the process, so a user request can never force its prompt
        or completion into the configured backend while capture is disabled.
        """
        return self.capture_message_content in (
            CaptureMessageContent.SPAN_ONLY,
            CaptureMessageContent.SPAN_AND_EVENT,
        )

    @classmethod
    def from_env(cls) -> "OpenTelemetryV2Config":
        return cls()
