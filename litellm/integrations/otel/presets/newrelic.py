"""New Relic preset — OTLP/HTTP exporter to New Relic + GenAI vocabulary."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from litellm._logging import verbose_logger
from litellm.integrations.otel.model.config import (
    ExporterOwner,
    ExporterSpec,
    OpenTelemetryV2Config,
)
from litellm.integrations.otel.presets.utils import ensure_mappers
from litellm.types.utils import StandardCallbackDynamicParams

#: Region -> OTLP base endpoint. A fixed table by design: team config picks a
#: region enum rather than a free-form endpoint, so callback vars can never
#: redirect telemetry to an arbitrary host.
NEWRELIC_OTLP_ENDPOINT_BY_REGION: Final[Mapping[str, str]] = MappingProxyType(
    {
        "us": "https://otlp.nr-data.net",
        "eu": "https://otlp.eu01.nr-data.net",
    }
)

_DEFAULT_REGION: Final = "us"


class _NewRelicSettings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    # The same env vars the agent-based integration documents; the key is the
    # operator-level fallback for traffic without team credentials, the region
    # picks that fallback's data center, and the record-content flag keeps its
    # documented meaning when the OTel path replaces the agent.
    license_key: str | None = Field(default=None, validation_alias="NEW_RELIC_LICENSE_KEY")
    region: str | None = Field(default=None, validation_alias="NEW_RELIC_REGION")
    record_content: bool | None = Field(default=None, validation_alias="NEW_RELIC_AI_MONITORING_RECORD_CONTENT_ENABLED")


def newrelic_preset(
    *,
    config_overrides: OpenTelemetryV2Config | None = None,
) -> OpenTelemetryV2Config:
    settings: Final = _NewRelicSettings()
    base: Final = config_overrides or OpenTelemetryV2Config()
    endpoint: Final = NEWRELIC_OTLP_ENDPOINT_BY_REGION.get(
        (settings.region or _DEFAULT_REGION).lower(), NEWRELIC_OTLP_ENDPOINT_BY_REGION[_DEFAULT_REGION]
    )
    return base.model_copy(
        update={
            "exporters": [
                *base.exporters,
                ExporterSpec(
                    kind="otlp_http",
                    endpoint=endpoint,
                    headers=(f"api-key={settings.license_key}" if settings.license_key else None),
                    owner=ExporterOwner.NEWRELIC,
                    requires_headers=True,
                ),
            ],
            # New Relic ingests the OTLP GenAI semantic conventions natively.
            "mapper_names": ensure_mappers(base.mapper_names, "genai"),
            **(
                {"capture_message_content": ("span_only" if settings.record_content else "no_content")}
                if settings.record_content is not None
                else {}
            ),
        }
    )


def newrelic_dynamic_headers(params: StandardCallbackDynamicParams) -> dict[str, str]:
    """Per-request New Relic OTLP headers from team/key dynamic params."""
    api_key: Final = params.get("newrelic_api_key")
    return {header: value for header, value in (("api-key", api_key),) if value}


def newrelic_dynamic_endpoint(params: StandardCallbackDynamicParams) -> str:
    """Per-request OTLP endpoint for the team's ``newrelic_region``.

    Always the team's own region endpoint, defaulting to US when the team left
    the region unset. It never falls through to the preset's endpoint, which
    follows the operator's ``NEW_RELIC_REGION`` env; a team that saved only its
    ingest key must not inherit the operator's region and have its US-account
    spans rejected by an EU-configured default (or vice versa). An unknown
    region likewise resolves to the documented US default rather than a guess.
    """
    region: Final = params.get("newrelic_region")
    default_endpoint: Final = NEWRELIC_OTLP_ENDPOINT_BY_REGION[_DEFAULT_REGION]
    if not region:
        return default_endpoint
    endpoint: Final = NEWRELIC_OTLP_ENDPOINT_BY_REGION.get(region.lower())
    if endpoint is None:
        verbose_logger.warning(
            "New Relic: unknown newrelic_region %r; supported regions: %s. Using the default (US) endpoint.",
            region,
            ", ".join(sorted(NEWRELIC_OTLP_ENDPOINT_BY_REGION)),
        )
        return default_endpoint
    return endpoint
