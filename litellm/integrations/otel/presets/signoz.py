"""SigNoz preset — OTLP/HTTP exporter to SigNoz + GenAI vocabulary."""

from typing import Final

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from litellm.integrations.otel.model.config import (
    ExporterOwner,
    ExporterSpec,
    OpenTelemetryV2Config,
)
from litellm.integrations.otel.presets.utils import ensure_mappers
from litellm.types.utils import StandardCallbackDynamicParams

SIGNOZ_INGESTION_ENDPOINT_ENV: Final = "SIGNOZ_INGESTION_ENDPOINT"


class _SigNozSettings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    # One endpoint for both SigNoz Cloud and self-hosted, with no default host, so
    # nothing exports until the operator names a destination.
    endpoint: str | None = Field(default=None, validation_alias=SIGNOZ_INGESTION_ENDPOINT_ENV)
    ingestion_key: str | None = Field(default=None, validation_alias="SIGNOZ_INGESTION_KEY")


def signoz_preset(
    *,
    config_overrides: OpenTelemetryV2Config | None = None,
) -> OpenTelemetryV2Config:
    settings: Final = _SigNozSettings()
    base: Final = config_overrides or OpenTelemetryV2Config()
    key: Final = settings.ingestion_key
    return base.model_copy(
        update={  # mutable-ok: model_copy takes a dict of field updates
            "exporters": [  # mutable-ok: matches the config's exporters list
                *base.exporters,
                ExporterSpec(
                    kind="otlp_http",
                    endpoint=settings.endpoint,
                    headers=(f"signoz-ingestion-key={key}" if key else None),
                    owner=ExporterOwner.SIGNOZ,
                    # Cloud rejects keyless exports; a self-hosted collector accepts them.
                    requires_headers=bool(key),
                ),
            ],
            # SigNoz ingests the OTLP GenAI semantic conventions natively.
            "mapper_names": ensure_mappers(base.mapper_names, "genai"),
        }
    )


def signoz_dynamic_endpoint(params: StandardCallbackDynamicParams) -> str | None:
    """Per-request SigNoz endpoint from team/key dynamic params.

    ``None`` keeps the operator's endpoint, so a team that saved only a key
    still exports to the configured destination.
    """
    return params.get("signoz_ingestion_endpoint")


def signoz_dynamic_headers(params: StandardCallbackDynamicParams) -> dict[str, str]:  # mutable-ok: registry type
    """Per-request SigNoz OTLP headers from team/key dynamic params."""
    key: Final = params.get("signoz_ingestion_key")
    return {header: value for header, value in (("signoz-ingestion-key", key),) if value}  # mutable-ok: registry type
