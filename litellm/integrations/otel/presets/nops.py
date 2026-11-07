from types import MappingProxyType
from typing import Final

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from litellm.integrations.otel.model.config import (
    ExporterOwner,
    ExporterSpec,
    OpenTelemetryV2Config,
)
from litellm.integrations.otel.presets.utils import ensure_mappers

NOPS_API_KEY_ENV: Final = "NOPS_API_KEY"
NOPS_OTEL_ENDPOINT_ENV: Final = "NOPS_OTEL_ENDPOINT"
NOPS_DEFAULT_OTEL_ENDPOINT: Final = "https://api.nops.io/otel"


class _NopsSettings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore", frozen=True)

    endpoint: str | None = Field(default=None, validation_alias=NOPS_OTEL_ENDPOINT_ENV)
    api_key: str | None = Field(default=None, validation_alias=NOPS_API_KEY_ENV)


def nops_preset(
    *,
    config_overrides: OpenTelemetryV2Config | None = None,
    allow_missing_credentials: bool = False,
) -> OpenTelemetryV2Config:
    settings: Final = _NopsSettings()
    base: Final = config_overrides or OpenTelemetryV2Config()
    key: Final = settings.api_key
    spec: Final = ExporterSpec(
        kind="otlp_http",
        endpoint=settings.endpoint or NOPS_DEFAULT_OTEL_ENDPOINT,
        headers=(f"Authorization=Bearer {key}" if key else None),
        owner=ExporterOwner.NOPS,
        requires_headers=True,
    )
    return base.model_copy(
        update=MappingProxyType(
            {
                "exporters": (*base.exporters, spec),
                "mapper_names": ensure_mappers(base.mapper_names, "genai"),
            }
        )
    )
