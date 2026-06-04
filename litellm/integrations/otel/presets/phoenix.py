"""Arize-Phoenix preset."""

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from litellm.integrations.arize.arize_phoenix import (
    ArizePhoenixLogger as _V1Phoenix,
)
from litellm.integrations.otel.model.config import ExporterSpec, OpenTelemetryV2Config
from litellm.integrations.otel.presets.utils import ensure_mappers


class _PhoenixSettings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    project_name: str = Field(
        default="default",
        validation_alias=AliasChoices(
            "PHOENIX_PROJECT_NAME", "PHOENIX_COLLECTOR_PROJECT_NAME"
        ),
    )


def phoenix_preset(
    *,
    config_overrides: OpenTelemetryV2Config | None = None,
) -> OpenTelemetryV2Config:
    cfg = _V1Phoenix.get_arize_phoenix_config()
    headers = cfg.otlp_auth_headers if hasattr(cfg, "otlp_auth_headers") else None
    project_name = _PhoenixSettings().project_name
    base = config_overrides or OpenTelemetryV2Config()
    return base.model_copy(
        update={
            "exporters": [
                *base.exporters,
                ExporterSpec(
                    kind=cfg.protocol if hasattr(cfg, "protocol") else "otlp_http",
                    endpoint=cfg.endpoint,
                    headers=headers,
                ),
            ],
            "mapper_names": ensure_mappers(base.mapper_names, "openinference"),
            "resource_attributes": {
                **base.resource_attributes,
                "openinference.project.name": project_name,
            },
        }
    )
