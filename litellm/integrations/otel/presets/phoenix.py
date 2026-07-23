"""Arize-Phoenix preset."""

import os

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from litellm.integrations.arize.arize_phoenix import (
    ArizePhoenixLogger as _V1Phoenix,
)
from litellm.integrations.otel.model.config import (
    ExporterOwner,
    ExporterSpec,
    OpenTelemetryV2Config,
)
from litellm.integrations.otel.presets.utils import ensure_mappers


class _PhoenixSettings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    project_name: str = Field(
        default="default",
        validation_alias=AliasChoices("PHOENIX_PROJECT_NAME", "PHOENIX_COLLECTOR_PROJECT_NAME"),
    )


_PHOENIX_ENV_VARS = (
    "PHOENIX_API_KEY",
    "PHOENIX_COLLECTOR_ENDPOINT",
    "PHOENIX_COLLECTOR_HTTP_ENDPOINT",
)


def phoenix_preset(
    *,
    config_overrides: OpenTelemetryV2Config | None = None,
    allow_missing_credentials: bool = False,
) -> OpenTelemetryV2Config:
    project_name = _PhoenixSettings().project_name
    base = config_overrides or OpenTelemetryV2Config()
    # Contribute the global Phoenix exporter only when Phoenix is configured (a
    # cloud API key or a collector endpoint). Otherwise the config defaults to
    # http://localhost:6006 and would export there even when the operator only
    # uses admin-owned Phoenix destinations.
    if any(os.environ.get(v) for v in _PHOENIX_ENV_VARS):
        cfg = _V1Phoenix.get_arize_phoenix_config()
        headers = cfg.otlp_auth_headers if hasattr(cfg, "otlp_auth_headers") else None
        global_exporter = (
            ExporterSpec(
                kind=cfg.protocol if hasattr(cfg, "protocol") else "otlp_http",
                endpoint=cfg.endpoint,
                headers=headers,
                owner=ExporterOwner.ARIZE_PHOENIX,
            ),
        )
    else:
        global_exporter = ()
    return base.model_copy(
        update={
            "exporters": [*base.exporters, *global_exporter],
            "mapper_names": ensure_mappers(base.mapper_names, "openinference"),
            "resource_attributes": {
                **base.resource_attributes,
                "openinference.project.name": project_name,
            },
        }
    )
