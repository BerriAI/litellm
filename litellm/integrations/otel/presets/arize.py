"""Arize preset — OTLP exporter to Arize + OpenInference vocabulary."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from litellm.integrations.arize.arize import ArizeLogger as _V1ArizeLogger
from litellm.integrations.otel.model.config import (
    ExporterOwner,
    ExporterSpec,
    OpenTelemetryV2Config,
)
from litellm.integrations.otel.presets.utils import ensure_mappers


class _ArizeSettings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    # Standard OTLP headers env var, used as the fallback when no Arize
    # credentials are configured.
    otlp_traces_headers: str | None = Field(default=None, validation_alias="OTEL_EXPORTER_OTLP_TRACES_HEADERS")


def arize_preset(
    *,
    config_overrides: OpenTelemetryV2Config | None = None,
    allow_missing_credentials: bool = False,
) -> OpenTelemetryV2Config:
    arize_cfg = _V1ArizeLogger.get_arize_config()
    headers = _arize_headers(arize_cfg)
    base = config_overrides or OpenTelemetryV2Config()
    # Contribute the global Arize exporter only when Arize credentials are
    # configured. Without them it points at the Arize cloud with no auth and every
    # export fails PERMISSION_DENIED; admin-owned destinations carry their own
    # credentials and are appended by the router instead.
    global_exporter = (
        (
            ExporterSpec(
                kind=arize_cfg.protocol or "otlp_grpc",
                endpoint=arize_cfg.endpoint or "https://otlp.arize.com/v1",
                headers=headers,
                owner=ExporterOwner.ARIZE_AX,
            ),
        )
        if headers
        else ()
    )
    return base.model_copy(
        update={
            "exporters": [*base.exporters, *global_exporter],
            "mapper_names": ensure_mappers(base.mapper_names, "openinference"),
            "resource_attributes": {
                **base.resource_attributes,
                **({"model_id": arize_cfg.project_name} if arize_cfg.project_name else {}),
            },
        }
    )


def _arize_headers(arize_cfg) -> str | None:
    pieces = []
    if arize_cfg.space_id or arize_cfg.space_key:
        pieces.append(f"space_id={arize_cfg.space_id or arize_cfg.space_key}")
    if arize_cfg.api_key:
        pieces.append(f"api_key={arize_cfg.api_key}")
    if not pieces:
        # Fall back to the standard OTLP headers env var when no Arize
        # credentials are configured.
        return _ArizeSettings().otlp_traces_headers
    return ",".join(pieces)
