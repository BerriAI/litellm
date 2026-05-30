"""Arize preset — OTLP exporter to Arize + OpenInference vocabulary."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from litellm.integrations.arize.arize import ArizeLogger as _V1ArizeLogger
from litellm.integrations.otel.model.config import ExporterSpec, OpenTelemetryV2Config
from litellm.integrations.otel.presets.utils import ensure_mappers
from litellm.types.utils import StandardCallbackDynamicParams


class _ArizeSettings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    # Standard OTLP headers env var, used as the fallback when no Arize
    # credentials are configured.
    otlp_traces_headers: str | None = Field(
        default=None, validation_alias="OTEL_EXPORTER_OTLP_TRACES_HEADERS"
    )


def arize_preset(
    *,
    config_overrides: OpenTelemetryV2Config | None = None,
) -> OpenTelemetryV2Config:
    arize_cfg = _V1ArizeLogger.get_arize_config()
    headers = _arize_headers(arize_cfg)
    base = config_overrides or OpenTelemetryV2Config()
    return base.model_copy(
        update={
            "exporters": [
                *base.exporters,
                ExporterSpec(
                    kind=arize_cfg.protocol or "otlp_grpc",
                    endpoint=arize_cfg.endpoint or "https://otlp.arize.com/v1",
                    headers=headers,
                ),
            ],
            "mapper_names": ensure_mappers(base.mapper_names, "openinference"),
            "resource_attributes": {
                **base.resource_attributes,
                **(
                    {"model_id": arize_cfg.project_name}
                    if arize_cfg.project_name
                    else {}
                ),
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


def arize_dynamic_headers(params: StandardCallbackDynamicParams) -> dict[str, str]:
    """Per-request Arize OTLP headers from team/key dynamic params."""
    headers: dict[str, str] = {}
    # ``arize_space_key`` is the suggested param and wins over ``arize_space_id``.
    space = params.get("arize_space_key") or params.get("arize_space_id")
    if space:
        headers["arize-space-id"] = space
    api_key = params.get("arize_api_key")
    if api_key:
        headers["api_key"] = api_key
    return headers
