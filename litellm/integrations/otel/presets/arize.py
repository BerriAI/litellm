"""Arize preset — OTLP exporter to Arize + OpenInference vocabulary."""

from typing import Final

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from litellm.integrations.arize.arize import ArizeLogger as _V1ArizeLogger
from litellm.integrations.otel.model.config import (
    ExporterOwner,
    ExporterSpec,
    OpenTelemetryV2Config,
)
from litellm.integrations.otel.presets.utils import credential_gated_exporters, ensure_mappers
from litellm.types.utils import StandardCallbackDynamicParams

#: Arize routes an export to a project by the ``model_id`` resource attribute and
#: rejects one that names none, so this is the project when ``ARIZE_PROJECT_NAME``
#: is unset.
ARIZE_DEFAULT_PROJECT: Final = "litellm"


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
    base: Final = config_overrides or OpenTelemetryV2Config()
    mappers: Final = ensure_mappers(base.mapper_names, "openinference")
    arize_cfg: Final = _V1ArizeLogger.get_arize_config()
    headers: Final = _arize_headers(arize_cfg)
    resource_attributes: Final = {
        **base.resource_attributes,
        "model_id": arize_cfg.project_name or base.resource_attributes.get("model_id") or ARIZE_DEFAULT_PROJECT,
    }
    if headers is None:
        return base.model_copy(
            update={
                "exporters": credential_gated_exporters(base.exporters, ExporterOwner.ARIZE_AX),
                "mapper_names": mappers,
                "resource_attributes": resource_attributes,
            }
        )
    return base.model_copy(
        update={
            "exporters": [
                *base.exporters,
                ExporterSpec(
                    kind=arize_cfg.protocol or "otlp_grpc",
                    endpoint=arize_cfg.endpoint or "https://otlp.arize.com/v1",
                    headers=headers,
                    owner=ExporterOwner.ARIZE_AX,
                ),
            ],
            "mapper_names": mappers,
            "resource_attributes": resource_attributes,
        }
    )


def _arize_headers(arize_cfg) -> str | None:
    pieces: Final = []
    if arize_cfg.space_id or arize_cfg.space_key:
        pieces.append(f"space_id={arize_cfg.space_id or arize_cfg.space_key}")
    if arize_cfg.api_key:
        pieces.append(f"api_key={arize_cfg.api_key}")
    if not pieces:
        # Fall back to the standard OTLP headers env var when no Arize
        # credentials are configured.
        fallback: Final = _ArizeSettings().otlp_traces_headers
        return (fallback.strip() or None) if fallback else None
    return ",".join(pieces)


def arize_dynamic_headers(params: StandardCallbackDynamicParams) -> dict[str, str]:
    """Per-request Arize OTLP headers from team/key dynamic params."""
    headers: Final[dict[str, str]] = {}
    # ``arize_space_key`` is the suggested param and wins over ``arize_space_id``.
    space: Final = params.get("arize_space_key") or params.get("arize_space_id")
    if space:
        headers["arize-space-id"] = space
    api_key: Final = params.get("arize_api_key")
    if api_key:
        headers["api_key"] = api_key
    return headers
