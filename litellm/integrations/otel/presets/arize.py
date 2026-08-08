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
from litellm.integrations.otel.presets.utils import ensure_mappers
from litellm.types.utils import StandardCallbackDynamicParams

ARIZE_PUBLIC_OTLP_ENDPOINT = "https://otlp.arize.com/v1"


class _ArizeSettings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    # Standard OTLP headers env var, used as the fallback when no Arize
    # credentials are configured.
    otlp_traces_headers: str | None = Field(default=None, validation_alias="OTEL_EXPORTER_OTLP_TRACES_HEADERS")
    grpc_endpoint: str | None = Field(default=None, validation_alias="ARIZE_ENDPOINT")
    http_endpoint: str | None = Field(default=None, validation_alias="ARIZE_HTTP_ENDPOINT")


def arize_preset(
    *,
    config_overrides: OpenTelemetryV2Config | None = None,
    allow_missing_credentials: bool = False,
) -> OpenTelemetryV2Config:
    arize_cfg: Final = _V1ArizeLogger.get_arize_config()
    settings = _ArizeSettings()
    has_own_endpoint = bool(settings.grpc_endpoint or settings.http_endpoint)
    has_credentials = bool(arize_cfg.space_id or arize_cfg.space_key or arize_cfg.api_key)
    base: Final = config_overrides or OpenTelemetryV2Config()
    global_exporter = (
        ()
        if allow_missing_credentials and not has_credentials and not has_own_endpoint
        else (
            ExporterSpec(
                kind=arize_cfg.protocol or "otlp_grpc",
                endpoint=arize_cfg.endpoint or ARIZE_PUBLIC_OTLP_ENDPOINT,
                headers=_arize_headers(arize_cfg, settings, has_own_endpoint),
                owner=ExporterOwner.ARIZE_AX,
            ),
        )
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


def _arize_headers(arize_cfg, settings: "_ArizeSettings", has_own_endpoint: bool) -> str | None:
    space: Final = arize_cfg.space_id or arize_cfg.space_key
    pieces: Final = (
        *((f"space_id={space}",) if space else ()),
        *((f"api_key={arize_cfg.api_key}",) if arize_cfg.api_key else ()),
    )
    if pieces:
        return ",".join(pieces)
    # Fall back to the standard OTLP headers env var only for an operator's own
    # collector. Sending it to the public Arize endpoint would hand that collector's
    # auth header to a vendor the operator has no account with.
    return settings.otlp_traces_headers if has_own_endpoint else None


def arize_dynamic_headers(params: StandardCallbackDynamicParams) -> dict[str, str]:
    """Per-request Arize OTLP headers from team/key dynamic params."""
    headers: dict[str, str] = {}
    # ``arize_space_key`` is the suggested param and wins over ``arize_space_id``.
    space = params.get("arize_space_key") or params.get("arize_space_id")
    if space:
        headers["arize-space-id"] = space
    api_key: Final = params.get("arize_api_key")
    if api_key:
        headers["api_key"] = api_key
    return headers
