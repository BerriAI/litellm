"""SigNoz preset — OTLP/HTTP exporter to SigNoz + GenAI vocabulary."""

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
from litellm.types.utils import StandardCallbackDynamicParams

SIGNOZ_INGESTION_ENDPOINT_ENV: Final = "SIGNOZ_INGESTION_ENDPOINT"


class _SigNozSettings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    # No default host: nothing exports until the operator names a destination
    endpoint: str | None = Field(default=None, validation_alias=SIGNOZ_INGESTION_ENDPOINT_ENV)
    ingestion_key: str | None = Field(default=None, validation_alias="SIGNOZ_INGESTION_KEY")


def signoz_preset(
    *,
    config_overrides: OpenTelemetryV2Config | None = None,
) -> OpenTelemetryV2Config:
    settings: Final = _SigNozSettings()
    base: Final = config_overrides or OpenTelemetryV2Config()
    key: Final = settings.ingestion_key
    spec: Final = ExporterSpec(
        kind="otlp_http",
        endpoint=settings.endpoint,
        headers=(f"signoz-ingestion-key={key}" if key else None),
        owner=ExporterOwner.SIGNOZ,
        # Cloud rejects keyless exports; a self-hosted collector accepts them
        requires_headers=bool(key),
    )
    return base.model_copy(
        update=MappingProxyType(
            {
                "exporters": (*base.exporters, spec),
                "mapper_names": ensure_mappers(base.mapper_names, "genai"),
            }
        )
    )


def signoz_dynamic_endpoint(params: StandardCallbackDynamicParams) -> str | None:
    """Per-request SigNoz endpoint from team/key dynamic params; ``None`` keeps the operator's."""
    endpoint: Final = params.get("signoz_ingestion_endpoint")
    if not endpoint or not endpoint.startswith(("http://", "https://")):
        return None
    return endpoint


def signoz_dynamic_headers(
    params: StandardCallbackDynamicParams,
) -> dict[str, str]:  # mutable-ok: DYNAMIC_HEADERS_BY_CALLBACK returns a dict
    """Per-request SigNoz OTLP headers from team/key dynamic params."""
    key: Final = params.get("signoz_ingestion_key")
    return {"signoz-ingestion-key": key} if key else {}  # mutable-ok: same registry contract
