import base64
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
from litellm.integrations.otel.model.semconv import User
from litellm.integrations.otel.presets.utils import ensure_mappers
from litellm.types.utils import StandardCallbackDynamicParams

MIXPANEL_PROJECT_TOKEN_ENV: Final = "MIXPANEL_PROJECT_TOKEN"

MIXPANEL_OTLP_ENDPOINT_BY_REGION: Final[Mapping[str, str]] = MappingProxyType(
    {
        "us": "https://ingestion-us.mixpanel.com/v1/ai/otel/traces",
        "eu": "https://ingestion-eu.mixpanel.com/v1/ai/otel/traces",
        "in": "https://ingestion-in.mixpanel.com/v1/ai/otel/traces",
    }
)

_DEFAULT_REGION: Final = "us"


class _MixpanelSettings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore", frozen=True)

    project_token: str | None = Field(default=None, validation_alias=MIXPANEL_PROJECT_TOKEN_ENV)
    region: str | None = Field(default=None, validation_alias="MIXPANEL_REGION")
    endpoint: str | None = Field(default=None, validation_alias="MIXPANEL_ENDPOINT")


def mixpanel_authorization_header(project_token: str) -> str:
    encoded: Final = base64.b64encode(f"{project_token}:".encode()).decode()
    return f"Basic {encoded}"


def _resolve_region_endpoint(region: str | None) -> str:
    default_endpoint: Final = MIXPANEL_OTLP_ENDPOINT_BY_REGION[_DEFAULT_REGION]
    if not region:
        return default_endpoint
    endpoint: Final = MIXPANEL_OTLP_ENDPOINT_BY_REGION.get(region.strip().lower())
    if endpoint is None:
        verbose_logger.warning(
            "Mixpanel: unknown mixpanel_region %r; supported regions: %s. Using the default (US) endpoint.",
            region,
            ", ".join(sorted(MIXPANEL_OTLP_ENDPOINT_BY_REGION)),
        )
        return default_endpoint
    return endpoint


def mixpanel_preset(
    *,
    config_overrides: OpenTelemetryV2Config | None = None,
    allow_missing_credentials: bool = False,
) -> OpenTelemetryV2Config:
    settings: Final = _MixpanelSettings()
    base: Final = config_overrides or OpenTelemetryV2Config()
    token: Final = (settings.project_token or "").strip() or None
    endpoint: Final = (settings.endpoint or "").strip() or _resolve_region_endpoint(settings.region)
    spec: Final = ExporterSpec(
        kind="otlp_http",
        endpoint=endpoint,
        headers=(f"Authorization={mixpanel_authorization_header(token)}" if token else None),
        owner=ExporterOwner.MIXPANEL,
        requires_headers=True,
    )
    return base.model_copy(
        update=MappingProxyType(
            {
                "exporters": (*base.exporters, spec),
                "mapper_names": ensure_mappers(base.mapper_names, "genai", "mixpanel"),
                "baggage_promoted_keys": ensure_mappers(base.baggage_promoted_keys, User.ID),
            }
        )
    )


def mixpanel_dynamic_headers(
    params: StandardCallbackDynamicParams,
) -> dict[str, str]:  # mutable-ok: DYNAMIC_HEADERS_BY_CALLBACK returns a dict
    token: Final = (params.get("mixpanel_project_token") or "").strip()
    if not token:
        return {}
    return {"Authorization": mixpanel_authorization_header(token)}


def mixpanel_dynamic_endpoint(params: StandardCallbackDynamicParams) -> str:
    return _resolve_region_endpoint(params.get("mixpanel_region"))
