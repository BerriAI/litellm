from functools import lru_cache
from types import MappingProxyType
from typing import Final

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.otel.model.config import (
    ExporterOwner,
    ExporterSpec,
    OpenTelemetryV2Config,
)
from litellm.integrations.otel.presets.utils import ensure_mappers
from litellm.litellm_core_utils.url_utils import is_url_destination_allowed_by_host
from litellm.types.utils import StandardCallbackDynamicParams

SIGNOZ_INGESTION_ENDPOINT_ENV: Final = "SIGNOZ_INGESTION_ENDPOINT"


class _SigNozSettings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    endpoint: str | None = Field(default=None, validation_alias=SIGNOZ_INGESTION_ENDPOINT_ENV)
    ingestion_key: str | None = Field(default=None, validation_alias="SIGNOZ_INGESTION_KEY")


def signoz_preset(
    *,
    config_overrides: OpenTelemetryV2Config | None = None,
    allow_missing_credentials: bool = False,
) -> OpenTelemetryV2Config:
    settings: Final = _SigNozSettings()
    base: Final = config_overrides or OpenTelemetryV2Config()
    key: Final = settings.ingestion_key
    spec: Final = ExporterSpec(
        kind="otlp_http",
        endpoint=settings.endpoint,
        headers=(f"signoz-ingestion-key={key}" if key else None),
        owner=ExporterOwner.SIGNOZ,
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


@lru_cache(maxsize=128)
def _warn_host_not_allowlisted(endpoint: str) -> None:
    verbose_logger.warning(
        "SigNoz: not exporting to key/team endpoint '%s'. Add its host to "
        "litellm_settings.provider_url_destination_allowed_hosts to permit it",
        endpoint,
    )


def _tenant_endpoint_is_unusable(params: StandardCallbackDynamicParams) -> bool:
    return bool(params.get("signoz_ingestion_endpoint")) and signoz_dynamic_endpoint(params) is None


def signoz_dynamic_endpoint(params: StandardCallbackDynamicParams) -> str | None:
    endpoint: Final = params.get("signoz_ingestion_endpoint")
    if not endpoint or not endpoint.startswith(("http://", "https://")):
        return None
    if not is_url_destination_allowed_by_host(endpoint, litellm.provider_url_destination_allowed_hosts):
        _warn_host_not_allowlisted(endpoint)
        return None
    return endpoint


def signoz_dynamic_headers(
    params: StandardCallbackDynamicParams,
) -> dict[str, str]:  # mutable-ok: DYNAMIC_HEADERS_BY_CALLBACK returns a dict
    key: Final = params.get("signoz_ingestion_key")
    if not key or _tenant_endpoint_is_unusable(params):
        return {}  # mutable-ok: same registry contract
    return {"signoz-ingestion-key": key}  # mutable-ok: same registry contract
