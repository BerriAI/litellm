"""LangWatch preset: OTLP/HTTP to LangWatch with a bearer API key.

LangWatch (https://langwatch.ai) ingests the canonical OpenTelemetry GenAI
semantic conventions, so the default ``genai`` vocabulary is all it needs.
Self-hosted deployments serve the same path on their own host, set through
``LANGWATCH_ENDPOINT``.
"""

from typing import Final

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from litellm.integrations.otel.model.config import (
    ExporterOwner,
    ExporterSpec,
    OpenTelemetryV2Config,
)
from litellm.integrations.otel.presets.utils import (
    ensure_mappers,
    is_unconfigured_placeholder,
)

LANGWATCH_DEFAULT_ENDPOINT: Final = "https://app.langwatch.ai"
_LANGWATCH_OTEL_TRACES_PATH: Final = "/api/otel/v1/traces"


class _LangWatchSettings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    api_key: str | None = Field(default=None, validation_alias="LANGWATCH_API_KEY")
    endpoint: str | None = Field(default=None, validation_alias="LANGWATCH_ENDPOINT")


def langwatch_otel_traces_endpoint(base_url: str | None) -> str:
    """The OTLP/HTTP traces URL for a LangWatch base URL (cloud when unset)."""
    base: Final = (base_url or "").strip().rstrip("/") or LANGWATCH_DEFAULT_ENDPOINT
    normalized: Final = base if base.startswith(("http://", "https://")) else f"https://{base}"
    if normalized.endswith(_LANGWATCH_OTEL_TRACES_PATH):
        return normalized
    return f"{normalized}{_LANGWATCH_OTEL_TRACES_PATH}"


def get_langwatch_otel_config() -> tuple[str, str]:
    """Return ``(traces_endpoint, otlp_headers)`` from the LangWatch env vars.

    Raises ``ValueError`` when ``LANGWATCH_API_KEY`` is unset: LangWatch rejects
    unauthenticated exports, so a keyless exporter would only produce 401s.
    """
    settings: Final = _LangWatchSettings()
    if not settings.api_key:
        raise ValueError("LANGWATCH_API_KEY environment variable is required for the LangWatch integration.")
    return (
        langwatch_otel_traces_endpoint(settings.endpoint),
        f"Authorization=Bearer {settings.api_key}",
    )


def langwatch_preset(
    *,
    config_overrides: OpenTelemetryV2Config | None = None,
    allow_missing_credentials: bool = False,
) -> OpenTelemetryV2Config:
    endpoint, headers = get_langwatch_otel_config()
    base: Final = config_overrides or OpenTelemetryV2Config()
    return base.model_copy(
        update={
            # Drop the console placeholder ``OpenTelemetryV2Config`` folds in when
            # nothing was configured, so spans go to LangWatch and not to stdout.
            # An exporter the operator configured on purpose is kept.
            "exporters": [
                *(spec for spec in base.exporters if not is_unconfigured_placeholder(spec)),
                ExporterSpec(
                    kind="otlp_http",
                    endpoint=endpoint,
                    headers=headers,
                    owner=ExporterOwner.LANGWATCH,
                ),
            ],
            # LangWatch reads the OTel GenAI semantic conventions natively.
            "mapper_names": ensure_mappers(base.mapper_names, "genai"),
        }
    )
