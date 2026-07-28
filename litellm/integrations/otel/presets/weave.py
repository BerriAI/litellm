"""Weave (W&B) preset."""

from litellm.integrations.otel.model.config import (
    ExporterOwner,
    ExporterSpec,
    OpenTelemetryV2Config,
)
from litellm.integrations.otel.presets.utils import ensure_mappers
from litellm.integrations.weave.weave_otel import get_weave_otel_config


def weave_preset(
    *,
    config_overrides: OpenTelemetryV2Config | None = None,
    allow_missing_credentials: bool = False,
) -> OpenTelemetryV2Config:
    base = config_overrides or OpenTelemetryV2Config()
    mappers = ensure_mappers(base.mapper_names, "openinference", "weave")
    try:
        weave_cfg = get_weave_otel_config()
    except Exception:
        if not allow_missing_credentials:
            raise
        return base.model_copy(update={"mapper_names": mappers})
    return base.model_copy(
        update={
            "exporters": [
                *base.exporters,
                ExporterSpec(
                    kind=weave_cfg.protocol or "otlp_http",
                    endpoint=weave_cfg.endpoint,
                    headers=weave_cfg.otlp_auth_headers,
                    owner=ExporterOwner.WEAVE_OTEL,
                ),
            ],
            "mapper_names": mappers,
        }
    )
