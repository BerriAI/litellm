"""Langfuse-OTEL preset."""

from litellm.integrations.langfuse.langfuse_otel import (
    LangfuseOtelLogger as _V1Langfuse,
)
from litellm.integrations.otel.model.config import (
    ExporterOwner,
    ExporterSpec,
    OpenTelemetryV2Config,
)
from litellm.integrations.otel.presets.utils import ensure_mappers


def langfuse_preset(
    *,
    config_overrides: OpenTelemetryV2Config | None = None,
) -> OpenTelemetryV2Config:
    cfg = _V1Langfuse.get_langfuse_otel_config()
    kind = cfg.exporter if isinstance(cfg.exporter, str) else "otlp_http"
    base = config_overrides or OpenTelemetryV2Config()
    return base.model_copy(
        update={
            "exporters": [
                *base.exporters,
                ExporterSpec(
                    kind=kind,
                    endpoint=cfg.endpoint,
                    headers=cfg.headers,
                    owner=ExporterOwner.LANGFUSE_OTEL,
                ),
            ],
            "mapper_names": ensure_mappers(base.mapper_names, "langfuse"),
        }
    )
