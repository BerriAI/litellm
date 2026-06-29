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
    allow_missing_credentials: bool = False,
) -> OpenTelemetryV2Config:
    base = config_overrides or OpenTelemetryV2Config()
    mappers = ensure_mappers(base.mapper_names, "langfuse")
    # ``get_langfuse_otel_config()`` raises without Langfuse keys. Propagate that raise
    # for a global callback so a misconfigured deployment fails loud, but when an
    # admin-owned Langfuse destination is the reason for construction it carries its own
    # per-tenant keys, so degrade to a (global-exporter-less) mapper-only config -- or
    # the gen-AI span falls to the generic logger and never reaches it.
    try:
        cfg = _V1Langfuse.get_langfuse_otel_config()
    except Exception:
        if not allow_missing_credentials:
            raise
        return base.model_copy(update={"mapper_names": mappers})
    kind = cfg.exporter if isinstance(cfg.exporter, str) else "otlp_http"
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
            "mapper_names": mappers,
        }
    )
