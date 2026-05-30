"""Langfuse-OTEL preset."""

from litellm.integrations.langfuse.langfuse_otel import (
    LangfuseOtelLogger as _V1Langfuse,
)
from litellm.integrations.otel.model.config import ExporterSpec, OpenTelemetryV2Config
from litellm.integrations.otel.presets.utils import ensure_mappers
from litellm.types.utils import StandardCallbackDynamicParams


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
                ),
            ],
            "mapper_names": ensure_mappers(base.mapper_names, "langfuse"),
        }
    )


def langfuse_dynamic_headers(params: StandardCallbackDynamicParams) -> dict[str, str]:
    """Per-request Langfuse OTLP headers from team/key dynamic params."""
    public_key = params.get("langfuse_public_key")
    secret_key = params.get("langfuse_secret_key")
    if public_key and secret_key:
        return {
            "Authorization": _V1Langfuse._get_langfuse_authorization_header(
                public_key=public_key, secret_key=secret_key
            )
        }
    return {}
