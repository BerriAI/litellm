"""Langfuse-OTEL preset."""

from typing import Final

from litellm.integrations.langfuse.langfuse_otel import (
    LangfuseOtelLogger as _V1Langfuse,
)
from litellm.integrations.otel.model.config import (
    ExporterOwner,
    ExporterSpec,
    OpenTelemetryV2Config,
)
from litellm.integrations.otel.presets.utils import ensure_mappers
from litellm.types.utils import StandardCallbackDynamicParams


def langfuse_dynamic_headers(params: StandardCallbackDynamicParams) -> dict[str, str]:
    """Per-request Langfuse OTLP headers from team/key dynamic params."""
    public_key: Final = params.get("langfuse_public_key")
    secret_key: Final = params.get("langfuse_secret_key")
    if public_key and secret_key:
        return {
            "Authorization": _V1Langfuse._get_langfuse_authorization_header(
                public_key=public_key, secret_key=secret_key
            )
        }
    return {}


def langfuse_preset(
    *,
    config_overrides: OpenTelemetryV2Config | None = None,
    allow_missing_credentials: bool = False,
) -> OpenTelemetryV2Config:
    base: Final = config_overrides or OpenTelemetryV2Config()
    mappers = ensure_mappers(base.mapper_names, "langfuse")
    try:
        cfg = _V1Langfuse.get_langfuse_otel_config()
    except Exception:
        if not allow_missing_credentials:
            raise
        return base.model_copy(update={"mapper_names": mappers})
    kind: Final = cfg.exporter if isinstance(cfg.exporter, str) else "otlp_http"
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
