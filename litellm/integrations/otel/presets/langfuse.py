"""Langfuse-OTEL preset."""

import os
from collections.abc import Mapping
from typing import Final

from litellm.integrations.langfuse.langfuse_otel import (
    LangfuseOtelLogger as _V1Langfuse,
)
from litellm.integrations.otel.model.config import (
    ExporterOwner,
    ExporterSpec,
    OpenTelemetryV2Config,
)
from litellm.integrations.otel.presets.utils import (
    credential_gated_exporters,
    ensure_mappers,
)
from litellm.types.integrations.langfuse_otel import LangfuseSpanAttributes
from litellm.types.utils import StandardCallbackDynamicParams


def langfuse_environment_attributes(environment: str | None = None) -> Mapping[str, str]:
    """``langfuse.environment`` for ``environment``, else for ``LANGFUSE_TRACING_ENVIRONMENT``. Langfuse reads it off
    the resource."""
    resolved: Final = environment or os.environ.get("LANGFUSE_TRACING_ENVIRONMENT")
    return {LangfuseSpanAttributes.LANGFUSE_ENVIRONMENT.value: resolved} if resolved else {}


def langfuse_preset(
    *,
    config_overrides: OpenTelemetryV2Config | None = None,
    allow_missing_credentials: bool = False,
) -> OpenTelemetryV2Config:
    base: Final = config_overrides or OpenTelemetryV2Config()
    mappers: Final = ensure_mappers(base.mapper_names, "langfuse")
    try:
        cfg: Final = _V1Langfuse.get_langfuse_otel_config()
    except Exception:
        if not allow_missing_credentials:
            raise
        return base.model_copy(
            update={
                "exporters": credential_gated_exporters(base.exporters, ExporterOwner.LANGFUSE_OTEL),
                "mapper_names": mappers,
            }
        )
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
                    resource_attributes=langfuse_environment_attributes(),
                ),
            ],
            "mapper_names": mappers,
        }
    )


def langfuse_dynamic_headers(params: StandardCallbackDynamicParams) -> dict[str, str]:
    """Per-request Langfuse OTLP headers from team/key dynamic params."""
    public_key: Final = params.get("langfuse_public_key")
    secret_key: Final = params.get("langfuse_secret_key")
    if public_key and secret_key:
        return _V1Langfuse._build_langfuse_otel_headers(
            _V1Langfuse._get_langfuse_authorization_header(public_key=public_key, secret_key=secret_key)
        )
    return {}
