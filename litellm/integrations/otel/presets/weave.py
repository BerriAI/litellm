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
    # Weave consumes OpenInference + a small Weave-specific overlay.
    mappers = ensure_mappers(base.mapper_names, "openinference", "weave")
    # ``get_weave_otel_config()`` raises without W&B credentials. Propagate that raise
    # for a global callback so a misconfigured deployment fails loud, but when an
    # admin-owned Weave destination is the reason for construction it carries its own
    # per-tenant credentials, so degrade to a (global-exporter-less) mapper-only config
    # -- otherwise the gen-AI span falls to the generic logger and never reaches Weave.
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
