"""Levo preset — OTLP/HTTP to a Levo collector with org+workspace headers."""

from litellm.integrations.levo.levo import LevoLogger as _V1Levo
from litellm.integrations.otel.model.config import (
    ExporterOwner,
    ExporterSpec,
    OpenTelemetryV2Config,
)


def levo_preset(
    *,
    config_overrides: OpenTelemetryV2Config | None = None,
    allow_missing_credentials: bool = False,
) -> OpenTelemetryV2Config:
    base = config_overrides or OpenTelemetryV2Config()
    # ``get_levo_config()`` raises without Levo credentials. Propagate that raise for a
    # global callback so a misconfigured deployment fails loud, but when an admin-owned
    # Levo destination is the reason for construction it carries its own per-tenant
    # credentials, so degrade to a global-exporter-less config rather than raising.
    try:
        cfg = _V1Levo.get_levo_config()
    except Exception:
        if not allow_missing_credentials:
            raise
        return base
    return base.model_copy(
        update={
            "exporters": [
                *base.exporters,
                ExporterSpec(
                    kind="otlp_http",
                    endpoint=cfg.endpoint,
                    headers=cfg.otlp_auth_headers,
                    owner=ExporterOwner.LEVO,
                ),
            ],
        }
    )
