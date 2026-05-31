"""Levo preset — OTLP/HTTP to a Levo collector with org+workspace headers."""

from litellm.integrations.levo.levo import LevoLogger as _V1Levo
from litellm.integrations.otel.model.config import ExporterSpec, OpenTelemetryV2Config


def levo_preset(
    *,
    config_overrides: OpenTelemetryV2Config | None = None,
) -> OpenTelemetryV2Config:
    cfg = _V1Levo.get_levo_config()
    base = config_overrides or OpenTelemetryV2Config()
    return base.model_copy(
        update={
            "exporters": [
                *base.exporters,
                ExporterSpec(
                    kind="otlp_http",
                    endpoint=cfg.endpoint,
                    headers=cfg.otlp_auth_headers,
                ),
            ],
        }
    )
