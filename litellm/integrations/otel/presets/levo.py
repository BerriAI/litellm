"""Levo preset — OTLP/HTTP to a Levo collector with org+workspace headers."""

from typing import Final

from litellm.integrations.levo.levo import LevoLogger as _V1Levo
from litellm.integrations.otel.model.config import (
    ExporterOwner,
    ExporterSpec,
    OpenTelemetryV2Config,
)
from litellm.integrations.otel.presets.utils import operator_exporters


def levo_preset(
    *,
    config_overrides: OpenTelemetryV2Config | None = None,
    allow_missing_credentials: bool = False,
) -> OpenTelemetryV2Config:
    cfg: Final = _V1Levo.get_levo_config()
    base: Final = config_overrides or OpenTelemetryV2Config()
    return base.model_copy(
        update={
            "exporters": [
                *operator_exporters(base.exporters),
                ExporterSpec(
                    kind="otlp_http",
                    endpoint=cfg.endpoint,
                    headers=cfg.otlp_auth_headers,
                    owner=ExporterOwner.LEVO,
                ),
            ],
        }
    )
