"""Openlayer preset.

Reads its config through ``OpenlayerLogger.get_openlayer_config`` so the v1 and
v2 paths cannot drift. Adds no mapper vocabulary: Openlayer reads the canonical
GenAI attributes the always-present ``genai`` mapper emits, and ``mapper_names``
is logger-wide rather than per-exporter, so a vocabulary added here would also
be stamped on spans bound for a co-configured backend.
"""

from typing import Final

from litellm.integrations.openlayer.openlayer import OpenlayerLogger as _V1Openlayer
from litellm.integrations.otel.model.config import (
    ExporterOwner,
    ExporterSpec,
    OpenTelemetryV2Config,
)


def openlayer_preset(
    *,
    config_overrides: OpenTelemetryV2Config | None = None,
) -> OpenTelemetryV2Config:
    cfg: Final = _V1Openlayer.get_openlayer_config()
    base: Final = config_overrides or OpenTelemetryV2Config()
    return base.model_copy(
        update={  # mutable-ok: model_copy(update=) requires a dict
            "exporters": [  # mutable-ok: field is declared list[ExporterSpec] and update= does not coerce
                *base.exporters,
                ExporterSpec(
                    kind="otlp_http",
                    endpoint=cfg.endpoint,
                    headers=cfg.otlp_auth_headers,
                    owner=ExporterOwner.OPENLAYER,
                ),
            ],
        }
    )
