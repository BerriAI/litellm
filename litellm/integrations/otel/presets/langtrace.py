"""Langtrace preset — Langtrace consumes generic OTLP + a vendor mapper."""

from litellm.integrations.otel.model.config import OpenTelemetryV2Config
from litellm.integrations.otel.presets.utils import ensure_mappers


def langtrace_preset(
    *,
    config_overrides: OpenTelemetryV2Config | None = None,
) -> OpenTelemetryV2Config:
    """Compose the Langtrace mapper on top of the customer's OTLP destination.

    Unlike Arize / Phoenix / Langfuse, Langtrace doesn't ship its own endpoint
    — users point their existing OTLP collector at Langtrace and just
    need the vendor attribute schema applied to outgoing spans.
    """
    base = config_overrides or OpenTelemetryV2Config()
    return base.model_copy(
        update={
            "mapper_names": ensure_mappers(base.mapper_names, "langtrace"),
        }
    )
