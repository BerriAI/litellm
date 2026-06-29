"""Generic OTLP passthrough preset.

A ``generic`` admin-owned destination carries only an ``otel_endpoint`` /
``otel_headers`` (no vendor adapter), so this preset is deliberately vendor-neutral:
it yields a base ``OpenTelemetryV2`` config with the standard ``gen_ai.*`` (+ ``legacy``)
span vocabulary, NO vendor mapper, and NO exporter of its own. The per-destination
exporter is appended by ``TenantTracerCache`` from the resolved ``OtelDestination``, so a
generic destination receives the same complete trace tree as Arize/Langfuse/Weave --
root server span, auth/db internal spans, the ``chat <model>`` gen-AI span, and
cost/error spans. It needs no global OTEL env vars and never raises.
"""

from litellm.integrations.otel.model.config import OpenTelemetryV2Config


def generic_preset(
    *,
    config_overrides: OpenTelemetryV2Config | None = None,
    allow_missing_credentials: bool = False,
) -> OpenTelemetryV2Config:
    return config_overrides or OpenTelemetryV2Config()
