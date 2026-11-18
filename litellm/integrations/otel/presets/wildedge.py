import os
from types import MappingProxyType
from typing import Final
from urllib.parse import quote

from litellm.integrations.otel.model.config import ExporterOwner, ExporterSpec, OpenTelemetryV2Config
from litellm.integrations.otel.presets.utils import ensure_mappers


def wildedge_preset(
    *,
    config_overrides: OpenTelemetryV2Config | None = None,
    allow_missing_credentials: bool = False,
) -> OpenTelemetryV2Config:
    secret: Final = os.environ.get("WILDEDGE_PROJECT_SECRET", "")
    if not secret.strip():
        raise ValueError("Set WILDEDGE_PROJECT_SECRET to the WildEdge project secret.")
    endpoint: Final = os.environ.get("WILDEDGE_INGEST_URL", "https://ingest.wildedge.dev/api/otlp").strip()
    if not endpoint:
        raise ValueError("Set WILDEDGE_INGEST_URL to the WildEdge OTLP base URL or unset it for WildEdge Cloud.")
    base: Final = config_overrides or OpenTelemetryV2Config()
    exporter: Final = ExporterSpec(
        kind="otlp_http",
        endpoint=endpoint,
        headers=f"X-Project-Secret={quote(secret, safe='')}",
        owner=ExporterOwner.WILDEDGE,
        requires_headers=True,
    )
    return base.model_copy(
        update=MappingProxyType(
            {"exporters": (*base.exporters, exporter), "mapper_names": ensure_mappers(base.mapper_names, "genai")}
        )
    )
