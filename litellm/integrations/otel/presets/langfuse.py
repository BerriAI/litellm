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


def langfuse_preset(
    *,
    config_overrides: OpenTelemetryV2Config | None = None,
) -> OpenTelemetryV2Config:
    cfg: Final = _V1Langfuse.get_langfuse_otel_config()
    kind: Final = cfg.exporter if isinstance(cfg.exporter, str) else "otlp_http"
    base: Final = config_overrides or OpenTelemetryV2Config()
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
            "mapper_names": ensure_mappers(base.mapper_names, "langfuse"),
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


def langfuse_dynamic_endpoint(params: StandardCallbackDynamicParams) -> str | None:
    """Per-request Langfuse OTLP endpoint when the key or team pins its own host.

    ``None`` means the request does not move the destination, so the preset's
    env-resolved endpoint stands (V1 parity: ``construct_dynamic_otel_config``
    falls back to the env host when the dynamic params carry no ``langfuse_host``).

    A host only counts alongside the key pair it belongs to, which is the same
    precondition ``langfuse_dynamic_headers`` applies and the same one V1 applies.
    A host on its own would move the endpoint while the headers builder returned
    nothing, so the exporter would keep the operator's env-derived Authorization
    header and post it to the caller's host.
    """
    host: Final = params.get("langfuse_host")
    if not host or not params.get("langfuse_public_key") or not params.get("langfuse_secret_key"):
        return None
    return _V1Langfuse.get_langfuse_otel_endpoint(host)
