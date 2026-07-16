"""Langfuse-OTEL preset."""

import os

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


def _langfuse_static_headers() -> str | None:
    """OTLP auth header from proxy env keys, or ``None`` when only dynamic creds are used."""
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        return None
    auth = _V1Langfuse._get_langfuse_authorization_header(public_key=public_key, secret_key=secret_key)
    return f"Authorization={auth}"


def langfuse_preset(
    *,
    config_overrides: OpenTelemetryV2Config | None = None,
) -> OpenTelemetryV2Config:
    base = config_overrides or OpenTelemetryV2Config()
    return base.model_copy(
        update={
            "exporters": [
                *base.exporters,
                ExporterSpec(
                    kind="otlp_http",
                    endpoint=_V1Langfuse.get_langfuse_otel_endpoint(),
                    headers=_langfuse_static_headers(),
                    owner=ExporterOwner.LANGFUSE_OTEL,
                ),
            ],
            "mapper_names": ensure_mappers(base.mapper_names, "langfuse"),
        }
    )


def langfuse_dynamic_headers(params: StandardCallbackDynamicParams) -> dict[str, str]:
    """Per-request Langfuse OTLP headers from team/key dynamic params."""
    public_key = params.get("langfuse_public_key")
    secret_key = params.get("langfuse_secret_key")
    if public_key and secret_key:
        return {
            "Authorization": _V1Langfuse._get_langfuse_authorization_header(
                public_key=public_key, secret_key=secret_key
            )
        }
    return {}


def langfuse_dynamic_endpoint(params: StandardCallbackDynamicParams) -> str | None:
    """Per-request Langfuse OTLP endpoint from a team/key ``langfuse_host``, or ``None``."""
    host = params.get("langfuse_host")
    return _V1Langfuse.get_langfuse_otel_endpoint(host) if host else None
