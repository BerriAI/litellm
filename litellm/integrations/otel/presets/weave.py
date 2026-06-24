"""Weave (W&B) preset."""

from litellm.integrations.otel.model.config import (
    ExporterOwner,
    ExporterSpec,
    OpenTelemetryV2Config,
)
from litellm.integrations.otel.presets.utils import ensure_mappers
from litellm.integrations.weave.weave_otel import (
    _get_weave_authorization_header,
    get_weave_otel_config,
)
from litellm.types.utils import StandardCallbackDynamicParams


def weave_preset(
    *,
    config_overrides: OpenTelemetryV2Config | None = None,
) -> OpenTelemetryV2Config:
    weave_cfg = get_weave_otel_config()
    base = config_overrides or OpenTelemetryV2Config()
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
            # Weave consumes OpenInference + a small Weave-specific overlay.
            "mapper_names": ensure_mappers(base.mapper_names, "openinference", "weave"),
        }
    )


def weave_dynamic_headers(params: StandardCallbackDynamicParams) -> dict[str, str]:
    """Per-request Weave OTLP headers from team/key dynamic params."""
    headers: dict[str, str] = {}
    api_key = params.get("wandb_api_key")
    if api_key:
        headers["Authorization"] = _get_weave_authorization_header(api_key=api_key)
    project_id = params.get("weave_project_id")
    if project_id:
        headers["project_id"] = project_id
    return headers
