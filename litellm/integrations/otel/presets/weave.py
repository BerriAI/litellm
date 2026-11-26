"""Weave (W&B) preset."""

from collections.abc import Mapping
from typing import Final

from litellm.integrations.otel.model.config import (
    ExporterOwner,
    ExporterSpec,
    OpenTelemetryV2Config,
)
from litellm.integrations.otel.presets.utils import (
    credential_gated_exporters,
    dynamic_param_text,
    ensure_mappers,
)
from litellm.integrations.weave.weave_otel import (
    get_weave_authorization_header,
    get_weave_otel_config,
)


def weave_preset(
    *,
    config_overrides: OpenTelemetryV2Config | None = None,
    allow_missing_credentials: bool = False,
) -> OpenTelemetryV2Config:
    base: Final = config_overrides or OpenTelemetryV2Config()
    mappers: Final = ensure_mappers(base.mapper_names, "openinference", "weave")
    try:
        weave_cfg: Final = get_weave_otel_config()
    except Exception:
        if not allow_missing_credentials:
            raise
        return base.model_copy(
            update={
                "exporters": credential_gated_exporters(base.exporters, ExporterOwner.WEAVE_OTEL),
                "mapper_names": mappers,
            }
        )
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
            "mapper_names": mappers,
        }
    )


def weave_dynamic_headers(params: Mapping[str, object]) -> dict[str, str]:
    """Per-request Weave OTLP headers from team/key dynamic params."""
    headers: Final[dict[str, str]] = {}
    api_key: Final = dynamic_param_text(params, "wandb_api_key")
    if api_key:
        headers["Authorization"] = get_weave_authorization_header(api_key=api_key)
    project_id: Final = dynamic_param_text(params, "weave_project_id")
    if project_id:
        headers["project_id"] = project_id
    return headers
