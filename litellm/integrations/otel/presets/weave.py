"""Weave (W&B) preset."""

from typing import Final

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


def weave_dynamic_headers(params: StandardCallbackDynamicParams) -> dict[str, str]:
    """Per-request Weave OTLP headers from team/key dynamic params."""
    headers: dict[  # rebind-ok: reassigned on the branch below
        str, str
    ] = {}  # rebind-ok: populated by the optional-credential branches below  # mutable-ok: construction is handed to an API that needs a concrete dict
    api_key: Final = params.get("wandb_api_key")
    if api_key:
        headers["Authorization"] = _get_weave_authorization_header(api_key=api_key)
    project_id: Final = params.get("weave_project_id")
    if project_id:
        headers["project_id"] = project_id
    return headers


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
            update={"mapper_names": mappers}  # mutable-ok: handed to a model or SDK that needs a concrete dict
        )  # mutable-ok: pydantic model_copy takes a plain update mapping
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
            "mapper_names": mappers,
        }
    )
