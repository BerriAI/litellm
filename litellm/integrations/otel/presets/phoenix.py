"""Arize-Phoenix preset."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from litellm.integrations.arize.arize_phoenix import (
    ArizePhoenixLogger as _V1Phoenix,
)
from litellm.integrations.otel.model.config import (
    ExporterOwner,
    ExporterSpec,
    OpenTelemetryV2Config,
)
from litellm.integrations.otel.presets.utils import ensure_mappers


class _PhoenixSettings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    project_name: str = Field(
        default="default",
        validation_alias=AliasChoices("PHOENIX_PROJECT_NAME", "PHOENIX_COLLECTOR_PROJECT_NAME"),
    )


#: Phoenix routes an OTLP/HTTP export to a project by this header, which takes
#: precedence over the ``openinference.project.name`` resource attribute the env
#: var sets. Requires arize-phoenix 15.5.0+; older collectors ignore it and the
#: spans land in the resource attribute's project.
PHOENIX_PROJECT_HEADER: Final = "x-project-name"

#: Key/team config fields naming the target project, highest precedence first.
_PROJECT_KEYS: Final = ("phoenix_project_name_override", "phoenix_project_name")

_NO_PROJECT: Final[Mapping[str, str]] = MappingProxyType({})


def phoenix_project_headers(auth_metadata: Mapping[str, str] | None) -> Mapping[str, str]:
    """The per-request Phoenix project header for this key/team, if any.

    ``auth_metadata`` must be the key/team config the proxy resolved at auth
    (``user_api_key_auth_metadata``), never client-supplied request metadata:
    choosing the destination project is a data-exfiltration primitive, so a
    caller must not be able to name one. Returns an empty mapping when the key
    and team name no project, leaving the request on the env-configured default.
    """
    if not auth_metadata:
        return _NO_PROJECT
    project: Final = next(
        (stripped for key in _PROJECT_KEYS if (stripped := (auth_metadata.get(key) or "").strip())),
        "",
    )
    return MappingProxyType({PHOENIX_PROJECT_HEADER: project}) if project else _NO_PROJECT


def phoenix_preset(
    *,
    config_overrides: OpenTelemetryV2Config | None = None,
) -> OpenTelemetryV2Config:
    cfg: Final = _V1Phoenix.get_arize_phoenix_config()
    headers: Final = cfg.otlp_auth_headers if hasattr(cfg, "otlp_auth_headers") else None
    project_name: Final = _PhoenixSettings().project_name
    base: Final = config_overrides or OpenTelemetryV2Config()
    return base.model_copy(
        update={
            "exporters": [
                *base.exporters,
                ExporterSpec(
                    kind=cfg.protocol if hasattr(cfg, "protocol") else "otlp_http",
                    endpoint=cfg.endpoint,
                    headers=headers,
                    owner=ExporterOwner.ARIZE_PHOENIX,
                ),
            ],
            "mapper_names": ensure_mappers(base.mapper_names, "openinference"),
            "resource_attributes": {
                **base.resource_attributes,
                "openinference.project.name": project_name,
            },
        }
    )
