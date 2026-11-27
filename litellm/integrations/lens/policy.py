from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, ConfigDict, StrictBool, ValidationError

from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth

LENS_ENABLED_METADATA_KEY: Final = "lens_enabled"
LENS_LOGGING_POLICY_PARAM: Final = "_lens_logging_policy"


@dataclass(frozen=True, slots=True)
class LensLoggingPolicy:
    enabled: bool


class LensLoggingSettings(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    lens_enabled: StrictBool | None = None


class _AuthMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, from_attributes=True)

    metadata: Mapping[str, object] | None = None
    team_metadata: Mapping[str, object] | None = None


def lens_logging_metadata_error(metadata: Mapping[str, object] | None) -> str | None:
    try:
        LensLoggingSettings.model_validate(metadata or {})
    except ValidationError:
        return "metadata.lens_enabled must be a boolean or null"
    return None


def _enabled_override(metadata: Mapping[str, object] | None) -> bool | None:
    try:
        return LensLoggingSettings.model_validate(metadata or {}).lens_enabled
    except ValidationError:
        return None


def lens_logging_enabled(auth: UserAPIKeyAuth) -> bool:
    authenticated: Final = _AuthMetadata.model_validate(auth)
    key_override: Final = _enabled_override(authenticated.metadata)
    if key_override is not None:
        return key_override
    team_default: Final = _enabled_override(authenticated.team_metadata)
    if team_default is not None:
        return team_default
    return True


def _authenticated_key(metadata: object) -> UserAPIKeyAuth | None:
    fields: Final = as_str_mapping(metadata)
    if fields is None:
        return None
    auth: Final = fields.get("user_api_key_auth")
    return auth if isinstance(auth, UserAPIKeyAuth) else None


def lens_logging_enabled_for_call(kwargs: Mapping[str, object]) -> bool:
    params: Final = as_str_mapping(kwargs.get("litellm_params")) or {}
    auth: Final = _authenticated_key(params.get("metadata")) or _authenticated_key(params.get("litellm_metadata"))
    if auth is not None:
        return lens_logging_enabled(auth)
    policy: Final = params.get(LENS_LOGGING_POLICY_PARAM)
    if isinstance(policy, LensLoggingPolicy):
        return policy.enabled
    return True
