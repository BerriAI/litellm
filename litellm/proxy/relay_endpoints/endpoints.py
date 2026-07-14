import os
from pathlib import Path

import yaml
from fastapi import APIRouter, Depends, HTTPException

from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.proxy.relay_endpoints import RelayManagedConfigResponse

router = APIRouter()

RELAY_SETTINGS_PATH_ENV = "LITELLM_RELAY_SETTINGS_PATH"
DEFAULT_RELAY_SETTINGS_PATH = "relay_settings.yaml"


def _relay_settings_path() -> Path:
    return Path(os.getenv(RELAY_SETTINGS_PATH_ENV, DEFAULT_RELAY_SETTINGS_PATH))


def _load_managed_config(path: Path) -> RelayManagedConfigResponse:
    if not path.exists():
        return RelayManagedConfigResponse()

    try:
        parsed = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to parse Relay settings at {path}: {e}"},
        )

    if parsed is None:
        return RelayManagedConfigResponse()
    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=500,
            detail={"error": f"Relay settings at {path} must be a YAML mapping, got {type(parsed).__name__}"},
        )
    return RelayManagedConfigResponse.model_validate(parsed)


@router.get(
    "/relay/managed-config",
    tags=["LiteLLM Relay"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=RelayManagedConfigResponse,
)
async def get_relay_managed_config() -> RelayManagedConfigResponse:
    return _load_managed_config(_relay_settings_path())
