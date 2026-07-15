"""Custom validation of team metadata on team create/update.

Operators point `general_settings.custom_team_metadata_validate` at an async
Python function (loaded via `get_instance_fn`, like `custom_key_generate`).
The function receives a `TeamMetadataValidationPayload` and returns a
`TeamMetadataValidationResult`. The proxy awaits it before committing a team
write and fails closed: a rejected value surfaces the function's own message
(HTTP 400), while any raised exception or timeout blocks the write with a
generic message (HTTP 503).
"""

import asyncio
import inspect
from typing import Awaitable, Callable, Literal, Optional

from fastapi import HTTPException, status
from pydantic import BaseModel, JsonValue

from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth

DEFAULT_TEAM_METADATA_VALIDATION_TIMEOUT_SECONDS = 5.0
DEFAULT_TEAM_METADATA_VALIDATION_UNAVAILABLE_MESSAGE = (
    "Team metadata validation is currently unavailable, so the team was not saved. Contact your proxy admin."
)
DEFAULT_TEAM_METADATA_VALIDATION_REJECTED_MESSAGE = "Team metadata failed validation."


class TeamMetadataRequester(BaseModel):
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    user_role: Optional[str] = None


class TeamMetadataValidationPayload(BaseModel):
    operation: Literal["create", "update"]
    metadata: dict[str, JsonValue]
    existing_metadata: Optional[dict[str, JsonValue]] = None
    team_id: Optional[str] = None
    team_alias: Optional[str] = None
    requester: TeamMetadataRequester


class TeamMetadataValidationResult(BaseModel):
    valid: bool
    error_message: Optional[str] = None


TeamMetadataValidator = Callable[[TeamMetadataValidationPayload], Awaitable[TeamMetadataValidationResult]]


class TeamMetadataValidatorRegistry:
    def __init__(self) -> None:
        self._validator: Optional[TeamMetadataValidator] = None

    def set(self, validator: Optional[TeamMetadataValidator]) -> None:
        self._validator = validator

    def get(self) -> Optional[TeamMetadataValidator]:
        return self._validator


TEAM_METADATA_VALIDATOR_REGISTRY = TeamMetadataValidatorRegistry()


async def run_team_metadata_validation(
    validator: TeamMetadataValidator,
    payload: TeamMetadataValidationPayload,
    premium_user: bool,
    timeout_seconds: float,
    unavailable_message: str,
) -> None:
    if premium_user is not True:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": f"custom_team_metadata_validate is an Enterprise feature. {CommonProxyErrors.not_premium_user.value}"
            },
        )
    if not inspect.iscoroutinefunction(validator):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "custom_team_metadata_validate must be an async function"},
        )

    try:
        raw_result = await asyncio.wait_for(validator(payload), timeout=timeout_seconds)
        result = TeamMetadataValidationResult.model_validate(raw_result)
    except Exception:  # noqa: BLE001  # fail closed: any validator failure must block the team write
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": unavailable_message},
        )

    if not result.valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": result.error_message or DEFAULT_TEAM_METADATA_VALIDATION_REJECTED_MESSAGE},
        )


def _read_timeout_seconds(general_settings: dict) -> float:
    raw_timeout = general_settings.get("team_metadata_validation_timeout")
    if isinstance(raw_timeout, (int, float)) and not isinstance(raw_timeout, bool) and raw_timeout > 0:
        return float(raw_timeout)
    return DEFAULT_TEAM_METADATA_VALIDATION_TIMEOUT_SECONDS


def _read_unavailable_message(general_settings: dict) -> str:
    raw_message = general_settings.get("team_metadata_validation_error_message")
    if isinstance(raw_message, str) and raw_message.strip():
        return raw_message
    return DEFAULT_TEAM_METADATA_VALIDATION_UNAVAILABLE_MESSAGE


async def validate_team_metadata_if_configured(
    operation: Literal["create", "update"],
    metadata: Optional[dict],
    existing_metadata: Optional[dict],
    team_id: Optional[str],
    team_alias: Optional[str],
    user_api_key_dict: UserAPIKeyAuth,
    registry: TeamMetadataValidatorRegistry = TEAM_METADATA_VALIDATOR_REGISTRY,
) -> None:
    from litellm.proxy.proxy_server import general_settings, premium_user

    validator = registry.get()
    if validator is None:
        return

    payload = TeamMetadataValidationPayload(
        operation=operation,
        metadata=metadata if isinstance(metadata, dict) else {},
        existing_metadata=existing_metadata if isinstance(existing_metadata, dict) else None,
        team_id=team_id,
        team_alias=team_alias,
        requester=TeamMetadataRequester(
            user_id=user_api_key_dict.user_id,
            user_email=user_api_key_dict.user_email,
            user_role=user_api_key_dict.user_role.value if user_api_key_dict.user_role is not None else None,
        ),
    )
    await run_team_metadata_validation(
        validator=validator,
        payload=payload,
        premium_user=premium_user,
        timeout_seconds=_read_timeout_seconds(general_settings),
        unavailable_message=_read_unavailable_message(general_settings),
    )
