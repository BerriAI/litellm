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
from collections.abc import Awaitable, Mapping
from types import MappingProxyType
from typing import Final, Literal, Protocol

from fastapi import HTTPException, status
from pydantic import BaseModel, JsonValue, TypeAdapter

from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.types.proxy.management_endpoints.team_endpoints import (
    TeamMetadataFieldSchema,
)

DEFAULT_TEAM_METADATA_VALIDATION_TIMEOUT_SECONDS: Final = 5.0
DEFAULT_TEAM_METADATA_VALIDATION_UNAVAILABLE_MESSAGE: Final = (
    "Team metadata validation is currently unavailable, so the team was not saved. Contact your proxy admin."
)
DEFAULT_TEAM_METADATA_VALIDATION_REJECTED_MESSAGE: Final = "Team metadata failed validation."


class TeamMetadataRequester(BaseModel):
    user_id: str | None = None
    user_email: str | None = None
    user_role: str | None = None


class TeamMetadataValidationPayload(BaseModel):
    operation: Literal["create", "update"]
    metadata: Mapping[str, JsonValue]
    existing_metadata: Mapping[str, JsonValue] | None = None
    team_id: str | None = None
    team_alias: str | None = None
    requester: TeamMetadataRequester


class TeamMetadataValidationResult(BaseModel):
    valid: bool
    error_message: str | None = None


_EMPTY_METADATA: Final[Mapping[str, JsonValue]] = MappingProxyType({})


class TeamMetadataValidator(Protocol):
    def __call__(self, payload: TeamMetadataValidationPayload, /) -> Awaitable[TeamMetadataValidationResult]: ...


class TeamMetadataValidatorRegistry:
    def __init__(self) -> None:
        self._validator: TeamMetadataValidator | None = None

    def set(self, validator: TeamMetadataValidator | None) -> None:
        self._validator = validator

    def get(self) -> TeamMetadataValidator | None:
        return self._validator


TEAM_METADATA_VALIDATOR_REGISTRY: Final = TeamMetadataValidatorRegistry()

_TEAM_METADATA_SCHEMA_ADAPTER: Final[TypeAdapter[tuple[TeamMetadataFieldSchema, ...]]] = TypeAdapter(
    tuple[TeamMetadataFieldSchema, ...]
)


def parse_team_metadata_schema(raw_schema: object) -> tuple[TeamMetadataFieldSchema, ...]:
    """Parse ``general_settings.team_metadata_schema``; raises on a malformed schema so config load fails fast."""
    if raw_schema is None:
        return ()
    fields: Final = _TEAM_METADATA_SCHEMA_ADAPTER.validate_python(raw_schema)
    keys: Final = tuple(field.key for field in fields)
    duplicate_keys: Final = sorted(frozenset(key for key in keys if keys.count(key) > 1))
    if duplicate_keys:
        raise ValueError(f"team_metadata_schema contains duplicate keys: {', '.join(duplicate_keys)}")
    return fields


class TeamMetadataSchemaRegistry:
    def __init__(self) -> None:
        self._fields: tuple[TeamMetadataFieldSchema, ...] = ()

    def set(self, fields: tuple[TeamMetadataFieldSchema, ...]) -> None:
        self._fields = fields

    def get(self) -> tuple[TeamMetadataFieldSchema, ...]:
        return self._fields


TEAM_METADATA_SCHEMA_REGISTRY: Final = TeamMetadataSchemaRegistry()


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
            detail={  # mutable-ok: HTTPException.detail has no immutable form
                "error": f"custom_team_metadata_validate is an Enterprise feature. {CommonProxyErrors.not_premium_user.value}"
            },
        )
    if not (
        inspect.iscoroutinefunction(validator) or inspect.iscoroutinefunction(getattr(validator, "__call__", None))
    ):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={  # mutable-ok: HTTPException.detail has no immutable form
                "error": "custom_team_metadata_validate must be an async function"
            },
        )

    try:
        raw_result: Final = await asyncio.wait_for(validator(payload), timeout=timeout_seconds)
        result: Final = TeamMetadataValidationResult.model_validate(raw_result)
    except Exception:  # noqa: BLE001  # fail closed: any validator failure must block the team write
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": unavailable_message},  # mutable-ok: HTTPException.detail has no immutable form
        )

    if not result.valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={  # mutable-ok: HTTPException.detail has no immutable form
                "error": result.error_message or DEFAULT_TEAM_METADATA_VALIDATION_REJECTED_MESSAGE
            },
        )


def _read_timeout_seconds(general_settings: Mapping[str, object]) -> float:
    raw_timeout: Final = general_settings.get("team_metadata_validation_timeout")
    if isinstance(raw_timeout, (int, float)) and not isinstance(raw_timeout, bool) and raw_timeout > 0:
        return float(raw_timeout)
    return DEFAULT_TEAM_METADATA_VALIDATION_TIMEOUT_SECONDS


def _read_unavailable_message(general_settings: Mapping[str, object]) -> str:
    raw_message: Final = general_settings.get("team_metadata_validation_error_message")
    if isinstance(raw_message, str) and raw_message.strip():
        return raw_message
    return DEFAULT_TEAM_METADATA_VALIDATION_UNAVAILABLE_MESSAGE


async def validate_team_metadata_if_configured(
    operation: Literal["create", "update"],
    metadata: Mapping[str, JsonValue] | None,
    existing_metadata: Mapping[str, JsonValue] | None,
    team_id: str | None,
    team_alias: str | None,
    user_api_key_dict: UserAPIKeyAuth,
    registry: TeamMetadataValidatorRegistry = TEAM_METADATA_VALIDATOR_REGISTRY,
) -> None:
    from litellm.proxy.proxy_server import general_settings, premium_user

    validator: Final = registry.get()
    if validator is None:
        return

    payload: Final = TeamMetadataValidationPayload(
        operation=operation,
        metadata=metadata if isinstance(metadata, dict) else _EMPTY_METADATA,
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
