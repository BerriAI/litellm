import asyncio
import os
import sys
from unittest.mock import patch

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.management_helpers.team_metadata_validation import (
    DEFAULT_TEAM_METADATA_VALIDATION_REJECTED_MESSAGE,
    DEFAULT_TEAM_METADATA_VALIDATION_TIMEOUT_SECONDS,
    DEFAULT_TEAM_METADATA_VALIDATION_UNAVAILABLE_MESSAGE,
    TeamMetadataRequester,
    TeamMetadataValidationPayload,
    TeamMetadataValidationResult,
    TeamMetadataValidatorRegistry,
    _read_timeout_seconds,
    _read_unavailable_message,
    run_team_metadata_validation,
    validate_team_metadata_if_configured,
)


def _registry_with(validator):
    registry = TeamMetadataValidatorRegistry()
    registry.set(validator)
    return registry

UNAVAILABLE_MESSAGE = "validation system down, contact ops"


def _payload(**overrides):
    values = {
        "operation": "create",
        "metadata": {"cost_center": "CC-1001"},
        "existing_metadata": None,
        "team_id": "team-1",
        "team_alias": "alias-1",
        "requester": TeamMetadataRequester(user_id="u1"),
    }
    values.update(overrides)
    return TeamMetadataValidationPayload(**values)


async def _run(validator, payload=None, premium_user=True, timeout_seconds=1.0):
    await run_team_metadata_validation(
        validator=validator,
        payload=payload or _payload(),
        premium_user=premium_user,
        timeout_seconds=timeout_seconds,
        unavailable_message=UNAVAILABLE_MESSAGE,
    )


@pytest.mark.asyncio
async def test_valid_result_passes():
    async def validator(payload):
        return TeamMetadataValidationResult(valid=True)

    await _run(validator)


@pytest.mark.asyncio
async def test_rejection_raises_400_with_validator_message():
    async def validator(payload):
        return TeamMetadataValidationResult(valid=False, error_message="cost center rejected, contact FinOps")

    with pytest.raises(HTTPException) as exc_info:
        await _run(validator)
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == {"error": "cost center rejected, contact FinOps"}


@pytest.mark.asyncio
async def test_rejection_without_message_uses_default():
    async def validator(payload):
        return TeamMetadataValidationResult(valid=False)

    with pytest.raises(HTTPException) as exc_info:
        await _run(validator)
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == {"error": DEFAULT_TEAM_METADATA_VALIDATION_REJECTED_MESSAGE}


@pytest.mark.asyncio
async def test_dict_shaped_return_is_accepted():
    async def validator(payload):
        return {"valid": False, "error_message": "rejected via dict"}

    with pytest.raises(HTTPException) as exc_info:
        await _run(validator)
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == {"error": "rejected via dict"}


@pytest.mark.asyncio
async def test_validator_exception_fails_closed_with_generic_message():
    async def validator(payload):
        raise RuntimeError("internal validation service is down")

    with pytest.raises(HTTPException) as exc_info:
        await _run(validator)
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == {"error": UNAVAILABLE_MESSAGE}


@pytest.mark.asyncio
async def test_validator_timeout_fails_closed():
    async def validator(payload):
        await asyncio.sleep(1.0)
        return TeamMetadataValidationResult(valid=True)

    with pytest.raises(HTTPException) as exc_info:
        await _run(validator, timeout_seconds=0.01)
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == {"error": UNAVAILABLE_MESSAGE}


@pytest.mark.asyncio
async def test_malformed_return_shape_fails_closed():
    async def validator(payload):
        return "not-a-validation-result"

    with pytest.raises(HTTPException) as exc_info:
        await _run(validator)
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == {"error": UNAVAILABLE_MESSAGE}


@pytest.mark.asyncio
async def test_non_premium_user_is_rejected():
    async def validator(payload):
        return TeamMetadataValidationResult(valid=True)

    with pytest.raises(HTTPException) as exc_info:
        await _run(validator, premium_user=False)
    assert exc_info.value.status_code == 400
    assert CommonProxyErrors.not_premium_user.value in exc_info.value.detail["error"]


@pytest.mark.asyncio
async def test_non_coroutine_validator_is_rejected():
    def validator(payload):
        return TeamMetadataValidationResult(valid=True)

    with pytest.raises(HTTPException) as exc_info:
        await _run(validator)
    assert exc_info.value.status_code == 500
    assert "async" in exc_info.value.detail["error"]


@pytest.mark.parametrize(
    "general_settings, expected",
    [
        ({}, DEFAULT_TEAM_METADATA_VALIDATION_TIMEOUT_SECONDS),
        ({"team_metadata_validation_timeout": 2}, 2.0),
        ({"team_metadata_validation_timeout": 0.5}, 0.5),
        ({"team_metadata_validation_timeout": True}, DEFAULT_TEAM_METADATA_VALIDATION_TIMEOUT_SECONDS),
        ({"team_metadata_validation_timeout": -1}, DEFAULT_TEAM_METADATA_VALIDATION_TIMEOUT_SECONDS),
        ({"team_metadata_validation_timeout": "3"}, DEFAULT_TEAM_METADATA_VALIDATION_TIMEOUT_SECONDS),
    ],
)
def test_read_timeout_seconds(general_settings, expected):
    assert _read_timeout_seconds(general_settings) == expected


@pytest.mark.parametrize(
    "general_settings, expected",
    [
        ({}, DEFAULT_TEAM_METADATA_VALIDATION_UNAVAILABLE_MESSAGE),
        (
            {"team_metadata_validation_error_message": "call the help desk"},
            "call the help desk",
        ),
        ({"team_metadata_validation_error_message": "  "}, DEFAULT_TEAM_METADATA_VALIDATION_UNAVAILABLE_MESSAGE),
        ({"team_metadata_validation_error_message": None}, DEFAULT_TEAM_METADATA_VALIDATION_UNAVAILABLE_MESSAGE),
    ],
)
def test_read_unavailable_message(general_settings, expected):
    assert _read_unavailable_message(general_settings) == expected


@pytest.mark.asyncio
async def test_adapter_is_noop_when_unconfigured():
    calls = []

    async def validator(payload):
        calls.append(payload)
        return TeamMetadataValidationResult(valid=True)

    await validate_team_metadata_if_configured(
        operation="create",
        metadata={"cost_center": "CC-1001"},
        existing_metadata=None,
        team_id="team-1",
        team_alias="alias-1",
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="u1"),
        registry=TeamMetadataValidatorRegistry(),
    )
    assert calls == []


@pytest.mark.asyncio
async def test_adapter_builds_payload_and_reads_settings():
    recorded = []

    async def validator(payload):
        recorded.append(payload)
        return TeamMetadataValidationResult(valid=True)

    with (
        patch("litellm.proxy.proxy_server.premium_user", True),
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {"team_metadata_validation_timeout": 3, "team_metadata_validation_error_message": "ops msg"},
        ),
    ):
        await validate_team_metadata_if_configured(
            operation="update",
            metadata={"cost_center": "CC-2001"},
            existing_metadata={"cost_center": "CC-1001", "keep": 1},
            team_id="team-9",
            team_alias="alias-9",
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.INTERNAL_USER,
                user_id="user-9",
                user_email="user-9@example.com",
            ),
            registry=_registry_with(validator),
        )

    assert len(recorded) == 1
    payload = recorded[0]
    assert payload.operation == "update"
    assert payload.metadata == {"cost_center": "CC-2001"}
    assert payload.existing_metadata == {"cost_center": "CC-1001", "keep": 1}
    assert payload.team_id == "team-9"
    assert payload.team_alias == "alias-9"
    assert payload.requester.user_id == "user-9"
    assert payload.requester.user_email == "user-9@example.com"
    assert payload.requester.user_role == LitellmUserRoles.INTERNAL_USER.value


@pytest.mark.asyncio
async def test_adapter_normalizes_non_dict_metadata_to_empty_dict():
    recorded = []

    async def validator(payload):
        recorded.append(payload)
        return TeamMetadataValidationResult(valid=True)

    with (
        patch("litellm.proxy.proxy_server.premium_user", True),
        patch("litellm.proxy.proxy_server.general_settings", {}),
    ):
        await validate_team_metadata_if_configured(
            operation="create",
            metadata=None,
            existing_metadata=None,
            team_id="team-1",
            team_alias=None,
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="u1"),
            registry=_registry_with(validator),
        )

    assert len(recorded) == 1
    assert recorded[0].metadata == {}
    assert recorded[0].existing_metadata is None
