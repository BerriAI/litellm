"""
KEY MANAGEMENT

All /key management endpoints

/key/generate
/key/info
/key/update
/key/delete
"""

import asyncio
import copy
import inspect
import json
import os
import secrets
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

import fastapi
import yaml
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status

import litellm
from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.caching import DualCache
from litellm.constants import (
    LENGTH_OF_LITELLM_GENERATED_KEY,
    LITELLM_PROXY_ADMIN_NAME,
    UI_SESSION_TOKEN_TEAM_ID,
)
from litellm.litellm_core_utils.duration_parser import duration_in_seconds
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy._experimental.mcp_server.db import (
    rotate_mcp_server_credentials_master_key,
)
from litellm.proxy._types import *
from litellm.proxy._types import LiteLLM_VerificationToken
from litellm.proxy.auth.auth_checks import (
    _cache_key_object,
    _delete_cache_key_object,
    can_team_access_model,
    get_key_object,
    get_org_object,
    get_team_object,
)
from litellm.proxy.auth.auth_utils import abbreviate_api_key
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.timezone_utils import get_budget_reset_time
from litellm.proxy.hooks.key_management_event_hooks import KeyManagementEventHooks
from litellm.proxy.management_endpoints.common_utils import (
    _is_user_team_admin,
    _set_object_metadata_field,
)
from litellm.proxy.management_endpoints.model_management_endpoints import (
    _add_model_to_db,
)
from litellm.proxy.management_helpers.object_permission_utils import (
    _set_object_permission,
    attach_object_permission_to_dict,
    handle_update_object_permission_common,
)
from litellm.proxy.management_helpers.team_member_permission_checks import (
    TeamMemberPermissionChecks,
)
from litellm.proxy.management_helpers.utils import management_endpoint_wrapper
from litellm.proxy.spend_tracking.spend_tracking_utils import _is_master_key
from litellm.proxy.utils import (
    PrismaClient,
    ProxyLogging,
    _hash_token_if_needed,
    handle_exception_on_proxy,
    is_valid_api_key,
    jsonify_object,
)
from litellm.router import Router
from litellm.secret_managers.main import get_secret
from litellm.types.proxy.management_endpoints.key_management_endpoints import (
    BulkUpdateKeyRequest,
    BulkUpdateKeyRequestItem,
    BulkUpdateKeyResponse,
    FailedKeyUpdate,
    SuccessfulKeyUpdate,
)
from litellm.types.router import Deployment
from litellm.types.utils import (
    BudgetConfig,
    PersonalUIKeyGenerationConfig,
    TeamUIKeyGenerationConfig,
)


def _is_team_key(data: Union[GenerateKeyRequest, LiteLLM_VerificationToken]):
    return data.team_id is not None


def _get_user_in_team(
    team_table: LiteLLM_TeamTableCachedObj, user_id: Optional[str]
) -> Optional[Member]:
    if user_id is None:
        return None
    for member in team_table.members_with_roles:
        if member.user_id is not None and member.user_id == user_id:
            return member

    return None


def _calculate_key_rotation_time(rotation_interval: str) -> datetime:
    """
    Helper function to calculate the next rotation time for a key based on the rotation interval.

    Args:
        rotation_interval: String representing the rotation interval (e.g., '30d', '90d', '1h')

    Returns:
        datetime: The calculated next rotation time in UTC
    """
    now = datetime.now(timezone.utc)
    interval_seconds = duration_in_seconds(rotation_interval)
    return now + timedelta(seconds=interval_seconds)


def _set_key_rotation_fields(
    data: dict, auto_rotate: bool, rotation_interval: Optional[str]
) -> None:
    """
    Helper function to set rotation fields in key data if auto_rotate is enabled.

    Args:
        data: Dictionary to update with rotation fields
        auto_rotate: Whether auto rotation is enabled
        rotation_interval: The rotation interval string (required if auto_rotate is True)
    """
    if auto_rotate and rotation_interval:
        data.update(
            {
                "auto_rotate": auto_rotate,
                "rotation_interval": rotation_interval,
                "key_rotation_at": _calculate_key_rotation_time(rotation_interval),
            }
        )


def _is_allowed_to_make_key_request(
    user_api_key_dict: UserAPIKeyAuth,
    user_id: Optional[str],
    team_id: Optional[str],
) -> bool:
    """
    Assert user only creates/updates keys for themselves

    Relevant issue: https://github.com/BerriAI/litellm/issues/7336
    """
    ## BASE CASE - PROXY ADMIN
    if (
        user_api_key_dict.user_role is not None
        and user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value
    ):
        return True

    if user_id is not None:
        assert (
            user_id == user_api_key_dict.user_id
        ), "User can only create keys for themselves. Got user_id={}, Your ID={}".format(
            user_id, user_api_key_dict.user_id
        )

    if team_id is not None:
        if (
            user_api_key_dict.team_id is not None
            and user_api_key_dict.team_id == UI_TEAM_ID
        ):
            return True  # handle https://github.com/BerriAI/litellm/issues/7482

    return True


def _team_key_operation_team_member_check(
    assigned_user_id: Optional[str],
    team_table: LiteLLM_TeamTableCachedObj,
    user_api_key_dict: UserAPIKeyAuth,
    team_key_generation: TeamUIKeyGenerationConfig,
    route: KeyManagementRoutes,
):
    if assigned_user_id is not None:
        key_assigned_user_in_team = _get_user_in_team(
            team_table=team_table, user_id=assigned_user_id
        )

        if key_assigned_user_in_team is None:
            raise HTTPException(
                status_code=400,
                detail=f"User={assigned_user_id} not assigned to team={team_table.team_id}",
            )

    team_member_object = _get_user_in_team(
        team_table=team_table, user_id=user_api_key_dict.user_id
    )

    is_admin = (
        user_api_key_dict.user_role is not None
        and user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value
    )

    if is_admin:
        return True
    elif team_member_object is None:
        raise HTTPException(
            status_code=400,
            detail=f"User={user_api_key_dict.user_id} not assigned to team={team_table.team_id}",
        )
    elif (
        "allowed_team_member_roles" in team_key_generation
        and team_member_object.role
        not in team_key_generation["allowed_team_member_roles"]
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Team member role {team_member_object.role} not in allowed_team_member_roles={team_key_generation['allowed_team_member_roles']}",
        )

    TeamMemberPermissionChecks.does_team_member_have_permissions_for_endpoint(
        team_member_object=team_member_object,
        team_table=team_table,
        route=route,
    )
    return True


def _key_generation_required_param_check(
    data: GenerateKeyRequest, required_params: Optional[List[str]]
):
    if required_params is None:
        return True

    data_dict = data.model_dump(exclude_unset=True)
    for param in required_params:
        if param not in data_dict:
            raise HTTPException(
                status_code=400,
                detail=f"Required param {param} not in data",
            )
    return True


def _team_key_generation_check(
    team_table: LiteLLM_TeamTableCachedObj,
    user_api_key_dict: UserAPIKeyAuth,
    data: GenerateKeyRequest,
    route: KeyManagementRoutes,
):
    if user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value:
        return True
    if (
        litellm.key_generation_settings is not None
        and "team_key_generation" in litellm.key_generation_settings
    ):
        _team_key_generation = litellm.key_generation_settings["team_key_generation"]
    else:
        _team_key_generation = TeamUIKeyGenerationConfig(
            allowed_team_member_roles=["admin", "user"],
        )

    _team_key_operation_team_member_check(
        assigned_user_id=data.user_id,
        team_table=team_table,
        user_api_key_dict=user_api_key_dict,
        team_key_generation=_team_key_generation,
        route=route,
    )
    _key_generation_required_param_check(
        data,
        _team_key_generation.get("required_params"),
    )

    return True


def _personal_key_membership_check(
    user_api_key_dict: UserAPIKeyAuth,
    personal_key_generation: Optional[PersonalUIKeyGenerationConfig],
):
    if (
        personal_key_generation is None
        or "allowed_user_roles" not in personal_key_generation
    ):
        return True

    if user_api_key_dict.user_role not in personal_key_generation["allowed_user_roles"]:
        raise HTTPException(
            status_code=400,
            detail=f"Personal key creation has been restricted by admin. Allowed roles={litellm.key_generation_settings['personal_key_generation']['allowed_user_roles']}. Your role={user_api_key_dict.user_role}",  # type: ignore
        )

    return True


def _personal_key_generation_check(
    user_api_key_dict: UserAPIKeyAuth, data: GenerateKeyRequest
):
    if (
        litellm.key_generation_settings is None
        or litellm.key_generation_settings.get("personal_key_generation") is None
    ):
        return True

    _personal_key_generation = litellm.key_generation_settings["personal_key_generation"]  # type: ignore

    _personal_key_membership_check(
        user_api_key_dict,
        personal_key_generation=_personal_key_generation,
    )

    _key_generation_required_param_check(
        data,
        _personal_key_generation.get("required_params"),
    )

    return True


def key_generation_check(
    team_table: Optional[LiteLLM_TeamTableCachedObj],
    user_api_key_dict: UserAPIKeyAuth,
    data: GenerateKeyRequest,
    route: KeyManagementRoutes,
) -> bool:
    """
    Check if admin has restricted key creation to certain roles for teams or individuals
    """

    ## check if key is for team or individual
    is_team_key = _is_team_key(data=data)
    if is_team_key:
        if team_table is None and litellm.key_generation_settings is not None:
            raise HTTPException(
                status_code=400,
                detail=f"Unable to find team object in database. Team ID: {data.team_id}",
            )
        elif team_table is None:
            return True  # assume user is assigning team_id without using the team table
        return _team_key_generation_check(
            team_table=team_table,
            user_api_key_dict=user_api_key_dict,
            data=data,
            route=route,
        )
    else:
        return _personal_key_generation_check(
            user_api_key_dict=user_api_key_dict, data=data
        )


def common_key_access_checks(
    user_api_key_dict: UserAPIKeyAuth,
    data: Union[GenerateKeyRequest, UpdateKeyRequest],
    llm_router: Optional[Router],
    premium_user: bool,
    user_id: Optional[str] = None,
) -> Literal[True]:
    """
    Check if user is allowed to make a key request, for this key
    """
    try:
        _is_allowed_to_make_key_request(
            user_api_key_dict=user_api_key_dict,
            user_id=user_id or data.user_id,
            team_id=data.team_id,
        )
    except AssertionError as e:
        raise HTTPException(
            status_code=403,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    _check_model_access_group(
        models=data.models,
        llm_router=llm_router,
        premium_user=premium_user,
    )
    return True


router = APIRouter()


def handle_key_type(data: GenerateKeyRequest, data_json: dict) -> dict:
    """
    Handle the key type.
    """
    key_type = data.key_type
    data_json.pop("key_type", None)
    if key_type == LiteLLMKeyType.LLM_API:
        data_json["allowed_routes"] = ["llm_api_routes"]
    elif key_type == LiteLLMKeyType.MANAGEMENT:
        data_json["allowed_routes"] = ["management_routes"]
    elif key_type == LiteLLMKeyType.READ_ONLY:
        data_json["allowed_routes"] = ["info_routes"]
    return data_json


async def validate_team_id_used_in_service_account_request(
    team_id: Optional[str],
    prisma_client: Optional[PrismaClient],
):
    """
    Validate team_id is used in the request body for generating a service account key
    """
    if team_id is None:
        raise HTTPException(
            status_code=400,
            detail="team_id is required for service account keys. Please specify `team_id` in the request body.",
        )

    if prisma_client is None:
        raise HTTPException(
            status_code=400,
            detail="prisma_client is required for service account keys. Please specify `prisma_client` in the request body.",
        )

    # check if team_id exists in the database
    team = await prisma_client.db.litellm_teamtable.find_unique(
        where={"team_id": team_id},
    )
    if team is None:
        raise HTTPException(
            status_code=400,
            detail="team_id does not exist in the database. Please specify a valid `team_id` in the request body.",
        )
    return True


async def _common_key_generation_helper(  # noqa: PLR0915
    data: GenerateKeyRequest,
    user_api_key_dict: UserAPIKeyAuth,
    litellm_changed_by: Optional[str],
    team_table: Optional[LiteLLM_TeamTableCachedObj],
) -> GenerateKeyResponse:
    from litellm.proxy.proxy_server import (
        litellm_proxy_admin_name,
        llm_router,
        premium_user,
        prisma_client,
    )

    common_key_access_checks(
        user_api_key_dict=user_api_key_dict,
        data=data,
        llm_router=llm_router,
        premium_user=premium_user,
    )

    if (
        data.metadata is not None
        and data.metadata.get("service_account_id") is not None
        and data.team_id is None
    ):
        await validate_team_id_used_in_service_account_request(
            team_id=data.team_id,
            prisma_client=prisma_client,
        )

    # check if user set default key/generate params on config.yaml
    if litellm.default_key_generate_params is not None:
        for elem in data:
            key, value = elem
            if value is None and key in [
                "max_budget",
                "user_id",
                "team_id",
                "max_parallel_requests",
                "tpm_limit",
                "rpm_limit",
                "budget_duration",
                "duration",
            ]:
                setattr(data, key, litellm.default_key_generate_params.get(key, None))
            elif key == "models" and value == []:
                setattr(data, key, litellm.default_key_generate_params.get(key, []))
            elif key == "metadata" and value == {}:
                setattr(data, key, litellm.default_key_generate_params.get(key, {}))

    # check if user set default key/generate params on config.yaml
    if litellm.upperbound_key_generate_params is not None:
        for elem in data:
            key, value = elem
            upperbound_value = getattr(
                litellm.upperbound_key_generate_params, key, None
            )
            if upperbound_value is not None:
                if value is None:
                    # Use the upperbound value if user didn't provide a value
                    setattr(data, key, upperbound_value)
                else:
                    # Compare with upperbound for numeric fields
                    if key in [
                        "max_budget",
                        "max_parallel_requests",
                        "tpm_limit",
                        "rpm_limit",
                    ]:
                        if value > upperbound_value:
                            raise HTTPException(
                                status_code=400,
                                detail={
                                    "error": f"{key} is over max limit set in config - user_value={value}; max_value={upperbound_value}"
                                },
                            )
                    # Compare durations
                    elif key in ["budget_duration", "duration"]:
                        upperbound_duration = duration_in_seconds(
                            duration=upperbound_value
                        )
                        # Handle special case where duration is "-1" (never expires)
                        if value == "-1":
                            user_duration = float("inf")  # Infinite duration
                        else:
                            user_duration = duration_in_seconds(duration=value)
                        if user_duration > upperbound_duration:
                            raise HTTPException(
                                status_code=400,
                                detail={
                                    "error": f"{key} is over max limit set in config - user_value={value}; max_value={upperbound_value}"
                                },
                            )

    # APPLY ENTERPRISE KEY MANAGEMENT PARAMS
    try:
        from litellm_enterprise.proxy.management_endpoints.key_management_endpoints import (
            apply_enterprise_key_management_params,
        )

        data = apply_enterprise_key_management_params(data, team_table)
    except Exception as e:
        verbose_proxy_logger.debug(
            "litellm.proxy.proxy_server.generate_key_fn(): Enterprise key management params not applied - {}".format(
                str(e)
            )
        )

    # TODO: @ishaan-jaff: Migrate all budget tracking to use LiteLLM_BudgetTable
    _budget_id = data.budget_id
    if prisma_client is not None and data.soft_budget is not None:
        # create the Budget Row for the LiteLLM Verification Token
        budget_row = LiteLLM_BudgetTable(
            soft_budget=data.soft_budget,
            model_max_budget=data.model_max_budget or {},
        )
        new_budget = prisma_client.jsonify_object(budget_row.json(exclude_none=True))

        _budget = await prisma_client.db.litellm_budgettable.create(
            data={
                **new_budget,  # type: ignore
                "created_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
                "updated_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
            }
        )
        _budget_id = getattr(_budget, "budget_id", None)

    # ADD METADATA FIELDS
    # Set Management Endpoint Metadata Fields
    for field in LiteLLM_ManagementEndpoint_MetadataFields_Premium:
        if getattr(data, field, None) is not None:
            _set_object_metadata_field(
                object_data=data,
                field_name=field,
                value=getattr(data, field),
            )
            delattr(data, field)

    for field in LiteLLM_ManagementEndpoint_MetadataFields:
        if getattr(data, field, None) is not None:
            _set_object_metadata_field(
                object_data=data,
                field_name=field,
                value=getattr(data, field),
            )
            delattr(data, field)

    data_json = data.model_dump(exclude_unset=True, exclude_none=True)  # type: ignore

    data_json = handle_key_type(data, data_json)

    # if we get max_budget passed to /key/generate, then use it as key_max_budget. Since generate_key_helper_fn is used to make new users
    if "max_budget" in data_json:
        data_json["key_max_budget"] = data_json.pop("max_budget", None)
    if _budget_id is not None:
        data_json["budget_id"] = _budget_id

    if "budget_duration" in data_json:
        data_json["key_budget_duration"] = data_json.pop("budget_duration", None)

    if user_api_key_dict.user_id is not None:
        data_json["created_by"] = user_api_key_dict.user_id
        data_json["updated_by"] = user_api_key_dict.user_id

    # Set tags on the new key
    if "tags" in data_json:
        from litellm.proxy.proxy_server import premium_user

        if premium_user is not True and data_json["tags"] is not None:
            raise ValueError(
                f"Only premium users can add tags to keys. {CommonProxyErrors.not_premium_user.value}"
            )

        _metadata = data_json.get("metadata")
        if not _metadata:
            data_json["metadata"] = {"tags": data_json["tags"]}
        else:
            data_json["metadata"]["tags"] = data_json["tags"]

        data_json.pop("tags")

    data_json = await _set_object_permission(
        data_json=data_json,
        prisma_client=prisma_client,
    )

    await _enforce_unique_key_alias(
        key_alias=data_json.get("key_alias", None),
        prisma_client=prisma_client,
    )

    # Validate user-provided key format
    if data.key is not None and not data.key.startswith("sk-"):
        _masked = (
            "{}****{}".format(data.key[:4], data.key[-4:])
            if len(data.key) > 8
            else "****"
        )
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Invalid key format. LiteLLM Virtual Key must start with 'sk-'. Received: {_masked}"
            },
        )

    # check org key limits - done here to handle inheriting org id from team
    if data.organization_id is not None:
        from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

        if prisma_client:
            org_table = await get_org_object(
                org_id=data.organization_id,
                user_api_key_cache=user_api_key_cache,
                prisma_client=prisma_client,
            )
            if org_table is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Organization not found for organization_id={data.organization_id}",
                )
            await _check_org_key_limits(
                org_table=org_table,
                data=data,
                prisma_client=prisma_client,
            )

    response = await generate_key_helper_fn(
        request_type="key", **data_json, table_name="key"
    )

    response[
        "soft_budget"
    ] = data.soft_budget  # include the user-input soft budget in the response

    response = GenerateKeyResponse(**response)

    response.token = (
        response.token_id
    )  # remap token to use the hash, and leave the key in the `key` field [TODO]: clean up generate_key_helper_fn to do this

    asyncio.create_task(
        KeyManagementEventHooks.async_key_generated_hook(
            data=data,
            response=response,
            user_api_key_dict=user_api_key_dict,
            litellm_changed_by=litellm_changed_by,
        )
    )

    return response


def _check_key_model_specific_limits(
    keys: List[LiteLLM_VerificationToken],
    data: Union[GenerateKeyRequest, UpdateKeyRequest],
    entity_rpm_limit: Optional[int],
    entity_tpm_limit: Optional[int],
    entity_model_rpm_limit_dict: Dict[str, int],
    entity_model_tpm_limit_dict: Dict[str, int],
    entity_type: str,  # "team" or "organization"
) -> None:
    """
    Generic function to check if a key is allocating model specific limits.
    Raises an error if we're overallocating.
    """
    model_rpm_limit = getattr(data, "model_rpm_limit", None) or (
        data.metadata.get("model_rpm_limit", None) if data.metadata else None
    )
    model_tpm_limit = getattr(data, "model_tpm_limit", None) or (
        data.metadata.get("model_tpm_limit", None) if data.metadata else None
    )
    if model_rpm_limit is None and model_tpm_limit is None:
        return

    # get total model specific tpm/rpm limit
    model_specific_rpm_limit: Dict[str, int] = {}
    model_specific_tpm_limit: Dict[str, int] = {}

    for key in keys:
        if key.metadata.get("model_rpm_limit", None) is not None:
            for model, rpm_limit in key.metadata.get("model_rpm_limit", {}).items():
                model_specific_rpm_limit[model] = (
                    model_specific_rpm_limit.get(model, 0) + rpm_limit
                )
        if key.metadata.get("model_tpm_limit", None) is not None:
            for model, tpm_limit in key.metadata.get("model_tpm_limit", {}).items():
                model_specific_tpm_limit[model] = (
                    model_specific_tpm_limit.get(model, 0) + tpm_limit
                )

    if model_rpm_limit is not None:
        for model, rpm_limit in model_rpm_limit.items():
            if (
                entity_rpm_limit is not None
                and model_specific_rpm_limit.get(model, 0) + rpm_limit
                > entity_rpm_limit
            ):
                raise HTTPException(
                    status_code=400,
                    detail=f"Allocated RPM limit={model_specific_rpm_limit.get(model, 0)} + Key RPM limit={rpm_limit} is greater than {entity_type} RPM limit={entity_rpm_limit}",
                )
            elif entity_model_rpm_limit_dict:
                entity_model_specific_rpm_limit = entity_model_rpm_limit_dict.get(model)
                if (
                    entity_model_specific_rpm_limit
                    and model_specific_rpm_limit.get(model, 0) + rpm_limit
                    > entity_model_specific_rpm_limit
                ):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Allocated RPM limit={model_specific_rpm_limit.get(model, 0)} + Key RPM limit={rpm_limit} is greater than {entity_type} RPM limit={entity_model_specific_rpm_limit}",
                    )

    if model_tpm_limit is not None:
        for model, tpm_limit in model_tpm_limit.items():
            if (
                entity_tpm_limit is not None
                and model_specific_tpm_limit.get(model, 0) + tpm_limit
                > entity_tpm_limit
            ):
                raise HTTPException(
                    status_code=400,
                    detail=f"Allocated TPM limit={model_specific_tpm_limit.get(model, 0)} + Key TPM limit={tpm_limit} is greater than {entity_type} TPM limit={entity_tpm_limit}",
                )
            elif entity_model_tpm_limit_dict:
                entity_model_specific_tpm_limit = entity_model_tpm_limit_dict.get(model)
                if (
                    entity_model_specific_tpm_limit
                    and model_specific_tpm_limit.get(model, 0) + tpm_limit
                    > entity_model_specific_tpm_limit
                ):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Allocated TPM limit={model_specific_tpm_limit.get(model, 0)} + Key TPM limit={tpm_limit} is greater than {entity_type} TPM limit={entity_model_specific_tpm_limit}",
                    )


def _check_key_rpm_tpm_limits(
    keys: List[LiteLLM_VerificationToken],
    data: Union[GenerateKeyRequest, UpdateKeyRequest],
    entity_rpm_limit: Optional[int],
    entity_tpm_limit: Optional[int],
    entity_type: str,  # "team" or "organization"
) -> None:
    """
    Generic function to check if a key is allocating rpm/tpm limits.
    Raises an error if we're overallocating.
    """
    if keys is not None and len(keys) > 0:
        allocated_tpm = sum(key.tpm_limit for key in keys if key.tpm_limit is not None)
        allocated_rpm = sum(key.rpm_limit for key in keys if key.rpm_limit is not None)
    else:
        allocated_tpm = 0
        allocated_rpm = 0

    if (
        data.tpm_limit is not None
        and entity_tpm_limit is not None
        and data.tpm_limit + allocated_tpm > entity_tpm_limit
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Allocated TPM limit={allocated_tpm} + Key TPM limit={data.tpm_limit} is greater than {entity_type} TPM limit={entity_tpm_limit}",
        )
    if (
        data.rpm_limit is not None
        and entity_rpm_limit is not None
        and data.rpm_limit + allocated_rpm > entity_rpm_limit
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Allocated RPM limit={allocated_rpm} + Key RPM limit={data.rpm_limit} is greater than {entity_type} RPM limit={entity_rpm_limit}",
        )


def check_team_key_model_specific_limits(
    keys: List[LiteLLM_VerificationToken],
    team_table: LiteLLM_TeamTableCachedObj,
    data: Union[GenerateKeyRequest, UpdateKeyRequest],
) -> None:
    """
    Check if the team key is allocating model specific limits. If so, raise an error if we're overallocating.
    """
    entity_model_rpm_limit_dict = {}
    entity_model_tpm_limit_dict = {}
    if team_table.metadata:
        entity_model_rpm_limit_dict = team_table.metadata.get("model_rpm_limit", {})
        entity_model_tpm_limit_dict = team_table.metadata.get("model_tpm_limit", {})

    _check_key_model_specific_limits(
        keys=keys,
        data=data,
        entity_rpm_limit=team_table.rpm_limit,
        entity_tpm_limit=team_table.tpm_limit,
        entity_model_rpm_limit_dict=entity_model_rpm_limit_dict,
        entity_model_tpm_limit_dict=entity_model_tpm_limit_dict,
        entity_type="team",
    )


def check_team_key_rpm_tpm_limits(
    keys: List[LiteLLM_VerificationToken],
    team_table: LiteLLM_TeamTableCachedObj,
    data: Union[GenerateKeyRequest, UpdateKeyRequest],
) -> None:
    """
    Check if the team key is allocating rpm/tpm limits. If so, raise an error if we're overallocating.
    """
    _check_key_rpm_tpm_limits(
        keys=keys,
        data=data,
        entity_rpm_limit=team_table.rpm_limit,
        entity_tpm_limit=team_table.tpm_limit,
        entity_type="team",
    )


async def _check_team_key_limits(
    team_table: LiteLLM_TeamTableCachedObj,
    data: Union[GenerateKeyRequest, UpdateKeyRequest],
    prisma_client: PrismaClient,
) -> None:
    """
    Check if the team key is allocating guaranteed throughput limits. If so, raise an error if we're overallocating.

    Only runs check if tpm_limit_type or rpm_limit_type is "guaranteed_throughput"
    """
    if (
        data.tpm_limit_type != "guaranteed_throughput"
        and data.rpm_limit_type != "guaranteed_throughput"
    ):
        return
    # get all team keys
    # calculate allocated tpm/rpm limit
    # check if specified tpm/rpm limit is greater than allocated tpm/rpm limit

    keys = await prisma_client.db.litellm_verificationtoken.find_many(
        where={"team_id": team_table.team_id},
    )
    check_team_key_model_specific_limits(
        keys=keys,
        team_table=team_table,
        data=data,
    )
    check_team_key_rpm_tpm_limits(
        keys=keys,
        team_table=team_table,
        data=data,
    )


def check_org_key_model_specific_limits(
    keys: List[LiteLLM_VerificationToken],
    org_table: LiteLLM_OrganizationTable,
    data: Union[GenerateKeyRequest, UpdateKeyRequest],
) -> None:
    """
    Check if the organization key is allocating model specific limits. If so, raise an error if we're overallocating.
    """
    # Get org limits from budget table if available
    entity_rpm_limit = None
    entity_tpm_limit = None
    entity_model_rpm_limit_dict = {}
    entity_model_tpm_limit_dict = {}

    if org_table.litellm_budget_table is not None:
        entity_rpm_limit = org_table.litellm_budget_table.rpm_limit
        entity_tpm_limit = org_table.litellm_budget_table.tpm_limit

    if org_table.metadata:
        entity_model_rpm_limit_dict = org_table.metadata.get("model_rpm_limit", {})
        entity_model_tpm_limit_dict = org_table.metadata.get("model_tpm_limit", {})

    _check_key_model_specific_limits(
        keys=keys,
        data=data,
        entity_rpm_limit=entity_rpm_limit,
        entity_tpm_limit=entity_tpm_limit,
        entity_model_rpm_limit_dict=entity_model_rpm_limit_dict,
        entity_model_tpm_limit_dict=entity_model_tpm_limit_dict,
        entity_type="organization",
    )


def check_org_key_rpm_tpm_limits(
    keys: List[LiteLLM_VerificationToken],
    org_table: LiteLLM_OrganizationTable,
    data: Union[GenerateKeyRequest, UpdateKeyRequest],
) -> None:
    """
    Check if the organization key is allocating rpm/tpm limits. If so, raise an error if we're overallocating.
    """
    # Get org limits from budget table if available
    entity_rpm_limit = None
    entity_tpm_limit = None

    if org_table.litellm_budget_table is not None:
        entity_rpm_limit = org_table.litellm_budget_table.rpm_limit
        entity_tpm_limit = org_table.litellm_budget_table.tpm_limit

    _check_key_rpm_tpm_limits(
        keys=keys,
        data=data,
        entity_rpm_limit=entity_rpm_limit,
        entity_tpm_limit=entity_tpm_limit,
        entity_type="organization",
    )


async def _check_org_key_limits(
    org_table: LiteLLM_OrganizationTable,
    data: Union[GenerateKeyRequest, UpdateKeyRequest],
    prisma_client: PrismaClient,
) -> None:
    """
    Check if the organization key is allocating guaranteed throughput limits. If so, raise an error if we're overallocating.

    Only runs check if tpm_limit_type or rpm_limit_type is "guaranteed_throughput"
    """

    rpm_limit_type = getattr(data, "rpm_limit_type", None) or (
        data.metadata.get("rpm_limit_type", None) if data.metadata else None
    )
    tpm_limit_type = getattr(data, "tpm_limit_type", None) or (
        data.metadata.get("tpm_limit_type", None) if data.metadata else None
    )

    if (
        tpm_limit_type != "guaranteed_throughput"
        and rpm_limit_type != "guaranteed_throughput"
    ):
        return
    # get all organization keys
    # calculate allocated tpm/rpm limit
    # check if specified tpm/rpm limit is greater than allocated tpm/rpm limit
    keys = await prisma_client.db.litellm_verificationtoken.find_many(
        where={"organization_id": org_table.organization_id},
    )
    check_org_key_model_specific_limits(
        keys=keys,
        org_table=org_table,
        data=data,
    )
    check_org_key_rpm_tpm_limits(
        keys=keys,
        org_table=org_table,
        data=data,
    )


@router.post(
    "/key/generate",
    tags=["key management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=GenerateKeyResponse,
)
@management_endpoint_wrapper
async def generate_key_fn(
    data: GenerateKeyRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    litellm_changed_by: Optional[str] = Header(
        None,
        description="The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of other users, providing an audit trail for accountability",
    ),
):
    """
    Generate an API key based on the provided data.

    Docs: https://docs.litellm.ai/docs/proxy/virtual_keys

    Parameters:
    - duration: Optional[str] - Specify the length of time the token is valid for. You can set duration as seconds ("30s"), minutes ("30m"), hours ("30h"), days ("30d").
    - key_alias: Optional[str] - User defined key alias
    - key: Optional[str] - User defined key value. If not set, a 16-digit unique sk-key is created for you.
    - team_id: Optional[str] - The team id of the key
    - user_id: Optional[str] - The user id of the key
    - organization_id: Optional[str] - The organization id of the key. If not set, and team_id is set, the organization id will be the same as the team id. If conflict, an error will be raised.
    - budget_id: Optional[str] - The budget id associated with the key. Created by calling `/budget/new`.
    - models: Optional[list] - Model_name's a user is allowed to call. (if empty, key is allowed to call all models)
    - aliases: Optional[dict] - Any alias mappings, on top of anything in the config.yaml model list. - https://docs.litellm.ai/docs/proxy/virtual_keys#managing-auth---upgradedowngrade-models
    - config: Optional[dict] - any key-specific configs, overrides config in config.yaml
    - spend: Optional[int] - Amount spent by key. Default is 0. Will be updated by proxy whenever key is used. https://docs.litellm.ai/docs/proxy/virtual_keys#managing-auth---tracking-spend
    - send_invite_email: Optional[bool] - Whether to send an invite email to the user_id, with the generate key
    - max_budget: Optional[float] - Specify max budget for a given key.
    - budget_duration: Optional[str] - Budget is reset at the end of specified duration. If not set, budget is never reset. You can set duration as seconds ("30s"), minutes ("30m"), hours ("30h"), days ("30d").
    - max_parallel_requests: Optional[int] - Rate limit a user based on the number of parallel requests. Raises 429 error, if user's parallel requests > x.
    - metadata: Optional[dict] - Metadata for key, store information for key. Example metadata = {"team": "core-infra", "app": "app2", "email": "ishaan@berri.ai" }
    - guardrails: Optional[List[str]] - List of active guardrails for the key
    - policies: Optional[List[str]] - List of policy names to apply to the key. Policies define guardrails, conditions, and inheritance rules.
    - disable_global_guardrails: Optional[bool] - Whether to disable global guardrails for the key.
    - permissions: Optional[dict] - key-specific permissions. Currently just used for turning off pii masking (if connected). Example - {"pii": false}
    - model_max_budget: Optional[Dict[str, BudgetConfig]] - Model-specific budgets {"gpt-4": {"budget_limit": 0.0005, "time_period": "30d"}}}. IF null or {} then no model specific budget.
    - model_rpm_limit: Optional[dict] - key-specific model rpm limit. Example - {"text-davinci-002": 1000, "gpt-3.5-turbo": 1000}. IF null or {} then no model specific rpm limit.
    - model_tpm_limit: Optional[dict] - key-specific model tpm limit. Example - {"text-davinci-002": 1000, "gpt-3.5-turbo": 1000}. IF null or {} then no model specific tpm limit.
    - tpm_limit_type: Optional[str] - Type of tpm limit. Options: "best_effort_throughput" (no error if we're overallocating tpm), "guaranteed_throughput" (raise an error if we're overallocating tpm), "dynamic" (dynamically exceed limit when no 429 errors). Defaults to "best_effort_throughput".
    - rpm_limit_type: Optional[str] - Type of rpm limit. Options: "best_effort_throughput" (no error if we're overallocating rpm), "guaranteed_throughput" (raise an error if we're overallocating rpm), "dynamic" (dynamically exceed limit when no 429 errors). Defaults to "best_effort_throughput".
    - allowed_cache_controls: Optional[list] - List of allowed cache control values. Example - ["no-cache", "no-store"]. See all values - https://docs.litellm.ai/docs/proxy/caching#turn-on--off-caching-per-request
    - blocked: Optional[bool] - Whether the key is blocked.
    - rpm_limit: Optional[int] - Specify rpm limit for a given key (Requests per minute)
    - tpm_limit: Optional[int] - Specify tpm limit for a given key (Tokens per minute)
    - soft_budget: Optional[float] - Specify soft budget for a given key. Will trigger a slack alert when this soft budget is reached.
    - tags: Optional[List[str]] - Tags for [tracking spend](https://litellm.vercel.app/docs/proxy/enterprise#tracking-spend-for-custom-tags) and/or doing [tag-based routing](https://litellm.vercel.app/docs/proxy/tag_routing).
    - prompts: Optional[List[str]] - List of prompts that the key is allowed to use.
    - enforced_params: Optional[List[str]] - List of enforced params for the key (Enterprise only). [Docs](https://docs.litellm.ai/docs/proxy/enterprise#enforce-required-params-for-llm-requests)
    - prompts: Optional[List[str]] - List of prompts that the key is allowed to use.
    - allowed_routes: Optional[list] - List of allowed routes for the key. Store the actual route or store a wildcard pattern for a set of routes. Example - ["/chat/completions", "/embeddings", "/keys/*"]
    - allowed_passthrough_routes: Optional[list] - List of allowed pass through endpoints for the key. Store the actual endpoint or store a wildcard pattern for a set of endpoints. Example - ["/my-custom-endpoint"]. Use this instead of allowed_routes, if you just want to specify which pass through endpoints the key can access, without specifying the routes. If allowed_routes is specified, allowed_pass_through_endpoints is ignored.
    - object_permission: Optional[LiteLLM_ObjectPermissionBase] - key-specific object permission. Example - {"vector_stores": ["vector_store_1", "vector_store_2"], "agents": ["agent_1", "agent_2"], "agent_access_groups": ["dev_group"]}. IF null or {} then no object permission.
    - key_type: Optional[str] - Type of key that determines default allowed routes. Options: "llm_api" (can call LLM API routes), "management" (can call management routes), "read_only" (can only call info/read routes), "default" (uses default allowed routes). Defaults to "default".
    - prompts: Optional[List[str]] - List of allowed prompts for the key. If specified, the key will only be able to use these specific prompts.
    - auto_rotate: Optional[bool] - Whether this key should be automatically rotated (regenerated)
    - rotation_interval: Optional[str] - How often to auto-rotate this key (e.g., '30s', '30m', '30h', '30d'). Required if auto_rotate=True.
    - allowed_vector_store_indexes: Optional[List[dict]] - List of allowed vector store indexes for the key. Example - [{"index_name": "my-index", "index_permissions": ["write", "read"]}]. If specified, the key will only be able to use these specific vector store indexes. Create index, using `/v1/indexes` endpoint.
    - router_settings: Optional[UpdateRouterConfig] - key-specific router settings. Example - {"model_group_retry_policy": {"max_retries": 5}}. IF null or {} then no router settings.
    - access_group_ids: Optional[List[str]] - List of access group IDs to associate with the key. Access groups define which models a key can access. Example - ["access_group_1", "access_group_2"].

    Examples:

    1. Allow users to turn on/off pii masking

    ```bash
    curl --location 'http://0.0.0.0:4000/key/generate' \
        --header 'Authorization: Bearer sk-1234' \
        --header 'Content-Type: application/json' \
        --data '{
            "permissions": {"allow_pii_controls": true}
    }'
    ```

    Returns:
    - key: (str) The generated api key
    - expires: (datetime) Datetime object for when key expires.
    - user_id: (str) Unique user id - used for tracking spend across multiple keys for same user id.
    """
    try:
        from litellm.proxy._types import CommonProxyErrors
        from litellm.proxy.proxy_server import (
            prisma_client,
            user_api_key_cache,
            user_custom_key_generate,
        )

        if prisma_client is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )

        verbose_proxy_logger.debug("entered /key/generate")

        # Validate budget values are not negative
        if data.max_budget is not None and data.max_budget < 0:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"max_budget cannot be negative. Received: {data.max_budget}"
                },
            )
        if data.soft_budget is not None and data.soft_budget < 0:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"soft_budget cannot be negative. Received: {data.soft_budget}"
                },
            )

        if user_custom_key_generate is not None:
            if inspect.iscoroutinefunction(user_custom_key_generate):
                result = await user_custom_key_generate(data)  # type: ignore
            else:
                raise ValueError("user_custom_key_generate must be a coroutine")
            decision = result.get("decision", True)
            message = result.get("message", "Authentication Failed - Custom Auth Rule")
            if not decision:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN, detail=message
                )
        team_table: Optional[LiteLLM_TeamTableCachedObj] = None
        if data.team_id is not None:
            try:
                team_table = await get_team_object(
                    team_id=data.team_id,
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                    parent_otel_span=user_api_key_dict.parent_otel_span,
                    check_db_only=True,
                )
            except Exception as e:
                verbose_proxy_logger.debug(
                    f"Error getting team object in `/key/generate`: {e}"
                )

        key_generation_check(
            team_table=team_table,
            user_api_key_dict=user_api_key_dict,
            data=data,
            route=KeyManagementRoutes.KEY_GENERATE,
        )

        if team_table is not None:
            await _check_team_key_limits(
                team_table=team_table,
                data=data,
                prisma_client=prisma_client,
            )

        return await _common_key_generation_helper(
            data=data,
            user_api_key_dict=user_api_key_dict,
            litellm_changed_by=litellm_changed_by,
            team_table=team_table,
        )

    except Exception as e:
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.generate_key_fn(): Exception occured - {}".format(
                str(e)
            )
        )
        raise handle_exception_on_proxy(e)


@router.post(
    "/key/service-account/generate",
    tags=["key management"],
    dependencies=[Depends(user_api_key_auth)],
)
@management_endpoint_wrapper
async def generate_service_account_key_fn(
    data: GenerateKeyRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    litellm_changed_by: Optional[str] = Header(
        None,
        description="The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of other users, providing an audit trail for accountability",
    ),
):
    """
    Generate a Service Account API key based on the provided data. This key does not belong to any user. It belongs to the team.

    Why use a service account key?
    - Prevent key from being deleted when user is deleted.
    - Apply team limits, not team member limits to key.

    Docs: https://docs.litellm.ai/docs/proxy/virtual_keys

    Parameters:
    - duration: Optional[str] - Specify the length of time the token is valid for. You can set duration as seconds ("30s"), minutes ("30m"), hours ("30h"), days ("30d").
    - key_alias: Optional[str] - User defined key alias
    - key: Optional[str] - User defined key value. If not set, a 16-digit unique sk-key is created for you.
    - team_id: Optional[str] - The team id of the key
    - user_id: Optional[str] - [NON-FUNCTIONAL] THIS WILL BE IGNORED. The user id of the key
    - budget_id: Optional[str] - The budget id associated with the key. Created by calling `/budget/new`.
    - models: Optional[list] - Model_name's a user is allowed to call. (if empty, key is allowed to call all models)
    - aliases: Optional[dict] - Any alias mappings, on top of anything in the config.yaml model list. - https://docs.litellm.ai/docs/proxy/virtual_keys#managing-auth---upgradedowngrade-models
    - config: Optional[dict] - any key-specific configs, overrides config in config.yaml
    - spend: Optional[int] - Amount spent by key. Default is 0. Will be updated by proxy whenever key is used. https://docs.litellm.ai/docs/proxy/virtual_keys#managing-auth---tracking-spend
    - send_invite_email: Optional[bool] - Whether to send an invite email to the user_id, with the generate key
    - max_budget: Optional[float] - Specify max budget for a given key.
    - budget_duration: Optional[str] - Budget is reset at the end of specified duration. If not set, budget is never reset. You can set duration as seconds ("30s"), minutes ("30m"), hours ("30h"), days ("30d").
    - max_parallel_requests: Optional[int] - Rate limit a user based on the number of parallel requests. Raises 429 error, if user's parallel requests > x.
    - metadata: Optional[dict] - Metadata for key, store information for key. Example metadata = {"team": "core-infra", "app": "app2", "email": "ishaan@berri.ai" }
    - guardrails: Optional[List[str]] - List of active guardrails for the key
    - permissions: Optional[dict] - key-specific permissions. Currently just used for turning off pii masking (if connected). Example - {"pii": false}
    - model_max_budget: Optional[Dict[str, BudgetConfig]] - Model-specific budgets {"gpt-4": {"budget_limit": 0.0005, "time_period": "30d"}}}. IF null or {} then no model specific budget.
    - model_rpm_limit: Optional[dict] - key-specific model rpm limit. Example - {"text-davinci-002": 1000, "gpt-3.5-turbo": 1000}. IF null or {} then no model specific rpm limit.
    - model_tpm_limit: Optional[dict] - key-specific model tpm limit. Example - {"text-davinci-002": 1000, "gpt-3.5-turbo": 1000}. IF null or {} then no model specific tpm limit.
    - tpm_limit_type: Optional[str] - TPM rate limit type - "best_effort_throughput", "guaranteed_throughput", or "dynamic"
    - rpm_limit_type: Optional[str] - RPM rate limit type - "best_effort_throughput", "guaranteed_throughput", or "dynamic"
    - allowed_cache_controls: Optional[list] - List of allowed cache control values. Example - ["no-cache", "no-store"]. See all values - https://docs.litellm.ai/docs/proxy/caching#turn-on--off-caching-per-request
    - blocked: Optional[bool] - Whether the key is blocked.
    - rpm_limit: Optional[int] - Specify rpm limit for a given key (Requests per minute)
    - tpm_limit: Optional[int] - Specify tpm limit for a given key (Tokens per minute)
    - soft_budget: Optional[float] - Specify soft budget for a given key. Will trigger a slack alert when this soft budget is reached.
    - tags: Optional[List[str]] - Tags for [tracking spend](https://litellm.vercel.app/docs/proxy/enterprise#tracking-spend-for-custom-tags) and/or doing [tag-based routing](https://litellm.vercel.app/docs/proxy/tag_routing).
    - enforced_params: Optional[List[str]] - List of enforced params for the key (Enterprise only). [Docs](https://docs.litellm.ai/docs/proxy/enterprise#enforce-required-params-for-llm-requests)
    - allowed_routes: Optional[list] - List of allowed routes for the key. Store the actual route or store a wildcard pattern for a set of routes. Example - ["/chat/completions", "/embeddings", "/keys/*"]
    - object_permission: Optional[LiteLLM_ObjectPermissionBase] - key-specific object permission. Example - {"vector_stores": ["vector_store_1", "vector_store_2"], "agents": ["agent_1", "agent_2"], "agent_access_groups": ["dev_group"]}. IF null or {} then no object permission.
    Examples:
    - allowed_vector_store_indexes: Optional[List[dict]] - List of allowed vector store indexes for the key. Example - [{"index_name": "my-index", "index_permissions": ["write", "read"]}]. If specified, the key will only be able to use these specific vector store indexes. Create index, using `/v1/indexes` endpoint.


    1. Allow users to turn on/off pii masking

    ```bash
    curl --location 'http://0.0.0.0:4000/key/generate' \
        --header 'Authorization: Bearer sk-1234' \
        --header 'Content-Type: application/json' \
        --data '{
            "permissions": {"allow_pii_controls": true}
    }'
    ```

    Returns:
    - key: (str) The generated api key
    - expires: (datetime) Datetime object for when key expires.
    - user_id: (str) Unique user id - used for tracking spend across multiple keys for same user id.

    """
    from litellm.proxy._types import CommonProxyErrors
    from litellm.proxy.proxy_server import (
        prisma_client,
        user_api_key_cache,
        user_custom_key_generate,
    )

    if prisma_client is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    await validate_team_id_used_in_service_account_request(
        team_id=data.team_id,
        prisma_client=prisma_client,
    )

    verbose_proxy_logger.debug("entered /key/generate")

    if user_custom_key_generate is not None:
        if inspect.iscoroutinefunction(user_custom_key_generate):
            result = await user_custom_key_generate(data)  # type: ignore
        else:
            raise ValueError("user_custom_key_generate must be a coroutine")
        decision = result.get("decision", True)
        message = result.get("message", "Authentication Failed - Custom Auth Rule")
        if not decision:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=message)
    team_table: Optional[LiteLLM_TeamTableCachedObj] = None
    if data.team_id is not None:
        try:
            team_table = await get_team_object(
                team_id=data.team_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=user_api_key_dict.parent_otel_span,
                check_db_only=True,
            )
        except Exception as e:
            verbose_proxy_logger.debug(
                f"Error getting team object in `/key/generate`: {e}"
            )
            team_table = None

    if team_table is not None:
        await _check_team_key_limits(
            team_table=team_table,
            data=data,
            prisma_client=prisma_client,
        )

    key_generation_check(
        team_table=team_table,
        user_api_key_dict=user_api_key_dict,
        data=data,
        route=KeyManagementRoutes.KEY_GENERATE_SERVICE_ACCOUNT,
    )

    data.user_id = None  # do not allow user_id to be set for service account keys

    return await _common_key_generation_helper(
        data=data,
        user_api_key_dict=user_api_key_dict,
        litellm_changed_by=litellm_changed_by,
        team_table=team_table,
    )


def prepare_metadata_fields(
    data: BaseModel, non_default_values: dict, existing_metadata: dict
) -> dict:
    """
    Check LiteLLM_ManagementEndpoint_MetadataFields (proxy/_types.py) for fields that are allowed to be updated
    """
    if "metadata" not in non_default_values:  # allow user to set metadata to none
        non_default_values["metadata"] = existing_metadata.copy()

    casted_metadata = cast(dict, non_default_values["metadata"])

    data_json = data.model_dump(exclude_unset=True, exclude_none=True)

    try:
        for k, v in data_json.items():
            if k in LiteLLM_ManagementEndpoint_MetadataFields:
                if isinstance(v, datetime):
                    casted_metadata[k] = v.isoformat()
                else:
                    casted_metadata[k] = v
            if k in LiteLLM_ManagementEndpoint_MetadataFields_Premium:
                from litellm.proxy.utils import _premium_user_check

                _premium_user_check(k)
                casted_metadata[k] = v

    except Exception as e:
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.prepare_metadata_fields(): Exception occured - {}".format(
                str(e)
            )
        )

    non_default_values["metadata"] = casted_metadata
    return non_default_values


async def prepare_key_update_data(
    data: Union[UpdateKeyRequest, RegenerateKeyRequest],
    existing_key_row: LiteLLM_VerificationToken,
):
    data_json: dict = data.model_dump(exclude_unset=True)
    data_json.pop("key", None)
    data_json.pop("new_key", None)
    data_json.pop("grace_period", None)  # Request-only param, not a DB column
    if (
        data.metadata is not None
        and data.metadata.get("service_account_id") is not None
        and (data.team_id or existing_key_row.team_id) is None
    ):
        raise HTTPException(
            status_code=400,
            detail="team_id is required for service account keys. Please specify `team_id` in the request body.",
        )
    non_default_values = {}
    # ADD METADATA FIELDS
    # Set Management Endpoint Metadata Fields
    for field in LiteLLM_ManagementEndpoint_MetadataFields_Premium:
        if getattr(data, field, None) is not None:
            _set_object_metadata_field(
                object_data=data,
                field_name=field,
                value=getattr(data, field),
            )
    for k, v in data_json.items():
        if (
            k in LiteLLM_ManagementEndpoint_MetadataFields
            or k in LiteLLM_ManagementEndpoint_MetadataFields_Premium
        ):
            continue
        non_default_values[k] = v

    if "duration" in non_default_values:
        duration = non_default_values.pop("duration")
        if duration == "-1":
            # Set expires to None to indicate the key never expires
            non_default_values["expires"] = None
        elif duration and (isinstance(duration, str)) and len(duration) > 0:
            duration_s = duration_in_seconds(duration=duration)
            expires = datetime.now(timezone.utc) + timedelta(seconds=duration_s)
            non_default_values["expires"] = expires

    if "budget_duration" in non_default_values:
        budget_duration = non_default_values.pop("budget_duration")
        if (
            budget_duration
            and (isinstance(budget_duration, str))
            and len(budget_duration) > 0
        ):
            from litellm.proxy.common_utils.timezone_utils import get_budget_reset_time

            key_reset_at = get_budget_reset_time(budget_duration=budget_duration)
            non_default_values["budget_reset_at"] = key_reset_at
            non_default_values["budget_duration"] = budget_duration

    if "object_permission" in non_default_values:
        non_default_values = await _handle_update_object_permission(
            data_json=non_default_values,
            existing_key_row=existing_key_row,
        )

    _metadata = existing_key_row.metadata or {}

    # validate model_max_budget
    if "model_max_budget" in non_default_values:
        validate_model_max_budget(non_default_values["model_max_budget"])

    # Serialize router_settings to JSON if present
    if (
        "router_settings" in non_default_values
        and non_default_values["router_settings"] is not None
    ):
        non_default_values["router_settings"] = safe_dumps(
            non_default_values["router_settings"]
        )

    non_default_values = prepare_metadata_fields(
        data=data, non_default_values=non_default_values, existing_metadata=_metadata
    )

    return non_default_values


async def _handle_update_object_permission(
    data_json: dict,
    existing_key_row: LiteLLM_VerificationToken,
) -> dict:
    """
    Handle the update of object permission.
    """
    from litellm.proxy.proxy_server import prisma_client

    # Use the common helper to handle the object permission update
    object_permission_id = await handle_update_object_permission_common(
        data_json=data_json,
        existing_object_permission_id=existing_key_row.object_permission_id,
        prisma_client=prisma_client,
    )

    # Add the object_permission_id to data_json if one was created/updated
    if object_permission_id is not None:
        data_json["object_permission_id"] = object_permission_id
        verbose_proxy_logger.debug(
            f"updated object_permission_id: {object_permission_id}"
        )

    return data_json


def is_different_team(
    data: UpdateKeyRequest, existing_key_row: LiteLLM_VerificationToken
) -> bool:
    if data.team_id is None:
        return False
    if existing_key_row.team_id is None:
        return True
    return data.team_id != existing_key_row.team_id


def _validate_max_budget(max_budget: Optional[float]) -> None:
    """
    Validate that max_budget is not negative.

    Args:
        max_budget: The max_budget value to validate

    Raises:
        HTTPException: If max_budget is negative
    """
    if max_budget is not None and max_budget < 0:
        raise HTTPException(
            status_code=400,
            detail={"error": f"max_budget cannot be negative. Received: {max_budget}"},
        )


async def _get_and_validate_existing_key(
    token: str, prisma_client: Optional[PrismaClient]
) -> LiteLLM_VerificationToken:
    """
    Get existing key from database and validate it exists.

    Args:
        token: The key token to look up
        prisma_client: Prisma client instance

    Returns:
        LiteLLM_VerificationToken: The existing key row

    Raises:
        HTTPException: If key is not found
    """
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": "Database not connected"},
        )

    existing_key_row = await prisma_client.get_data(
        token=token,
        table_name="key",
        query_type="find_unique",
    )

    if existing_key_row is None:
        raise HTTPException(
            status_code=404,
            detail={"error": f"Key not found: {token}"},
        )

    return existing_key_row


async def _process_single_key_update(
    key_update_item: BulkUpdateKeyRequestItem,
    user_api_key_dict: UserAPIKeyAuth,
    litellm_changed_by: Optional[str],
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: DualCache,
    proxy_logging_obj: Any,
    llm_router: Optional[Router],
) -> Dict[str, Any]:
    """
    Process a single key update with all validations and checks.

    This function encapsulates all the logic for updating a single key,
    including validation, permission checks, team checks, and database updates.

    Args:
        key_update_item: The key update request item
        user_api_key_dict: The authenticated user's API key info
        litellm_changed_by: Optional header for tracking who made the change
        prisma_client: Prisma client instance
        user_api_key_cache: User API key cache
        proxy_logging_obj: Proxy logging object
        llm_router: LLM router instance

    Returns:
        Dict containing the updated key information

    Raises:
        HTTPException: For various validation and permission errors
    """
    # Validate max_budget
    _validate_max_budget(key_update_item.max_budget)

    # Get and validate existing key
    existing_key_row = await _get_and_validate_existing_key(
        token=key_update_item.key,
        prisma_client=prisma_client,
    )

    # Check team member permissions
    if prisma_client is not None:
        await TeamMemberPermissionChecks.can_team_member_execute_key_management_endpoint(
            user_api_key_dict=user_api_key_dict,
            route=KeyManagementRoutes.KEY_UPDATE,
            prisma_client=prisma_client,
            existing_key_row=existing_key_row,
            user_api_key_cache=user_api_key_cache,
        )

    # Create UpdateKeyRequest from BulkUpdateKeyRequestItem
    update_key_request = UpdateKeyRequest(
        key=key_update_item.key,
        budget_id=key_update_item.budget_id,
        max_budget=key_update_item.max_budget,
        team_id=key_update_item.team_id,
        tags=key_update_item.tags,
    )

    # Get team object and check team limits if team_id is provided
    team_obj: Optional[LiteLLM_TeamTableCachedObj] = None
    if update_key_request.team_id is not None:
        team_obj = await get_team_object(
            team_id=update_key_request.team_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            check_db_only=True,
        )

        if team_obj is not None and prisma_client is not None:
            await _check_team_key_limits(
                team_table=team_obj,
                data=update_key_request,
                prisma_client=prisma_client,
            )

    # Validate team change if team is being changed
    if is_different_team(data=update_key_request, existing_key_row=existing_key_row):
        if llm_router is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "LLM router not found. Please set it up by passing in a valid config.yaml or adding models via the UI."
                },
            )
        if team_obj is None:
            raise HTTPException(
                status_code=500,
                detail={"error": "Team object not found for team change validation"},
            )
        await validate_key_team_change(
            key=existing_key_row,
            team=team_obj,
            change_initiated_by=user_api_key_dict,
            llm_router=llm_router,
        )

    # Prepare update data
    non_default_values = await prepare_key_update_data(
        data=update_key_request, existing_key_row=existing_key_row
    )

    # Update key in database
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": "Database not connected"},
        )

    _data = {**non_default_values, "token": key_update_item.key}
    response = await prisma_client.update_data(token=key_update_item.key, data=_data)

    # Delete cache
    await _delete_cache_key_object(
        hashed_token=hash_token(key_update_item.key),
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )

    # Trigger async hook
    asyncio.create_task(
        KeyManagementEventHooks.async_key_updated_hook(
            data=update_key_request,
            existing_key_row=existing_key_row,
            response=response,
            user_api_key_dict=user_api_key_dict,
            litellm_changed_by=litellm_changed_by,
        )
    )

    if response is None:
        raise ValueError("Failed to update key got response = None")

    # Extract and format updated key info
    updated_key_info = response.get("data", {})
    if hasattr(updated_key_info, "model_dump"):
        updated_key_info = updated_key_info.model_dump()
    elif hasattr(updated_key_info, "dict"):
        updated_key_info = updated_key_info.dict()

    updated_key_info.pop("token", None)

    return updated_key_info


@router.post(
    "/key/update", tags=["key management"], dependencies=[Depends(user_api_key_auth)]
)
@management_endpoint_wrapper
async def update_key_fn(
    request: Request,
    data: UpdateKeyRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    litellm_changed_by: Optional[str] = Header(
        None,
        description="The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of other users, providing an audit trail for accountability",
    ),
):
    """
    Update an existing API key's parameters.

    Parameters:
    - key: str - The key to update
    - key_alias: Optional[str] - User-friendly key alias
    - user_id: Optional[str] - User ID associated with key
    - team_id: Optional[str] - Team ID associated with key
    - budget_id: Optional[str] - The budget id associated with the key. Created by calling `/budget/new`.
    - models: Optional[list] - Model_name's a user is allowed to call
    - tags: Optional[List[str]] - Tags for organizing keys (Enterprise only)
    - prompts: Optional[List[str]] - List of prompts that the key is allowed to use.
    - enforced_params: Optional[List[str]] - List of enforced params for the key (Enterprise only). [Docs](https://docs.litellm.ai/docs/proxy/enterprise#enforce-required-params-for-llm-requests)
    - spend: Optional[float] - Amount spent by key
    - max_budget: Optional[float] - Max budget for key
    - model_max_budget: Optional[Dict[str, BudgetConfig]] - Model-specific budgets {"gpt-4": {"budget_limit": 0.0005, "time_period": "30d"}}
    - budget_duration: Optional[str] - Budget reset period ("30d", "1h", etc.)
    - soft_budget: Optional[float] - [TODO] Soft budget limit (warning vs. hard stop). Will trigger a slack alert when this soft budget is reached.
    - max_parallel_requests: Optional[int] - Rate limit for parallel requests
    - metadata: Optional[dict] - Metadata for key. Example {"team": "core-infra", "app": "app2"}
    - tpm_limit: Optional[int] - Tokens per minute limit
    - rpm_limit: Optional[int] - Requests per minute limit
    - model_rpm_limit: Optional[dict] - Model-specific RPM limits {"gpt-4": 100, "claude-v1": 200}
    - model_tpm_limit: Optional[dict] - Model-specific TPM limits {"gpt-4": 100000, "claude-v1": 200000}
    - tpm_limit_type: Optional[str] - TPM rate limit type - "best_effort_throughput", "guaranteed_throughput", or "dynamic"
    - rpm_limit_type: Optional[str] - RPM rate limit type - "best_effort_throughput", "guaranteed_throughput", or "dynamic"
    - allowed_cache_controls: Optional[list] - List of allowed cache control values
    - duration: Optional[str] - Key validity duration ("30d", "1h", etc.) or "-1" to never expire
    - permissions: Optional[dict] - Key-specific permissions
    - send_invite_email: Optional[bool] - Send invite email to user_id
    - guardrails: Optional[List[str]] - List of active guardrails for the key
    - policies: Optional[List[str]] - List of policy names to apply to the key. Policies define guardrails, conditions, and inheritance rules.
    - disable_global_guardrails: Optional[bool] - Whether to disable global guardrails for the key.
    - prompts: Optional[List[str]] - List of prompts that the key is allowed to use.
    - blocked: Optional[bool] - Whether the key is blocked
    - aliases: Optional[dict] - Model aliases for the key - [Docs](https://litellm.vercel.app/docs/proxy/virtual_keys#model-aliases)
    - config: Optional[dict] - [DEPRECATED PARAM] Key-specific config.
    - temp_budget_increase: Optional[float] - Temporary budget increase for the key (Enterprise only).
    - temp_budget_expiry: Optional[str] - Expiry time for the temporary budget increase (Enterprise only).
    - allowed_routes: Optional[list] - List of allowed routes for the key. Store the actual route or store a wildcard pattern for a set of routes. Example - ["/chat/completions", "/embeddings", "/keys/*"]
    - allowed_passthrough_routes: Optional[list] - List of allowed pass through routes for the key. Store the actual route or store a wildcard pattern for a set of routes. Example - ["/my-custom-endpoint"]. Use this instead of allowed_routes, if you just want to specify which pass through routes the key can access, without specifying the routes. If allowed_routes is specified, allowed_passthrough_routes is ignored.
    - prompts: Optional[List[str]] - List of allowed prompts for the key. If specified, the key will only be able to use these specific prompts.
    - object_permission: Optional[LiteLLM_ObjectPermissionBase] - key-specific object permission. Example - {"vector_stores": ["vector_store_1", "vector_store_2"], "agents": ["agent_1", "agent_2"], "agent_access_groups": ["dev_group"]}. IF null or {} then no object permission.
    - auto_rotate: Optional[bool] - Whether this key should be automatically rotated
    - rotation_interval: Optional[str] - How often to rotate this key (e.g., '30d', '90d'). Required if auto_rotate=True
    - allowed_vector_store_indexes: Optional[List[dict]] - List of allowed vector store indexes for the key. Example - [{"index_name": "my-index", "index_permissions": ["write", "read"]}]. If specified, the key will only be able to use these specific vector store indexes. Create index, using `/v1/indexes` endpoint.
    - router_settings: Optional[UpdateRouterConfig] - key-specific router settings. Example - {"model_group_retry_policy": {"max_retries": 5}}. IF null or {} then no router settings.
    - access_group_ids: Optional[List[str]] - List of access group IDs to associate with the key. Access groups define which models a key can access. Example - ["access_group_1", "access_group_2"].

    Example:
    ```bash
    curl --location 'http://0.0.0.0:4000/key/update' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
        "key": "sk-1234",
        "key_alias": "my-key",
        "user_id": "user-1234",
        "team_id": "team-1234",
        "max_budget": 100,
        "metadata": {"any_key": "any-val"},
    }'
    ```
    """
    from litellm.proxy.proxy_server import (
        llm_router,
        premium_user,
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
    )

    try:
        # Validate budget values are not negative
        if data.max_budget is not None and data.max_budget < 0:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"max_budget cannot be negative. Received: {data.max_budget}"
                },
            )

        data_json: dict = data.model_dump(exclude_unset=True, exclude_none=True)
        key = data_json.pop("key")

        # get the row from db
        if prisma_client is None:
            raise Exception("Not connected to DB!")

        existing_key_row = await prisma_client.get_data(
            token=data.key, table_name="key", query_type="find_unique"
        )

        if existing_key_row is None:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Team not found, passed team_id={data.team_id}"},
            )

        ## sanity check - prevent non-proxy admin user from updating key to belong to a different user
        if (
            data.user_id is not None
            and data.user_id != existing_key_row.user_id
            and user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN.value
        ):
            raise HTTPException(
                status_code=403,
                detail=f"User={data.user_id} is not allowed to update key={key} to belong to user={existing_key_row.user_id}",
            )

        common_key_access_checks(
            user_api_key_dict=user_api_key_dict,
            data=data,
            user_id=existing_key_row.user_id,
            llm_router=llm_router,
            premium_user=premium_user,
        )

        # check if user has permission to update key
        await TeamMemberPermissionChecks.can_team_member_execute_key_management_endpoint(
            user_api_key_dict=user_api_key_dict,
            route=KeyManagementRoutes.KEY_UPDATE,
            prisma_client=prisma_client,
            existing_key_row=existing_key_row,
            user_api_key_cache=user_api_key_cache,
        )

        # Only check team limits if key has a team_id
        team_obj: Optional[LiteLLM_TeamTableCachedObj] = None
        if data.team_id is not None:
            team_obj = await get_team_object(
                team_id=data.team_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                check_db_only=True,
            )

            if team_obj is not None:
                await _check_team_key_limits(
                    team_table=team_obj,
                    data=data,
                    prisma_client=prisma_client,
                )

        # if team change - check if this is possible
        if is_different_team(data=data, existing_key_row=existing_key_row):
            if llm_router is None:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "LLM router not found. Please set it up by passing in a valid config.yaml or adding models via the UI."
                    },
                )
            # team_obj should be set since is_different_team() returns True only when data.team_id is not None
            if team_obj is None:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "Team object not found for team change validation"
                    },
                )
            await validate_key_team_change(
                key=existing_key_row,
                team=team_obj,
                change_initiated_by=user_api_key_dict,
                llm_router=llm_router,
            )

            # Set Management Endpoint Metadata Fields

        non_default_values = await prepare_key_update_data(
            data=data, existing_key_row=existing_key_row
        )

        await _enforce_unique_key_alias(
            key_alias=non_default_values.get("key_alias", None),
            prisma_client=prisma_client,
            existing_key_token=existing_key_row.token,
        )

        # Handle rotation fields if auto_rotate is being enabled
        _set_key_rotation_fields(
            non_default_values,
            non_default_values.get("auto_rotate", False),
            non_default_values.get("rotation_interval"),
        )

        _data = {**non_default_values, "token": key}
        response = await prisma_client.update_data(token=key, data=_data)

        # Delete - key from cache, since it's been updated!
        # key updated - a new model could have been added to this key. it should not block requests after this is done
        await _delete_cache_key_object(
            hashed_token=hash_token(key),
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

        asyncio.create_task(
            KeyManagementEventHooks.async_key_updated_hook(
                data=data,
                existing_key_row=existing_key_row,
                response=response,
                user_api_key_dict=user_api_key_dict,
                litellm_changed_by=litellm_changed_by,
            )
        )

        if response is None:
            raise ValueError("Failed to update key got response = None")

        return {"key": key, **response["data"]}
        # update based on remaining passed in values
    except Exception as e:
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.update_key_fn(): Exception occured - {}".format(
                str(e)
            )
        )
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Authentication Error({str(e)})"),
                type=ProxyErrorTypes.auth_error,
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Authentication Error, " + str(e),
            type=ProxyErrorTypes.auth_error,
            param=getattr(e, "param", "None"),
            code=status.HTTP_400_BAD_REQUEST,
        )


@router.post(
    "/key/bulk_update",
    tags=["key management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=BulkUpdateKeyResponse,
)
@management_endpoint_wrapper
async def bulk_update_keys(
    data: BulkUpdateKeyRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    litellm_changed_by: Optional[str] = Header(
        None,
        description="The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of other users, providing an audit trail for accountability",
    ),
):
    """
    Bulk update multiple keys at once.
    
    This endpoint allows updating multiple keys in a single request. Each key update
    is processed independently - if some updates fail, others will still succeed.
    
    Parameters:
    - keys: List[BulkUpdateKeyRequestItem] - List of key update requests, each containing:
        - key: str - The key identifier (token) to update
        - budget_id: Optional[str] - Budget ID associated with the key
        - max_budget: Optional[float] - Max budget for key
        - team_id: Optional[str] - Team ID associated with key
        - tags: Optional[List[str]] - Tags for organizing keys
    
    Returns:
    - total_requested: int - Total number of keys requested for update
    - successful_updates: List[SuccessfulKeyUpdate] - List of successfully updated keys with their updated info
    - failed_updates: List[FailedKeyUpdate] - List of failed updates with key_info and failed_reason
    
    Example request:
    ```bash
    curl --location 'http://0.0.0.0:4000/key/bulk_update' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
        "keys": [
            {
                "key": "sk-1234",
                "max_budget": 100.0,
                "team_id": "team-123",
                "tags": ["production", "api"]
            },
            {
                "key": "sk-5678",
                "budget_id": "budget-456",
                "tags": ["staging"]
            }
        ]
    }'
    ```
    """
    from litellm.proxy.proxy_server import (
        llm_router,
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
    )

    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN.value:
        raise HTTPException(
            status_code=403,
            detail={"error": "Only proxy admins can perform bulk key updates"},
        )

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": "Database not connected"},
        )

    if not data.keys:
        raise HTTPException(
            status_code=400,
            detail={"error": "No keys provided for update"},
        )

    MAX_BATCH_SIZE = 500
    if len(data.keys) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Maximum {MAX_BATCH_SIZE} keys can be updated at once. Found {len(data.keys)} keys."
            },
        )

    successful_updates: List[SuccessfulKeyUpdate] = []
    failed_updates: List[FailedKeyUpdate] = []

    for key_update_item in data.keys:
        try:
            # Process single key update using reusable function
            updated_key_info = await _process_single_key_update(
                key_update_item=key_update_item,
                user_api_key_dict=user_api_key_dict,
                litellm_changed_by=litellm_changed_by,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                proxy_logging_obj=proxy_logging_obj,
                llm_router=llm_router,
            )

            successful_updates.append(
                SuccessfulKeyUpdate(
                    key=key_update_item.key,
                    key_info=updated_key_info,
                )
            )

        except Exception as e:
            verbose_proxy_logger.exception(
                f"Failed to update key {key_update_item.key}: {e}"
            )

            if isinstance(e, HTTPException):
                error_detail = e.detail
                if isinstance(error_detail, dict):
                    error_message = error_detail.get("error", str(e))
                else:
                    error_message = str(error_detail)
            else:
                error_message = str(e)

            key_info = None
            try:
                existing_key_row = await prisma_client.get_data(
                    token=key_update_item.key,
                    table_name="key",
                    query_type="find_unique",
                )
                if existing_key_row is not None:
                    if hasattr(existing_key_row, "model_dump"):
                        key_info = existing_key_row.model_dump()
                    elif hasattr(existing_key_row, "dict"):
                        key_info = existing_key_row.dict()
                    if key_info:
                        key_info.pop("token", None)
            except Exception:
                pass

            failed_updates.append(
                FailedKeyUpdate(
                    key=key_update_item.key,
                    key_info=key_info,
                    failed_reason=error_message,
                )
            )

    return BulkUpdateKeyResponse(
        total_requested=len(data.keys),
        successful_updates=successful_updates,
        failed_updates=failed_updates,
    )


async def validate_key_team_change(
    key: LiteLLM_VerificationToken,
    team: LiteLLM_TeamTable,
    change_initiated_by: UserAPIKeyAuth,
    llm_router: Router,
):
    """
    Validate that a key can be moved to a new team.

    - The team must have access to the key's models
    - The key's user_id must be a member of the team
    - The key's tpm/rpm limit must be less than the team's tpm/rpm limit
    - The person initiating the change must be either Proxy Admin or Team Admin
    """
    # Check if the team has access to the key's models
    if len(key.models) > 0:
        for model in key.models:
            await can_team_access_model(
                model=model,
                team_object=team,
                llm_router=llm_router,
            )

    # Check if the key's user_id is a member of the team
    member_object = _get_user_in_team(
        team_table=cast(LiteLLM_TeamTableCachedObj, team), user_id=key.user_id
    )
    if key.user_id is not None:
        if not member_object:
            raise HTTPException(
                status_code=403,
                detail=f"User={key.user_id} is not a member of the team={team.team_id}. Check team members via `/team/info`.",
            )

    # Check if the key's tpm/rpm limit is less than the team's tpm/rpm limit
    if key.tpm_limit is not None:
        if team.tpm_limit and key.tpm_limit > team.tpm_limit:
            raise HTTPException(
                status_code=403,
                detail=f"Key={key.token} has a tpm_limit={key.tpm_limit} which is greater than the team's tpm_limit={team.tpm_limit}.",
            )
        if team.rpm_limit and key.rpm_limit and key.rpm_limit > team.rpm_limit:
            raise HTTPException(
                status_code=403,
                detail=f"Key={key.token} has a rpm_limit={key.rpm_limit} which is greater than the team's rpm_limit={team.rpm_limit}.",
            )

    # Check if the person initiating the change is a Proxy Admin or Team Admin
    if change_initiated_by.user_role == LitellmUserRoles.PROXY_ADMIN.value:
        return
    elif _is_user_team_admin(
        user_api_key_dict=change_initiated_by,
        team_obj=team,
    ):
        return
    # this teams member permissions allow updating a
    elif TeamMemberPermissionChecks.does_team_member_have_permissions_for_endpoint(
        team_member_object=member_object,
        team_table=cast(LiteLLM_TeamTableCachedObj, team),
        route=KeyManagementRoutes.KEY_UPDATE.value,
    ):
        return
    else:
        raise HTTPException(
            status_code=403,
            detail=f"User={change_initiated_by.user_id} is not a Proxy Admin or Team Admin for team={team.team_id}. Please ask your Proxy Admin to allow this action under 'Member Permissions' for this team.",
        )


@router.post(
    "/key/delete", tags=["key management"], dependencies=[Depends(user_api_key_auth)]
)
@management_endpoint_wrapper
async def delete_key_fn(
    data: KeyRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    litellm_changed_by: Optional[str] = Header(
        None,
        description="The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of other users, providing an audit trail for accountability",
    ),
):
    """
    Delete a key from the key management system.

    Parameters::
    - keys (List[str]): A list of keys or hashed keys to delete. Example {"keys": ["sk-QWrxEynunsNpV1zT48HIrw", "837e17519f44683334df5291321d97b8bf1098cd490e49e215f6fea935aa28be"]}
    - key_aliases (List[str]): A list of key aliases to delete. Can be passed instead of `keys`.Example {"key_aliases": ["alias1", "alias2"]}

    Returns:
    - deleted_keys (List[str]): A list of deleted keys. Example {"deleted_keys": ["sk-QWrxEynunsNpV1zT48HIrw", "837e17519f44683334df5291321d97b8bf1098cd490e49e215f6fea935aa28be"]}

    Example:
    ```bash
    curl --location 'http://0.0.0.0:4000/key/delete' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
        "keys": ["sk-QWrxEynunsNpV1zT48HIrw"]
    }'
    ```

    Raises:
        HTTPException: If an error occurs during key deletion.
    """
    try:
        from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

        if prisma_client is None:
            raise Exception("Not connected to DB!")

        # Normalize litellm_changed_by: if it's a Header object or not a string, convert to None
        if litellm_changed_by is not None and not isinstance(litellm_changed_by, str):
            litellm_changed_by = None

        ## only allow user to delete keys they own
        verbose_proxy_logger.debug(
            f"user_api_key_dict.user_role: {user_api_key_dict.user_role}"
        )

        num_keys_to_be_deleted = 0
        deleted_keys = []
        if data.keys:
            number_deleted_keys, _keys_being_deleted = await delete_verification_tokens(
                tokens=data.keys,
                user_api_key_cache=user_api_key_cache,
                user_api_key_dict=user_api_key_dict,
                litellm_changed_by=litellm_changed_by,
            )
            num_keys_to_be_deleted = len(data.keys)
            deleted_keys = data.keys
        elif data.key_aliases:
            number_deleted_keys, _keys_being_deleted = await delete_key_aliases(
                key_aliases=data.key_aliases,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                user_api_key_dict=user_api_key_dict,
                litellm_changed_by=litellm_changed_by,
            )
            num_keys_to_be_deleted = len(data.key_aliases)
            deleted_keys = data.key_aliases
        else:
            raise ValueError("Invalid request type")

        if number_deleted_keys is None:
            raise ProxyException(
                message="Failed to delete keys got None response from delete_verification_token",
                type=ProxyErrorTypes.internal_server_error,
                param="keys",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        verbose_proxy_logger.debug(f"/key/delete - deleted_keys={number_deleted_keys}")

        try:
            assert num_keys_to_be_deleted == len(deleted_keys)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"Not all keys passed in were deleted. This probably means you don't have access to delete all the keys passed in. Keys passed in={num_keys_to_be_deleted}, Deleted keys ={number_deleted_keys}"
                },
            )

        verbose_proxy_logger.debug(
            f"/keys/delete - cache after delete: {user_api_key_cache.in_memory_cache.cache_dict}"
        )

        asyncio.create_task(
            KeyManagementEventHooks.async_key_deleted_hook(
                data=data,
                keys_being_deleted=_keys_being_deleted,
                user_api_key_dict=user_api_key_dict,
                litellm_changed_by=litellm_changed_by,
                response=number_deleted_keys,
            )
        )

        return {"deleted_keys": deleted_keys}
    except Exception as e:
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.delete_key_fn(): Exception occured - {}".format(
                str(e)
            )
        )
        raise handle_exception_on_proxy(e)


@router.post(
    "/v2/key/info",
    tags=["key management"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
async def info_key_fn_v2(
    data: Optional[KeyRequest] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Retrieve information about a list of keys.

    **New endpoint**. Currently admin only.
    Parameters:
        keys: Optional[list] = body parameter representing the key(s) in the request
        user_api_key_dict: UserAPIKeyAuth = Dependency representing the user's API key
    Returns:
        Dict containing the key and its associated information

    Example Curl:
    ```
    curl -X GET "http://0.0.0.0:4000/key/info" \
    -H "Authorization: Bearer sk-1234" \
    -d {"keys": ["sk-1", "sk-2", "sk-3"]}
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    try:
        if prisma_client is None:
            raise Exception(
                "Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )
        if data is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"message": "Malformed request. No keys passed in."},
            )

        key_info = await prisma_client.get_data(
            token=data.keys, table_name="key", query_type="find_all"
        )
        if key_info is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"message": "No keys found"},
            )
        filtered_key_info = []
        for k in key_info:
            try:
                k = k.model_dump()  # noqa
            except Exception:
                # if using pydantic v1
                k = k.dict()
            filtered_key_info.append(k)
        return {"key": data.keys, "info": filtered_key_info}

    except Exception as e:
        raise handle_exception_on_proxy(e)


@router.get(
    "/key/info", tags=["key management"], dependencies=[Depends(user_api_key_auth)]
)
async def info_key_fn(
    key: Optional[str] = fastapi.Query(
        default=None, description="Key in the request parameters"
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Retrieve information about a key.
    Parameters:
        key: Optional[str] = Query parameter representing the key in the request
        user_api_key_dict: UserAPIKeyAuth = Dependency representing the user's API key
    Returns:
        Dict containing the key and its associated information

    Example Curl:
    ```
    curl -X GET "http://0.0.0.0:4000/key/info?key=sk-test-example-key-123" \
-H "Authorization: Bearer sk-1234"
    ```

    Example Curl - if no key is passed, it will use the Key Passed in Authorization Header
    ```
    curl -X GET "http://0.0.0.0:4000/key/info" \
-H "Authorization: Bearer sk-test-example-key-123"
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    try:
        if prisma_client is None:
            raise Exception(
                "Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )

        # default to using Auth token if no key is passed in
        key = key or user_api_key_dict.api_key
        hashed_key: Optional[str] = key
        if key is not None:
            hashed_key = _hash_token_if_needed(token=key)
        key_info = await prisma_client.db.litellm_verificationtoken.find_unique(
            where={"token": hashed_key},  # type: ignore
            include={"litellm_budget_table": True},
        )
        if key_info is None:
            raise ProxyException(
                message="Key not found in database",
                type=ProxyErrorTypes.not_found_error,
                param="key",
                code=status.HTTP_404_NOT_FOUND,
            )

        if (
            await _can_user_query_key_info(
                user_api_key_dict=user_api_key_dict,
                key=key,
                key_info=key_info,
            )
            is not True
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You are not allowed to access this key's info. Your role={}".format(
                    user_api_key_dict.user_role
                ),
            )
        ## REMOVE HASHED TOKEN INFO BEFORE RETURNING ##
        try:
            key_info = key_info.model_dump()  # noqa
        except Exception:
            # if using pydantic v1
            key_info = key_info.dict()
        key_info.pop("token")

        # Attach object_permission if object_permission_id is set
        key_info = await attach_object_permission_to_dict(key_info, prisma_client)

        return {"key": key, "info": key_info}
    except Exception as e:
        raise handle_exception_on_proxy(e)


def _check_model_access_group(
    models: Optional[List[str]], llm_router: Optional[Router], premium_user: bool
) -> Literal[True]:
    """
    if is_model_access_group is True + is_wildcard_route is True, check if user is a premium user

    Return True if user is a premium user, False otherwise
    """
    if models is None or llm_router is None:
        return True

    for model in models:
        if llm_router._is_model_access_group_for_wildcard_route(
            model_access_group=model
        ):
            if not premium_user:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={
                        "error": "Setting a model access group on a wildcard model is only available for LiteLLM Enterprise users.{}".format(
                            CommonProxyErrors.not_premium_user.value
                        )
                    },
                )

    return True


async def generate_key_helper_fn(  # noqa: PLR0915
    request_type: Literal[
        "user", "key"
    ],  # identifies if this request is from /user/new or /key/generate
    duration: Optional[str] = None,
    models: list = [],
    aliases: dict = {},
    config: dict = {},
    spend: float = 0.0,
    key_max_budget: Optional[float] = None,  # key_max_budget is used to Budget Per key
    key_budget_duration: Optional[str] = None,
    budget_id: Optional[float] = None,  # budget id <-> LiteLLM_BudgetTable
    soft_budget: Optional[
        float
    ] = None,  # soft_budget is used to set soft Budgets Per user
    max_budget: Optional[float] = None,  # max_budget is used to Budget Per user
    blocked: Optional[bool] = None,
    budget_duration: Optional[str] = None,  # max_budget is used to Budget Per user
    token: Optional[str] = None,
    key: Optional[
        str
    ] = None,  # dev-friendly alt param for 'token'. Exposed on `/key/generate` for setting key value yourself.
    user_id: Optional[str] = None,
    user_alias: Optional[str] = None,
    team_id: Optional[str] = None,
    user_email: Optional[str] = None,
    user_role: Optional[str] = None,
    max_parallel_requests: Optional[int] = None,
    metadata: Optional[dict] = {},
    tpm_limit: Optional[int] = None,
    rpm_limit: Optional[int] = None,
    query_type: Literal["insert_data", "update_data"] = "insert_data",
    update_key_values: Optional[dict] = None,
    key_alias: Optional[str] = None,
    allowed_cache_controls: Optional[list] = [],
    permissions: Optional[dict] = {},
    model_max_budget: Optional[dict] = {},
    model_rpm_limit: Optional[dict] = None,
    model_tpm_limit: Optional[dict] = None,
    guardrails: Optional[list] = None,
    policies: Optional[list] = None,
    prompts: Optional[list] = None,
    teams: Optional[list] = None,
    organization_id: Optional[str] = None,
    table_name: Optional[Literal["key", "user"]] = None,
    send_invite_email: Optional[bool] = None,
    created_by: Optional[str] = None,
    updated_by: Optional[str] = None,
    allowed_routes: Optional[list] = None,
    sso_user_id: Optional[str] = None,
    object_permission_id: Optional[
        str
    ] = None,  # object_permission_id <-> LiteLLM_ObjectPermissionTable
    object_permission: Optional[LiteLLM_ObjectPermissionBase] = None,
    auto_rotate: Optional[bool] = None,
    rotation_interval: Optional[str] = None,
    router_settings: Optional[dict] = None,
    access_group_ids: Optional[list] = None,
):
    from litellm.proxy.proxy_server import premium_user, prisma_client

    if prisma_client is None:
        raise Exception(
            "Connect Proxy to database to generate keys - https://docs.litellm.ai/docs/proxy/virtual_keys "
        )

    if token is None:
        if key is not None:
            token = key
        else:
            token = f"sk-{secrets.token_urlsafe(LENGTH_OF_LITELLM_GENERATED_KEY)}"

    if duration is None:  # allow tokens that never expire
        expires = None
    else:
        # Add duration to current time for exact expiration (not standardized reset time)
        duration_seconds = duration_in_seconds(duration)
        expires = datetime.now(timezone.utc) + timedelta(seconds=duration_seconds)

    if key_budget_duration is None:  # one-time budget
        key_reset_at = None
    else:
        key_reset_at = get_budget_reset_time(budget_duration=key_budget_duration)

    if budget_duration is None:  # one-time budget
        reset_at = None
    else:
        reset_at = get_budget_reset_time(budget_duration=budget_duration)

    aliases_json = json.dumps(aliases)
    config_json = json.dumps(config)
    permissions_json = json.dumps(permissions)
    router_settings_json = (
        safe_dumps(router_settings) if router_settings is not None else safe_dumps({})
    )

    # Add model_rpm_limit and model_tpm_limit to metadata
    if model_rpm_limit is not None:
        metadata = metadata or {}
        metadata["model_rpm_limit"] = model_rpm_limit
    if model_tpm_limit is not None:
        metadata = metadata or {}
        metadata["model_tpm_limit"] = model_tpm_limit
    if guardrails is not None:
        metadata = metadata or {}
        metadata["guardrails"] = guardrails
    if policies is not None:
        metadata = metadata or {}
        metadata["policies"] = policies
    if prompts is not None:
        metadata = metadata or {}
        metadata["prompts"] = prompts

    metadata_json = json.dumps(metadata)
    validate_model_max_budget(model_max_budget)
    model_max_budget_json = json.dumps(model_max_budget)
    user_role = user_role
    tpm_limit = tpm_limit
    rpm_limit = rpm_limit
    allowed_cache_controls = allowed_cache_controls

    try:
        # Create a new verification token (you may want to enhance this logic based on your needs)

        user_data = {
            "max_budget": max_budget,
            "user_email": user_email,
            "user_id": user_id,
            "user_alias": user_alias,
            "team_id": team_id,
            "organization_id": organization_id,
            "user_role": user_role,
            "spend": spend,
            "models": models,
            "metadata": metadata_json,
            "max_parallel_requests": max_parallel_requests,
            "tpm_limit": tpm_limit,
            "rpm_limit": rpm_limit,
            "budget_duration": budget_duration,
            "budget_reset_at": reset_at,
            "allowed_cache_controls": allowed_cache_controls,
            "sso_user_id": sso_user_id,
            "object_permission_id": object_permission_id,
        }
        if teams is not None:
            user_data["teams"] = teams
        key_data = {
            "token": token,
            "key_alias": key_alias,
            "expires": expires,
            "models": models,
            "aliases": aliases_json,
            "config": config_json,
            "spend": spend,
            "max_budget": key_max_budget,
            "user_id": user_id,
            "team_id": team_id,
            "max_parallel_requests": max_parallel_requests,
            "metadata": metadata_json,
            "tpm_limit": tpm_limit,
            "rpm_limit": rpm_limit,
            "budget_duration": key_budget_duration,
            "budget_reset_at": key_reset_at,
            "allowed_cache_controls": allowed_cache_controls,
            "permissions": permissions_json,
            "model_max_budget": model_max_budget_json,
            "organization_id": organization_id,
            "budget_id": budget_id,
            "blocked": blocked,
            "created_by": created_by,
            "updated_by": updated_by,
            "allowed_routes": allowed_routes or [],
            "object_permission_id": object_permission_id,
            "router_settings": router_settings_json,
            "access_group_ids": access_group_ids or [],
        }

        # Add rotation fields if auto_rotate is enabled
        _set_key_rotation_fields(
            data=key_data,
            auto_rotate=auto_rotate or False,
            rotation_interval=rotation_interval,
        )

        if (
            get_secret("DISABLE_KEY_NAME", False) is True
        ):  # allow user to disable storing abbreviated key name (shown in UI, to help figure out which key spent how much)
            pass
        else:
            key_data["key_name"] = abbreviate_api_key(api_key=token)
        saved_token = copy.deepcopy(key_data)
        if isinstance(saved_token["aliases"], str):
            saved_token["aliases"] = json.loads(saved_token["aliases"])
        if isinstance(saved_token["config"], str):
            saved_token["config"] = json.loads(saved_token["config"])
        if isinstance(saved_token["metadata"], str):
            saved_token["metadata"] = json.loads(saved_token["metadata"])
        if isinstance(saved_token["permissions"], str):
            if (
                "get_spend_routes" in saved_token["permissions"]
                and premium_user is not True
            ):
                raise ValueError(
                    "get_spend_routes permission is only available for LiteLLM Enterprise users"
                )

            saved_token["permissions"] = json.loads(saved_token["permissions"])
        if isinstance(saved_token["model_max_budget"], str):
            saved_token["model_max_budget"] = json.loads(
                saved_token["model_max_budget"]
            )
        router_settings = cast(Optional[dict], saved_token.get("router_settings"))
        if router_settings is not None and isinstance(router_settings, str):
            try:
                saved_token["router_settings"] = yaml.safe_load(router_settings)
            except yaml.YAMLError:
                # If it's not valid JSON/YAML, keep as is or set to empty dict
                saved_token["router_settings"] = {}

        if saved_token.get("expires", None) is not None and isinstance(
            saved_token["expires"], datetime
        ):
            saved_token["expires"] = saved_token["expires"].isoformat()
        if prisma_client is not None:
            if (
                table_name is None or table_name == "user"
            ):  # do not auto-create users for `/key/generate`
                ## CREATE USER (If necessary)
                if query_type == "insert_data":
                    user_row = await prisma_client.insert_data(
                        data=user_data, table_name="user"
                    )

                    if user_row is None:
                        raise Exception("Failed to create user")
                    ## use default user model list if no key-specific model list provided
                    if len(user_row.models) > 0 and len(key_data["models"]) == 0:  # type: ignore
                        key_data["models"] = user_row.models  # type: ignore
                elif query_type == "update_data":
                    user_row = await prisma_client.update_data(
                        data=user_data,
                        table_name="user",
                        update_key_values=update_key_values,
                    )
            if table_name is not None and table_name == "user":
                # do not create a key if table name is set to just 'user'
                # we only need to ensure this exists in the user table
                # the LiteLLM_VerificationToken table will increase in size if we don't do this check
                return user_data

            ## CREATE KEY
            verbose_proxy_logger.debug("prisma_client: Creating Key= %s", key_data)
            create_key_response = await prisma_client.insert_data(
                data=key_data, table_name="key"
            )

            key_data["token_id"] = getattr(create_key_response, "token", None)
            key_data["litellm_budget_table"] = getattr(
                create_key_response, "litellm_budget_table", None
            )
            key_data["created_at"] = getattr(create_key_response, "created_at", None)
            key_data["updated_at"] = getattr(create_key_response, "updated_at", None)

            # Deserialize router_settings from JSON string to dict for response
            router_settings_value = key_data.get("router_settings")
            if router_settings_value is not None and isinstance(
                router_settings_value, str
            ):
                try:
                    key_data["router_settings"] = yaml.safe_load(router_settings_value)
                except yaml.YAMLError:
                    # If it's not valid JSON/YAML, keep as is or set to empty dict
                    key_data["router_settings"] = {}
    except Exception as e:
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.generate_key_helper_fn(): Exception occured - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal Server Error."},
        )

    # Add budget related info in key_data - this ensures it's returned
    key_data["budget_id"] = budget_id

    if request_type == "user":
        # if this is a /user/new request update the key_date with user_data fields
        key_data.update(user_data)

    return key_data


async def _team_key_deletion_check(
    user_api_key_dict: UserAPIKeyAuth,
    key_info: LiteLLM_VerificationToken,
    prisma_client: PrismaClient,
    user_api_key_cache: DualCache,
):
    is_team_key = _is_team_key(data=key_info)

    if is_team_key and key_info.team_id is not None:
        team_table = await get_team_object(
            team_id=key_info.team_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            check_db_only=True,
        )
        if (
            litellm.key_generation_settings is not None
            and "team_key_generation" in litellm.key_generation_settings
        ):
            _team_key_generation = litellm.key_generation_settings[
                "team_key_generation"
            ]
        else:
            _team_key_generation = TeamUIKeyGenerationConfig(
                allowed_team_member_roles=["admin", "user"],
            )
        # check if user is team admin
        if team_table is not None:
            return _team_key_operation_team_member_check(
                assigned_user_id=user_api_key_dict.user_id,
                team_table=team_table,
                user_api_key_dict=user_api_key_dict,
                team_key_generation=_team_key_generation,
                route=KeyManagementRoutes.KEY_DELETE,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": f"Team not found in db, and user not proxy admin. Team id = {key_info.team_id}"
                },
            )
    return False


async def can_modify_verification_token(
    key_info: LiteLLM_VerificationToken,
    user_api_key_cache: DualCache,
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: PrismaClient,
) -> bool:
    """
    Check if user has permission to modify (delete/regenerate) a verification token.

    Rules:
    - Proxy admin can modify any key
    - Internal jobs service account can modify any key (for auto-rotation)
    - For team keys: only team admin or key owner can modify
    - For personal keys: only key owner can modify

    Args:
        key_info: The verification token to check
        user_api_key_cache: Cache for user API keys
        user_api_key_dict: The user making the request
        prisma_client: Prisma client for database access

    Returns:
        True if user can modify the key, False otherwise
    """
    from litellm.constants import LITELLM_INTERNAL_JOBS_SERVICE_ACCOUNT_NAME

    is_team_key = _is_team_key(data=key_info)

    # 1. Proxy admin can modify any key
    if user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value:
        return True

    # 2. Internal jobs service account can modify any key (for auto-rotation)
    if user_api_key_dict.api_key == LITELLM_INTERNAL_JOBS_SERVICE_ACCOUNT_NAME:
        return True

    # 3. For team keys: only team admin or key owner can modify
    if is_team_key and key_info.team_id is not None:
        # Get team object to check if user is team admin
        team_table = await get_team_object(
            team_id=key_info.team_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            check_db_only=True,
        )

        if team_table is None:
            return False

        # Check if user is team admin
        if _is_user_team_admin(
            user_api_key_dict=user_api_key_dict,
            team_obj=team_table,
        ):
            return True

        # Check if the key belongs to the user (they own it)
        if (
            key_info.user_id is not None
            and key_info.user_id == user_api_key_dict.user_id
        ):
            return True

        # Not team admin and doesn't own the key
        return False

    # 4. For personal keys: only key owner can modify
    if key_info.user_id is not None and key_info.user_id == user_api_key_dict.user_id:
        return True

    # Default: deny
    return False


async def delete_verification_tokens(
    tokens: List,
    user_api_key_cache: DualCache,
    user_api_key_dict: UserAPIKeyAuth,
    litellm_changed_by: Optional[str] = None,
) -> Tuple[Optional[Dict], List[LiteLLM_VerificationToken]]:
    """
    Helper that deletes the list of tokens from the database

    - check if user is proxy admin
    - check if user is team admin and key is a team key

    Args:
        tokens: List of tokens to delete
        user_id: Optional user_id to filter by

    Returns:
        Tuple[Optional[Dict], List[LiteLLM_VerificationToken]]:
            Optional[Dict]:
                - Number of deleted tokens
            List[LiteLLM_VerificationToken]:
                - List of keys being deleted, this contains information about the key_alias, token, and user_id being deleted,
                this is passed down to the KeyManagementEventHooks to delete the keys from the secret manager and handle audit logs
    """
    from litellm.proxy.proxy_server import prisma_client

    try:
        if prisma_client:
            tokens = [_hash_token_if_needed(token=key) for key in tokens]
            _keys_being_deleted: List[
                LiteLLM_VerificationToken
            ] = await prisma_client.db.litellm_verificationtoken.find_many(
                where={"token": {"in": tokens}}
            )

            if len(_keys_being_deleted) == 0:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"error": "No keys found"},
                )

            if user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value:
                authorized_keys = _keys_being_deleted
            else:
                authorized_keys = []
                for key in _keys_being_deleted:
                    if await can_modify_verification_token(
                        key_info=key,
                        user_api_key_cache=user_api_key_cache,
                        user_api_key_dict=user_api_key_dict,
                        prisma_client=prisma_client,
                    ):
                        authorized_keys.append(key)
                    else:
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail={
                                "error": "You are not authorized to delete this key"
                            },
                        )
            await _persist_deleted_verification_tokens(
                keys=authorized_keys,
                prisma_client=prisma_client,
                user_api_key_dict=user_api_key_dict,
                litellm_changed_by=litellm_changed_by,
            )

            if user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value:
                deleted_tokens = await prisma_client.delete_data(tokens=tokens)
            else:
                deletion_tasks = [
                    prisma_client.delete_data(tokens=[key.token])
                    for key in authorized_keys
                ]
                await asyncio.gather(*deletion_tasks)

                deleted_tokens = [key.token for key in authorized_keys]
                if len(deleted_tokens) != len(tokens):
                    failed_tokens = [
                        token for token in tokens if token not in deleted_tokens
                    ]
                    raise Exception(
                        "Failed to delete all tokens. Failed to delete tokens: "
                        + str(failed_tokens)
                    )
        else:
            raise Exception("DB not connected. prisma_client is None")
    except Exception as e:
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.delete_verification_tokens(): Exception occured - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        raise e

    for key in tokens:
        user_api_key_cache.delete_cache(key)
        # remove hash token from cache
        hashed_token = hash_token(cast(str, key))
        user_api_key_cache.delete_cache(hashed_token)

    return {"deleted_keys": deleted_tokens}, _keys_being_deleted


def _transform_verification_tokens_to_deleted_records(
    keys: List[LiteLLM_VerificationToken],
    user_api_key_dict: UserAPIKeyAuth,
    litellm_changed_by: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Transform verification tokens into deleted token records ready for persistence."""
    if not keys:
        return []

    deleted_at = datetime.now(timezone.utc)
    records = []
    for key in keys:
        key_payload = key.model_dump()
        deleted_record = LiteLLM_DeletedVerificationToken(
            **key_payload,
            deleted_at=deleted_at,
            deleted_by=user_api_key_dict.user_id,
            deleted_by_api_key=user_api_key_dict.api_key,
            litellm_changed_by=litellm_changed_by,
        )
        record = deleted_record.model_dump()

        # Map org_id to organization_id (model uses org_id, but schema expects organization_id)
        org_id_value = record.pop("org_id", None)
        if org_id_value is not None:
            record["organization_id"] = org_id_value

        for json_field in [
            "aliases",
            "config",
            "permissions",
            "metadata",
            "model_spend",
            "model_max_budget",
            "router_settings",
        ]:
            if json_field in record and record[json_field] is not None:
                record[json_field] = json.dumps(record[json_field])

        for rel_key in (
            "litellm_budget_table",
            "litellm_organization_table",
            "object_permission",
            "id",
        ):
            record.pop(rel_key, None)

        records.append(record)

    return records


async def _save_deleted_verification_token_records(
    records: List[Dict[str, Any]],
    prisma_client: PrismaClient,
) -> None:
    """Save deleted verification token records to the database."""
    if not records:
        return
    await prisma_client.db.litellm_deletedverificationtoken.create_many(data=records)


async def _persist_deleted_verification_tokens(
    keys: List[LiteLLM_VerificationToken],
    prisma_client: PrismaClient,
    user_api_key_dict: UserAPIKeyAuth,
    litellm_changed_by: Optional[str] = None,
) -> None:
    """Persist deleted verification token records by transforming and saving them."""
    records = _transform_verification_tokens_to_deleted_records(
        keys=keys,
        user_api_key_dict=user_api_key_dict,
        litellm_changed_by=litellm_changed_by,
    )
    await _save_deleted_verification_token_records(
        records=records,
        prisma_client=prisma_client,
    )


async def delete_key_aliases(
    key_aliases: List[str],
    user_api_key_cache: DualCache,
    prisma_client: PrismaClient,
    user_api_key_dict: UserAPIKeyAuth,
    litellm_changed_by: Optional[str] = None,
) -> Tuple[Optional[Dict], List[LiteLLM_VerificationToken]]:
    _keys_being_deleted = await prisma_client.db.litellm_verificationtoken.find_many(
        where={"key_alias": {"in": key_aliases}}
    )

    tokens = [key.token for key in _keys_being_deleted]
    return await delete_verification_tokens(
        tokens=tokens,
        user_api_key_cache=user_api_key_cache,
        user_api_key_dict=user_api_key_dict,
        litellm_changed_by=litellm_changed_by,
    )


async def _rotate_master_key(
    prisma_client: PrismaClient,
    user_api_key_dict: UserAPIKeyAuth,
    current_master_key: str,
    new_master_key: str,
) -> None:
    """
    Rotate the master key

    1. Get the values from the DB
        - Get models from DB
        - Get config from DB
    2. Decrypt the values
        - ModelTable
            - [{"model_name": "str", "litellm_params": {}}]
        - ConfigTable
    3. Encrypt the values with the new master key
    4. Update the values in the DB
    """
    from litellm.proxy.proxy_server import proxy_config

    try:
        models: Optional[
            List
        ] = await prisma_client.db.litellm_proxymodeltable.find_many()
    except Exception:
        models = None
    # 2. process model table
    if models:
        decrypted_models = proxy_config.decrypt_model_list_from_db(new_models=models)
        verbose_proxy_logger.debug(
            "ABLE TO DECRYPT MODELS - len(decrypted_models): %s", len(decrypted_models)
        )
        new_models = []
        for model in decrypted_models:
            new_model = await _add_model_to_db(
                model_params=Deployment(**model),
                user_api_key_dict=user_api_key_dict,
                prisma_client=prisma_client,
                new_encryption_key=new_master_key,
                should_create_model_in_db=False,
            )
            if new_model:
                new_models.append(jsonify_object(new_model.model_dump()))
        verbose_proxy_logger.debug("Resetting proxy model table")
        await prisma_client.db.litellm_proxymodeltable.delete_many()
        verbose_proxy_logger.debug("Creating %s models", len(new_models))
        await prisma_client.db.litellm_proxymodeltable.create_many(
            data=new_models,
        )
    # 3. process config table
    try:
        config = await prisma_client.db.litellm_config.find_many()
    except Exception:
        config = None

    if config:
        """If environment_variables is found, decrypt it and encrypt it with the new master key"""
        environment_variables_dict = {}
        for c in config:
            if c.param_name == "environment_variables":
                environment_variables_dict = c.param_value

        if environment_variables_dict:
            decrypted_env_vars = proxy_config._decrypt_and_set_db_env_variables(
                environment_variables=environment_variables_dict
            )
            encrypted_env_vars = proxy_config._encrypt_env_variables(
                environment_variables=decrypted_env_vars,
                new_encryption_key=new_master_key,
            )

            if encrypted_env_vars:
                await prisma_client.db.litellm_config.update(
                    where={"param_name": "environment_variables"},
                    data={"param_value": jsonify_object(encrypted_env_vars)},
                )

    # 4. process MCP server table
    await rotate_mcp_server_credentials_master_key(
        prisma_client=prisma_client,
        touched_by=user_api_key_dict.user_id or LITELLM_PROXY_ADMIN_NAME,
        new_master_key=new_master_key,
    )

    # 5. process credentials table
    try:
        credentials = await prisma_client.db.litellm_credentialstable.find_many()
    except Exception:
        credentials = None
    if credentials:
        from litellm.proxy.credential_endpoints.endpoints import update_db_credential

        for cred in credentials:
            try:
                decrypted_cred = proxy_config.decrypt_credentials(cred)
                encrypted_cred = update_db_credential(
                    db_credential=cred,
                    updated_patch=decrypted_cred,
                    new_encryption_key=new_master_key,
                )
                credential_object_jsonified = jsonify_object(
                    encrypted_cred.model_dump()
                )
                await prisma_client.db.litellm_credentialstable.update(
                    where={"credential_name": cred.credential_name},
                    data={
                        **credential_object_jsonified,
                        "updated_by": user_api_key_dict.user_id,
                    },
                )
            except Exception as e:
                verbose_proxy_logger.error(
                    f"Failed to re-encrypt credential {cred.credential_name}: {str(e)}"
                )
                # Continue with next credential instead of failing entire rotation
                continue
        verbose_proxy_logger.debug(
            f"Successfully re-encrypted {len(credentials)} credentials with new master key"
        )


def get_new_token(data: Optional[RegenerateKeyRequest]) -> str:
    if data and data.new_key is not None:
        new_token = data.new_key
        if not data.new_key.startswith("sk-"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "New key must start with 'sk-'. This is to distinguish a key hash (used by litellm for logging / internal logic) from the actual key."
                },
            )
    else:
        new_token = f"sk-{secrets.token_urlsafe(LENGTH_OF_LITELLM_GENERATED_KEY)}"
    return new_token


async def _insert_deprecated_key(
    prisma_client: "PrismaClient",
    old_token_hash: str,
    new_token_hash: str,
    grace_period: Optional[str],
) -> None:
    """
    Insert old key into deprecated table so it remains valid during grace period.

    Uses upsert to handle concurrent rotations gracefully.

    Parameters:
        prisma_client: DB client
        old_token_hash: Hash of the old key being rotated out
        new_token_hash: Hash of the new replacement key
        grace_period: Duration string (e.g. "24h", "2d") or None/empty for immediate revoke
    """
    grace_period_value = grace_period or os.getenv(
        "LITELLM_KEY_ROTATION_GRACE_PERIOD", ""
    )
    if not grace_period_value:
        return

    try:
        grace_seconds = duration_in_seconds(grace_period_value)
    except ValueError:
        verbose_proxy_logger.warning(
            "Invalid grace_period format: %s. Expected format like '24h', '2d'.",
            grace_period_value,
        )
        return

    if grace_seconds <= 0:
        return

    try:
        revoke_at = datetime.now(timezone.utc) + timedelta(seconds=grace_seconds)
        await prisma_client.db.litellm_deprecatedverificationtoken.upsert(
            where={"token": old_token_hash},
            data={
                "create": {
                    "token": old_token_hash,
                    "active_token_id": new_token_hash,
                    "revoke_at": revoke_at,
                },
                "update": {
                    "active_token_id": new_token_hash,
                    "revoke_at": revoke_at,
                },
            },
        )
        verbose_proxy_logger.debug(
            "Deprecated key retained for %s (revoke_at: %s)",
            grace_period_value,
            revoke_at,
        )
    except Exception as deprecated_err:
        verbose_proxy_logger.warning(
            "Failed to insert deprecated key for grace period: %s",
            deprecated_err,
        )
async def _execute_virtual_key_regeneration(
    *,
    prisma_client: PrismaClient,
    key_in_db: LiteLLM_VerificationToken,
    hashed_api_key: str,
    key: str,
    data: Optional[RegenerateKeyRequest],
    user_api_key_dict: UserAPIKeyAuth,
    litellm_changed_by: Optional[str],
    user_api_key_cache: DualCache,
    proxy_logging_obj: ProxyLogging,
) -> GenerateKeyResponse:
    """Generate new token, update DB, invalidate cache, and return response."""
    from litellm.proxy.proxy_server import hash_token

    new_token = get_new_token(data=data)
    new_token_hash = hash_token(new_token)
    new_token_key_name = f"sk-...{new_token[-4:]}"
    update_data = {"token": new_token_hash, "key_name": new_token_key_name}

    non_default_values = {}
    if data is not None:
        non_default_values = await prepare_key_update_data(
            data=data, existing_key_row=key_in_db
        )
        verbose_proxy_logger.debug("non_default_values: %s", non_default_values)
    update_data.update(non_default_values)
    update_data = prisma_client.jsonify_object(data=update_data)

    # If grace period set, insert deprecated key so old key remains valid
    await _insert_deprecated_key(
        prisma_client=prisma_client,
        old_token_hash=hashed_api_key,
        new_token_hash=new_token_hash,
        grace_period=data.grace_period if data else None,
    )

    updated_token = await prisma_client.db.litellm_verificationtoken.update(
        where={"token": hashed_api_key},
        data=update_data,  # type: ignore
    )
    updated_token_dict = dict(updated_token) if updated_token is not None else {}
    updated_token_dict["key"] = new_token
    updated_token_dict["token_id"] = updated_token_dict.pop("token")

    if hashed_api_key or key:
        await _delete_cache_key_object(
            hashed_token=hash_token(key),
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

    response = GenerateKeyResponse(**updated_token_dict)
    asyncio.create_task(
        KeyManagementEventHooks.async_key_rotated_hook(
            data=data,
            existing_key_row=key_in_db,
            response=response,
            user_api_key_dict=user_api_key_dict,
            litellm_changed_by=litellm_changed_by,
        )
    )
    return response


@router.post(
    "/key/{key:path}/regenerate",
    tags=["key management"],
    dependencies=[Depends(user_api_key_auth)],
)
@router.post(
    "/key/regenerate",
    tags=["key management"],
    dependencies=[Depends(user_api_key_auth)],
)
@management_endpoint_wrapper
async def regenerate_key_fn(  # noqa: PLR0915
    key: Optional[str] = None,
    data: Optional[RegenerateKeyRequest] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    litellm_changed_by: Optional[str] = Header(
        None,
        description="The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of other users, providing an audit trail for accountability",
    ),
) -> Optional[GenerateKeyResponse]:
    """
    Regenerate an existing API key while optionally updating its parameters.

    Parameters:
    - key: str (path parameter) - The key to regenerate
    - data: Optional[RegenerateKeyRequest] - Request body containing optional parameters to update
        - key: Optional[str] - The key to regenerate.
        - new_master_key: Optional[str] - The new master key to use, if key is the master key.
        - new_key: Optional[str] - The new key to use, if key is not the master key. If both set, new_master_key will be used.
        - key_alias: Optional[str] - User-friendly key alias
        - user_id: Optional[str] - User ID associated with key
        - team_id: Optional[str] - Team ID associated with key
        - models: Optional[list] - Model_name's a user is allowed to call
        - tags: Optional[List[str]] - Tags for organizing keys (Enterprise only)
        - spend: Optional[float] - Amount spent by key
        - max_budget: Optional[float] - Max budget for key
        - model_max_budget: Optional[Dict[str, BudgetConfig]] - Model-specific budgets {"gpt-4": {"budget_limit": 0.0005, "time_period": "30d"}}
        - budget_duration: Optional[str] - Budget reset period ("30d", "1h", etc.)
        - soft_budget: Optional[float] - Soft budget limit (warning vs. hard stop). Will trigger a slack alert when this soft budget is reached.
        - max_parallel_requests: Optional[int] - Rate limit for parallel requests
        - metadata: Optional[dict] - Metadata for key. Example {"team": "core-infra", "app": "app2"}
        - tpm_limit: Optional[int] - Tokens per minute limit
        - rpm_limit: Optional[int] - Requests per minute limit
        - model_rpm_limit: Optional[dict] - Model-specific RPM limits {"gpt-4": 100, "claude-v1": 200}
        - model_tpm_limit: Optional[dict] - Model-specific TPM limits {"gpt-4": 100000, "claude-v1": 200000}
        - allowed_cache_controls: Optional[list] - List of allowed cache control values
        - duration: Optional[str] - Key validity duration ("30d", "1h", etc.)
        - permissions: Optional[dict] - Key-specific permissions
        - guardrails: Optional[List[str]] - List of active guardrails for the key
        - blocked: Optional[bool] - Whether the key is blocked
        - grace_period: Optional[str] - Duration to keep old key valid after rotation (e.g. "24h", "2d"). Omitted = immediate revoke. Env: LITELLM_KEY_ROTATION_GRACE_PERIOD


    Returns:
    - GenerateKeyResponse containing the new key and its updated parameters

    Example:
    ```bash
    curl --location --request POST 'http://localhost:4000/key/sk-1234/regenerate' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data-raw '{
        "max_budget": 100,
        "metadata": {"team": "core-infra"},
        "models": ["gpt-4", "gpt-3.5-turbo"]
    }'
    ```

    Note: This is an Enterprise feature. It requires a premium license to use.
    """
    try:
        from litellm.proxy.proxy_server import (
            hash_token,
            master_key,
            premium_user,
            prisma_client,
            proxy_logging_obj,
            user_api_key_cache,
        )

        is_master_key_regeneration = data and data.new_master_key is not None

        if (
            premium_user is not True and not is_master_key_regeneration
        ):  # allow master key regeneration for non-premium users
            raise ValueError(
                f"Regenerating Virtual Keys is an Enterprise feature, {CommonProxyErrors.not_premium_user.value}"
            )

        # Check if key exists, raise exception if key is not in the DB
        key = data.key if data and data.key else key
        if not key:
            raise HTTPException(status_code=400, detail={"error": "No key passed in."})
        ### 1. Create New copy that is duplicate of existing key
        ######################################################################

        # create duplicate of existing key
        # set token = new token generated
        # insert new token in DB

        # create hash of token
        if prisma_client is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "DB not connected. prisma_client is None"},
            )

        _is_master_key_valid = _is_master_key(api_key=key, _master_key=master_key)

        if master_key is not None and data and _is_master_key_valid:
            if data.new_master_key is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"error": "New master key is required."},
                )
            await _rotate_master_key(
                prisma_client=prisma_client,
                user_api_key_dict=user_api_key_dict,
                current_master_key=master_key,
                new_master_key=data.new_master_key,
            )
            return GenerateKeyResponse(
                key=data.new_master_key,
                token=data.new_master_key,
                key_name=data.new_master_key,
                expires=None,
            )

        if "sk" not in key:
            hashed_api_key = key
        else:
            hashed_api_key = hash_token(key)

        _key_in_db = await prisma_client.db.litellm_verificationtoken.find_unique(
            where={"token": hashed_api_key},
        )
        if _key_in_db is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": f"Key {key} not found."},
            )

        # check if user has permission to regenerate key
        await TeamMemberPermissionChecks.can_team_member_execute_key_management_endpoint(
            user_api_key_dict=user_api_key_dict,
            route=KeyManagementRoutes.KEY_REGENERATE,
            prisma_client=prisma_client,
            existing_key_row=_key_in_db,
            user_api_key_cache=user_api_key_cache,
        )

        # check if user has ownership permission to regenerate key
        if not await can_modify_verification_token(
            key_info=_key_in_db,
            user_api_key_cache=user_api_key_cache,
            user_api_key_dict=user_api_key_dict,
            prisma_client=prisma_client,
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"error": "You are not authorized to regenerate this key"},
            )

        verbose_proxy_logger.info(
            "Key regeneration requested: key_alias=%s",
            getattr(_key_in_db, "key_alias", None),
        )
        verbose_proxy_logger.debug("key_in_db: %s", _key_in_db)

        # Normalize litellm_changed_by: if it's a Header object or not a string, convert to None
        if litellm_changed_by is not None and not isinstance(litellm_changed_by, str):
            litellm_changed_by = None

        # Save the old key record to deleted table before regeneration.
        # This preserves key_alias and team_id metadata for historical spend records.
        # If this fails, abort the regeneration to avoid permanently losing the
        # old hash→metadata mapping.
        await _persist_deleted_verification_tokens(
            keys=[_key_in_db],
            prisma_client=prisma_client,
            user_api_key_dict=user_api_key_dict,
            litellm_changed_by=litellm_changed_by,
        )

        return await _execute_virtual_key_regeneration(
            prisma_client=prisma_client,
            key_in_db=_key_in_db,
            hashed_api_key=hashed_api_key,
            key=key,
            data=data,
            user_api_key_dict=user_api_key_dict,
            litellm_changed_by=litellm_changed_by,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
    except Exception as e:
        verbose_proxy_logger.exception("Error regenerating key: %s", e)
        raise handle_exception_on_proxy(e)


async def _check_proxy_or_team_admin_for_key(
    key_in_db: LiteLLM_VerificationToken,
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: PrismaClient,
    user_api_key_cache: DualCache,
) -> None:
    if user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value:
        return

    if key_in_db.team_id is not None:
        team_table = await get_team_object(
            team_id=key_in_db.team_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            check_db_only=True,
        )
        if team_table is not None:
            if _is_user_team_admin(
                user_api_key_dict=user_api_key_dict,
                team_obj=team_table,
            ):
                return

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={"error": "You must be a proxy admin or team admin to reset key spend"},
    )


def _validate_reset_spend_value(
    reset_to: Any, key_in_db: LiteLLM_VerificationToken
) -> float:
    if not isinstance(reset_to, (int, float)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "reset_to must be a float"},
        )

    reset_to = float(reset_to)

    if reset_to < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "reset_to must be >= 0"},
        )

    current_spend = key_in_db.spend or 0.0
    if reset_to > current_spend:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": f"reset_to ({reset_to}) must be <= current spend ({current_spend})"
            },
        )

    max_budget = key_in_db.max_budget
    if key_in_db.litellm_budget_table is not None:
        budget_max_budget = getattr(key_in_db.litellm_budget_table, "max_budget", None)
        if budget_max_budget is not None:
            if max_budget is None or budget_max_budget < max_budget:
                max_budget = budget_max_budget

    if max_budget is not None and reset_to > max_budget:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": f"reset_to ({reset_to}) must be <= budget ({max_budget})"},
        )

    return reset_to


@router.post(
    "/key/{key:path}/reset_spend",
    tags=["key management"],
    dependencies=[Depends(user_api_key_auth)],
)
@management_endpoint_wrapper
async def reset_key_spend_fn(
    key: str,
    data: ResetSpendRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    litellm_changed_by: Optional[str] = Header(
        None,
        description="The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of other users, providing an audit trail for accountability",
    ),
) -> Dict[str, Any]:
    try:
        from litellm.proxy.proxy_server import (
            hash_token,
            prisma_client,
            proxy_logging_obj,
            user_api_key_cache,
        )

        if prisma_client is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "DB not connected. prisma_client is None"},
            )

        if "sk" not in key:
            hashed_api_key = key
        else:
            hashed_api_key = hash_token(key)

        _key_in_db = await prisma_client.db.litellm_verificationtoken.find_unique(
            where={"token": hashed_api_key},
            include={"litellm_budget_table": True},
        )
        if _key_in_db is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": f"Key {key} not found."},
            )

        current_spend = _key_in_db.spend or 0.0
        reset_to = _validate_reset_spend_value(data.reset_to, _key_in_db)

        await _check_proxy_or_team_admin_for_key(
            key_in_db=_key_in_db,
            user_api_key_dict=user_api_key_dict,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
        )

        updated_key = await prisma_client.db.litellm_verificationtoken.update(
            where={"token": hashed_api_key},
            data={"spend": reset_to},
        )

        if updated_key is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "Failed to update key spend"},
            )

        await _delete_cache_key_object(
            hashed_token=hashed_api_key,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

        max_budget = updated_key.max_budget
        budget_reset_at = updated_key.budget_reset_at

        return {
            "key_hash": hashed_api_key,
            "spend": reset_to,
            "previous_spend": current_spend,
            "max_budget": max_budget,
            "budget_reset_at": budget_reset_at,
        }
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error resetting key spend: %s", e)
        raise handle_exception_on_proxy(e)


async def validate_key_list_check(
    user_api_key_dict: UserAPIKeyAuth,
    user_id: Optional[str],
    team_id: Optional[str],
    organization_id: Optional[str],
    key_alias: Optional[str],
    key_hash: Optional[str],
    prisma_client: PrismaClient,
) -> Optional[LiteLLM_UserTable]:
    if user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value:
        return None

    if user_api_key_dict.user_id is None:
        raise ProxyException(
            message="You are not authorized to access this endpoint. No 'user_id' is associated with your API key.",
            type=ProxyErrorTypes.bad_request_error,
            param="user_id",
            code=status.HTTP_403_FORBIDDEN,
        )
    complete_user_info_db_obj: Optional[
        BaseModel
    ] = await prisma_client.db.litellm_usertable.find_unique(
        where={"user_id": user_api_key_dict.user_id},
        include={"organization_memberships": True},
    )

    if complete_user_info_db_obj is None:
        raise ProxyException(
            message="You are not authorized to access this endpoint. No 'user_id' is associated with your API key.",
            type=ProxyErrorTypes.bad_request_error,
            param="user_id",
            code=status.HTTP_403_FORBIDDEN,
        )

    complete_user_info = LiteLLM_UserTable(**complete_user_info_db_obj.model_dump())

    # internal user can only see their own keys
    if user_id:
        if complete_user_info.user_id != user_id:
            raise ProxyException(
                message="You are not authorized to check another user's keys",
                type=ProxyErrorTypes.bad_request_error,
                param="user_id",
                code=status.HTTP_403_FORBIDDEN,
            )

    if team_id:
        if team_id not in complete_user_info.teams:
            raise ProxyException(
                message="You are not authorized to check this team's keys",
                type=ProxyErrorTypes.bad_request_error,
                param="team_id",
                code=status.HTTP_403_FORBIDDEN,
            )

    if organization_id:
        if (
            complete_user_info.organization_memberships is None
            or organization_id
            not in [
                membership.organization_id
                for membership in complete_user_info.organization_memberships
            ]
        ):
            raise ProxyException(
                message="You are not authorized to check this organization's keys",
                type=ProxyErrorTypes.bad_request_error,
                param="organization_id",
                code=status.HTTP_403_FORBIDDEN,
            )

    if key_hash:
        try:
            key_info = await prisma_client.db.litellm_verificationtoken.find_unique(
                where={"token": key_hash},
            )
        except Exception:
            raise ProxyException(
                message="Key Hash not found.",
                type=ProxyErrorTypes.bad_request_error,
                param="key_hash",
                code=status.HTTP_403_FORBIDDEN,
            )
        can_user_query_key_info = await _can_user_query_key_info(
            user_api_key_dict=user_api_key_dict,
            key=key_hash,
            key_info=key_info,
        )
        if not can_user_query_key_info:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You are not allowed to access this key's info. Your role={}".format(
                    user_api_key_dict.user_role
                ),
            )
    return complete_user_info


async def get_admin_team_ids(
    complete_user_info: Optional[LiteLLM_UserTable],
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: PrismaClient,
) -> List[str]:
    """
    Get all team IDs where the user is an admin.
    """
    if complete_user_info is None:
        return []
    # Get all teams that user is an admin of
    teams: Optional[
        List[BaseModel]
    ] = await prisma_client.db.litellm_teamtable.find_many(
        where={"team_id": {"in": complete_user_info.teams}}
    )
    if teams is None:
        return []

    teams_pydantic_obj = [LiteLLM_TeamTable(**team.model_dump()) for team in teams]

    admin_team_ids = [
        team.team_id
        for team in teams_pydantic_obj
        if _is_user_team_admin(user_api_key_dict=user_api_key_dict, team_obj=team)
    ]
    return admin_team_ids


@router.get(
    "/key/list",
    tags=["key management"],
    dependencies=[Depends(user_api_key_auth)],
)
@management_endpoint_wrapper
async def list_keys(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    page: int = Query(1, description="Page number", ge=1),
    size: int = Query(10, description="Page size", ge=1, le=100),
    user_id: Optional[str] = Query(None, description="Filter keys by user ID"),
    team_id: Optional[str] = Query(None, description="Filter keys by team ID"),
    organization_id: Optional[str] = Query(
        None, description="Filter keys by organization ID"
    ),
    key_hash: Optional[str] = Query(None, description="Filter keys by key hash"),
    key_alias: Optional[str] = Query(None, description="Filter keys by key alias"),
    return_full_object: bool = Query(False, description="Return full key object"),
    include_team_keys: bool = Query(
        False, description="Include all keys for teams that user is an admin of."
    ),
    include_created_by_keys: bool = Query(
        False, description="Include keys created by the user"
    ),
    sort_by: Optional[str] = Query(
        default=None,
        description="Column to sort by (e.g. 'user_id', 'created_at', 'spend')",
    ),
    sort_order: str = Query(default="desc", description="Sort order ('asc' or 'desc')"),
    expand: Optional[List[str]] = Query(
        None, description="Expand related objects (e.g. 'user')"
    ),
    status: Optional[str] = Query(
        None, description="Filter by status (e.g. 'deleted')"
    ),
) -> KeyListResponseObject:
    """
    List all keys for a given user / team / organization.

    Parameters:
        expand: Optional[List[str]] - Expand related objects (e.g. 'user' to include user information)
        status: Optional[str] - Filter by status. Currently supports "deleted" to query deleted keys.

    Returns:
        {
            "keys": List[str] or List[UserAPIKeyAuth],
            "total_count": int,
            "current_page": int,
            "total_pages": int,
        }

    When expand includes "user", each key object will include a "user" field with the associated user object.
    Note: When expand=user is specified, full key objects are returned regardless of the return_full_object parameter.
    """
    try:
        from litellm.proxy.proxy_server import prisma_client

        verbose_proxy_logger.debug("Entering list_keys function")

        if prisma_client is None:
            verbose_proxy_logger.error("Database not connected")
            raise Exception("Database not connected")

        # Validate status parameter
        if status is not None and status != "deleted":
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid status value. Currently only 'deleted' is supported."
                },
            )

        complete_user_info = await validate_key_list_check(
            user_api_key_dict=user_api_key_dict,
            user_id=user_id,
            team_id=team_id,
            organization_id=organization_id,
            key_alias=key_alias,
            key_hash=key_hash,
            prisma_client=prisma_client,
        )

        if include_team_keys:
            admin_team_ids = await get_admin_team_ids(
                complete_user_info=complete_user_info,
                user_api_key_dict=user_api_key_dict,
                prisma_client=prisma_client,
            )
        else:
            admin_team_ids = None

        if not user_id and user_api_key_dict.user_role not in [
            LitellmUserRoles.PROXY_ADMIN.value,
            LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value,
        ]:
            user_id = user_api_key_dict.user_id

        response = await _list_key_helper(
            prisma_client=prisma_client,
            page=page,
            size=size,
            user_id=user_id,
            team_id=team_id,
            key_alias=key_alias,
            key_hash=key_hash,
            return_full_object=return_full_object,
            organization_id=organization_id,
            admin_team_ids=admin_team_ids,
            include_created_by_keys=include_created_by_keys,
            sort_by=sort_by,
            sort_order=sort_order,
            expand=expand,
            status=status,
        )

        verbose_proxy_logger.debug("Successfully prepared response")

        return response

    except Exception as e:
        verbose_proxy_logger.exception(f"Error in list_keys: {e}")
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"error({str(e)})"),
                type=ProxyErrorTypes.internal_server_error,
                param=getattr(e, "param", "None"),
                code=getattr(
                    e, "status_code", fastapi.status.HTTP_500_INTERNAL_SERVER_ERROR
                ),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Authentication Error, " + str(e),
            type=ProxyErrorTypes.internal_server_error,
            param=getattr(e, "param", "None"),
            code=fastapi.status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.get(
    "/key/aliases",
    tags=["key management"],
    dependencies=[Depends(user_api_key_auth)],
)
@management_endpoint_wrapper
async def key_aliases() -> Dict[str, List[str]]:
    """
    Lists all key aliases

    Returns:
        {
            "aliases": List[str]
        }
    """
    try:
        from litellm.proxy.proxy_server import prisma_client

        verbose_proxy_logger.debug("Entering key_aliases function")

        if prisma_client is None:
            verbose_proxy_logger.error("Database not connected")
            raise Exception("Database not connected")

        where: Dict[str, Any] = {}
        try:
            where.update(_get_condition_to_filter_out_ui_session_tokens())
        except NameError:
            # Helper may not exist in some builds; ignore if missing
            pass

        rows = await prisma_client.db.litellm_verificationtoken.find_many(
            where=where,
            order=[{"key_alias": "asc"}],
        )

        seen = set()
        aliases: List[str] = []
        for row in rows:
            alias = getattr(row, "key_alias", None)
            if alias is None and isinstance(row, dict):
                alias = row.get("key_alias")

            if not alias:
                continue

            alias_str = str(alias).strip()
            if alias_str and alias_str not in seen:
                seen.add(alias_str)
                aliases.append(alias_str)

        verbose_proxy_logger.debug(f"Returning {len(aliases)} key aliases")

        return {"aliases": aliases}

    except Exception as e:
        verbose_proxy_logger.exception(f"Error in key_aliases: {e}")
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"error({str(e)})"),
                type=ProxyErrorTypes.internal_server_error,
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Authentication Error, " + str(e),
            type=ProxyErrorTypes.internal_server_error,
            param=getattr(e, "param", "None"),
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


def _validate_sort_params(
    sort_by: Optional[str], sort_order: str
) -> Optional[Dict[str, str]]:
    order_by: Dict[str, str] = {}

    if sort_by is None:
        return None
    # Validate sort_by is a valid column
    valid_columns = [
        "spend",
        "max_budget",
        "created_at",
        "updated_at",
        "token",
        "key_alias",
    ]
    if sort_by not in valid_columns:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Invalid sort column. Must be one of: {', '.join(valid_columns)}"
            },
        )

    # Validate sort_order
    if sort_order.lower() not in ["asc", "desc"]:
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid sort order. Must be 'asc' or 'desc'"},
        )

    order_by[sort_by] = sort_order.lower()

    return order_by


def _build_key_filter_conditions(
    user_id: Optional[str],
    team_id: Optional[str],
    organization_id: Optional[str],
    key_alias: Optional[str],
    key_hash: Optional[str],
    exclude_team_id: Optional[str],
    admin_team_ids: Optional[List[str]],
    include_created_by_keys: bool,
) -> Dict[str, Union[str, Dict[str, Any], List[Dict[str, Any]]]]:
    """Build filter conditions for key listing."""
    # Prepare filter conditions
    where: Dict[str, Union[str, Dict[str, Any], List[Dict[str, Any]]]] = {}
    where.update(_get_condition_to_filter_out_ui_session_tokens())

    # Build the OR conditions for user's keys and admin team keys
    or_conditions: List[Dict[str, Any]] = []

    # Base conditions for user's own keys
    user_condition: Dict[str, Any] = {}
    if user_id and isinstance(user_id, str):
        user_condition["user_id"] = user_id
    if team_id and isinstance(team_id, str):
        user_condition["team_id"] = team_id
    if key_alias and isinstance(key_alias, str):
        user_condition["key_alias"] = key_alias
    if exclude_team_id and isinstance(exclude_team_id, str):
        user_condition["team_id"] = {"not": exclude_team_id}
    if organization_id and isinstance(organization_id, str):
        user_condition["organization_id"] = organization_id
    if key_hash and isinstance(key_hash, str):
        user_condition["token"] = key_hash

    if user_condition:
        or_conditions.append(user_condition)

    # Add condition for created by keys if provided
    if include_created_by_keys and user_id:
        or_conditions.append({"created_by": user_id})

    # Add condition for admin team keys if provided
    if admin_team_ids:
        or_conditions.append({"team_id": {"in": admin_team_ids}})

    # Combine conditions with OR if we have multiple conditions
    if len(or_conditions) > 1:
        where = {"AND": [where, {"OR": or_conditions}]}
    elif len(or_conditions) == 1:
        where.update(or_conditions[0])

    verbose_proxy_logger.debug(f"Filter conditions: {where}")
    return where


async def _list_key_helper(
    prisma_client: PrismaClient,
    page: int,
    size: int,
    user_id: Optional[str],
    team_id: Optional[str],
    organization_id: Optional[str],
    key_alias: Optional[str],
    key_hash: Optional[str],
    exclude_team_id: Optional[str] = None,
    return_full_object: bool = False,
    admin_team_ids: Optional[
        List[str]
    ] = None,  # New parameter for teams where user is admin
    include_created_by_keys: bool = False,
    sort_by: Optional[str] = None,
    sort_order: str = "desc",
    expand: Optional[List[str]] = None,
    status: Optional[str] = None,
) -> KeyListResponseObject:
    """
    Helper function to list keys
    Args:
        page: int
        size: int
        user_id: Optional[str]
        team_id: Optional[str]
        key_alias: Optional[str]
        exclude_team_id: Optional[str] # exclude a specific team_id
        return_full_object: bool # when true, will return UserAPIKeyAuth objects instead of just the token
        admin_team_ids: Optional[List[str]] # list of team IDs where the user is an admin

    Returns:
        KeyListResponseObject
        {
            "keys": List[str] or List[UserAPIKeyAuth],  # Updated to reflect possible return types
            "total_count": int,
            "current_page": int,
            "total_pages": int,
        }
    """
    where = _build_key_filter_conditions(
        user_id=user_id,
        team_id=team_id,
        organization_id=organization_id,
        key_alias=key_alias,
        key_hash=key_hash,
        exclude_team_id=exclude_team_id,
        admin_team_ids=admin_team_ids,
        include_created_by_keys=include_created_by_keys,
    )

    # Calculate skip for pagination
    skip = (page - 1) * size

    verbose_proxy_logger.debug(f"Pagination: skip={skip}, take={size}")

    order_by: Optional[Dict[str, str]] = (
        _validate_sort_params(sort_by, sort_order)
        if sort_by is not None and isinstance(sort_by, str)
        else None
    )

    # Determine which table to query based on status
    use_deleted_table = status == "deleted"

    # Fetch keys with pagination
    if use_deleted_table:
        keys = await prisma_client.db.litellm_deletedverificationtoken.find_many(
            where=where,  # type: ignore
            skip=skip,  # type: ignore
            take=size,  # type: ignore
            order=(
                order_by
                if order_by
                else [
                    {"created_at": "desc"},
                    {"token": "desc"},  # fallback sort
                ]
            ),
        )
    else:
        keys = await prisma_client.db.litellm_verificationtoken.find_many(
            where=where,  # type: ignore
            skip=skip,  # type: ignore
            take=size,  # type: ignore
            order=(
                order_by
                if order_by
                else [
                    {"created_at": "desc"},
                    {"token": "desc"},  # fallback sort
                ]
            ),
            include={"object_permission": True},
        )

    verbose_proxy_logger.debug(f"Fetched {len(keys)} keys")

    # Get total count of keys
    if use_deleted_table:
        total_count = await prisma_client.db.litellm_deletedverificationtoken.count(
            where=where  # type: ignore
        )
    else:
        total_count = await prisma_client.db.litellm_verificationtoken.count(
            where=where  # type: ignore
        )

    verbose_proxy_logger.debug(f"Total count of keys: {total_count}")

    # Calculate total pages
    total_pages = -(-total_count // size)  # Ceiling division

    # Fetch user information if expand includes "user"
    user_map = {}
    if expand and "user" in expand:
        user_ids = [key.user_id for key in keys if key.user_id]
        if user_ids:
            users = await prisma_client.db.litellm_usertable.find_many(
                where={"user_id": {"in": list(set(user_ids))}}  # Remove duplicates
            )
            user_map = {user.user_id: user for user in users}

    # Prepare response
    key_list: List[Union[str, UserAPIKeyAuth, LiteLLM_DeletedVerificationToken]] = []
    for key in keys:
        # Convert Prisma model to dict (supports both Pydantic v1 and v2)
        try:
            key_dict = key.model_dump()
        except Exception:
            # Fallback for Pydantic v1 compatibility
            key_dict = key.dict()
        # Attach object_permission if object_permission_id is set (only for non-deleted keys)
        if not use_deleted_table:
            key_dict = await attach_object_permission_to_dict(key_dict, prisma_client)

        # Include user information if expand includes "user"
        if expand and "user" in expand and key.user_id and key.user_id in user_map:
            try:
                key_dict["user"] = user_map[key.user_id].model_dump()
            except Exception:
                key_dict["user"] = user_map[key.user_id].dict()

        if return_full_object is True or (expand and "user" in expand):
            if use_deleted_table:
                # Use deleted key type to preserve deleted_at, deleted_by, etc.
                key_list.append(LiteLLM_DeletedVerificationToken(**key_dict))
            else:
                key_list.append(UserAPIKeyAuth(**key_dict))  # Return full key object
        else:
            _token = key_dict.get("token")
            key_list.append(cast(str, _token))  # Return only the token

    return KeyListResponseObject(
        keys=key_list,
        total_count=total_count,
        current_page=page,
        total_pages=total_pages,
    )


def _get_condition_to_filter_out_ui_session_tokens() -> Dict[str, Any]:
    """
    Condition to filter out UI session tokens
    """
    return {
        "OR": [
            {"team_id": None},  # Include records where team_id is null
            {
                "team_id": {"not": UI_SESSION_TOKEN_TEAM_ID}
            },  # Include records where team_id != UI_SESSION_TOKEN_TEAM_ID
        ]
    }


@router.post(
    "/key/block", tags=["key management"], dependencies=[Depends(user_api_key_auth)]
)
@management_endpoint_wrapper
async def block_key(
    data: BlockKeyRequest,
    http_request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    litellm_changed_by: Optional[str] = Header(
        None,
        description="The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of other users, providing an audit trail for accountability",
    ),
) -> Optional[LiteLLM_VerificationToken]:
    """
    Block an Virtual key from making any requests.

    Parameters:
    - key: str - The key to block. Can be either the unhashed key (sk-...) or the hashed key value

     Example:
    ```bash
    curl --location 'http://0.0.0.0:4000/key/block' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
        "key": "sk-Fn8Ej39NxjAXrvpUGKghGw"
    }'
    ```

    Note: This is an admin-only endpoint. Only proxy admins can block keys.
    """
    from litellm.proxy.proxy_server import (
        create_audit_log_for_update,
        hash_token,
        litellm_proxy_admin_name,
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
    )

    if prisma_client is None:
        raise Exception("{}".format(CommonProxyErrors.db_not_connected_error.value))

    if not is_valid_api_key(data.key):
        raise ProxyException(
            message="Invalid key format.",
            type=ProxyErrorTypes.bad_request_error,
            param="key",
            code=status.HTTP_400_BAD_REQUEST,
        )
    if data.key.startswith("sk-"):
        hashed_token = hash_token(token=data.key)
    else:
        hashed_token = data.key

    if litellm.store_audit_logs is True:
        # make an audit log for key update
        record = await prisma_client.db.litellm_verificationtoken.find_unique(
            where={"token": hashed_token}
        )
        if record is None:
            raise ProxyException(
                message=f"Key {data.key} not found",
                type=ProxyErrorTypes.bad_request_error,
                param="key",
                code=status.HTTP_404_NOT_FOUND,
            )
        asyncio.create_task(
            create_audit_log_for_update(
                request_data=LiteLLM_AuditLogs(
                    id=str(uuid.uuid4()),
                    updated_at=datetime.now(timezone.utc),
                    changed_by=litellm_changed_by
                    or user_api_key_dict.user_id
                    or litellm_proxy_admin_name,
                    changed_by_api_key=user_api_key_dict.api_key,
                    table_name=LitellmTableNames.KEY_TABLE_NAME,
                    object_id=hashed_token,
                    action="blocked",
                    updated_values="{}",
                    before_value=record.model_dump_json(),
                )
            )
        )

    record = await prisma_client.db.litellm_verificationtoken.update(
        where={"token": hashed_token}, data={"blocked": True}  # type: ignore
    )

    ## UPDATE KEY CACHE

    ### get cached object ###
    key_object = await get_key_object(
        hashed_token=hashed_token,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        parent_otel_span=None,
        proxy_logging_obj=proxy_logging_obj,
    )

    ### update cached object ###
    key_object.blocked = True

    ### store cached object ###
    await _cache_key_object(
        hashed_token=hashed_token,
        user_api_key_obj=key_object,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )

    return record


@router.post(
    "/key/unblock", tags=["key management"], dependencies=[Depends(user_api_key_auth)]
)
@management_endpoint_wrapper
async def unblock_key(
    data: BlockKeyRequest,
    http_request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    litellm_changed_by: Optional[str] = Header(
        None,
        description="The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of other users, providing an audit trail for accountability",
    ),
):
    """
    Unblock a Virtual key to allow it to make requests again.

    Parameters:
    - key: str - The key to unblock. Can be either the unhashed key (sk-...) or the hashed key value

    Example:
    ```bash
    curl --location 'http://0.0.0.0:4000/key/unblock' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
        "key": "sk-Fn8Ej39NxjAXrvpUGKghGw"
    }'
    ```

    Note: This is an admin-only endpoint. Only proxy admins can unblock keys.
    """
    from litellm.proxy.proxy_server import (
        create_audit_log_for_update,
        hash_token,
        litellm_proxy_admin_name,
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
    )

    if prisma_client is None:
        raise Exception("{}".format(CommonProxyErrors.db_not_connected_error.value))

    if not is_valid_api_key(data.key):
        raise ProxyException(
            message="Invalid key format.",
            type=ProxyErrorTypes.bad_request_error,
            param="key",
            code=status.HTTP_400_BAD_REQUEST,
        )
    if data.key.startswith("sk-"):
        hashed_token = hash_token(token=data.key)
    else:
        hashed_token = data.key

    if litellm.store_audit_logs is True:
        # make an audit log for key update
        record = await prisma_client.db.litellm_verificationtoken.find_unique(
            where={"token": hashed_token}
        )
        if record is None:
            raise ProxyException(
                message=f"Key {data.key} not found",
                type=ProxyErrorTypes.bad_request_error,
                param="key",
                code=status.HTTP_404_NOT_FOUND,
            )
        asyncio.create_task(
            create_audit_log_for_update(
                request_data=LiteLLM_AuditLogs(
                    id=str(uuid.uuid4()),
                    updated_at=datetime.now(timezone.utc),
                    changed_by=litellm_changed_by
                    or user_api_key_dict.user_id
                    or litellm_proxy_admin_name,
                    changed_by_api_key=user_api_key_dict.api_key,
                    table_name=LitellmTableNames.KEY_TABLE_NAME,
                    object_id=hashed_token,
                    action="blocked",
                    updated_values="{}",
                    before_value=record.model_dump_json(),
                )
            )
        )

    record = await prisma_client.db.litellm_verificationtoken.update(
        where={"token": hashed_token}, data={"blocked": False}  # type: ignore
    )

    ## UPDATE KEY CACHE

    ### get cached object ###
    key_object = await get_key_object(
        hashed_token=hashed_token,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        parent_otel_span=None,
        proxy_logging_obj=proxy_logging_obj,
    )

    ### update cached object ###
    key_object.blocked = False

    ### store cached object ###
    await _cache_key_object(
        hashed_token=hashed_token,
        user_api_key_obj=key_object,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )

    return record


@router.post(
    "/key/health",
    tags=["key management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=KeyHealthResponse,
)
@management_endpoint_wrapper
async def key_health(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Check the health of the key

    Checks:
    - If key based logging is configured correctly - sends a test log

    Usage 

    Pass the key in the request header

    ```bash
    curl -X POST "http://localhost:4000/key/health" \
     -H "Authorization: Bearer sk-1234" \
     -H "Content-Type: application/json"
    ```

    Response when logging callbacks are setup correctly:

    ```json
    {
      "key": "healthy",
      "logging_callbacks": {
        "callbacks": [
          "gcs_bucket"
        ],
        "status": "healthy",
        "details": "No logger exceptions triggered, system is healthy. Manually check if logs were sent to ['gcs_bucket']"
      }
    }
    ```


    Response when logging callbacks are not setup correctly:
    ```json
    {
      "key": "unhealthy",
      "logging_callbacks": {
        "callbacks": [
          "gcs_bucket"
        ],
        "status": "unhealthy",
        "details": "Logger exceptions triggered, system is unhealthy: Failed to load vertex credentials. Check to see if credentials containing partial/invalid information."
      }
    }
    ```
    """
    try:
        # Get the key's metadata
        key_metadata = user_api_key_dict.metadata

        health_status: KeyHealthResponse = KeyHealthResponse(
            key="healthy",
            logging_callbacks=None,
        )

        # Check if logging is configured in metadata
        if key_metadata and "logging" in key_metadata:
            logging_statuses = await test_key_logging(
                user_api_key_dict=user_api_key_dict,
                request=request,
                key_logging=key_metadata["logging"],
            )
            health_status["logging_callbacks"] = logging_statuses

            # Check if any logging callback is unhealthy
            if logging_statuses.get("status") == "unhealthy":
                health_status["key"] = "unhealthy"

        return KeyHealthResponse(**health_status)

    except Exception as e:
        raise ProxyException(
            message=f"Key health check failed: {str(e)}",
            type=ProxyErrorTypes.internal_server_error,
            param=getattr(e, "param", "None"),
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


async def _can_user_query_key_info(
    user_api_key_dict: UserAPIKeyAuth,
    key: Optional[str],
    key_info: LiteLLM_VerificationToken,
) -> bool:
    """
    Helper to check if the user has access to the key's info
    """
    if (
        user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value
        or user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value
    ):
        return True
    elif user_api_key_dict.api_key == key:
        return True
    # user can query their own key info
    elif key_info.user_id == user_api_key_dict.user_id:
        return True
    elif await TeamMemberPermissionChecks.user_belongs_to_keys_team(
        user_api_key_dict=user_api_key_dict,
        existing_key_row=key_info,
    ):
        return True
    return False


async def test_key_logging(
    user_api_key_dict: UserAPIKeyAuth,
    request: Request,
    key_logging: List[Dict[str, Any]],
) -> LoggingCallbackStatus:
    """
    Test the key-based logging

    - Test that key logging is correctly formatted and all args are passed correctly
    - Make a mock completion call -> user can check if it's correctly logged
    - Check if any logger.exceptions were triggered -> if they were then returns it to the user client side
    """
    import logging
    from io import StringIO

    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request
    from litellm.proxy.proxy_server import general_settings, proxy_config

    logging_callbacks: List[str] = []
    for callback in key_logging:
        if callback.get("callback_name") is not None:
            logging_callbacks.append(callback["callback_name"])
        else:
            raise ValueError("callback_name is required in key_logging")

    log_capture_string = StringIO()
    ch = logging.StreamHandler(log_capture_string)
    ch.setLevel(logging.ERROR)
    logger = logging.getLogger()
    logger.addHandler(ch)

    try:
        data = {
            "model": "openai/litellm-key-health-test",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, this is a test from litellm /key/health. No LLM API call was made for this",
                }
            ],
            "mock_response": "test response",
        }
        data = await add_litellm_data_to_request(
            data=data,
            user_api_key_dict=user_api_key_dict,
            proxy_config=proxy_config,
            general_settings=general_settings,
            request=request,
        )
        await litellm.acompletion(
            **data
        )  # make mock completion call to trigger key based callbacks
    except Exception as e:
        return LoggingCallbackStatus(
            callbacks=logging_callbacks,
            status="unhealthy",
            details=f"Logging test failed: {str(e)}",
        )

    await asyncio.sleep(
        2
    )  # wait for callbacks to run, callbacks use batching so wait for the flush event

    # Check if any logger exceptions were triggered
    log_contents = log_capture_string.getvalue()
    logger.removeHandler(ch)
    if log_contents:
        return LoggingCallbackStatus(
            callbacks=logging_callbacks,
            status="unhealthy",
            details=f"Logger exceptions triggered, system is unhealthy: {log_contents}",
        )
    else:
        return LoggingCallbackStatus(
            callbacks=logging_callbacks,
            status="healthy",
            details=f"No logger exceptions triggered, system is healthy. Manually check if logs were sent to {logging_callbacks} ",
        )


async def _enforce_unique_key_alias(
    key_alias: Optional[str],
    prisma_client: Any,
    existing_key_token: Optional[str] = None,
) -> None:
    """
    Helper to enforce unique key aliases across all keys.

    Args:
        key_alias (Optional[str]): The key alias to check
        prisma_client (Any): Prisma client instance
        existing_key_token (Optional[str]): ID of existing key being updated, to exclude from uniqueness check
            (The Admin UI passes key_alias, in all Edit key requests. So we need to be sure that if we find a key with the same alias, it's not the same key we're updating)

    Raises:
        ProxyException: If key alias already exists on a different key
    """
    if key_alias is not None and prisma_client is not None:
        where_clause: dict[str, Any] = {"key_alias": key_alias}
        if existing_key_token:
            # Exclude the current key from the uniqueness check
            where_clause["NOT"] = {"token": existing_key_token}

        existing_key = await prisma_client.db.litellm_verificationtoken.find_first(
            where=where_clause
        )
        if existing_key is not None:
            raise ProxyException(
                message=f"Key with alias '{key_alias}' already exists. Unique key aliases across all keys are required.",
                type=ProxyErrorTypes.bad_request_error,
                param="key_alias",
                code=status.HTTP_400_BAD_REQUEST,
            )


def validate_model_max_budget(model_max_budget: Optional[Dict]) -> None:
    """
    Validate the model_max_budget is GenericBudgetConfigType + enforce user has an enterprise license

    Raises:
        Exception: If model_max_budget is not a valid GenericBudgetConfigType
    """
    try:
        if model_max_budget is None:
            return
        if len(model_max_budget) == 0:
            return
        if model_max_budget is not None:
            from litellm.proxy.proxy_server import CommonProxyErrors, premium_user

            if premium_user is not True:
                raise ValueError(
                    f"You must have an enterprise license to set model_max_budget. {CommonProxyErrors.not_premium_user.value}"
                )
            for _model, _budget_info in model_max_budget.items():
                assert isinstance(_model, str)

                # Normalize to dict (Pydantic may already parse nested values as BudgetConfig)
                _info = (
                    _budget_info.model_dump()
                    if hasattr(_budget_info, "model_dump")
                    else dict(_budget_info)
                )
                # /CRUD endpoints can pass budget_limit as a string, so we need to convert it to a float
                if "budget_limit" in _info:
                    _info["budget_limit"] = float(_info["budget_limit"])
                BudgetConfig(**_info)
    except Exception as e:
        raise ValueError(
            f"Invalid model_max_budget: {str(e)}. Example of valid model_max_budget: https://docs.litellm.ai/docs/proxy/users"
        )
