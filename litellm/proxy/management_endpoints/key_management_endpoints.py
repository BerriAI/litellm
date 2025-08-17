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
import json
import secrets
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Literal, Optional, Tuple, cast

import fastapi
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching import DualCache
from litellm.constants import LENGTH_OF_LITELLM_GENERATED_KEY, UI_SESSION_TOKEN_TEAM_ID
from litellm.litellm_core_utils.duration_parser import duration_in_seconds
from litellm.proxy._types import *
from litellm.proxy.auth.auth_checks import (
    _cache_key_object,
    _delete_cache_key_object,
    can_team_access_model,
    get_key_object,
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
    _hash_token_if_needed,
    handle_exception_on_proxy,
    is_valid_api_key,
    jsonify_object,
)
from litellm.router import Router
from litellm.secret_managers.main import get_secret
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


def _is_allowed_to_make_key_request(
    user_api_key_dict: UserAPIKeyAuth, user_id: Optional[str], team_id: Optional[str]
) -> bool:
    """
    Assert user only creates keys for themselves

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
) -> Literal[True]:
    """
    Check if user is allowed to make a key request, for this key
    """
    try:
        _is_allowed_to_make_key_request(
            user_api_key_dict=user_api_key_dict,
            user_id=data.user_id,
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
        verbose_proxy_logger.info(
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

    response = await generate_key_helper_fn(
        request_type="key", **data_json, table_name="key"
    )

    response["soft_budget"] = (
        data.soft_budget
    )  # include the user-input soft budget in the response

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
    - object_permission: Optional[LiteLLM_ObjectPermissionBase] - key-specific object permission. Example - {"vector_stores": ["vector_store_1", "vector_store_2"]}. IF null or {} then no object permission.
    - key_type: Optional[str] - Type of key that determines default allowed routes. Options: "llm_api" (can call LLM API routes), "management" (can call management routes), "read_only" (can only call info/read routes), "default" (uses default allowed routes). Defaults to "default".
    - prompts: Optional[List[str]] - List of allowed prompts for the key. If specified, the key will only be able to use these specific prompts.
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
        from litellm.proxy.proxy_server import (
            prisma_client,
            user_api_key_cache,
            user_custom_key_generate,
        )

        verbose_proxy_logger.debug("entered /key/generate")

        if user_custom_key_generate is not None:
            if asyncio.iscoroutinefunction(user_custom_key_generate):
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
                team_table = None

        key_generation_check(
            team_table=team_table,
            user_api_key_dict=user_api_key_dict,
            data=data,
            route=KeyManagementRoutes.KEY_GENERATE,
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
    - allowed_cache_controls: Optional[list] - List of allowed cache control values. Example - ["no-cache", "no-store"]. See all values - https://docs.litellm.ai/docs/proxy/caching#turn-on--off-caching-per-request
    - blocked: Optional[bool] - Whether the key is blocked.
    - rpm_limit: Optional[int] - Specify rpm limit for a given key (Requests per minute)
    - tpm_limit: Optional[int] - Specify tpm limit for a given key (Tokens per minute)
    - soft_budget: Optional[float] - Specify soft budget for a given key. Will trigger a slack alert when this soft budget is reached.
    - tags: Optional[List[str]] - Tags for [tracking spend](https://litellm.vercel.app/docs/proxy/enterprise#tracking-spend-for-custom-tags) and/or doing [tag-based routing](https://litellm.vercel.app/docs/proxy/tag_routing).
    - enforced_params: Optional[List[str]] - List of enforced params for the key (Enterprise only). [Docs](https://docs.litellm.ai/docs/proxy/enterprise#enforce-required-params-for-llm-requests)
    - allowed_routes: Optional[list] - List of allowed routes for the key. Store the actual route or store a wildcard pattern for a set of routes. Example - ["/chat/completions", "/embeddings", "/keys/*"]
    - object_permission: Optional[LiteLLM_ObjectPermissionBase] - key-specific object permission. Example - {"vector_stores": ["vector_store_1", "vector_store_2"]}. IF null or {} then no object permission.
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
    from litellm.proxy.proxy_server import (
        prisma_client,
        user_api_key_cache,
        user_custom_key_generate,
    )

    verbose_proxy_logger.debug("entered /key/generate")

    if user_custom_key_generate is not None:
        if asyncio.iscoroutinefunction(user_custom_key_generate):
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

                _premium_user_check()
                casted_metadata[k] = v

    except Exception as e:
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.prepare_metadata_fields(): Exception occured - {}".format(
                str(e)
            )
        )

    non_default_values["metadata"] = casted_metadata
    return non_default_values


async def _set_object_permission(
    data_json: dict,
    prisma_client: Optional[PrismaClient],
):
    """
    Creates the LiteLLM_ObjectPermissionTable record for the key.
    - Handles permissions for vector stores and mcp servers.
    """
    if prisma_client is None:
        return data_json

    if "object_permission" in data_json:
        created_object_permission = (
            await prisma_client.db.litellm_objectpermissiontable.create(
                data=data_json["object_permission"],
            )
        )
        data_json["object_permission_id"] = (
            created_object_permission.object_permission_id
        )
        # delete the object_permission from the data_json
        data_json.pop("object_permission")
    return data_json


async def prepare_key_update_data(
    data: Union[UpdateKeyRequest, RegenerateKeyRequest],
    existing_key_row: LiteLLM_VerificationToken,
):

    data_json: dict = data.model_dump(exclude_unset=True)
    data_json.pop("key", None)
    data_json.pop("new_key", None)
    non_default_values = {}
    for k, v in data_json.items():
        if (
            k in LiteLLM_ManagementEndpoint_MetadataFields
            or k in LiteLLM_ManagementEndpoint_MetadataFields_Premium
        ):
            continue
        non_default_values[k] = v

    if "duration" in non_default_values:
        duration = non_default_values.pop("duration")
        if duration and (isinstance(duration, str)) and len(duration) > 0:
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
    - allowed_cache_controls: Optional[list] - List of allowed cache control values
    - duration: Optional[str] - Key validity duration ("30d", "1h", etc.)
    - permissions: Optional[dict] - Key-specific permissions
    - send_invite_email: Optional[bool] - Send invite email to user_id
    - guardrails: Optional[List[str]] - List of active guardrails for the key
    - prompts: Optional[List[str]] - List of prompts that the key is allowed to use.
    - blocked: Optional[bool] - Whether the key is blocked
    - aliases: Optional[dict] - Model aliases for the key - [Docs](https://litellm.vercel.app/docs/proxy/virtual_keys#model-aliases)
    - config: Optional[dict] - [DEPRECATED PARAM] Key-specific config.
    - temp_budget_increase: Optional[float] - Temporary budget increase for the key (Enterprise only).
    - temp_budget_expiry: Optional[str] - Expiry time for the temporary budget increase (Enterprise only).
    - allowed_routes: Optional[list] - List of allowed routes for the key. Store the actual route or store a wildcard pattern for a set of routes. Example - ["/chat/completions", "/embeddings", "/keys/*"]
    - prompts: Optional[List[str]] - List of allowed prompts for the key. If specified, the key will only be able to use these specific prompts.
    - object_permission: Optional[LiteLLM_ObjectPermissionBase] - key-specific object permission. Example - {"vector_stores": ["vector_store_1", "vector_store_2"]}. IF null or {} then no object permission.
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
        data_json: dict = data.model_dump(exclude_unset=True, exclude_none=True)
        key = data_json.pop("key")

        # get the row from db
        if prisma_client is None:
            raise Exception("Not connected to DB!")

        common_key_access_checks(
            user_api_key_dict=user_api_key_dict,
            data=data,
            llm_router=llm_router,
            premium_user=premium_user,
        )

        existing_key_row = await prisma_client.get_data(
            token=data.key, table_name="key", query_type="find_unique"
        )

        if existing_key_row is None:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Team not found, passed team_id={data.team_id}"},
            )

        # check if user has permission to update key
        await TeamMemberPermissionChecks.can_team_member_execute_key_management_endpoint(
            user_api_key_dict=user_api_key_dict,
            route=KeyManagementRoutes.KEY_UPDATE,
            prisma_client=prisma_client,
            existing_key_row=existing_key_row,
            user_api_key_cache=user_api_key_cache,
        )

        # if team change - check if this is possible
        if is_different_team(data=data, existing_key_row=existing_key_row):
            team_obj = await get_team_object(
                team_id=cast(str, data.team_id),
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                check_db_only=True,
            )
            if llm_router is None:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "LLM router not found. Please set it up by passing in a valid config.yaml or adding models via the UI."
                    },
                )
            validate_key_team_change(
                key=existing_key_row,
                team=team_obj,
                change_initiated_by=user_api_key_dict,
                llm_router=llm_router,
            )
        non_default_values = await prepare_key_update_data(
            data=data, existing_key_row=existing_key_row
        )

        await _enforce_unique_key_alias(
            key_alias=non_default_values.get("key_alias", None),
            prisma_client=prisma_client,
            existing_key_token=existing_key_row.token,
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


def validate_key_team_change(
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
            can_team_access_model(
                model=model,
                team_object=team,
                llm_router=llm_router,
            )

    # Check if the key's user_id is a member of the team
    if key.user_id is not None:
        is_member = False
        for member in team.members_with_roles:
            if member.user_id == key.user_id:
                is_member = True
                break
        if not is_member:
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
    else:
        raise HTTPException(
            status_code=403,
            detail=f"User={change_initiated_by.user_id} is not a Proxy Admin or Team Admin for team={team.team_id}.",
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
            )
            num_keys_to_be_deleted = len(data.keys)
            deleted_keys = data.keys
        elif data.key_aliases:
            number_deleted_keys, _keys_being_deleted = await delete_key_aliases(
                key_aliases=data.key_aliases,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                user_api_key_dict=user_api_key_dict,
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
    curl -X GET "http://0.0.0.0:4000/key/info?key=sk-02Wr4IAlN3NvPXvL5JVvDA" \
-H "Authorization: Bearer sk-1234"
    ```

    Example Curl - if no key is passed, it will use the Key Passed in Authorization Header
    ```
    curl -X GET "http://0.0.0.0:4000/key/info" \
-H "Authorization: Bearer sk-02Wr4IAlN3NvPXvL5JVvDA"
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
        duration_s = duration_in_seconds(duration=duration)
        expires = datetime.now(timezone.utc) + timedelta(seconds=duration_s)

    if key_budget_duration is None:  # one-time budget
        key_reset_at = None
    else:
        duration_s = duration_in_seconds(duration=key_budget_duration)
        key_reset_at = datetime.now(timezone.utc) + timedelta(seconds=duration_s)

    if budget_duration is None:  # one-time budget
        reset_at = None
    else:
        reset_at = get_budget_reset_time(budget_duration=budget_duration)

    aliases_json = json.dumps(aliases)
    config_json = json.dumps(config)
    permissions_json = json.dumps(permissions)

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
            "budget_id": budget_id,
            "blocked": blocked,
            "created_by": created_by,
            "updated_by": updated_by,
            "allowed_routes": allowed_routes or [],
            "object_permission_id": object_permission_id,
        }

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


async def can_delete_verification_token(
    key_info: LiteLLM_VerificationToken,
    user_api_key_cache: DualCache,
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: PrismaClient,
) -> bool:
    """
    - check if user is proxy admin
    - check if user is team admin and key is a team key
    - check if key is personal key
    """
    is_team_key = _is_team_key(data=key_info)
    if user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value:
        return True
    elif is_team_key and key_info.team_id is not None:
        return await _team_key_deletion_check(
            user_api_key_dict=user_api_key_dict,
            key_info=key_info,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
        )
    elif key_info.user_id is not None and key_info.user_id == user_api_key_dict.user_id:
        return True
    else:
        return False


async def delete_verification_tokens(
    tokens: List,
    user_api_key_cache: DualCache,
    user_api_key_dict: UserAPIKeyAuth,
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
            _keys_being_deleted: List[LiteLLM_VerificationToken] = (
                await prisma_client.db.litellm_verificationtoken.find_many(
                    where={"token": {"in": tokens}}
                )
            )

            if len(_keys_being_deleted) == 0:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"error": "No keys found"},
                )

            # Assuming 'db' is your Prisma Client instance
            # check if admin making request - don't filter by user-id
            if user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value:
                deleted_tokens = await prisma_client.delete_data(tokens=tokens)
            # else
            else:
                tasks = []
                deleted_tokens = []
                for key in _keys_being_deleted:

                    async def _delete_key(key: LiteLLM_VerificationToken):
                        if await can_delete_verification_token(
                            key_info=key,
                            user_api_key_cache=user_api_key_cache,
                            user_api_key_dict=user_api_key_dict,
                            prisma_client=prisma_client,
                        ):
                            await prisma_client.delete_data(tokens=[key.token])
                            deleted_tokens.append(key.token)
                        else:
                            raise HTTPException(
                                status_code=status.HTTP_403_FORBIDDEN,
                                detail={
                                    "error": "You are not authorized to delete this key"
                                },
                            )

                    tasks.append(_delete_key(key))
                await asyncio.gather(*tasks)

                _num_deleted_tokens = len(deleted_tokens)
                if _num_deleted_tokens != len(tokens):
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


async def delete_key_aliases(
    key_aliases: List[str],
    user_api_key_cache: DualCache,
    prisma_client: PrismaClient,
    user_api_key_dict: UserAPIKeyAuth,
) -> Tuple[Optional[Dict], List[LiteLLM_VerificationToken]]:
    _keys_being_deleted = await prisma_client.db.litellm_verificationtoken.find_many(
        where={"key_alias": {"in": key_aliases}}
    )

    tokens = [key.token for key in _keys_being_deleted]
    return await delete_verification_tokens(
        tokens=tokens,
        user_api_key_cache=user_api_key_cache,
        user_api_key_dict=user_api_key_dict,
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
        models: Optional[List] = (
            await prisma_client.db.litellm_proxymodeltable.find_many()
        )
    except Exception:
        models = None
    # 2. process model table
    if models:
        decrypted_models = proxy_config.decrypt_model_list_from_db(new_models=models)
        verbose_proxy_logger.info(
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
        verbose_proxy_logger.info("Resetting proxy model table")
        await prisma_client.db.litellm_proxymodeltable.delete_many()
        verbose_proxy_logger.info("Creating %s models", len(new_models))
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
async def regenerate_key_fn(
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

        verbose_proxy_logger.debug("key_in_db: %s", _key_in_db)

        new_token = get_new_token(data=data)

        new_token_hash = hash_token(new_token)
        new_token_key_name = f"sk-...{new_token[-4:]}"

        # Prepare the update data
        update_data = {
            "token": new_token_hash,
            "key_name": new_token_key_name,
        }

        non_default_values = {}
        if data is not None:
            # Update with any provided parameters from GenerateKeyRequest
            non_default_values = await prepare_key_update_data(
                data=data, existing_key_row=_key_in_db
            )
            verbose_proxy_logger.debug("non_default_values: %s", non_default_values)

        update_data.update(non_default_values)
        update_data = prisma_client.jsonify_object(data=update_data)
        # Update the token in the database
        updated_token = await prisma_client.db.litellm_verificationtoken.update(
            where={"token": hashed_api_key},
            data=update_data,  # type: ignore
        )

        updated_token_dict = {}
        if updated_token is not None:
            updated_token_dict = dict(updated_token)

        updated_token_dict["key"] = new_token
        updated_token_dict["token_id"] = updated_token_dict.pop("token")

        ### 3. remove existing key entry from cache
        ######################################################################
        if key:
            await _delete_cache_key_object(
                hashed_token=hash_token(key),
                user_api_key_cache=user_api_key_cache,
                proxy_logging_obj=proxy_logging_obj,
            )

        if hashed_api_key:
            await _delete_cache_key_object(
                hashed_token=hash_token(key),
                user_api_key_cache=user_api_key_cache,
                proxy_logging_obj=proxy_logging_obj,
            )

        response = GenerateKeyResponse(
            **updated_token_dict,
        )

        asyncio.create_task(
            KeyManagementEventHooks.async_key_rotated_hook(
                data=data,
                existing_key_row=_key_in_db,
                response=response,
                user_api_key_dict=user_api_key_dict,
                litellm_changed_by=litellm_changed_by,
            )
        )

        return response
    except Exception as e:
        verbose_proxy_logger.exception("Error regenerating key: %s", e)
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
    complete_user_info_db_obj: Optional[BaseModel] = (
        await prisma_client.db.litellm_usertable.find_unique(
            where={"user_id": user_api_key_dict.user_id},
            include={"organization_memberships": True},
        )
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
    teams: Optional[List[BaseModel]] = (
        await prisma_client.db.litellm_teamtable.find_many(
            where={"team_id": {"in": complete_user_info.teams}}
        )
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
    sort_by: Optional[str] = Query(
        default=None,
        description="Column to sort by (e.g. 'user_id', 'created_at', 'spend')",
    ),
    sort_order: str = Query(default="desc", description="Sort order ('asc' or 'desc')"),
) -> KeyListResponseObject:
    """
    List all keys for a given user / team / organization.

    Returns:
        {
            "keys": List[str] or List[UserAPIKeyAuth],
            "total_count": int,
            "current_page": int,
            "total_pages": int,
        }
    """
    try:
        from litellm.proxy.proxy_server import prisma_client

        verbose_proxy_logger.debug("Entering list_keys function")

        if prisma_client is None:
            verbose_proxy_logger.error("Database not connected")
            raise Exception("Database not connected")

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

        if user_id is None and user_api_key_dict.user_role not in [
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
            sort_by=sort_by,
            sort_order=sort_order,
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
    sort_by: Optional[str] = None,
    sort_order: str = "desc",
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

    # Add condition for admin team keys if provided
    if admin_team_ids:
        or_conditions.append({"team_id": {"in": admin_team_ids}})

    # Combine conditions with OR if we have multiple conditions
    if len(or_conditions) > 1:
        where = {"AND": [where, {"OR": or_conditions}]}
    elif len(or_conditions) == 1:
        where.update(or_conditions[0])

    verbose_proxy_logger.debug(f"Filter conditions: {where}")

    # Calculate skip for pagination
    skip = (page - 1) * size

    verbose_proxy_logger.debug(f"Pagination: skip={skip}, take={size}")

    order_by: Optional[Dict[str, str]] = (
        _validate_sort_params(sort_by, sort_order)
        if sort_by is not None and isinstance(sort_by, str)
        else None
    )

    # Fetch keys with pagination
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
    total_count = await prisma_client.db.litellm_verificationtoken.count(
        where=where  # type: ignore
    )

    verbose_proxy_logger.debug(f"Total count of keys: {total_count}")

    # Calculate total pages
    total_pages = -(-total_count // size)  # Ceiling division

    # Prepare response
    key_list: List[Union[str, UserAPIKeyAuth]] = []
    for key in keys:
        key_dict = key.dict()
        # Attach object_permission if object_permission_id is set
        key_dict = await attach_object_permission_to_dict(key_dict, prisma_client)
        if return_full_object is True:
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
    hashed_token = hash_token(token=data.key)

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

                # /CRUD endpoints can pass budget_limit as a string, so we need to convert it to a float
                if "budget_limit" in _budget_info:
                    _budget_info["budget_limit"] = float(_budget_info["budget_limit"])
                BudgetConfig(**_budget_info)
    except Exception as e:
        raise ValueError(
            f"Invalid model_max_budget: {str(e)}. Example of valid model_max_budget: https://docs.litellm.ai/docs/proxy/users"
        )
