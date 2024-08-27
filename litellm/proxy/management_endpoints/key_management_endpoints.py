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
import re
import secrets
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import fastapi
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_helpers.utils import management_endpoint_wrapper
from litellm.proxy.utils import _duration_in_seconds

router = APIRouter()


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
    - team_id: Optional[str] - The team id of the key
    - user_id: Optional[str] - The user id of the key
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
    - model_max_budget: Optional[dict] - key-specific model budget in USD. Example - {"text-davinci-002": 0.5, "gpt-3.5-turbo": 0.5}. IF null or {} then no model specific budget.
    - model_rpm_limit: Optional[dict] - key-specific model rpm limit. Example - {"text-davinci-002": 1000, "gpt-3.5-turbo": 1000}. IF null or {} then no model specific rpm limit.
    - model_tpm_limit: Optional[dict] - key-specific model tpm limit. Example - {"text-davinci-002": 1000, "gpt-3.5-turbo": 1000}. IF null or {} then no model specific tpm limit.
    Examples:

    1. Allow users to turn on/off pii masking

    ```bash
    curl --location 'http://0.0.0.0:8000/key/generate' \
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
            create_audit_log_for_update,
            general_settings,
            litellm_proxy_admin_name,
            prisma_client,
            proxy_logging_obj,
            user_custom_key_generate,
        )

        verbose_proxy_logger.debug("entered /key/generate")

        if user_custom_key_generate is not None:
            result = await user_custom_key_generate(data)
            decision = result.get("decision", True)
            message = result.get("message", "Authentication Failed - Custom Auth Rule")
            if not decision:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN, detail=message
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
                    setattr(
                        data, key, litellm.default_key_generate_params.get(key, None)
                    )
                elif key == "models" and value == []:
                    setattr(data, key, litellm.default_key_generate_params.get(key, []))
                elif key == "metadata" and value == {}:
                    setattr(data, key, litellm.default_key_generate_params.get(key, {}))

        # check if user set default key/generate params on config.yaml
        if litellm.upperbound_key_generate_params is not None:
            for elem in data:
                # if key in litellm.upperbound_key_generate_params, use the min of value and litellm.upperbound_key_generate_params[key]
                key, value = elem
                if (
                    value is not None
                    and getattr(litellm.upperbound_key_generate_params, key, None)
                    is not None
                ):
                    # if value is float/int
                    if key in [
                        "max_budget",
                        "max_parallel_requests",
                        "tpm_limit",
                        "rpm_limit",
                    ]:
                        if value > getattr(litellm.upperbound_key_generate_params, key):
                            raise HTTPException(
                                status_code=400,
                                detail={
                                    "error": f"{key} is over max limit set in config - user_value={value}; max_value={getattr(litellm.upperbound_key_generate_params, key)}"
                                },
                            )
                    elif key == "budget_duration":
                        # budgets are in 1s, 1m, 1h, 1d, 1m (30s, 30m, 30h, 30d, 30m)
                        # compare the duration in seconds and max duration in seconds
                        upperbound_budget_duration = _duration_in_seconds(
                            duration=getattr(
                                litellm.upperbound_key_generate_params, key
                            )
                        )
                        user_set_budget_duration = _duration_in_seconds(duration=value)
                        if user_set_budget_duration > upperbound_budget_duration:
                            raise HTTPException(
                                status_code=400,
                                detail={
                                    "error": f"Budget duration is over max limit set in config - user_value={user_set_budget_duration}; max_value={upperbound_budget_duration}"
                                },
                            )

        # TODO: @ishaan-jaff: Migrate all budget tracking to use LiteLLM_BudgetTable
        _budget_id = None
        if prisma_client is not None and data.soft_budget is not None:
            # create the Budget Row for the LiteLLM Verification Token
            budget_row = LiteLLM_BudgetTable(
                soft_budget=data.soft_budget,
                model_max_budget=data.model_max_budget or {},
            )
            new_budget = prisma_client.jsonify_object(
                budget_row.json(exclude_none=True)
            )

            _budget = await prisma_client.db.litellm_budgettable.create(
                data={
                    **new_budget,  # type: ignore
                    "created_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
                    "updated_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
                }
            )
            _budget_id = getattr(_budget, "budget_id", None)
        data_json = data.json()  # type: ignore
        # if we get max_budget passed to /key/generate, then use it as key_max_budget. Since generate_key_helper_fn is used to make new users
        if "max_budget" in data_json:
            data_json["key_max_budget"] = data_json.pop("max_budget", None)
        if _budget_id is not None:
            data_json["budget_id"] = _budget_id

        if "budget_duration" in data_json:
            data_json["key_budget_duration"] = data_json.pop("budget_duration", None)

        response = await generate_key_helper_fn(
            request_type="key", **data_json, table_name="key"
        )

        response["soft_budget"] = (
            data.soft_budget
        )  # include the user-input soft budget in the response

        if data.send_invite_email is True:
            if "email" not in general_settings.get("alerting", []):
                raise ValueError(
                    "Email alerting not setup on config.yaml. Please set `alerting=['email']. \nDocs: https://docs.litellm.ai/docs/proxy/email`"
                )
            event = WebhookEvent(
                event="key_created",
                event_group="key",
                event_message=f"API Key Created",
                token=response.get("token", ""),
                spend=response.get("spend", 0.0),
                max_budget=response.get("max_budget", 0.0),
                user_id=response.get("user_id", None),
                team_id=response.get("team_id", "Default Team"),
                key_alias=response.get("key_alias", None),
            )

            # If user configured email alerting - send an Email letting their end-user know the key was created
            asyncio.create_task(
                proxy_logging_obj.slack_alerting_instance.send_key_created_or_user_invited_email(
                    webhook_event=event,
                )
            )

        # Enterprise Feature - Audit Logging. Enable with litellm.store_audit_logs = True
        if litellm.store_audit_logs is True:
            _updated_values = json.dumps(response, default=str)
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
                        object_id=response.get("token_id", ""),
                        action="created",
                        updated_values=_updated_values,
                        before_value=None,
                    )
                )
            )

        return GenerateKeyResponse(**response)
    except Exception as e:
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.generate_key_fn(): Exception occured - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
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
    Update an existing key
    """
    from litellm.proxy.proxy_server import (
        create_audit_log_for_update,
        general_settings,
        litellm_proxy_admin_name,
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
        user_custom_key_generate,
    )

    try:
        data_json: dict = data.json()
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

        _metadata_fields = ["model_rpm_limit", "model_tpm_limit", "guardrails"]
        # get non default values for key
        non_default_values = {}
        for k, v in data_json.items():
            # this field gets stored in metadata
            if key in _metadata_fields:
                continue
            if v is not None and v not in (
                [],
                {},
                0,
            ):  # models default to [], spend defaults to 0, we should not reset these values
                non_default_values[k] = v

        if "duration" in non_default_values:
            duration = non_default_values.pop("duration")
            duration_s = _duration_in_seconds(duration=duration)
            expires = datetime.now(timezone.utc) + timedelta(seconds=duration_s)
            non_default_values["expires"] = expires

        if "budget_duration" in non_default_values:
            duration_s = _duration_in_seconds(
                duration=non_default_values["budget_duration"]
            )
            key_reset_at = datetime.now(timezone.utc) + timedelta(seconds=duration_s)
            non_default_values["budget_reset_at"] = key_reset_at

        # Update metadata for virtual Key
        if data.model_tpm_limit:
            _metadata = existing_key_row.metadata or {}
            if "model_tpm_limit" not in _metadata:
                _metadata["model_tpm_limit"] = {}

            _metadata["model_tpm_limit"].update(data.model_tpm_limit)
            non_default_values["metadata"] = _metadata
            non_default_values.pop("model_tpm_limit", None)

        if data.model_rpm_limit:
            _metadata = existing_key_row.metadata or {}
            if "model_rpm_limit" not in _metadata:
                _metadata["model_rpm_limit"] = {}

            _metadata["model_rpm_limit"].update(data.model_rpm_limit)
            non_default_values["metadata"] = _metadata
            non_default_values.pop("model_rpm_limit", None)

        if data.guardrails:
            _metadata = existing_key_row.metadata or {}
            _metadata["guardrails"] = data.guardrails

            # update values that will be written to the DB
            non_default_values["metadata"] = _metadata
            non_default_values.pop("guardrails", None)

        response = await prisma_client.update_data(
            token=key, data={**non_default_values, "token": key}
        )

        # Delete - key from cache, since it's been updated!
        # key updated - a new model could have been added to this key. it should not block requests after this is done
        user_api_key_cache.delete_cache(key)
        hashed_token = hash_token(key)
        user_api_key_cache.delete_cache(hashed_token)

        # Enterprise Feature - Audit Logging. Enable with litellm.store_audit_logs = True
        if litellm.store_audit_logs is True:
            _updated_values = json.dumps(data_json, default=str)

            _before_value = existing_key_row.json(exclude_none=True)
            _before_value = json.dumps(_before_value, default=str)

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
                        object_id=data.key,
                        action="updated",
                        updated_values=_updated_values,
                        before_value=_before_value,
                    )
                )
            )

        return {"key": key, **response["data"]}
        # update based on remaining passed in values
    except Exception as e:
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

    Returns:
    - deleted_keys (List[str]): A list of deleted keys. Example {"deleted_keys": ["sk-QWrxEynunsNpV1zT48HIrw", "837e17519f44683334df5291321d97b8bf1098cd490e49e215f6fea935aa28be"]}


    Raises:
        HTTPException: If an error occurs during key deletion.
    """
    try:
        from litellm.proxy.proxy_server import (
            create_audit_log_for_update,
            general_settings,
            litellm_proxy_admin_name,
            prisma_client,
            proxy_logging_obj,
            user_api_key_cache,
            user_custom_key_generate,
        )

        keys = data.keys
        if len(keys) == 0:
            raise ProxyException(
                message=f"No keys provided, passed in: keys={keys}",
                type=ProxyErrorTypes.auth_error,
                param="keys",
                code=status.HTTP_400_BAD_REQUEST,
            )

        ## only allow user to delete keys they own
        user_id = user_api_key_dict.user_id
        verbose_proxy_logger.debug(
            f"user_api_key_dict.user_role: {user_api_key_dict.user_role}"
        )
        if (
            user_api_key_dict.user_role is not None
            and user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN
        ):
            user_id = None  # unless they're admin

        # Enterprise Feature - Audit Logging. Enable with litellm.store_audit_logs = True
        # we do this after the first for loop, since first for loop is for validation. we only want this inserted after validation passes
        if litellm.store_audit_logs is True:
            # make an audit log for each team deleted
            for key in data.keys:
                key_row = await prisma_client.get_data(  # type: ignore
                    token=key, table_name="key", query_type="find_unique"
                )

                key_row = key_row.json(exclude_none=True)
                _key_row = json.dumps(key_row, default=str)

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
                            object_id=key,
                            action="deleted",
                            updated_values="{}",
                            before_value=_key_row,
                        )
                    )
                )

        number_deleted_keys = await delete_verification_token(
            tokens=keys, user_id=user_id
        )
        verbose_proxy_logger.debug(
            f"/key/delete - deleted_keys={number_deleted_keys['deleted_keys']}"
        )

        try:
            assert len(keys) == number_deleted_keys["deleted_keys"]
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"Not all keys passed in were deleted. This probably means you don't have access to delete all the keys passed in. Keys passed in={len(keys)}, Deleted keys ={number_deleted_keys['deleted_keys']}"
                },
            )

        for key in keys:
            user_api_key_cache.delete_cache(key)
            # remove hash token from cache
            hashed_token = hash_token(key)
            user_api_key_cache.delete_cache(hashed_token)

        verbose_proxy_logger.debug(
            f"/keys/delete - cache after delete: {user_api_key_cache.in_memory_cache.cache_dict}"
        )

        return {"deleted_keys": keys}
    except Exception as e:
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
    "/v2/key/info", tags=["key management"], dependencies=[Depends(user_api_key_auth)]
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
    curl -X GET "http://0.0.0.0:8000/key/info" \
    -H "Authorization: Bearer sk-1234" \
    -d {"keys": ["sk-1", "sk-2", "sk-3"]}
    ```
    """
    from litellm.proxy.proxy_server import (
        create_audit_log_for_update,
        general_settings,
        litellm_proxy_admin_name,
        prisma_client,
        proxy_logging_obj,
        user_custom_key_generate,
    )

    try:
        if prisma_client is None:
            raise Exception(
                f"Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )
        if data is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"message": "Malformed request. No keys passed in."},
            )

        key_info = await prisma_client.get_data(
            token=data.keys, table_name="key", query_type="find_all"
        )
        filtered_key_info = []
        for k in key_info:
            try:
                k = k.model_dump()  # noqa
            except:
                # if using pydantic v1
                k = k.dict()
            filtered_key_info.append(k)
        return {"key": data.keys, "info": filtered_key_info}

    except Exception as e:
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
    curl -X GET "http://0.0.0.0:8000/key/info?key=sk-02Wr4IAlN3NvPXvL5JVvDA" \
-H "Authorization: Bearer sk-1234"
    ```

    Example Curl - if no key is passed, it will use the Key Passed in Authorization Header
    ```
    curl -X GET "http://0.0.0.0:8000/key/info" \
-H "Authorization: Bearer sk-02Wr4IAlN3NvPXvL5JVvDA"
    ```
    """
    from litellm.proxy.proxy_server import (
        create_audit_log_for_update,
        general_settings,
        litellm_proxy_admin_name,
        prisma_client,
        proxy_logging_obj,
        user_custom_key_generate,
    )

    try:
        if prisma_client is None:
            raise Exception(
                f"Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )
        if key == None:
            key = user_api_key_dict.api_key
        key_info = await prisma_client.get_data(token=key)
        ## REMOVE HASHED TOKEN INFO BEFORE RETURNING ##
        try:
            key_info = key_info.model_dump()  # noqa
        except:
            # if using pydantic v1
            key_info = key_info.dict()
        key_info.pop("token")
        return {"key": key, "info": key_info}
    except Exception as e:
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


async def generate_key_helper_fn(
    request_type: Literal[
        "user", "key"
    ],  # identifies if this request is from /user/new or /key/generate
    duration: Optional[str],
    models: list,
    aliases: dict,
    config: dict,
    spend: float,
    key_max_budget: Optional[float] = None,  # key_max_budget is used to Budget Per key
    key_budget_duration: Optional[str] = None,
    budget_id: Optional[float] = None,  # budget id <-> LiteLLM_BudgetTable
    soft_budget: Optional[
        float
    ] = None,  # soft_budget is used to set soft Budgets Per user
    max_budget: Optional[float] = None,  # max_budget is used to Budget Per user
    budget_duration: Optional[str] = None,  # max_budget is used to Budget Per user
    token: Optional[str] = None,
    user_id: Optional[str] = None,
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
    model_rpm_limit: Optional[dict] = {},
    model_tpm_limit: Optional[dict] = {},
    guardrails: Optional[list] = None,
    teams: Optional[list] = None,
    organization_id: Optional[str] = None,
    table_name: Optional[Literal["key", "user"]] = None,
    send_invite_email: Optional[bool] = None,
):
    from litellm.proxy.proxy_server import (
        custom_db_client,
        litellm_proxy_budget_name,
        premium_user,
        prisma_client,
    )

    if prisma_client is None and custom_db_client is None:
        raise Exception(
            f"Connect Proxy to database to generate keys - https://docs.litellm.ai/docs/proxy/virtual_keys "
        )

    if token is None:
        token = f"sk-{secrets.token_urlsafe(16)}"

    if duration is None:  # allow tokens that never expire
        expires = None
    else:
        duration_s = _duration_in_seconds(duration=duration)
        expires = datetime.now(timezone.utc) + timedelta(seconds=duration_s)

    if key_budget_duration is None:  # one-time budget
        key_reset_at = None
    else:
        duration_s = _duration_in_seconds(duration=key_budget_duration)
        key_reset_at = datetime.now(timezone.utc) + timedelta(seconds=duration_s)

    if budget_duration is None:  # one-time budget
        reset_at = None
    else:
        duration_s = _duration_in_seconds(duration=budget_duration)
        reset_at = datetime.now(timezone.utc) + timedelta(seconds=duration_s)

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

    metadata_json = json.dumps(metadata)
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
            "team_id": team_id,
            "organization_id": organization_id,
            "user_role": user_role,
            "spend": spend,
            "models": models,
            "max_parallel_requests": max_parallel_requests,
            "tpm_limit": tpm_limit,
            "rpm_limit": rpm_limit,
            "budget_duration": budget_duration,
            "budget_reset_at": reset_at,
            "allowed_cache_controls": allowed_cache_controls,
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
        }

        if (
            litellm.get_secret("DISABLE_KEY_NAME", False) is True
        ):  # allow user to disable storing abbreviated key name (shown in UI, to help figure out which key spent how much)
            pass
        else:
            key_data["key_name"] = f"sk-...{token[-4:]}"
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
                and premium_user != True
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
                    ## use default user model list if no key-specific model list provided
                    if len(user_row.models) > 0 and len(key_data["models"]) == 0:  # type: ignore
                        key_data["models"] = user_row.models
                elif query_type == "update_data":
                    user_row = await prisma_client.update_data(
                        data=user_data,
                        table_name="user",
                        update_key_values=update_key_values,
                    )
            if user_id == litellm_proxy_budget_name or (
                table_name is not None and table_name == "user"
            ):
                # do not create a key for litellm_proxy_budget_name or if table name is set to just 'user'
                # we only need to ensure this exists in the user table
                # the LiteLLM_VerificationToken table will increase in size if we don't do this check
                return user_data

            ## CREATE KEY
            verbose_proxy_logger.debug("prisma_client: Creating Key= %s", key_data)
            create_key_response = await prisma_client.insert_data(
                data=key_data, table_name="key"
            )
            key_data["token_id"] = getattr(create_key_response, "token", None)
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


async def delete_verification_token(tokens: List, user_id: Optional[str] = None):
    from litellm.proxy.proxy_server import litellm_proxy_admin_name, prisma_client

    try:
        if prisma_client:
            # Assuming 'db' is your Prisma Client instance
            # check if admin making request - don't filter by user-id
            if user_id == litellm_proxy_admin_name:
                deleted_tokens = await prisma_client.delete_data(tokens=tokens)
            # else
            else:
                deleted_tokens = await prisma_client.delete_data(
                    tokens=tokens, user_id=user_id
                )
                _num_deleted_tokens = deleted_tokens.get("deleted_keys", 0)
                if _num_deleted_tokens != len(tokens):
                    raise Exception(
                        "Failed to delete all tokens. Tried to delete tokens that don't belong to user: "
                        + str(user_id)
                    )
        else:
            raise Exception("DB not connected. prisma_client is None")
    except Exception as e:
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.delete_verification_token(): Exception occured - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        raise e
    return deleted_tokens


@router.post(
    "/key/{key:path}/regenerate",
    tags=["key management"],
    dependencies=[Depends(user_api_key_auth)],
)
@management_endpoint_wrapper
async def regenerate_key_fn(
    key: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    litellm_changed_by: Optional[str] = Header(
        None,
        description="The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of other users, providing an audit trail for accountability",
    ),
) -> GenerateKeyResponse:
    from litellm.proxy.proxy_server import (
        hash_token,
        premium_user,
        prisma_client,
        user_api_key_cache,
    )

    """
    Endpoint for regenerating a key
    """

    if premium_user is not True:
        raise ValueError(
            f"Regenerating Virtual Keys is an Enterprise feature, {CommonProxyErrors.not_premium_user.value}"
        )

    # Check if key exists, raise exception if key is not in the DB

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

    verbose_proxy_logger.debug("key_in_db: %s", _key_in_db)

    new_token = f"sk-{secrets.token_urlsafe(16)}"
    new_token_hash = hash_token(new_token)
    new_token_key_name = f"sk-...{new_token[-4:]}"

    # update new token in DB
    updated_token = await prisma_client.db.litellm_verificationtoken.update(
        where={"token": hashed_api_key},
        data={
            "token": new_token_hash,
            "key_name": new_token_key_name,
        },
    )
    updated_token_dict = {}
    if updated_token is not None:
        updated_token_dict = dict(updated_token)

    updated_token_dict["token"] = new_token

    ### 3. remove existing key entry from cache
    ######################################################################
    if key:
        user_api_key_cache.delete_cache(key)

    if hashed_api_key:
        user_api_key_cache.delete_cache(hashed_api_key)

    return GenerateKeyResponse(
        **updated_token_dict,
    )
