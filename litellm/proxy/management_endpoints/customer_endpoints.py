"""
CUSTOMER MANAGEMENT

All /customer management endpoints

/customer/new
/customer/info
/customer/update
/customer/delete
"""

#### END-USER/CUSTOMER MANAGEMENT ####
from datetime import datetime, timedelta
from typing import List, Optional

import fastapi
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.duration_parser import duration_in_seconds
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.timezone_utils import get_budget_reset_time
from litellm.proxy.management_endpoints.common_daily_activity import get_daily_activity
from litellm.proxy.management_helpers.object_permission_utils import (
    _set_object_permission,
    handle_update_object_permission_common,
)
from litellm.proxy.utils import handle_exception_on_proxy
from litellm.repositories.budget_repository import BudgetRepository
from litellm.repositories.table_repositories import EndUserRepository
from litellm.types.proxy.management_endpoints.common_daily_activity import (
    SpendAnalyticsPaginatedResponse,
)
from litellm.types.proxy.management_endpoints.customer_endpoints import (
    BlockUsersResponse,
    CustomerResponse,
    DeleteCustomersResponse,
    UnblockUsersResponse,
)

router = APIRouter()

_CUSTOMER_UPDATE_ZERO_ALLOWED_BUDGET_FIELDS = frozenset(
    {
        "soft_budget",
        "max_parallel_requests",
        "tpm_limit",
        "rpm_limit",
    }
)


def _to_customer_response(record: BaseModel) -> CustomerResponse:
    """Validate a raw end-user DB row into the typed customer response.

    object_permission reverse relations and the budget's audit fields are
    dropped here by the response model's field set, so callers need no manual
    cleanup.
    """
    return CustomerResponse.model_validate(record.model_dump())


@router.post(
    "/end_user/block",
    tags=["Customer Management"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
@router.post(
    "/customer/block",
    tags=["Customer Management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=BlockUsersResponse,
)
async def block_user(data: BlockUsers):
    """
    [BETA] Reject calls with this end-user id

    Parameters:
    - user_ids (List[str], required): The unique `user_id`s for the users to block

        (any /chat/completion call with this user={end-user-id} param, will be rejected.)

        ```
        curl -X POST "http://0.0.0.0:8000/user/block"
        -H "Authorization: Bearer sk-1234"
        -d '{
        "user_ids": [<user_id>, ...]
        }'
        ```
    """
    from litellm.proxy.proxy_server import prisma_client

    try:
        records = []
        if prisma_client is not None:
            for id in data.user_ids:
                record = await EndUserRepository(prisma_client).table.upsert(
                    where={"user_id": id},  # type: ignore
                    data={
                        "create": {"user_id": id, "blocked": True},  # type: ignore
                        "update": {"blocked": True},
                    },
                )
                records.append(record)
        else:
            raise HTTPException(
                status_code=500,
                detail={"error": "Postgres DB Not connected"},
            )

        return {"blocked_users": records}
    except Exception as e:
        verbose_proxy_logger.error(f"An error occurred - {str(e)}")
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.post(
    "/end_user/unblock",
    tags=["Customer Management"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
@router.post(
    "/customer/unblock",
    tags=["Customer Management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=UnblockUsersResponse,
)
async def unblock_user(data: BlockUsers):
    """
    [BETA] Unblock calls with this user id

    Example
    ```
    curl -X POST "http://0.0.0.0:8000/user/unblock"
    -H "Authorization: Bearer sk-1234"
    -d '{
    "user_ids": [<user_id>, ...]
    }'
    ```
    """
    try:
        from enterprise.enterprise_hooks.blocked_user_list import (
            _ENTERPRISE_BlockedUserList,
        )
    except ImportError:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Blocked user check was never set. This call has no effect."
                + CommonProxyErrors.missing_enterprise_package_docker.value
            },
        )

    if (
        not any(isinstance(x, _ENTERPRISE_BlockedUserList) for x in litellm.callbacks)
        or litellm.blocked_user_list is None
    ):
        raise HTTPException(
            status_code=400,
            detail={"error": "Blocked user check was never set. This call has no effect."},
        )

    if isinstance(litellm.blocked_user_list, list):
        for id in data.user_ids:
            litellm.blocked_user_list.remove(id)
    else:
        raise HTTPException(
            status_code=500,
            detail={"error": "`blocked_user_list` must be set as a list. Filepaths can't be updated."},
        )

    return {"blocked_users": litellm.blocked_user_list}


def new_budget_request(data: NewCustomerRequest) -> Optional[BudgetNewRequest]:
    """
    Return a new budget object if new budget params are passed.
    """
    budget_params = BudgetNewRequest.model_fields.keys()
    budget_kv_pairs = {}

    # Get the actual values from the data object using getattr
    for field_name in budget_params:
        if field_name == "budget_id":
            continue
        value = getattr(data, field_name, None)
        if value is not None:
            budget_kv_pairs[field_name] = value

    if budget_kv_pairs:
        budget_request = BudgetNewRequest(**budget_kv_pairs)
        if budget_request.budget_reset_at is None and budget_request.budget_duration is not None:
            budget_request.budget_reset_at = datetime.utcnow() + timedelta(
                seconds=duration_in_seconds(duration=budget_request.budget_duration)
            )
        return budget_request
    return None


async def _handle_customer_object_permission_update(
    non_default_values: dict,
    end_user_table_data_typed: Optional[LiteLLM_EndUserTable],
    update_end_user_table_data: dict,
    prisma_client,
) -> None:
    """
    Handle object permission updates for customer endpoints.

    Updates the update_end_user_table_data dict in place with the new object_permission_id.

    Args:
        non_default_values: Dictionary containing the update values including object_permission
        end_user_table_data_typed: Existing end user table data
        update_end_user_table_data: Dictionary to update with new object_permission_id
        prisma_client: Prisma database client
    """
    if "object_permission" in non_default_values:
        existing_object_permission_id = (
            end_user_table_data_typed.object_permission_id if end_user_table_data_typed is not None else None
        )
        object_permission_id = await handle_update_object_permission_common(
            data_json=non_default_values,
            existing_object_permission_id=existing_object_permission_id,
            prisma_client=prisma_client,
        )
        if object_permission_id is not None:
            update_end_user_table_data["object_permission_id"] = object_permission_id


@router.post(
    "/end_user/new",
    tags=["Customer Management"],
    include_in_schema=False,
    dependencies=[Depends(user_api_key_auth)],
)
@router.post(
    "/customer/new",
    tags=["Customer Management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=CustomerResponse,
)
async def new_end_user(
    data: NewCustomerRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> CustomerResponse:
    """
    Allow creating a new Customer 


    Parameters:
    - user_id: str - The unique identifier for the user.
    - alias: Optional[str] - A human-friendly alias for the user.
    - blocked: bool - Flag to allow or disallow requests for this end-user. Default is False.
    - max_budget: Optional[float] - The maximum budget allocated to the user. Either 'max_budget' or 'budget_id' should be provided, not both.
    - budget_id: Optional[str] - The identifier for an existing budget allocated to the user. Either 'max_budget' or 'budget_id' should be provided, not both.
    - allowed_model_region: Optional[Union[Literal["eu"], Literal["us"]]] - Require all user requests to use models in this specific region.
    - default_model: Optional[str] - If no equivalent model in the allowed region, default all requests to this model.
    - metadata: Optional[dict] = Metadata for customer, store information for customer. Example metadata = {"data_training_opt_out": True}
    - budget_duration: Optional[str] - Budget is reset at the end of specified duration. If not set, budget is never reset. You can set duration as seconds ("30s"), minutes ("30m"), hours ("30h"), days ("30d").
    - tpm_limit: Optional[int] - [Not Implemented Yet] Specify tpm limit for a given customer (Tokens per minute)
    - rpm_limit: Optional[int] - [Not Implemented Yet] Specify rpm limit for a given customer (Requests per minute)
    - model_max_budget: Optional[dict] - [Not Implemented Yet] Specify max budget for a given model. Example: {"openai/gpt-4o-mini": {"max_budget": 100.0, "budget_duration": "1d"}}
    - max_parallel_requests: Optional[int] - [Not Implemented Yet] Specify max parallel requests for a given customer.
    - soft_budget: Optional[float] - [Not Implemented Yet] Get alerts when customer crosses given budget, doesn't block requests.
    - spend: Optional[float] - Specify initial spend for a given customer.
    - budget_reset_at: Optional[str] - Specify the date and time when the budget should be reset.
    - object_permission: Optional[LiteLLM_ObjectPermissionBase] - Customer-specific object permissions to control access to resources.
        Supported fields:
        * mcp_servers: List[str] - List of allowed MCP server IDs
        * mcp_access_groups: List[str] - List of MCP access group names
        * mcp_tool_permissions: Dict[str, List[str]] - Map of server ID to allowed tool names (e.g., {"server_1": ["tool_a", "tool_b"]})
        * vector_stores: List[str] - List of allowed vector store IDs
        * agents: List[str] - List of allowed agent IDs
        * agent_access_groups: List[str] - List of agent access group names
        Example: {"mcp_servers": ["server_1", "server_2"], "vector_stores": ["vector_store_1"], "agents": ["agent_1"]}
        IF null or {} then no object-level restrictions apply.
    
    
    - Allow specifying allowed regions 
    - Allow specifying default model

    Example curl:
    ```
    curl --location 'http://0.0.0.0:4000/customer/new' \
        --header 'Authorization: Bearer sk-1234' \
        --header 'Content-Type: application/json' \
        --data '{
            "user_id" : "ishaan-jaff-3",
            "allowed_region": "eu",
            "budget_id": "free_tier",
            "default_model": "azure/gpt-3.5-turbo-eu"
        }'

    # With object permissions
    curl -L -X POST 'http://localhost:4000/customer/new' \
        -H 'Authorization: Bearer sk-1234' \
        -H 'Content-Type: application/json' \
        -d '{
            "user_id": "user_1",
            "object_permission": {
              "mcp_servers": ["server_1"],
              "mcp_access_groups": ["public_group"],
              "vector_stores": ["vector_store_1"]
            }
          }'

        # return end-user object
    ```

    NOTE: This used to be called `/end_user/new`, we will still be maintaining compatibility for /end_user/XXX for these endpoints
    """
    """
    Validation:
        - check if default model exists 
        - create budget object if not already created
    
    - Add user to end user table 

    Return 
    - end-user object
    - currently allowed models 
    """
    from litellm.proxy.proxy_server import (
        litellm_proxy_admin_name,
        llm_router,
        prisma_client,
    )

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )
    try:
        ## VALIDATION ##
        if data.default_model is not None:
            if llm_router is None:
                raise HTTPException(
                    status_code=422,
                    detail={"error": CommonProxyErrors.no_llm_router.value},
                )
            elif data.default_model not in llm_router.get_model_names():
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "Default Model not on proxy. Configure via `/model/new` or config.yaml. Default_model={}, proxy_model_names={}".format(
                            data.default_model, set(llm_router.get_model_names())
                        )
                    },
                )

        new_end_user_obj: Dict = {}

        ## CREATE BUDGET ## if set
        _new_budget = new_budget_request(data)
        if _new_budget is not None:
            try:
                budget_record = await BudgetRepository(prisma_client).table.create(
                    data={
                        **_new_budget.model_dump(exclude_unset=True),
                        "created_by": user_api_key_dict.user_id or litellm_proxy_admin_name,  # type: ignore
                        "updated_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
                    }
                )
            except Exception as e:
                raise HTTPException(status_code=422, detail={"error": str(e)})

            new_end_user_obj["budget_id"] = budget_record.budget_id
        elif data.budget_id is not None:
            new_end_user_obj["budget_id"] = data.budget_id

        _user_data = data.dict(exclude_none=True)

        for k, v in _user_data.items():
            if k not in BudgetNewRequest.model_fields.keys():
                new_end_user_obj[k] = v

        ## Handle Object Permission - MCP Servers, Vector Stores etc.
        new_end_user_obj = await _set_object_permission(
            data_json=new_end_user_obj,
            prisma_client=prisma_client,
        )

        # Ensure object_permission is not in the data being sent to create
        # It should have been converted to object_permission_id by _set_object_permission
        if "object_permission" in new_end_user_obj:
            verbose_proxy_logger.warning(
                f"object_permission still in new_end_user_obj after _set_object_permission: {new_end_user_obj.get('object_permission')}"
            )
            new_end_user_obj.pop("object_permission", None)

        ## WRITE TO DB ##
        end_user_record = await EndUserRepository(prisma_client).table.create(
            data=new_end_user_obj,  # type: ignore
            include={"litellm_budget_table": True, "object_permission": True},
        )

        return _to_customer_response(end_user_record)
    except Exception as e:
        verbose_proxy_logger.exception(
            "litellm.proxy.management_endpoints.customer_endpoints.new_end_user(): Exception occured - {}".format(
                str(e)
            )
        )
        if "Unique constraint failed on the fields: (`user_id`)" in str(e):
            raise ProxyException(
                message=f"Customer already exists, passed user_id={data.user_id}. Please pass a new user_id.",
                type="bad_request",
                code=400,
                param="user_id",
            )
        raise handle_exception_on_proxy(e)


@router.get(
    "/customer/info",
    tags=["Customer Management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=CustomerResponse,
)
@router.get(
    "/end_user/info",
    tags=["Customer Management"],
    include_in_schema=False,
    dependencies=[Depends(user_api_key_auth)],
)
async def end_user_info(
    end_user_id: str = fastapi.Query(description="End User ID in the request parameters"),
) -> CustomerResponse:
    """
    Get information about an end-user. An `end_user` is a customer (external user) of the proxy.

    Parameters:
    - end_user_id (str, required): The unique identifier for the end-user

    Example curl:
    ```
    curl -X GET 'http://localhost:4000/customer/info?end_user_id=test-litellm-user-4' \
        -H 'Authorization: Bearer sk-1234'
    ```
    """
    try:
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )

        user_info = await EndUserRepository(prisma_client).table.find_first(
            where={"user_id": end_user_id},
            include={"litellm_budget_table": True, "object_permission": True},
        )

        if user_info is None:
            raise ProxyException(
                message="End User Id={} does not exist in db".format(end_user_id),
                type="not_found",
                code=404,
                param="end_user_id",
            )

        return _to_customer_response(user_info)

    except Exception as e:
        verbose_proxy_logger.exception(
            "litellm.proxy.management_endpoints.customer_endpoints.end_user_info(): Exception occured - {}".format(
                str(e)
            )
        )
        raise handle_exception_on_proxy(e)


@router.post(
    "/customer/update",
    tags=["Customer Management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=CustomerResponse,
)
@router.post(
    "/end_user/update",
    tags=["Customer Management"],
    include_in_schema=False,
    dependencies=[Depends(user_api_key_auth)],
)
async def update_end_user(
    data: UpdateCustomerRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> CustomerResponse:
    """
    Example curl 

    Parameters:
    - user_id: str
    - alias: Optional[str] = None  # human-friendly alias
    - blocked: bool = False  # allow/disallow requests for this end-user
    - max_budget: Optional[float] = None
    - soft_budget: Optional[float] = None
    - max_parallel_requests: Optional[int] = None
    - tpm_limit: Optional[int] = None
    - rpm_limit: Optional[int] = None
    - model_max_budget: Optional[dict] = None
    - budget_duration: Optional[str] = None  # e.g. "30d", "1mo"; also sets budget_reset_at when creating/updating a budget
    - budget_id: Optional[str] = None  # give either a budget_id or max_budget
    - allowed_model_region: Optional[AllowedModelRegion] = (
        None  # require all user requests to use models in this specific region
    )
    - default_model: Optional[str] = (
        None  # if no equivalent model in allowed region - default all requests to this model
    )
    - object_permission: Optional[LiteLLM_ObjectPermissionBase] - Customer-specific object permissions to control access to resources.
        Supported fields:
        * mcp_servers: List[str] - List of allowed MCP server IDs
        * mcp_access_groups: List[str] - List of MCP access group names
        * mcp_tool_permissions: Dict[str, List[str]] - Map of server ID to allowed tool names
        * vector_stores: List[str] - List of allowed vector store IDs
        * agents: List[str] - List of allowed agent IDs
        * agent_access_groups: List[str] - List of agent access group names
        Example: {"mcp_servers": ["server_1"], "vector_stores": ["vector_store_1"]}
        IF null or {} then no object-level restrictions apply.

    Example curl:
    ```
    curl --location 'http://0.0.0.0:4000/customer/update' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
        "user_id": "test-litellm-user-4",
        "budget_id": "paid_tier"
    }'

    # Updating object permissions
    curl -L -X POST 'http://localhost:4000/customer/update' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
        "user_id": "user_1",
        "object_permission": {
          "mcp_servers": ["server_3"],
          "vector_stores": ["vector_store_2", "vector_store_3"]
        }
      }'

    See below for all params
    ```
    """

    from litellm.proxy.proxy_server import litellm_proxy_admin_name, prisma_client

    try:
        data_json: dict = data.json()
        # get the row from db
        if prisma_client is None:
            raise Exception("Not connected to DB!")

        # get non default values for key
        explicit_fields = data.fields_set()
        non_default_values = {}
        for k, v in data_json.items():
            if v is None:
                continue
            if v in ([], {}, 0):
                if k in _CUSTOMER_UPDATE_ZERO_ALLOWED_BUDGET_FIELDS and k in explicit_fields:
                    non_default_values[k] = v
                continue
            non_default_values[k] = v

        ## Get end user table data ##
        end_user_table_data = await EndUserRepository(prisma_client).table.find_first(
            where={"user_id": data.user_id}, include={"litellm_budget_table": True}
        )

        if end_user_table_data is None:
            raise ProxyException(
                message="End User Id={} does not exist in db".format(data.user_id),
                type="not_found",
                code=404,
                param="user_id",
            )

        end_user_table_data_typed = LiteLLM_EndUserTable(**end_user_table_data.model_dump())

        ## Get budget table data ##
        end_user_budget_table = end_user_table_data_typed.litellm_budget_table

        ## Get all params for budget table ##
        budget_table_data = {}
        update_end_user_table_data = {}
        for k, v in non_default_values.items():
            # budget_id is for linking to existing budget, not for creating new budget
            if k == "budget_id":
                update_end_user_table_data[k] = v
            elif k in LiteLLM_BudgetTable.model_fields.keys():
                budget_table_data[k] = v

            elif k in LiteLLM_EndUserTable.model_fields.keys():
                update_end_user_table_data[k] = v

        ## Handle object permission updates (MCP servers, vector stores, etc.)
        await _handle_customer_object_permission_update(
            non_default_values=non_default_values,
            end_user_table_data_typed=end_user_table_data_typed,
            update_end_user_table_data=update_end_user_table_data,
            prisma_client=prisma_client,
        )

        ## Check if we need to create a new budget (only if budget fields are provided, not just budget_id) ##
        if budget_table_data:
            model_max_budget = budget_table_data.get("model_max_budget")
            if model_max_budget is not None and len(model_max_budget) > 0:
                from litellm.proxy.management_endpoints.key_management_endpoints import (
                    validate_model_max_budget,
                )

                try:
                    validate_model_max_budget(model_max_budget)
                except ValueError as e:
                    raise ProxyException(
                        message={"error": str(e)},
                        type="bad_request",
                        param="model_max_budget",
                        code=400,
                    )

            # Mirror /budget/new and /budget/update: when budget_duration is set and
            # budget_reset_at is not, compute the next reset so reset_budget_job can
            # renew spend. budget_reset_at is server-managed (not on LiteLLM_BudgetTable
            # allowlist), so it must be injected here.
            if budget_table_data.get("budget_duration") is not None and "budget_reset_at" not in budget_table_data:
                budget_table_data["budget_reset_at"] = get_budget_reset_time(
                    budget_duration=budget_table_data["budget_duration"]
                )

            if end_user_budget_table is None:
                ## Create new budget ##
                budget_table_data_record = await BudgetRepository(prisma_client).table.create(
                    data={
                        **budget_table_data,
                        "created_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
                        "updated_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
                    },
                    include={"end_users": True},
                )

                update_end_user_table_data["budget_id"] = budget_table_data_record.budget_id
            else:
                ## Update existing budget ##
                budget_table_data_record = await BudgetRepository(prisma_client).table.update(
                    where={"budget_id": end_user_budget_table.budget_id},
                    data=budget_table_data,
                )

        ## Update user table, with update params + new budget id (if set) ##
        verbose_proxy_logger.debug("/customer/update: Received data = %s", data)

        # Ensure object_permission is not in the update data
        # It should have been converted to object_permission_id by handle_update_object_permission_common
        if "object_permission" in update_end_user_table_data:
            verbose_proxy_logger.warning(
                f"object_permission still in update_end_user_table_data: {update_end_user_table_data.get('object_permission')}"
            )
            update_end_user_table_data.pop("object_permission", None)

        if data.user_id is not None and len(data.user_id) > 0:
            update_end_user_table_data["user_id"] = data.user_id  # type: ignore
            verbose_proxy_logger.debug("In update customer, user_id condition block.")
            response = await EndUserRepository(prisma_client).table.update(
                where={"user_id": data.user_id},
                data=update_end_user_table_data,
                include={"litellm_budget_table": True, "object_permission": True},  # type: ignore
            )
            if response is None:
                raise ValueError(f"Failed updating customer data. User ID does not exist passed user_id={data.user_id}")
            verbose_proxy_logger.debug(f"received response from updating prisma client. response={response}")

            return _to_customer_response(response)
        else:
            raise ValueError(f"user_id is required, passed user_id = {data.user_id}")

        # update based on remaining passed in values

    except Exception as e:
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.update_end_user(): Exception occured - {}".format(str(e))
        )
        raise handle_exception_on_proxy(e)


@router.post(
    "/customer/delete",
    tags=["Customer Management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=DeleteCustomersResponse,
)
@router.post(
    "/end_user/delete",
    tags=["Customer Management"],
    include_in_schema=False,
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_end_user(
    data: DeleteCustomerRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> DeleteCustomersResponse:
    """
    Delete multiple end-users.

    Parameters:
    - user_ids (List[str], required): The unique `user_id`s for the users to delete

    Example curl:
    ```
    curl --location 'http://0.0.0.0:4000/customer/delete' \
        --header 'Authorization: Bearer sk-1234' \
        --header 'Content-Type: application/json' \
        --data '{
            "user_ids" :["ishaan-jaff-5"]
    }'

    See below for all params 
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    try:
        if prisma_client is None:
            raise Exception("Not connected to DB!")

        verbose_proxy_logger.debug("/customer/delete: Received data = %s", data)
        if data.user_ids is not None and isinstance(data.user_ids, list) and len(data.user_ids) > 0:
            # First check if all users exist
            existing_users = await EndUserRepository(prisma_client).table.find_many(
                where={"user_id": {"in": data.user_ids}}
            )
            existing_user_ids = {user.user_id for user in existing_users}
            missing_user_ids = [user_id for user_id in data.user_ids if user_id not in existing_user_ids]

            if missing_user_ids:
                raise ProxyException(
                    message="End User Id(s)={} do not exist in db".format(", ".join(missing_user_ids)),
                    type="not_found",
                    code=404,
                    param="user_ids",
                )

            # All users exist, proceed with deletion
            response = await EndUserRepository(prisma_client).table.delete_many(
                where={"user_id": {"in": data.user_ids}}
            )
            verbose_proxy_logger.debug(f"received response from updating prisma client. response={response}")
            return DeleteCustomersResponse(
                deleted_customers=response,
                message="Successfully deleted customers with ids: " + str(data.user_ids),
            )
        else:
            raise ValueError(f"user_id is required, passed user_id = {data.user_ids}")

        # update based on remaining passed in values
    except Exception as e:
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.delete_end_user(): Exception occured - {}".format(str(e))
        )
        raise handle_exception_on_proxy(e)


@router.get(
    "/customer/list",
    tags=["Customer Management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=List[CustomerResponse],
)
@router.get(
    "/end_user/list",
    tags=["Customer Management"],
    include_in_schema=False,
    dependencies=[Depends(user_api_key_auth)],
)
async def list_end_user(
    http_request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> List[CustomerResponse]:
    """
    [Admin-only] List all available customers

    Example curl:
    ```
    curl --location --request GET 'http://0.0.0.0:4000/customer/list' \
        --header 'Authorization: Bearer sk-1234'
    ```

    """
    try:
        from litellm.proxy.proxy_server import prisma_client

        if (
            user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN
            and user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY
        ):
            raise HTTPException(
                status_code=401,
                detail={"error": "Admin-only endpoint. Your user role={}".format(user_api_key_dict.user_role)},
            )

        if prisma_client is None:
            raise HTTPException(
                status_code=400,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )

        response = await EndUserRepository(prisma_client).table.find_many(
            include={"litellm_budget_table": True, "object_permission": True}
        )

        return [_to_customer_response(item) for item in response]

    except Exception as e:
        verbose_proxy_logger.exception(
            "litellm.proxy.management_endpoints.customer_endpoints.list_end_user(): Exception occured - {}".format(
                str(e)
            )
        )
        raise handle_exception_on_proxy(e)


@router.get(
    "/customer/daily/activity",
    tags=["Customer Management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=SpendAnalyticsPaginatedResponse,
)
@router.get(
    "/end_user/daily/activity",
    tags=["Customer Management"],
    include_in_schema=False,
    dependencies=[Depends(user_api_key_auth)],
)
async def get_customer_daily_activity(
    end_user_ids: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    page: int = 1,
    page_size: int = 10,
    exclude_end_user_ids: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get daily activity for specific organizations or all accessible organizations.
    """
    if (
        user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN
        and user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY
    ):
        raise HTTPException(
            status_code=401,
            detail={"error": "Admin-only endpoint. Your user role={}".format(user_api_key_dict.user_role)},
        )

    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    # Parse comma-separated ids
    end_user_ids_list = end_user_ids.split(",") if end_user_ids else None
    exclude_end_user_ids_list: Optional[List[str]] = None
    if exclude_end_user_ids:
        exclude_end_user_ids_list = exclude_end_user_ids.split(",") if exclude_end_user_ids else None

    # Fetch organization aliases for metadata
    where_condition = {}
    if end_user_ids_list:
        where_condition["user_id"] = {"in": list(end_user_ids_list)}
    end_user_aliases = await EndUserRepository(prisma_client).table.find_many(where=where_condition)
    end_user_alias_metadata = {e.user_id: {"alias": e.alias} for e in end_user_aliases}

    # Query daily activity for organizations
    return await get_daily_activity(
        prisma_client=prisma_client,
        table_name="litellm_dailyenduserspend",
        entity_id_field="end_user_id",
        entity_id=end_user_ids_list,
        entity_metadata_field=end_user_alias_metadata,
        exclude_entity_ids=exclude_end_user_ids_list,
        start_date=start_date,
        end_date=end_date,
        model=model,
        api_key=api_key,
        page=page,
        page_size=page_size,
    )
