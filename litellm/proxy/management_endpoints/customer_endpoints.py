"""
CUSTOMER MANAGEMENT

All /customer management endpoints

/customer/new
/customer/info
/customer/update
/customer/delete
/customer/spend - List all customers with aggregated spend (paginated)
/customer/{end_user_id}/spend - Get detailed spend for a specific customer with model breakdown
"""

#### END-USER/CUSTOMER MANAGEMENT ####
from typing import List, Optional
from datetime import datetime

import fastapi
from fastapi import APIRouter, Depends, HTTPException, Request

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.utils import handle_exception_on_proxy

router = APIRouter()


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
                record = await prisma_client.db.litellm_endusertable.upsert(
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
            detail={
                "error": "Blocked user check was never set. This call has no effect."
            },
        )

    if isinstance(litellm.blocked_user_list, list):
        for id in data.user_ids:
            litellm.blocked_user_list.remove(id)
    else:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "`blocked_user_list` must be set as a list. Filepaths can't be updated."
            },
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
        return BudgetNewRequest(**budget_kv_pairs)
    return None


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
)
async def new_end_user(
    data: NewCustomerRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
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
            "default_model": "azure/gpt-3.5-turbo-eu" <- all calls from this user, use this model? 
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
                budget_record = await prisma_client.db.litellm_budgettable.create(
                    data={
                        **_new_budget.model_dump(exclude_unset=True),
                        "created_by": user_api_key_dict.user_id or litellm_proxy_admin_name,  # type: ignore
                        "updated_by": user_api_key_dict.user_id
                        or litellm_proxy_admin_name,
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

        ## WRITE TO DB ##
        end_user_record = await prisma_client.db.litellm_endusertable.create(
            data=new_end_user_obj,  # type: ignore
            include={"litellm_budget_table": True},
        )

        return end_user_record
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
    response_model=LiteLLM_EndUserTable,
)
@router.get(
    "/end_user/info",
    tags=["Customer Management"],
    include_in_schema=False,
    dependencies=[Depends(user_api_key_auth)],
)
async def end_user_info(
    end_user_id: str = fastapi.Query(
        description="End User ID in the request parameters"
    ),
):
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

        user_info = await prisma_client.db.litellm_endusertable.find_first(
            where={"user_id": end_user_id}, include={"litellm_budget_table": True}
        )

        if user_info is None:
            raise ProxyException(
                message="End User Id={} does not exist in db".format(end_user_id),
                type="not_found",
                code=404,
                param="end_user_id",
            )
        return user_info.model_dump(exclude_none=True)
    
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
):
    """
    Example curl 

    Parameters:
    - user_id: str
    - alias: Optional[str] = None  # human-friendly alias
    - blocked: bool = False  # allow/disallow requests for this end-user
    - max_budget: Optional[float] = None
    - budget_id: Optional[str] = None  # give either a budget_id or max_budget
    - allowed_model_region: Optional[AllowedModelRegion] = (
        None  # require all user requests to use models in this specific region
    )
    - default_model: Optional[str] = (
        None  # if no equivalent model in allowed region - default all requests to this model
    )

    Example curl:
    ```
    curl --location 'http://0.0.0.0:4000/customer/update' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
        "user_id": "test-litellm-user-4",
        "budget_id": "paid_tier"
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
        non_default_values = {}
        for k, v in data_json.items():
            if v is not None and v not in (
                [],
                {},
                0,
            ):  # models default to [], spend defaults to 0, we should not reset these values
                non_default_values[k] = v

        ## Get end user table data ##
        end_user_table_data = await prisma_client.db.litellm_endusertable.find_first(
            where={"user_id": data.user_id}, include={"litellm_budget_table": True}
        )

        if end_user_table_data is None:
            raise ProxyException(
                message="End User Id={} does not exist in db".format(data.user_id),
                type="not_found",
                code=404,
                param="user_id",
            )

        end_user_table_data_typed = LiteLLM_EndUserTable(
            **end_user_table_data.model_dump()
        )

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

        ## Check if we need to create a new budget (only if budget fields are provided, not just budget_id) ##
        if budget_table_data:
            if end_user_budget_table is None:
                ## Create new budget ##
                budget_table_data_record = (
                    await prisma_client.db.litellm_budgettable.create(
                        data={
                            **budget_table_data,
                            "created_by": user_api_key_dict.user_id
                            or litellm_proxy_admin_name,
                            "updated_by": user_api_key_dict.user_id
                            or litellm_proxy_admin_name,
                        },
                        include={"end_users": True},
                    )
                )

                update_end_user_table_data[
                    "budget_id"
                ] = budget_table_data_record.budget_id
            else:
                ## Update existing budget ##
                budget_table_data_record = (
                    await prisma_client.db.litellm_budgettable.update(
                        where={"budget_id": end_user_budget_table.budget_id},
                        data=budget_table_data,
                    )
                )

        ## Update user table, with update params + new budget id (if set) ##
        verbose_proxy_logger.debug("/customer/update: Received data = %s", data)
        if data.user_id is not None and len(data.user_id) > 0:
            update_end_user_table_data["user_id"] = data.user_id  # type: ignore
            verbose_proxy_logger.debug("In update customer, user_id condition block.")
            response = await prisma_client.db.litellm_endusertable.update(
                where={"user_id": data.user_id}, data=update_end_user_table_data, include={"litellm_budget_table": True}  # type: ignore
            )
            if response is None:
                raise ValueError(
                    f"Failed updating customer data. User ID does not exist passed user_id={data.user_id}"
                )
            verbose_proxy_logger.debug(
                f"received response from updating prisma client. response={response}"
            )
            return response
        else:
            raise ValueError(f"user_id is required, passed user_id = {data.user_id}")

        # update based on remaining passed in values

    except Exception as e:
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.update_end_user(): Exception occured - {}".format(
                str(e)
            )
        )
        raise handle_exception_on_proxy(e)


@router.post(
    "/customer/delete",
    tags=["Customer Management"],
    dependencies=[Depends(user_api_key_auth)],
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
):
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
        if (
            data.user_ids is not None
            and isinstance(data.user_ids, list)
            and len(data.user_ids) > 0
        ):
            # First check if all users exist
            existing_users = await prisma_client.db.litellm_endusertable.find_many(
                where={"user_id": {"in": data.user_ids}}
            )
            existing_user_ids = {user.user_id for user in existing_users}
            missing_user_ids = [
                user_id for user_id in data.user_ids if user_id not in existing_user_ids
            ]

            if missing_user_ids:
                raise ProxyException(
                    message="End User Id(s)={} do not exist in db".format(
                        ", ".join(missing_user_ids)
                    ),
                    type="not_found",
                    code=404,
                    param="user_ids",
                )

            # All users exist, proceed with deletion
            response = await prisma_client.db.litellm_endusertable.delete_many(
                where={"user_id": {"in": data.user_ids}}
            )
            verbose_proxy_logger.debug(
                f"received response from updating prisma client. response={response}"
            )
            return {
                "deleted_customers": response,
                "message": "Successfully deleted customers with ids: "
                + str(data.user_ids),
            }
        else:
            raise ValueError(f"user_id is required, passed user_id = {data.user_ids}")

        # update based on remaining passed in values
    except Exception as e:
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.delete_end_user(): Exception occured - {}".format(
                str(e)
            )
        )
        raise handle_exception_on_proxy(e)

@router.get(
    "/customer/list",
    tags=["Customer Management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=List[LiteLLM_EndUserTable],
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
):
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
                detail={
                    "error": "Admin-only endpoint. Your user role={}".format(
                        user_api_key_dict.user_role
                    )
                },
            )

        if prisma_client is None:
            raise HTTPException(
                status_code=400,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )

        response = await prisma_client.db.litellm_endusertable.find_many(
            include={"litellm_budget_table": True}
        )

        returned_response: List[LiteLLM_EndUserTable] = []
        for item in response:
            returned_response.append(LiteLLM_EndUserTable(**item.model_dump()))
        return returned_response
    
    except Exception as e:
        verbose_proxy_logger.exception(
            "litellm.proxy.management_endpoints.customer_endpoints.list_end_user(): Exception occured - {}".format(
                str(e)
            )
        )
        raise handle_exception_on_proxy(e)

@router.get(
    "/customer/spend",
    tags=["Customer Management"],
    dependencies=[Depends(user_api_key_auth)],
)
@router.get(
    "/end_user/spend",
    tags=["Customer Management"],
    include_in_schema=False,
    dependencies=[Depends(user_api_key_auth)],
)
async def get_customer_spend_report(
    http_request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    start_date: Optional[str] = fastapi.Query(
        default=None,
        description="Start date in YYYY-MM-DD format. If not provided, returns all-time spend.",
    ),
    end_date: Optional[str] = fastapi.Query(
        default=None,
        description="End date in YYYY-MM-DD format. If not provided, uses current date.",
    ),
    end_user_id: Optional[str] = fastapi.Query(
        default=None,
        description="Filter by specific end user ID. If not provided, returns spend for all end users.",
    ),
    alias: Optional[str] = fastapi.Query(
        default=None,
        description="Filter by customer alias. If not provided, returns spend for all end users.",
    ),
    page: int = fastapi.Query(
        default=1,
        ge=1,
        description="Page number (starts at 1).",
    ),
    page_size: int = fastapi.Query(
        default=50,
        ge=1,
        le=1000,
        description="Number of results per page (max 1000).",
    ),
):
    """
    [Admin-only] Get spend report for customers/ end users over a specified time period.

    This endpoint aggregates spend data from LiteLLM_SpendLogs and joins it with
    the LiteLLM_EndUserTable to include user aliases.

    Parameters:
    - start_date: Optional[str] - Start date for the report in YYYY-MM-DD format
    - end_date: Optional[str] - End date for the report in YYYY-MM-DD format
    - end_user_id: Optional[str] - Filter by specific end user ID
    - alias: Optional[str] - Filter by customer alias
    - page: int - Page number (default: 1, min: 1)
    - page_size: int - Number of results per page (default: 50, min: 1, max: 1000)

    Returns:
    A paginated list of end user spend records with:
    - end_user_id: The end user ID
    - alias: The admin-facing alias for the end user (null if not set)
    - total_spend: Total spend for the end user
    - total_requests: Total number of requests
    - total_tokens: Total tokens used
    - total_prompt_tokens: Total prompt tokens used
    - total_completion_tokens: Total completion tokens used

    Response also includes pagination metadata:
    - total_customers: Total number of customers matching the criteria
    - page: Current page number
    - page_size: Number of results per page
    - total_pages: Total number of pages

    Example curl:
    ```
    curl --location 'http://0.0.0.0:4000/customer/spend?start_date=2024-01-01&end_date=2024-12-31&page=1&page_size=50' \
        --header 'Authorization: Bearer sk-1234'
    ```

    Example with specific end user:
    ```
    curl --location 'http://0.0.0.0:4000/customer/spend?end_user_id=user-123' \
        --header 'Authorization: Bearer sk-1234'
    ```

    Example with alias filter:
    ```
    curl --location 'http://0.0.0.0:4000/customer/spend?alias=acme-corp' \
        --header 'Authorization: Bearer sk-1234'
    ```

    Example with pagination:
    ```
    curl --location 'http://0.0.0.0:4000/customer/spend?page=2&page_size=100' \
        --header 'Authorization: Bearer sk-1234'
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    if (
        user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN
        and user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY
    ):
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Admin-only endpoint. Your user role={}".format(
                    user_api_key_dict.user_role
                )
            },
        )

    if prisma_client is None:
        raise HTTPException(
            status_code=400,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    # Build query parameters and filters
    date_query_params = []
    user_query_params = []
    date_filter_conditions = []

    # Add date range filters
    if start_date is not None:
        date_filter_conditions.append(f'sl."startTime" >= ${len(date_query_params) + 1}::timestamp')
        date_query_params.append(datetime.strptime(start_date, "%Y-%m-%d"))

    if end_date is not None:
        end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
        end_datetime = end_datetime.replace(hour=23, minute=59, second=59)
        date_filter_conditions.append(f'sl."startTime" <= ${len(date_query_params) + 1}::timestamp')
        date_query_params.append(end_datetime)

    # Build date filter SQL fragment for JOIN
    date_filter_sql = (
        " AND " + " AND ".join(date_filter_conditions) if date_filter_conditions else ""
    )

    # Add user filter
    user_filter_conditions = []
    if end_user_id is not None and end_user_id.strip():
        user_filter_conditions.append(f"eu.user_id = ${len(user_query_params) + 1}")  # Exact match for user_id
        user_query_params.append(end_user_id)
    if alias is not None and alias.strip():
        user_filter_conditions.append(f"eu.alias ILIKE ${len(user_query_params) + 1}")  # Partial match for alias
        user_query_params.append(f"%{alias}%")

    user_filter_sql = ""
    if user_filter_conditions:
        user_filter_sql = "WHERE " + " AND ".join(user_filter_conditions)  # Changed to AND

    # Get total count for pagination (only uses user filters, not date filters)
    count_query = f"""
        SELECT COUNT(*) as total
        FROM "LiteLLM_EndUserTable" eu
        {user_filter_sql}
    """
    count_result = await prisma_client.db.query_raw(count_query, *user_query_params)
    total_customers = int(count_result[0]["total"]) if count_result else 0

    total_pages = (total_customers + page_size - 1) // page_size if total_customers > 0 else 0
    offset = (page - 1) * page_size

    # Get paginated customer spend
    # Combine parameters: date params first (used in JOIN), then user params (used in WHERE), then pagination
    all_query_params = date_query_params + user_query_params

    # Adjust user filter parameter indices if there are date parameters
    if date_query_params and user_filter_conditions:
        # Rebuild user filter with adjusted indices
        user_filter_conditions_adjusted = []
        param_offset = len(date_query_params)
        if end_user_id is not None and end_user_id.strip():
            user_filter_conditions_adjusted.append(f"eu.user_id = ${param_offset + 1}")
            param_offset += 1
        if alias is not None and alias.strip():
            user_filter_conditions_adjusted.append(f"eu.alias ILIKE ${param_offset + 1}")
        user_filter_sql = "WHERE " + " AND ".join(user_filter_conditions_adjusted)

    report_query = f"""
        SELECT
            eu.user_id as end_user_id,
            eu.alias,
            COALESCE(SUM(sl.spend), 0) as total_spend,
            COALESCE(COUNT(sl.request_id), 0) as total_requests,
            COALESCE(SUM(sl.total_tokens), 0) as total_tokens,
            COALESCE(SUM(sl.prompt_tokens), 0) as total_prompt_tokens,
            COALESCE(SUM(sl.completion_tokens), 0) as total_completion_tokens
        FROM "LiteLLM_EndUserTable" eu
        LEFT JOIN "LiteLLM_SpendLogs" sl
            ON eu.user_id = sl.end_user{date_filter_sql}
        {user_filter_sql}
        GROUP BY eu.user_id, eu.alias
        ORDER BY total_spend DESC
        LIMIT ${len(all_query_params) + 1} OFFSET ${len(all_query_params) + 2}
    """

    rows = await prisma_client.db.query_raw(
        report_query, *all_query_params, page_size, offset
    )

    # Format results
    result = []
    for row in rows:
        if row["end_user_id"] is None:
            continue
        result.append({
            "end_user_id": row["end_user_id"],
            "alias": row["alias"],
            "total_spend": float(row["total_spend"] or 0.0),
            "total_requests": int(row["total_requests"] or 0),
            "total_tokens": int(row["total_tokens"] or 0),
            "total_prompt_tokens": int(row["total_prompt_tokens"] or 0),
            "total_completion_tokens": int(row["total_completion_tokens"] or 0),
        })

    return {
        "spend_report": result,
        "total_customers": total_customers,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "date_range": {
            "start_date": start_date,
            "end_date": end_date,
        },
    }


@router.get(
    "/customer/{end_user_id}/spend",
    tags=["Customer Management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_customer_spend_detail(
    end_user_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    start_date: Optional[str] = fastapi.Query(
        default=None,
        description="Start date in YYYY-MM-DD format. If not provided, returns all-time spend.",
    ),
    end_date: Optional[str] = fastapi.Query(
        default=None,
        description="End date in YYYY-MM-DD format. If not provided, uses current date.",
    ),
):
    """
    [Admin-only] Get detailed spend information for a specific customer, including model breakdown.

    This endpoint provides comprehensive spend analytics for a single end user,
    including aggregated totals and per-model breakdowns.

    Parameters:
    - end_user_id: str (path parameter) - The end user ID to get spend details for
    - start_date: Optional[str] - Start date for the report in YYYY-MM-DD format
    - end_date: Optional[str] - End date for the report in YYYY-MM-DD format

    Returns:
    Detailed spend information including:
    - end_user_id: The end user ID
    - alias: The admin-facing alias for the end user
    - total_spend: Total spend for the end user
    - total_requests: Total number of requests
    - total_tokens: Total tokens used
    - total_prompt_tokens: Total prompt tokens used
    - total_completion_tokens: Total completion tokens used
    - spend_by_model: Array of per-model spend breakdowns with:
        - model: Model name
        - total_spend: Spend for this model
        - total_requests: Number of requests for this model
        - total_tokens: Total tokens for this model
        - total_prompt_tokens: Prompt tokens for this model
        - total_completion_tokens: Completion tokens for this model

    Example curl:
    ```
    curl --location 'http://0.0.0.0:4000/customer/user-123/spend?start_date=2024-01-01&end_date=2024-12-31' \\
        --header 'Authorization: Bearer sk-1234'
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    if (
        user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN
        and user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY
    ):
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Admin-only endpoint. Your user role={}".format(
                    user_api_key_dict.user_role
                )
            },
        )

    if prisma_client is None:
        raise HTTPException(
            status_code=400,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    # Check if end user exists
    end_user = await prisma_client.db.litellm_endusertable.find_unique(
        where={"user_id": end_user_id}
    )

    if end_user is None:
        raise HTTPException(
            status_code=404,
            detail={"error": f"End user with ID '{end_user_id}' not found"},
        )

    # Build where conditions for spend logs
    where_conditions = {"end_user": end_user_id}

    if start_date is not None or end_date is not None:
        where_conditions["startTime"] = {}

        if start_date is not None:
            where_conditions["startTime"]["gte"] = datetime.strptime(
                start_date, "%Y-%m-%d"
            )

        if end_date is not None:
            end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
            end_datetime = end_datetime.replace(hour=23, minute=59, second=59)
            where_conditions["startTime"]["lte"] = end_datetime

    # Get total spend aggregation using Prisma
    total_aggregation = await prisma_client.db.litellm_spendlogs.group_by(
        by=["end_user"],
        where=where_conditions,
        sum={
            "spend": True,
            "total_tokens": True,
            "prompt_tokens": True,
            "completion_tokens": True,
        },
        count={"_all": True},
    )

    # Get per-model spend breakdown using Prisma
    model_aggregations = await prisma_client.db.litellm_spendlogs.group_by(
        by=["model"],
        where=where_conditions,
        sum={
            "spend": True,
            "total_tokens": True,
            "prompt_tokens": True,
            "completion_tokens": True,
        },
        count={"_all": True},
    )

    # Format total spend
    if total_aggregation and len(total_aggregation) > 0:
        total = total_aggregation[0]
        total_spend = float(total["_sum"]["spend"] or 0.0)
        total_requests = total["_count"]["_all"]
        total_tokens = total["_sum"]["total_tokens"] or 0
        total_prompt_tokens = total["_sum"]["prompt_tokens"] or 0
        total_completion_tokens = total["_sum"]["completion_tokens"] or 0
    else:
        # No spend logs found for this user in the date range
        total_spend = 0.0
        total_requests = 0
        total_tokens = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0

    # Format model breakdown
    spend_by_model = []
    for agg in model_aggregations:
        spend_by_model.append({
            "model": agg["model"] or "unknown",
            "total_spend": float(agg["_sum"]["spend"] or 0.0),
            "total_requests": agg["_count"]["_all"],
            "total_tokens": agg["_sum"]["total_tokens"] or 0,
            "total_prompt_tokens": agg["_sum"]["prompt_tokens"] or 0,
            "total_completion_tokens": agg["_sum"]["completion_tokens"] or 0,
        })

    # Sort by spend descending
    spend_by_model.sort(key=lambda x: x["total_spend"], reverse=True)

    return {
        "end_user_id": end_user_id,
        "alias": end_user.alias,
        "total_spend": total_spend,
        "total_requests": total_requests,
        "total_tokens": total_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "spend_by_model": spend_by_model,
        "date_range": {
            "start_date": start_date,
            "end_date": end_date,
        },
    }