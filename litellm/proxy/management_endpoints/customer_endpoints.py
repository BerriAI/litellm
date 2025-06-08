"""
CUSTOMER MANAGEMENT

All /customer management endpoints 

/customer/new   
/customer/info
/customer/update
/customer/delete
"""

#### END-USER/CUSTOMER MANAGEMENT ####
import traceback
from typing import List, Optional

import fastapi
from fastapi import APIRouter, Depends, HTTPException, Request, status

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

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

        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Internal Server Error({str(e)})"),
                type="internal_error",
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Internal Server Error, " + str(e),
            type="internal_error",
            param=getattr(e, "param", "None"),
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


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
        raise HTTPException(
            status_code=400,
            detail={"error": "End User Id={} does not exist in db".format(end_user_id)},
        )
    return user_info.model_dump(exclude_none=True)


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

    from litellm.proxy.proxy_server import prisma_client

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
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "End User Id={} does not exist in db".format(data.user_id)
                },
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
            if k in LiteLLM_BudgetTable.model_fields.keys():
                budget_table_data[k] = v

            if k in LiteLLM_EndUserTable.model_fields.keys():
                update_end_user_table_data[k] = v

        ## Check if budget id is set ##
        if budget_table_data:
            if end_user_budget_table is None:
                ## Create new budget ##
                budget_table_data_record = (
                    await prisma_client.db.litellm_budgettable.create(
                        data=budget_table_data, include={"litellm_endusertable": True}
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
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Internal Server Error({str(e)})"),
                type="internal_error",
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Internal Server Error, " + str(e),
            type="internal_error",
            param=getattr(e, "param", "None"),
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    pass


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
            response = await prisma_client.db.litellm_endusertable.delete_many(
                where={"user_id": {"in": data.user_ids}}
            )
            if response is None:
                raise ValueError(
                    f"Failed deleting customer data. User ID does not exist passed user_id={data.user_ids}"
                )
            if response != len(data.user_ids):
                raise ValueError(
                    f"Failed deleting all customer data. User ID does not exist passed user_id={data.user_ids}. Deleted {response} customers, passed {len(data.user_ids)} customers"
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
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Internal Server Error({str(e)})"),
                type="internal_error",
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Internal Server Error, " + str(e),
            type="internal_error",
            param=getattr(e, "param", "None"),
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    pass


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
