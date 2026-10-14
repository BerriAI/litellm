"""
CUSTOMER MANAGEMENT

All /customer management endpoints

/customer/new
/customer/info
/customer/update
/customer/bulk_update
/customer/delete
"""

#### END-USER/CUSTOMER MANAGEMENT ####
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Final, NamedTuple, Protocol, TypeVar, overload

import fastapi
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, TypeAdapter

if TYPE_CHECKING:
    from prisma import Prisma
    from prisma.models import LiteLLM_BudgetTable as PrismaBudgetRow
    from prisma.models import LiteLLM_EndUserTable as PrismaEndUserRow

    from litellm.proxy.utils import PrismaClient

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.duration_parser import duration_in_seconds
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.user_api_key_cache import (
    end_user_cache_key,
    end_user_restricted_registry_cache_key,
)
from litellm.proxy.management_endpoints.common_daily_activity import get_daily_activity
from litellm.proxy.management_endpoints.common_utils import validate_budget_duration
from litellm.proxy.management_helpers.object_permission_utils import (  # noqa: F401  # legacy module exports
    ObjectPermissionUpsert,
    _set_object_permission,  # pyright: ignore[reportPrivateUsage,reportUnusedImport]  # backwards-compatible package export
    prepare_object_permission_upsert,
    set_object_permission,
)
from litellm.proxy.utils import handle_exception_on_proxy
from litellm.repositories.budget_repository import BudgetRepository
from litellm.repositories.chunked_in import find_many_in
from litellm.repositories.object_permission_repository import ObjectPermissionRepository
from litellm.repositories.table_repositories import EndUserRepository
from litellm.types.proxy.management_endpoints.common_daily_activity import (
    SpendAnalyticsPaginatedResponse,
)
from litellm.types.proxy.management_endpoints.customer_endpoints import (
    BlockUsersResponse,
    BulkUpdateCustomerResponse,
    CustomerResponse,
    CustomerUpdateResult,
    DeleteCustomersResponse,
    UnblockUsersResponse,
)


@dataclass(frozen=True, slots=True)
class TransactionClient:
    db: "Prisma"


_RowT_co: Final = TypeVar("_RowT_co", covariant=True)
_STR_OBJECT_DICT: Final = TypeAdapter(dict[str, object])
_CLEARABLE_LIST_FIELDS: Final = frozenset({"models"})
_CUSTOMER_BUDGET_LOCK_SQL: Final = 'SELECT "budget_id" FROM "LiteLLM_BudgetTable" WHERE "budget_id" = $1 FOR UPDATE'
_CUSTOMER_SHARED_BUDGET_SQL: Final = """
SELECT "budget_id" FROM "LiteLLM_BudgetTable" WHERE "budget_id" = $1 AND EXISTS (
    SELECT 1 FROM "LiteLLM_EndUserTable" WHERE "budget_id" = $1 AND "user_id" <> $2
    UNION ALL SELECT 1 FROM "LiteLLM_VerificationToken" WHERE "budget_id" = $1
        OR "metadata"->>'end_user_budget_id' = $1
    UNION ALL SELECT 1 FROM "LiteLLM_OrganizationTable" WHERE "budget_id" = $1
    UNION ALL SELECT 1 FROM "LiteLLM_ProjectTable" WHERE "budget_id" = $1
    UNION ALL SELECT 1 FROM "LiteLLM_TagTable" WHERE "budget_id" = $1
    UNION ALL SELECT 1 FROM "LiteLLM_ModelAccessGroupBudgetTable" WHERE "budget_id" = $1
    UNION ALL SELECT 1 FROM "LiteLLM_TeamMembership" WHERE "budget_id" = $1
    UNION ALL SELECT 1 FROM "LiteLLM_OrganizationMembership" WHERE "budget_id" = $1
    UNION ALL SELECT 1 FROM "LiteLLM_TeamTable" WHERE "metadata"->>'team_member_budget_id' = $1
)
"""
_CUSTOMER_UPDATE_LOCK_SQL: Final = 'SELECT "user_id" FROM "LiteLLM_EndUserTable" WHERE "user_id" = $1 FOR UPDATE'


def _should_update_field(field: str, value: object, sent_fields: AbstractSet[str]) -> bool:
    if value is None:
        return False
    if field in sent_fields and (isinstance(value, bool) or field in _CLEARABLE_LIST_FIELDS):
        return True
    if isinstance(value, (list, dict)) and not value:
        return False
    return value != 0


if TYPE_CHECKING:

    class _TableOps(Protocol[_RowT_co]):
        async def find_first(
            self,
            where: Mapping[str, object] | None = None,
            include: Mapping[str, bool] | None = None,
        ) -> _RowT_co | None: ...

        async def find_many(
            self,
            where: Mapping[str, object] | None = None,
            include: Mapping[str, bool] | None = None,
        ) -> Sequence[_RowT_co]: ...

        async def create(
            self,
            data: Mapping[str, object],
            include: Mapping[str, bool] | None = None,
        ) -> _RowT_co: ...

        async def update(
            self,
            where: Mapping[str, object],
            data: Mapping[str, object],
            include: Mapping[str, bool] | None = None,
        ) -> _RowT_co | None: ...

        async def upsert(
            self,
            where: Mapping[str, object],
            data: Mapping[str, Mapping[str, object]],
        ) -> _RowT_co: ...

        async def delete_many(self, where: Mapping[str, object]) -> int: ...


@overload
def _typed_table(repo: EndUserRepository) -> "_TableOps[PrismaEndUserRow]": ...
@overload
def _typed_table(repo: BudgetRepository) -> "_TableOps[PrismaBudgetRow]": ...
def _typed_table(repo: EndUserRepository | BudgetRepository) -> object:
    return repo.table


router: Final = APIRouter()


async def _evict_end_user_cache_keys(cache_keys: Sequence[str]) -> None:
    """
    Every endpoint that mutates an end-user row must call this, or a newly blocked or budgeted
    customer keeps being served unrestricted until the TTL expires: auth reads end users
    cache-first, and the cached restricted-id registry decides whether the row is read at all.
    """
    from litellm.proxy.common_utils.auth_cache_invalidation_pubsub import (
        evict_and_broadcast,
    )
    from litellm.proxy.proxy_server import user_api_key_cache

    await evict_and_broadcast(cache_keys=cache_keys, user_api_key_cache=user_api_key_cache)


def _end_user_cache_keys(user_ids: Sequence[str]) -> tuple[str, ...]:
    """The per-id entries plus the registry, which any restriction change can move ids in or out of."""
    return (*(end_user_cache_key(user_id) for user_id in user_ids), end_user_restricted_registry_cache_key())


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
        -H "Authorization: Bearer $LITELLM_MASTER_KEY"
        -d '{
        "user_ids": [<user_id>, ...]
        }'
        ```
    """
    from litellm.proxy.proxy_server import prisma_client

    try:
        records: Final = []
        if prisma_client is not None:
            for id in data.user_ids:
                record = await _typed_table(EndUserRepository(prisma_client)).upsert(
                    where={"user_id": id},
                    data={
                        "create": {"user_id": id, "blocked": True},
                        "update": {"blocked": True},
                    },
                )
                records.append(record)
            await _evict_end_user_cache_keys(_end_user_cache_keys(data.user_ids))
        else:
            raise HTTPException(
                status_code=500,
                detail={"error": "Postgres DB Not connected"},
            )

        return {"blocked_users": records}
    except Exception as e:
        verbose_proxy_logger.error("An error occurred - %s", e)
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
    -H "Authorization: Bearer $LITELLM_MASTER_KEY"
    -d '{
    "user_ids": [<user_id>, ...]
    }'
    ```
    """
    try:
        from enterprise.enterprise_hooks.blocked_user_list import (
            ENTERPRISE_BlockedUserList,
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
        not any(isinstance(x, ENTERPRISE_BlockedUserList) for x in litellm.callbacks)
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


def new_budget_request(data: NewCustomerRequest) -> BudgetNewRequest | None:
    """
    Return a new budget object if new budget params are passed.
    """
    budget_params: Final = BudgetNewRequest.model_fields.keys()
    budget_kv_pairs: Final = {}

    # Get the actual values from the data object using getattr
    for field_name in budget_params:
        if field_name == "budget_id":
            continue
        value = getattr(data, field_name, None)
        if value is not None:
            budget_kv_pairs[field_name] = value

    if budget_kv_pairs:
        budget_request: Final = BudgetNewRequest.model_validate(budget_kv_pairs)
        validate_budget_duration(budget_request.budget_duration)
        if budget_request.budget_reset_at is None and budget_request.budget_duration is not None:
            budget_request.budget_reset_at = datetime.utcnow() + timedelta(
                seconds=duration_in_seconds(duration=budget_request.budget_duration)
            )
        return budget_request
    return None


async def _prepare_customer_object_permission_update(
    data: UpdateCustomerRequest,
    existing_object_permission_id: str | None,
    prisma_client: TransactionClient,
) -> ObjectPermissionUpsert | None:
    if data.object_permission is None:
        return None
    return await prepare_object_permission_upsert(
        new_object_permission=_STR_OBJECT_DICT.validate_python(data.object_permission.model_dump()),
        existing_object_permission_id=existing_object_permission_id,
        prisma_client=prisma_client,
    )


async def _customer_budget_update_error(
    tx: TransactionClient,
    budget_id: str,
    user_id: str,
) -> ProxyException | None:
    budget_locked: Final = bool(await tx.db.query_raw(_CUSTOMER_BUDGET_LOCK_SQL, budget_id))
    if not budget_locked:
        return ProxyException(message="Budget does not exist", type="not_found", code=404, param="budget_id")
    shared: Final = budget_id == litellm.max_end_user_budget_id or bool(
        await tx.db.query_raw(_CUSTOMER_SHARED_BUDGET_SQL, budget_id, user_id)
    )
    if shared:
        return ProxyException(
            message=(
                "Cannot update max_budget for a shared budget. Update the budget directly or assign a different budget."
            ),
            type="bad_request",
            code=400,
            param="max_budget",
        )
    return None


async def _write_customer_budget_update(
    tx: TransactionClient,
    budget_data: Mapping[str, object],
    existing_budget: LiteLLM_BudgetTable | None,
    updated_by: str,
) -> str | None:
    if not budget_data:
        return None
    if existing_budget is not None:
        await _typed_table(BudgetRepository(tx)).update(
            where={"budget_id": existing_budget.budget_id},
            data=budget_data,
        )
        return None
    budget: Final = await _typed_table(BudgetRepository(tx)).create(
        data={**budget_data, "created_by": updated_by, "updated_by": updated_by},
        include={"end_users": True},
    )
    return budget.budget_id


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
    - models: Optional[list[str]] - Restrict this customer's access to the listed models.
    - metadata: Optional[dict] = Metadata for customer, store information for customer. Example metadata = {"data_training_opt_out": True}
    - budget_duration: Optional[str] - Budget is reset at the end of specified duration. If not set, budget is never reset. You can set duration as seconds ("30s"), minutes ("30m"), hours ("30h"), days ("30d").
    - tpm_limit: Optional[int] - [Not Implemented Yet] Specify tpm limit for a given customer (Tokens per minute)
    - rpm_limit: Optional[int] - [Not Implemented Yet] Specify rpm limit for a given customer (Requests per minute)
    - tpd_limit: Optional[int] - Specify tpd limit for a given customer (Tokens per day). Batch submissions are charged against it instead of tpm_limit/rpm_limit
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
        --header "Authorization: Bearer $LITELLM_MASTER_KEY" \
        --header 'Content-Type: application/json' \
        --data '{
            "user_id" : "ishaan-jaff-3",
            "allowed_region": "eu",
            "budget_id": "free_tier",
            "models": ["gpt-4o-mini"],
            "default_model": "azure/gpt-3.5-turbo-eu"
        }'

    # With object permissions
    curl -L -X POST 'http://localhost:4000/customer/new' \
        -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
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
                        "error": f"Default Model not on proxy. Configure via `/model/new` or config.yaml. Default_model={data.default_model}, proxy_model_names={set(llm_router.get_model_names())}"
                    },
                )

        new_end_user_obj: dict[str, object] = {}

        ## CREATE BUDGET ## if set
        _new_budget: Final = new_budget_request(data)
        if _new_budget is not None:
            try:
                budget_record: Final = await _typed_table(BudgetRepository(prisma_client)).create(
                    data={
                        **_new_budget.model_dump(exclude_unset=True),
                        "created_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
                        "updated_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
                    }
                )
            except Exception as e:
                raise HTTPException(status_code=422, detail={"error": str(e)})

            new_end_user_obj["budget_id"] = budget_record.budget_id
        elif data.budget_id is not None:
            new_end_user_obj["budget_id"] = data.budget_id

        _user_data: Final = _STR_OBJECT_DICT.validate_python(data.dict(exclude_none=True))

        for k, v in _user_data.items():
            if k not in BudgetNewRequest.model_fields:
                new_end_user_obj[k] = v

        ## Handle Object Permission - MCP Servers, Vector Stores etc.
        new_end_user_obj = _STR_OBJECT_DICT.validate_python(
            await set_object_permission(
                data_json=new_end_user_obj,
                prisma_client=prisma_client,
            )
        )

        # Ensure object_permission is not in the data being sent to create
        # It should have been converted to object_permission_id by _set_object_permission
        if "object_permission" in new_end_user_obj:
            verbose_proxy_logger.warning(
                "object_permission still in new_end_user_obj after _set_object_permission: %s",
                new_end_user_obj.get("object_permission"),
            )
            new_end_user_obj.pop("object_permission", None)

        ## WRITE TO DB ##
        end_user_record: Final = await _typed_table(EndUserRepository(prisma_client)).create(
            data=new_end_user_obj,
            include={"litellm_budget_table": True, "object_permission": True},
        )

        await _evict_end_user_cache_keys(_end_user_cache_keys((data.user_id,)))

        return _to_customer_response(end_user_record)
    except Exception as e:
        verbose_proxy_logger.exception(
            "litellm.proxy.management_endpoints.customer_endpoints.new_end_user(): Exception occured - %s", e
        )
        if "Unique constraint failed on the fields: (`user_id`)" in str(e):
            raise ProxyException(
                message=f"Customer already exists, passed user_id={data.user_id}. Please pass a new user_id.",
                type="bad_request",
                code=400,
                param="user_id",
            )
        raise handle_exception_on_proxy(e)


class _CustomerDailyActivityScope(NamedTuple):
    end_user_ids: tuple[str, ...] | None
    end_user_metadata: Mapping[str, dict[str, object]]


async def resolve_customer_daily_activity_scope(
    *,
    end_user_ids: tuple[str, ...] | None,
    prisma_client: "PrismaClient",
) -> _CustomerDailyActivityScope:
    end_user_table: Final = _typed_table(EndUserRepository(prisma_client))
    end_user_aliases: Final = (
        await find_many_in(end_user_table, "user_id", end_user_ids)
        if end_user_ids is not None
        else await end_user_table.find_many(where={})
    )
    metadata: Final = MappingProxyType({end_user.user_id: {"alias": end_user.alias} for end_user in end_user_aliases})
    return _CustomerDailyActivityScope(end_user_ids, metadata)


def customer_daily_activity_is_admin(user_api_key_dict: UserAPIKeyAuth) -> bool:
    return user_api_key_dict.user_role in (
        LitellmUserRoles.PROXY_ADMIN,
        LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
    )


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
        -H "Authorization: Bearer $LITELLM_MASTER_KEY"
    ```
    """
    try:
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )

        user_info: Final = await _typed_table(EndUserRepository(prisma_client)).find_first(
            where={"user_id": end_user_id},
            include={"litellm_budget_table": True, "object_permission": True},
        )

        if user_info is None:
            raise ProxyException(
                message=f"End User Id={end_user_id} does not exist in db",
                type="not_found",
                code=404,
                param="end_user_id",
            )

        return _to_customer_response(user_info)

    except Exception as e:
        verbose_proxy_logger.exception(
            "litellm.proxy.management_endpoints.customer_endpoints.end_user_info(): Exception occured - %s", e
        )
        raise handle_exception_on_proxy(e)


async def _update_single_customer(
    data: UpdateCustomerRequest,
    user_api_key_dict: UserAPIKeyAuth,
) -> CustomerResponse | ProxyException:
    from litellm.proxy.proxy_server import litellm_proxy_admin_name, prisma_client

    try:
        data_json: Final = _STR_OBJECT_DICT.validate_python(data.json())
        if prisma_client is None:
            raise Exception("Not connected to DB!")

        sent_fields: Final = data.fields_set()
        non_default_values: Final[dict[str, object]] = {
            k: v for k, v in data_json.items() if _should_update_field(k, v, sent_fields)
        }

        budget_table_data: Final = {
            field: value
            for field, value in non_default_values.items()
            if field != "budget_id" and field in LiteLLM_BudgetTable.model_fields
        }
        customer_table_data: Final = {
            field: value
            for field, value in non_default_values.items()
            if field == "budget_id"
            or (
                field in LiteLLM_EndUserTable.model_fields
                and field not in LiteLLM_BudgetTable.model_fields
                and field != "object_permission"
            )
        }
        async with prisma_client.tx() as transaction:
            tx: Final = TransactionClient(db=transaction)
            customer_locked: Final = bool(await transaction.query_raw(_CUSTOMER_UPDATE_LOCK_SQL, data.user_id))
            end_user_table_data: Final = (
                await _typed_table(EndUserRepository(tx)).find_first(
                    where={"user_id": data.user_id}, include={"litellm_budget_table": True}
                )
                if customer_locked
                else None
            )
            if end_user_table_data is None:
                raise ProxyException(
                    message=f"End User Id={data.user_id} does not exist in db",
                    type="not_found",
                    code=404,
                    param="user_id",
                )
            end_user_table_data_typed: Final = LiteLLM_EndUserTable.model_validate(end_user_table_data.model_dump())
            budget_error: Final = (
                await _customer_budget_update_error(tx, end_user_table_data_typed.budget_id, data.user_id)
                if "max_budget" in budget_table_data and end_user_table_data_typed.budget_id is not None
                else None
            )
            if budget_error is not None:
                return budget_error
            permission_upsert: Final = await _prepare_customer_object_permission_update(
                data=data,
                existing_object_permission_id=end_user_table_data_typed.object_permission_id,
                prisma_client=tx,
            )
            permission: Final = (
                await ObjectPermissionRepository(tx).table.upsert(
                    where={"object_permission_id": permission_upsert.object_permission_id},
                    data={"create": permission_upsert.record, "update": permission_upsert.update_record},
                )
                if permission_upsert is not None
                else None
            )
            budget_id: Final = await _write_customer_budget_update(
                tx=tx,
                budget_data=budget_table_data,
                existing_budget=end_user_table_data_typed.litellm_budget_table,
                updated_by=user_api_key_dict.user_id or litellm_proxy_admin_name,
            )
            update_data: Final = {
                **customer_table_data,
                "user_id": data.user_id,
                **({"object_permission_id": permission.object_permission_id} if permission is not None else {}),
                **({"budget_id": budget_id} if budget_id is not None else {}),
            }
            response: Final = await _typed_table(EndUserRepository(tx)).update(
                where={"user_id": data.user_id},
                data=update_data,
                include={"litellm_budget_table": True, "object_permission": True},
            )
            if response is None:
                raise ValueError(f"Failed updating customer data. User ID does not exist passed user_id={data.user_id}")
            customer_response: Final = _to_customer_response(response)
        await _evict_end_user_cache_keys(_end_user_cache_keys((data.user_id,)))

    except Exception as e:
        verbose_proxy_logger.exception("litellm.proxy.proxy_server.update_end_user(): Exception occured - %s", e)
        return handle_exception_on_proxy(e)
    else:
        return customer_response


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
    - budget_id: Optional[str] = None  # give either a budget_id or max_budget
    - allowed_model_region: Optional[AllowedModelRegion] = (
        None  # require all user requests to use models in this specific region
    )
    - default_model: Optional[str] = (
        None  # if no equivalent model in allowed region - default all requests to this model
    )
    - models: Optional[list[str]] = None  # omitted or null leaves the allowlist unchanged; an empty list clears it
    - object_permission: Optional[LiteLLM_ObjectPermissionBase] - Customer-specific object permissions to control access to resources.
        Supported fields:
        * mcp_servers: List[str] - List of allowed MCP server IDs
        * mcp_access_groups: List[str] - List of MCP access group names
        * mcp_toolsets: List[str] - List of allowed MCP toolset IDs
        * mcp_tool_permissions: Dict[str, List[str]] - Map of server ID to allowed tool names
        * vector_stores: List[str] - List of allowed vector store IDs
        * agents: List[str] - List of allowed agent IDs
        * agent_access_groups: List[str] - List of agent access group names
        Example: {"mcp_servers": ["server_1"], "vector_stores": ["vector_store_1"]}
        Omitted permission fields retain their existing values. Supplying null or {} leaves permissions unchanged.

    Example curl:
    ```
    curl --location 'http://0.0.0.0:4000/customer/update' \
    --header "Authorization: Bearer $LITELLM_MASTER_KEY" \
    --header 'Content-Type: application/json' \
    --data '{
        "user_id": "test-litellm-user-4",
        "budget_id": "paid_tier",
        "models": ["gpt-4o-mini"]
    }'

    # Updating object permissions
    curl -L -X POST 'http://localhost:4000/customer/update' \
    --header "Authorization: Bearer $LITELLM_MASTER_KEY" \
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

    result: Final = await _update_single_customer(data=data, user_api_key_dict=user_api_key_dict)
    if isinstance(result, ProxyException):
        raise result
    return result


async def _update_customer_result(
    data: UpdateCustomerRequest,
    user_api_key_dict: UserAPIKeyAuth,
) -> CustomerUpdateResult:
    result: Final = await _update_single_customer(data=data, user_api_key_dict=user_api_key_dict)
    if isinstance(result, ProxyException):
        return CustomerUpdateResult(user_id=data.user_id, success=False, error=str(result))
    return CustomerUpdateResult(user_id=data.user_id, success=True, updated_customer=result)


@router.post(
    "/customer/bulk_update",
    tags=["Customer Management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=BulkUpdateCustomerResponse,
)
@router.post(
    "/end_user/bulk_update",
    tags=["Customer Management"],
    include_in_schema=False,
    dependencies=[Depends(user_api_key_auth)],
    response_model=BulkUpdateCustomerResponse,
)
async def bulk_customer_update(
    data: BulkUpdateCustomerRequest,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> BulkUpdateCustomerResponse:
    """Update up to 500 customers independently using the same fields as `/customer/update`.

    Supply a `customers` array of customer updates. Each result contains the `user_id`,
    `success`, and either `updated_customer` or `error`, in request order. A failed update
    does not prevent the remaining customers from being updated. Object permission fields
    omitted from an update retain their existing values, and supplied permission lists
    replace that field. Null permission fields retain their stored values and explicit empty
    lists or objects clear that field. Each item is a locked transaction, so failed items
    roll back their permission, budget, and customer writes. Only proxy admins can update customers.

    `max_budget` can update a budget used only by that customer. Shared budgets and default
    budgets must be edited through the budget endpoints. Sending both `budget_id` and
    `max_budget`, even with a null value, fails request validation with HTTP 422 before
    any item is processed. A valid item rejected for sharing a budget does not prevent
    the other items from succeeding.

    Example request body:
    ```json
    {
      "customers": [
        {"user_id": "customer-1", "object_permission": {"mcp_toolsets": ["toolset-1"]}},
        {"user_id": "customer-2", "models": ["model-1"]}
      ]
    }
    ```
    """
    results: Final = tuple(
        [
            await _update_customer_result(data=customer, user_api_key_dict=user_api_key_dict)
            for customer in data.customers
        ]
    )
    successful_updates: Final = sum(result.success for result in results)
    return BulkUpdateCustomerResponse(
        results=results,
        total_requested=len(results),
        successful_updates=successful_updates,
        failed_updates=len(results) - successful_updates,
    )


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
        --header "Authorization: Bearer $LITELLM_MASTER_KEY" \
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
            existing_users: Final = await _typed_table(EndUserRepository(prisma_client)).find_many(
                where={"user_id": {"in": data.user_ids}}
            )
            existing_user_ids: Final = {user.user_id for user in existing_users}
            missing_user_ids: Final = [user_id for user_id in data.user_ids if user_id not in existing_user_ids]

            if missing_user_ids:
                raise ProxyException(
                    message="End User Id(s)={} do not exist in db".format(", ".join(missing_user_ids)),
                    type="not_found",
                    code=404,
                    param="user_ids",
                )

            # All users exist, proceed with deletion
            response: Final = await _typed_table(EndUserRepository(prisma_client)).delete_many(
                where={"user_id": {"in": data.user_ids}}
            )
            verbose_proxy_logger.debug("received response from updating prisma client. response=%s", response)

            await _evict_end_user_cache_keys(_end_user_cache_keys(data.user_ids))

            return DeleteCustomersResponse(
                deleted_customers=response,
                message="Successfully deleted customers with ids: " + str(data.user_ids),
            )
        else:
            raise ValueError(f"user_id is required, passed user_id = {data.user_ids}")

        # update based on remaining passed in values
    except Exception as e:
        verbose_proxy_logger.error("litellm.proxy.proxy_server.delete_end_user(): Exception occured - %s", e)
        raise handle_exception_on_proxy(e)


@router.get(
    "/customer/list",
    tags=["Customer Management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=list[CustomerResponse],
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
) -> list[CustomerResponse]:
    """
    [Admin-only] List all available customers

    Example curl:
    ```
    curl --location --request GET 'http://0.0.0.0:4000/customer/list' \
        --header "Authorization: Bearer $LITELLM_MASTER_KEY"
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
                detail={"error": f"Admin-only endpoint. Your user role={user_api_key_dict.user_role}"},
            )

        if prisma_client is None:
            raise HTTPException(
                status_code=400,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )

        response: Final = await _typed_table(EndUserRepository(prisma_client)).find_many(
            include={"litellm_budget_table": True, "object_permission": True}
        )

        return [_to_customer_response(item) for item in response]

    except Exception as e:
        verbose_proxy_logger.exception(
            "litellm.proxy.management_endpoints.customer_endpoints.list_end_user(): Exception occured - %s", e
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
    end_user_ids: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    page: int = 1,
    page_size: int = 10,
    exclude_end_user_ids: str | None = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get daily activity for specific organizations or all accessible organizations.
    """
    if not customer_daily_activity_is_admin(user_api_key_dict):
        raise HTTPException(
            status_code=401,
            detail={"error": f"Admin-only endpoint. Your user role={user_api_key_dict.user_role}"},
        )

    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    # Parse comma-separated ids
    end_user_ids_list: Final = tuple(end_user_ids.split(",")) if end_user_ids else None
    exclude_end_user_ids_list: list[str] | None = None
    if exclude_end_user_ids:
        exclude_end_user_ids_list = exclude_end_user_ids.split(",") if exclude_end_user_ids else None

    customer_scope: Final = await resolve_customer_daily_activity_scope(
        end_user_ids=end_user_ids_list,
        prisma_client=prisma_client,
    )

    return await get_daily_activity(
        prisma_client=prisma_client,
        table_name="litellm_dailyenduserspend",
        entity_id_field="end_user_id",
        entity_id=None if customer_scope.end_user_ids is None else list(customer_scope.end_user_ids),
        entity_metadata_field=customer_scope.end_user_metadata,
        exclude_entity_ids=exclude_end_user_ids_list,
        start_date=start_date,
        end_date=end_date,
        model=model,
        api_key=api_key,
        page=page,
        page_size=page_size,
    )
