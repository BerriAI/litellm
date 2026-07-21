# What is this?
## Helper utils for the management endpoints (keys/users/teams)
from datetime import datetime
from functools import wraps
from typing import Any, Callable, List, Optional, Tuple

from fastapi import HTTPException, Request
from pydantic import BaseModel

import litellm
from litellm._logging import verbose_logger
from litellm._uuid import uuid
from litellm.integrations.otel.model.config import is_otel_v2_enabled
from litellm.proxy._types import (  # key request types; user request types; team request types; customer request types
    BudgetNewRequest,
    DeleteCustomerRequest,
    DeleteTeamRequest,
    DeleteUserRequest,
    KeyRequest,
    LiteLLM_BudgetTable,
    LiteLLM_TeamMembership,
    LiteLLM_UserTable,
    ManagementEndpointLoggingPayload,
    Member,
    SSOUserDefinedValues,
    UpdateCustomerRequest,
    UpdateKeyRequest,
    UpdateTeamRequest,
    UpdateUserRequest,
    UserAPIKeyAuth,
    VirtualKeyEvent,
)
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.proxy.common_utils.timezone_utils import get_budget_reset_time
from litellm.proxy.utils import PrismaClient
from litellm.repositories.budget_repository import BudgetRepository
from litellm.repositories.table_repositories import TeamMembershipRepository
from litellm.repositories.user_repository import UserRepository


def get_new_internal_user_defaults(user_id: str, user_email: Optional[str] = None) -> dict:
    user_info = litellm.default_internal_user_params or {}

    returned_dict: SSOUserDefinedValues = {
        "models": user_info.get("models") or [],
        "max_budget": user_info.get("max_budget", litellm.max_internal_user_budget),
        "budget_duration": user_info.get("budget_duration", litellm.internal_user_budget_duration),
        "user_email": user_email or user_info.get("user_email", None),
        "user_id": user_id,
        "user_role": "internal_user",
    }

    non_null_dict = {}
    for k, v in returned_dict.items():
        if v is not None:
            non_null_dict[k] = v
    return non_null_dict


async def handle_budget_for_entity(
    data,
    existing_budget_id: Optional[str],
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: PrismaClient,
    litellm_proxy_admin_name: str,
) -> Optional[str]:
    """
    Common helper to handle budget creation/updates for entities (organizations, tags, etc).

    This function:
    1. Creates a new budget if budget_id is None but budget fields are provided
    2. Updates an existing budget if budget fields are provided and budget_id exists
    3. Returns the budget_id to use (existing or newly created)

    Args:
        data: The request object (e.g., TagNewRequest, NewOrganizationRequest, etc.) containing budget fields
        existing_budget_id: The existing budget_id if updating an entity, None if creating new
        user_api_key_dict: User authentication info
        prisma_client: Database client
        litellm_proxy_admin_name: Admin name for audit trail

    Returns:
        Optional[str]: The budget_id to use, or None if no budget was created/updated
    """
    from litellm.proxy.management_endpoints.budget_management_endpoints import (
        update_budget,
    )

    # Get all budget field names
    budget_params = LiteLLM_BudgetTable.model_fields.keys()

    # Extract budget fields from data
    _json_data = data.model_dump(exclude_none=True) if hasattr(data, "model_dump") else data
    _budget_data = {k: v for k, v in _json_data.items() if k in budget_params}

    # Check if budget_id is explicitly provided in the data
    data_budget_id = getattr(data, "budget_id", None)

    # Case 1: Creating new entity - no existing budget_id
    if existing_budget_id is None:
        if data_budget_id is not None:
            # Use the provided budget_id
            return data_budget_id
        elif _budget_data:
            # Create a new budget with the provided fields
            budget_row = LiteLLM_BudgetTable(**_budget_data)
            new_budget_data = prisma_client.jsonify_object(budget_row.model_dump(exclude_none=True))

            _budget = await BudgetRepository(prisma_client).table.create(
                data={
                    **new_budget_data,  # type: ignore
                    "created_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
                    "updated_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
                }
            )  # type: ignore

            return _budget.budget_id
        else:
            # No budget fields provided, no budget to create
            return None

    # Case 2: Updating existing entity - has existing budget_id
    else:
        # If budget fields are provided, update the existing budget
        if _budget_data:
            await update_budget(
                budget_obj=BudgetNewRequest(budget_id=existing_budget_id, **_budget_data),
                user_api_key_dict=user_api_key_dict,
            )

        # If a different budget_id is explicitly provided, use that instead
        if data_budget_id is not None and data_budget_id != existing_budget_id:
            return data_budget_id

        # Otherwise, keep using the existing budget_id
        return existing_budget_id


# Fields on LiteLLM_BudgetTable that represent the budget's *configuration*
# (i.e. the values an admin sets). We copy these when cloning a team's
# default member-budget into an individual member-budget so that the new
# row starts with the same limits as the default.
_CLONABLE_BUDGET_FIELDS: Tuple[str, ...] = (
    "max_budget",
    "soft_budget",
    "max_parallel_requests",
    "tpm_limit",
    "rpm_limit",
    "model_max_budget",
    "budget_duration",
    "allowed_models",
)


async def _clone_team_default_budget_for_member(
    prisma_client: PrismaClient,
    default_team_budget_id: str,
    user_api_key_dict: UserAPIKeyAuth,
    litellm_proxy_admin_name: str,
    budget_duration_override: Optional[str] = None,
) -> Optional[str]:
    """
    Create a new budget row that copies the values from the team's default
    member budget. Returns the new budget_id, or None if the default budget
    no longer exists in the DB.

    Used when adding a new team member without an explicit per-member budget,
    so the member starts with the team default's values but gets their own
    private budget row (which can be edited independently).

    ``budget_duration_override`` replaces the default's reset window for this
    member while keeping the default's other limits, so an admin can set a
    member's reset cadence without discarding the team default's max_budget.
    """
    default_budget = await BudgetRepository(prisma_client).table.find_unique(
        where={"budget_id": default_team_budget_id}
    )
    if default_budget is None:
        return None

    default_budget_dict = default_budget.model_dump()
    cloned_data: dict = {
        "created_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
        "updated_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
    }
    for field in _CLONABLE_BUDGET_FIELDS:
        value = default_budget_dict.get(field)
        if value is None:
            continue
        # Skip empty list defaults (e.g. allowed_models = []) so the cloned
        # row matches the "no value set" shape rather than carrying a default.
        if isinstance(value, list) and len(value) == 0:
            continue
        cloned_data[field] = value

    if budget_duration_override is not None:
        cloned_data["budget_duration"] = budget_duration_override

    # Start the member's budget window at clone time, not the pool's reset
    # timestamp — otherwise a member joining mid-cycle inherits a stale reset.
    if cloned_data.get("budget_duration"):
        cloned_data["budget_reset_at"] = get_budget_reset_time(cloned_data["budget_duration"])

    new_budget = await BudgetRepository(prisma_client).table.create(data=cloned_data)
    return new_budget.budget_id


async def _resolve_member_budget_id(
    prisma_client: PrismaClient,
    user_api_key_dict: UserAPIKeyAuth,
    litellm_proxy_admin_name: str,
    max_budget_in_team: Optional[float],
    allowed_models: Optional[list[str]],
    budget_duration: Optional[str],
    default_team_budget_id: Optional[str],
) -> Optional[str]:
    """
    Resolve the budget a new team member should be linked to.

    Explicit per-member limits create a fresh budget. Otherwise the team's
    default member budget is cloned (with ``budget_duration`` overriding its
    reset window while keeping its other limits). A lone ``budget_duration``
    with no team default creates a window-only budget. With nothing set the
    member gets no budget.
    """
    has_explicit_limit = max_budget_in_team is not None or allowed_models is not None

    if not has_explicit_limit and default_team_budget_id is not None:
        return await _clone_team_default_budget_for_member(
            prisma_client=prisma_client,
            default_team_budget_id=default_team_budget_id,
            user_api_key_dict=user_api_key_dict,
            litellm_proxy_admin_name=litellm_proxy_admin_name,
            budget_duration_override=budget_duration,
        )

    if not has_explicit_limit and budget_duration is None:
        return None

    budget_data: dict = {
        "created_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
        "updated_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
    }
    if max_budget_in_team is not None:
        budget_data["max_budget"] = max_budget_in_team
    if allowed_models is not None:
        budget_data["allowed_models"] = allowed_models
    if budget_duration is not None:
        budget_data["budget_duration"] = budget_duration
        budget_data["budget_reset_at"] = get_budget_reset_time(budget_duration=budget_duration)
    response = await BudgetRepository(prisma_client).table.create(data=budget_data)
    return response.budget_id


async def _append_team_id_to_user(prisma_client: PrismaClient, user_id: str, team_id: str) -> LiteLLM_UserTable | None:
    """Append team_id to an existing user's teams array, only if not already present.

    The filtered update makes the append a no-op once the team is present, so
    repeated or concurrent adds of the same team cannot accumulate duplicate
    team ids in user.teams (which would also break auth logic that keys off the
    number of teams a user belongs to). Teams added concurrently for a different
    team id are unaffected, since each update filters on its own team id.
    """
    await UserRepository(prisma_client).table.update_many(
        where={"user_id": user_id, "NOT": {"teams": {"has": team_id}}},
        data={"teams": {"push": [team_id]}},
    )
    updated_user = await UserRepository(prisma_client).table.find_unique(where={"user_id": user_id})
    if updated_user is None:
        return None
    return LiteLLM_UserTable(**updated_user.model_dump())


async def add_new_member(
    new_member: Member,
    max_budget_in_team: Optional[float],
    prisma_client: PrismaClient,
    team_id: str,
    user_api_key_dict: UserAPIKeyAuth,
    litellm_proxy_admin_name: str,
    default_team_budget_id: Optional[str] = None,
    allowed_models: Optional[List[str]] = None,
    budget_duration: Optional[str] = None,
) -> Tuple[LiteLLM_UserTable, Optional[LiteLLM_TeamMembership]]:
    """
    Add a new member to a team

    - add team id to user table
    - add team member w/ budget to team member table

    Returns created/existing user + team membership w/ budget id
    """
    returned_user: Optional[LiteLLM_UserTable] = None
    returned_team_membership: Optional[LiteLLM_TeamMembership] = None
    ## ADD TEAM ID, to USER TABLE IF NEW ##
    if new_member.user_id is not None:
        existing_user = await UserRepository(prisma_client).table.find_unique(where={"user_id": new_member.user_id})
        if existing_user is None:
            new_user_defaults = get_new_internal_user_defaults(user_id=new_member.user_id)
            _created_user = await UserRepository(prisma_client).table.create(
                data={"teams": [team_id], **new_user_defaults},
            )
            if _created_user is not None:
                returned_user = LiteLLM_UserTable(**_created_user.model_dump())
        else:
            returned_user = await _append_team_id_to_user(
                prisma_client=prisma_client, user_id=new_member.user_id, team_id=team_id
            )
    elif new_member.user_email is not None:
        new_user_defaults = get_new_internal_user_defaults(user_id=str(uuid.uuid4()), user_email=new_member.user_email)
        ## user email is not unique acc. to prisma schema -> future improvement
        ### for now: check if it exists in db, if not - insert it
        existing_user_row: Optional[list] = await prisma_client.get_data(
            key_val={"user_email": new_member.user_email},
            table_name="user",
            query_type="find_all",
        )
        if existing_user_row is None or (isinstance(existing_user_row, list) and len(existing_user_row) == 0):
            new_user_defaults["teams"] = [team_id]
            _returned_user = await prisma_client.insert_data(data=new_user_defaults, table_name="user")  # type: ignore

            if _returned_user is not None:
                returned_user = LiteLLM_UserTable(**_returned_user.model_dump())
        elif len(existing_user_row) == 1:
            user_info = existing_user_row[0]
            returned_user = await _append_team_id_to_user(
                prisma_client=prisma_client, user_id=user_info.user_id, team_id=team_id
            )
        elif len(existing_user_row) > 1:
            raise HTTPException(
                status_code=400,
                detail={"error": "Multiple users with this email found in db. Please use 'user_id' instead."},
            )

    _budget_id = await _resolve_member_budget_id(
        prisma_client=prisma_client,
        user_api_key_dict=user_api_key_dict,
        litellm_proxy_admin_name=litellm_proxy_admin_name,
        max_budget_in_team=max_budget_in_team,
        allowed_models=allowed_models,
        budget_duration=budget_duration,
        default_team_budget_id=default_team_budget_id,
    )

    if _budget_id and returned_user is not None and returned_user.user_id is not None:
        _returned_team_membership = await TeamMembershipRepository(prisma_client).table.create(
            data={
                "team_id": team_id,
                "user_id": returned_user.user_id,
                "budget_id": _budget_id,
            },
            include={"litellm_budget_table": True},
        )

        returned_team_membership = LiteLLM_TeamMembership(**_returned_team_membership.model_dump())

    if returned_user is None:
        raise Exception("Unable to update user table with membership information!")

    return returned_user, returned_team_membership


def _delete_user_id_from_cache(kwargs):
    from litellm.proxy.proxy_server import user_api_key_cache

    if kwargs.get("data") is not None:
        update_user_request = kwargs.get("data")
        if isinstance(update_user_request, UpdateUserRequest):
            user_api_key_cache.delete_cache(key=update_user_request.user_id)

        # delete user request
        if isinstance(update_user_request, DeleteUserRequest):
            for user_id in update_user_request.user_ids:
                user_api_key_cache.delete_cache(key=user_id)
    pass


def _delete_api_key_from_cache(kwargs):
    from litellm.proxy.proxy_server import user_api_key_cache

    if kwargs.get("data") is not None:
        update_request = kwargs.get("data")
        if isinstance(update_request, UpdateKeyRequest):
            user_api_key_cache.delete_cache(key=update_request.key)

        # delete key request
        if isinstance(update_request, KeyRequest) and update_request.keys:
            for key in update_request.keys:
                user_api_key_cache.delete_cache(key=key)
    pass


def _delete_team_id_from_cache(kwargs):
    from litellm.proxy.proxy_server import user_api_key_cache

    if kwargs.get("data") is not None:
        update_request = kwargs.get("data")
        if isinstance(update_request, UpdateTeamRequest):
            user_api_key_cache.delete_cache(key=update_request.team_id)

        # delete team request
        if isinstance(update_request, DeleteTeamRequest):
            for team_id in update_request.team_ids:
                user_api_key_cache.delete_cache(key=team_id)
    pass


def _delete_customer_id_from_cache(kwargs):
    from litellm.proxy.proxy_server import user_api_key_cache

    if kwargs.get("data") is not None:
        update_request = kwargs.get("data")
        if isinstance(update_request, UpdateCustomerRequest):
            user_api_key_cache.delete_cache(key=update_request.user_id)

        # delete customer request
        if isinstance(update_request, DeleteCustomerRequest):
            for user_id in update_request.user_ids:
                user_api_key_cache.delete_cache(key=user_id)
    pass


async def send_management_endpoint_alert(
    request_kwargs: dict,
    user_api_key_dict: UserAPIKeyAuth,
    function_name: str,
):
    """
    Sends a slack alert when:
    - A virtual key is created, updated, or deleted
    - An internal user is created, updated, or deleted
    - A team is created, updated, or deleted
    """
    from litellm.proxy.proxy_server import proxy_logging_obj
    from litellm.types.integrations.slack_alerting import AlertType

    management_function_to_event_name = {
        "generate_key_fn": AlertType.new_virtual_key_created,
        "update_key_fn": AlertType.virtual_key_updated,
        "delete_key_fn": AlertType.virtual_key_deleted,
        # Team events
        "new_team": AlertType.new_team_created,
        "update_team": AlertType.team_updated,
        "delete_team": AlertType.team_deleted,
        # Internal User events
        "new_user": AlertType.new_internal_user_created,
        "user_update": AlertType.internal_user_updated,
        "delete_user": AlertType.internal_user_deleted,
    }

    # Check if alerting is enabled
    if proxy_logging_obj is not None and proxy_logging_obj.slack_alerting_instance is not None:
        # Virtual Key Events
        if function_name in management_function_to_event_name:
            _event_name: AlertType = management_function_to_event_name[function_name]

            key_event = VirtualKeyEvent(
                created_by_user_id=user_api_key_dict.user_id or "Unknown",
                created_by_user_role=user_api_key_dict.user_role or "Unknown",
                created_by_key_alias=user_api_key_dict.key_alias,
                request_kwargs=request_kwargs,
            )

            # replace all "_" with " " and capitalize
            event_name = _event_name.replace("_", " ").title()
            await proxy_logging_obj.slack_alerting_instance.send_virtual_key_event_slack(
                key_event=key_event,
                event_name=event_name,
                alert_type=_event_name,
            )


def _redacted_env_var(entry: Any) -> dict:
    get = entry.get if isinstance(entry, dict) else lambda k: getattr(entry, k, None)
    return {
        "name": get("name"),
        "scope": get("scope"),
        "description": get("description"),
        "value": "",
    }


def _redact_record_env_vars(record: Any) -> Any:
    """Return ``record`` with its ``env_vars[].value`` blanked.

    Copies rather than mutating, because the record aliases the live response
    object that is also returned to the caller. Records without an ``env_vars``
    list are returned unchanged.
    """
    env_vars = record.get("env_vars") if isinstance(record, dict) else getattr(record, "env_vars", None)
    if not isinstance(env_vars, list):
        return record
    redacted = [_redacted_env_var(entry) for entry in env_vars]
    if isinstance(record, dict):
        return {**record, "env_vars": redacted}
    if isinstance(record, BaseModel):
        return record.model_copy(update={"env_vars": redacted})
    return record


def _redact_env_var_values(response: dict) -> None:
    """Blank ``env_vars[].value`` in a management response before telemetry.

    MCP endpoints return decrypted ``scope="global"`` env var values so the admin
    UI can pre-fill the edit form; those values are upstream credentials and must
    not be serialized verbatim into OTEL spans, where an observability user could
    read them. The values surface both at the top level (single-server
    create/update) and nested under ``items`` (the submissions queue), so both are
    scrubbed. Names, scopes, and descriptions are kept so traces stay useful.
    """
    if isinstance(response.get("env_vars"), list):
        response["env_vars"] = [_redacted_env_var(entry) for entry in response["env_vars"]]

    items = response.get("items")
    if isinstance(items, list):
        response["items"] = [_redact_record_env_vars(item) for item in items]


async def _emit_management_endpoint_otel_span(
    func: Callable,
    kwargs: dict,
    parent_otel_span: Any,
    start_time: datetime,
    end_time: datetime,
    result: Any = None,
    exception: Optional[Exception] = None,
) -> None:
    """Stamp + end the parent OTEL SERVER span for a management endpoint.

    Routes the request/response (or exception) through the OTEL success/failure
    hook. Falls back to ``func.__name__`` for the route when the handler has no
    ``http_request`` param — endpoints like ``/key/generate`` never receive one,
    and gating the hook on it leaked their SERVER span (created in auth, never
    ended → never exported). Always emitting keeps both success and failure
    paths consistent.
    """
    from litellm.proxy.proxy_server import open_telemetry_logger

    if open_telemetry_logger is None:
        return

    # Under V2 OTel, management endpoints are ordinary FastAPI routes already
    # spanned by the mounted instrumentor — there is no management hook to fire, so
    # skip the payload build entirely. The legacy logger still needs the hook.
    if is_otel_v2_enabled():
        return

    http_request: Optional[Request] = kwargs.get("http_request")
    if http_request is not None:
        # Inline import — auth_utils participates in a proxy import cycle.
        from litellm.proxy.auth.auth_utils import (  # noqa: PLC0415
            get_request_route,
        )

        route = get_request_route(http_request)
        request_body: dict = await _read_request_body(request=http_request)
    else:
        route = func.__name__
        request_body = {}

    _CREDENTIAL_FIELDS = frozenset(
        {
            "key",
            "token",
            "api_key",
            "secret",
            "password",
            "access_token",
            "refresh_token",
            "private_key",
            "service_account_key",
        }
    )

    _response: Optional[dict] = None
    if exception is None and result is not None:
        try:
            raw = dict(result)
            _response = {k: v for k, v in raw.items() if k not in _CREDENTIAL_FIELDS}
            _redact_env_var_values(_response)
        except Exception:
            _response = None

    logging_payload = ManagementEndpointLoggingPayload(
        route=route,
        request_data=request_body,
        response=_response,
        start_time=start_time,
        end_time=end_time,
        exception=exception,
    )

    if exception is None:
        await open_telemetry_logger.async_management_endpoint_success_hook(
            logging_payload=logging_payload,
            parent_otel_span=parent_otel_span,
        )
    else:
        await open_telemetry_logger.async_management_endpoint_failure_hook(
            logging_payload=logging_payload,
            parent_otel_span=parent_otel_span,
        )


def management_endpoint_wrapper(func):
    """
    This wrapper does the following:

    1. Log I/O, Exceptions to OTEL
    2. Create an Audit log for success calls
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = datetime.now()
        try:
            result = await func(*args, **kwargs)
            end_time = datetime.now()
            try:
                user_api_key_dict: UserAPIKeyAuth = kwargs.get("user_api_key_dict") or UserAPIKeyAuth()

                await send_management_endpoint_alert(
                    request_kwargs=kwargs,
                    user_api_key_dict=user_api_key_dict,
                    function_name=func.__name__,
                )
                parent_otel_span = getattr(user_api_key_dict, "parent_otel_span", None)
                if parent_otel_span is not None:
                    await _emit_management_endpoint_otel_span(
                        func=func,
                        kwargs=kwargs,
                        parent_otel_span=parent_otel_span,
                        start_time=start_time,
                        end_time=end_time,
                        result=result,
                    )

                # Delete updated/deleted info from cache
                _delete_api_key_from_cache(kwargs=kwargs)
                _delete_user_id_from_cache(kwargs=kwargs)
                _delete_team_id_from_cache(kwargs=kwargs)
                _delete_customer_id_from_cache(kwargs=kwargs)
            except Exception as e:
                # Non-Blocking Exception
                verbose_logger.debug("Error in management endpoint wrapper: %s", str(e))
                pass

            return result
        except Exception as e:
            end_time = datetime.now()

            user_api_key_dict: UserAPIKeyAuth = kwargs.get("user_api_key_dict") or UserAPIKeyAuth()
            parent_otel_span = getattr(user_api_key_dict, "parent_otel_span", None)
            if parent_otel_span is not None:
                try:
                    await _emit_management_endpoint_otel_span(
                        func=func,
                        kwargs=kwargs,
                        parent_otel_span=parent_otel_span,
                        start_time=start_time,
                        end_time=end_time,
                        exception=e,
                    )
                except Exception as otel_exc:
                    # Non-Blocking Exception - never let OTEL failures swallow
                    # the original management-endpoint exception.
                    verbose_logger.debug(
                        "Error emitting OTEL span in management endpoint wrapper failure path: %s",
                        str(otel_exc),
                    )

            raise e

    return wrapper
