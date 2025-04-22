# What is this?
## Helper utils for the management endpoints (keys/users/teams)
import uuid
from datetime import datetime
from functools import wraps
from typing import Optional, Tuple

from fastapi import HTTPException, Request

import litellm
from litellm._logging import verbose_logger
from litellm.proxy._types import (  # key request types; user request types; team request types; customer request types
    DeleteCustomerRequest,
    DeleteTeamRequest,
    DeleteUserRequest,
    KeyRequest,
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
from litellm.proxy.utils import PrismaClient


def get_new_internal_user_defaults(
    user_id: str, user_email: Optional[str] = None
) -> dict:
    user_info = litellm.default_internal_user_params or {}

    returned_dict: SSOUserDefinedValues = {
        "models": user_info.get("models", None),
        "max_budget": user_info.get("max_budget", litellm.max_internal_user_budget),
        "budget_duration": user_info.get(
            "budget_duration", litellm.internal_user_budget_duration
        ),
        "user_email": user_email or user_info.get("user_email", None),
        "user_id": user_id,
        "user_role": "internal_user",
    }

    non_null_dict = {}
    for k, v in returned_dict.items():
        if v is not None:
            non_null_dict[k] = v
    return non_null_dict


async def add_new_member(
    new_member: Member,
    max_budget_in_team: Optional[float],
    prisma_client: PrismaClient,
    team_id: str,
    user_api_key_dict: UserAPIKeyAuth,
    litellm_proxy_admin_name: str,
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
        new_user_defaults = get_new_internal_user_defaults(user_id=new_member.user_id)
        _returned_user = await prisma_client.db.litellm_usertable.upsert(
            where={"user_id": new_member.user_id},
            data={
                "update": {"teams": {"push": [team_id]}},
                "create": {"teams": [team_id], **new_user_defaults},  # type: ignore
            },
        )
        if _returned_user is not None:
            returned_user = LiteLLM_UserTable(**_returned_user.model_dump())
    elif new_member.user_email is not None:
        new_user_defaults = get_new_internal_user_defaults(
            user_id=str(uuid.uuid4()), user_email=new_member.user_email
        )
        ## user email is not unique acc. to prisma schema -> future improvement
        ### for now: check if it exists in db, if not - insert it
        existing_user_row: Optional[list] = await prisma_client.get_data(
            key_val={"user_email": new_member.user_email},
            table_name="user",
            query_type="find_all",
        )
        if existing_user_row is None or (
            isinstance(existing_user_row, list) and len(existing_user_row) == 0
        ):
            new_user_defaults["teams"] = [team_id]
            _returned_user = await prisma_client.insert_data(data=new_user_defaults, table_name="user")  # type: ignore

            if _returned_user is not None:
                returned_user = LiteLLM_UserTable(**_returned_user.model_dump())
        elif len(existing_user_row) == 1:
            user_info = existing_user_row[0]
            _returned_user = await prisma_client.db.litellm_usertable.update(
                where={"user_id": user_info.user_id},  # type: ignore
                data={"teams": {"push": [team_id]}},
            )
            if _returned_user is not None:
                returned_user = LiteLLM_UserTable(**_returned_user.model_dump())
        elif len(existing_user_row) > 1:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Multiple users with this email found in db. Please use 'user_id' instead."
                },
            )

    # Check if trying to set a budget for team member
    if (
        max_budget_in_team is not None
        and returned_user is not None
        and returned_user.user_id is not None
    ):
        # create a new budget item for this member
        response = await prisma_client.db.litellm_budgettable.create(
            data={
                "max_budget": max_budget_in_team,
                "created_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
                "updated_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
            }
        )

        _budget_id = response.budget_id
        _returned_team_membership = (
            await prisma_client.db.litellm_teammembership.create(
                data={
                    "team_id": team_id,
                    "user_id": returned_user.user_id,
                    "budget_id": _budget_id,
                },
                include={"litellm_budget_table": True},
            )
        )

        returned_team_membership = LiteLLM_TeamMembership(
            **_returned_team_membership.model_dump()
        )

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
    from litellm.proxy.proxy_server import premium_user, proxy_logging_obj
    from litellm.types.integrations.slack_alerting import AlertType

    if premium_user is not True:
        return

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
    if (
        proxy_logging_obj is not None
        and proxy_logging_obj.slack_alerting_instance is not None
    ):
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


def management_endpoint_wrapper(func):
    """
    This wrapper does the following:

    1. Log I/O, Exceptions to OTEL
    2. Create an Audit log for success calls
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = datetime.now()
        _http_request: Optional[Request] = None
        try:
            result = await func(*args, **kwargs)
            end_time = datetime.now()
            try:
                if kwargs is None:
                    kwargs = {}
                user_api_key_dict: UserAPIKeyAuth = (
                    kwargs.get("user_api_key_dict") or UserAPIKeyAuth()
                )

                await send_management_endpoint_alert(
                    request_kwargs=kwargs,
                    user_api_key_dict=user_api_key_dict,
                    function_name=func.__name__,
                )
                _http_request = kwargs.get("http_request", None)
                parent_otel_span = getattr(user_api_key_dict, "parent_otel_span", None)
                if parent_otel_span is not None:
                    from litellm.proxy.proxy_server import open_telemetry_logger

                    if open_telemetry_logger is not None:
                        if _http_request:
                            _route = _http_request.url.path
                            _request_body: dict = await _read_request_body(
                                request=_http_request
                            )
                            _response = dict(result) if result is not None else None

                            logging_payload = ManagementEndpointLoggingPayload(
                                route=_route,
                                request_data=_request_body,
                                response=_response,
                                start_time=start_time,
                                end_time=end_time,
                            )

                            await open_telemetry_logger.async_management_endpoint_success_hook(  # type: ignore
                                logging_payload=logging_payload,
                                parent_otel_span=parent_otel_span,
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

            if kwargs is None:
                kwargs = {}
            user_api_key_dict: UserAPIKeyAuth = (
                kwargs.get("user_api_key_dict") or UserAPIKeyAuth()
            )
            parent_otel_span = getattr(user_api_key_dict, "parent_otel_span", None)
            if parent_otel_span is not None:
                from litellm.proxy.proxy_server import open_telemetry_logger

                if open_telemetry_logger is not None:
                    _http_request = kwargs.get("http_request")
                    if _http_request:
                        _route = _http_request.url.path
                        _request_body: dict = await _read_request_body(
                            request=_http_request
                        )
                        logging_payload = ManagementEndpointLoggingPayload(
                            route=_route,
                            request_data=_request_body,
                            response=None,
                            start_time=start_time,
                            end_time=end_time,
                            exception=e,
                        )

                        await open_telemetry_logger.async_management_endpoint_failure_hook(  # type: ignore
                            logging_payload=logging_payload,
                            parent_otel_span=parent_otel_span,
                        )

            raise e

    return wrapper
