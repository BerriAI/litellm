# What is this?
## Helper utils for the management endpoints (keys/users/teams)
import uuid
from datetime import datetime
from functools import wraps
from typing import Optional

from fastapi import Request

import litellm
from litellm._logging import verbose_logger
from litellm.proxy._types import (  # key request types; user request types; team request types; customer request types
    DeleteCustomerRequest,
    DeleteTeamRequest,
    DeleteUserRequest,
    KeyRequest,
    LiteLLM_TeamTable,
    ManagementEndpointLoggingPayload,
    Member,
    SSOUserDefinedValues,
    UpdateCustomerRequest,
    UpdateKeyRequest,
    UpdateTeamRequest,
    UpdateUserRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.proxy.utils import PrismaClient


def get_new_internal_user_defaults(
    user_id: str, user_email: Optional[str] = None
) -> dict:
    user_info = litellm.default_user_params or {}

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
):
    """
    Add a new member to a team

    - add team id to user table
    - add team member w/ budget to team member table
    """
    ## ADD TEAM ID, to USER TABLE IF NEW ##
    if new_member.user_id is not None:
        new_user_defaults = get_new_internal_user_defaults(user_id=new_member.user_id)
        await prisma_client.db.litellm_usertable.upsert(
            where={"user_id": new_member.user_id},
            data={
                "update": {"teams": {"push": [team_id]}},
                "create": {"teams": [team_id], **new_user_defaults},  # type: ignore
            },
        )
    elif new_member.user_email is not None:
        new_user_defaults = get_new_internal_user_defaults(
            user_id=str(uuid.uuid4()), user_email=new_member.user_email
        )
        ## user email is not unique acc. to prisma schema -> future improvement
        ### for now: check if it exists in db, if not - insert it
        existing_user_row = await prisma_client.get_data(
            key_val={"user_email": new_member.user_email},
            table_name="user",
            query_type="find_all",
        )
        if existing_user_row is None or (
            isinstance(existing_user_row, list) and len(existing_user_row) == 0
        ):

            await prisma_client.insert_data(data=new_user_defaults, table_name="user")  # type: ignore

    # Check if trying to set a budget for team member
    if max_budget_in_team is not None and new_member.user_id is not None:
        # create a new budget item for this member
        response = await prisma_client.db.litellm_budgettable.create(
            data={
                "max_budget": max_budget_in_team,
                "created_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
                "updated_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
            }
        )

        _budget_id = response.budget_id
        await prisma_client.db.litellm_teammembership.create(
            data={
                "team_id": team_id,
                "user_id": new_member.user_id,
                "budget_id": _budget_id,
            }
        )


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
        if isinstance(update_request, KeyRequest):
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
                if kwargs is None:
                    kwargs = {}
                user_api_key_dict: UserAPIKeyAuth = (
                    kwargs.get("user_api_key_dict") or UserAPIKeyAuth()
                )
                _http_request: Request = kwargs.get("http_request")
                parent_otel_span = user_api_key_dict.parent_otel_span
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

                            await open_telemetry_logger.async_management_endpoint_success_hook(
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
            parent_otel_span = user_api_key_dict.parent_otel_span
            if parent_otel_span is not None:
                from litellm.proxy.proxy_server import open_telemetry_logger

                if open_telemetry_logger is not None:
                    _http_request: Request = kwargs.get("http_request")
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

                        await open_telemetry_logger.async_management_endpoint_failure_hook(
                            logging_payload=logging_payload,
                            parent_otel_span=parent_otel_span,
                        )

            raise e

    return wrapper
