# What is this?
## Helper utils for the management endpoints (keys/users/teams)
from datetime import datetime
from functools import wraps
from litellm.proxy._types import UserAPIKeyAuth, ManagementEndpointLoggingPayload
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm._logging import verbose_logger
from fastapi import Request

from litellm.proxy._types import LiteLLM_TeamTable, Member, UserAPIKeyAuth
from litellm.proxy.utils import PrismaClient
import uuid
from typing import Optional


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
        await prisma_client.db.litellm_usertable.update(
            where={"user_id": new_member.user_id},
            data={"teams": {"push": [team_id]}},
        )
    elif new_member.user_email is not None:
        user_data = {"user_id": str(uuid.uuid4()), "user_email": new_member.user_email}
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

            await prisma_client.insert_data(data=user_data, table_name="user")

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

                    if _http_request:
                        _route = _http_request.url.path
                        # Flush user_api_key cache if this was an update/delete call to /key, /team, or /user
                        if _route in [
                            "/key/update",
                            "/key/delete",
                            "/team/update",
                            "/team/delete",
                            "/user/update",
                            "/user/delete",
                            "/customer/update",
                            "/customer/delete",
                        ]:
                            from litellm.proxy.proxy_server import user_api_key_cache

                            user_api_key_cache.flush_cache()
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
