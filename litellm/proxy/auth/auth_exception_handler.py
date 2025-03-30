"""
Handles Authentication Errors
"""

import asyncio
from typing import TYPE_CHECKING, Any, Optional, Union

from fastapi import HTTPException, Request, status

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import ProxyErrorTypes, ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.auth_utils import _get_request_ip_address
from litellm.proxy.db.exception_handler import PrismaDBExceptionHandler
from litellm.types.services import ServiceTypes

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = Union[_Span, Any]
else:
    Span = Any


class UserAPIKeyAuthExceptionHandler:
    @staticmethod
    async def _handle_authentication_error(
        e: Exception,
        request: Request,
        request_data: dict,
        route: str,
        parent_otel_span: Optional[Span],
        api_key: str,
    ) -> UserAPIKeyAuth:
        """
        Handles Connection Errors when reading a Virtual Key from LiteLLM DB
        Use this if you don't want failed DB queries to block LLM API reqiests

        Reliability scenarios this covers:
        - DB is down and having an outage
        - Unable to read / recover a key from the DB

        Returns:
            - UserAPIKeyAuth: If general_settings.allow_requests_on_db_unavailable is True

        Raises:
            - Orignal Exception in all other cases
        """
        from litellm.proxy.proxy_server import (
            general_settings,
            litellm_proxy_admin_name,
            proxy_logging_obj,
        )

        if (
            PrismaDBExceptionHandler.should_allow_request_on_db_unavailable()
            and PrismaDBExceptionHandler.is_database_connection_error(e)
        ):
            # log this as a DB failure on prometheus
            proxy_logging_obj.service_logging_obj.service_failure_hook(
                service=ServiceTypes.DB,
                call_type="get_key_object",
                error=e,
                duration=0.0,
            )

            return UserAPIKeyAuth(
                key_name="failed-to-connect-to-db",
                token="failed-to-connect-to-db",
                user_id=litellm_proxy_admin_name,
            )
        else:
            # raise the exception to the caller
            requester_ip = _get_request_ip_address(
                request=request,
                use_x_forwarded_for=general_settings.get("use_x_forwarded_for", False),
            )
            verbose_proxy_logger.exception(
                "litellm.proxy.proxy_server.user_api_key_auth(): Exception occured - {}\nRequester IP Address:{}".format(
                    str(e),
                    requester_ip,
                ),
                extra={"requester_ip": requester_ip},
            )

            # Log this exception to OTEL, Datadog etc
            user_api_key_dict = UserAPIKeyAuth(
                parent_otel_span=parent_otel_span,
                api_key=api_key,
            )
            asyncio.create_task(
                proxy_logging_obj.post_call_failure_hook(
                    request_data=request_data,
                    original_exception=e,
                    user_api_key_dict=user_api_key_dict,
                    error_type=ProxyErrorTypes.auth_error,
                    route=route,
                )
            )

            if isinstance(e, litellm.BudgetExceededError):
                raise ProxyException(
                    message=e.message,
                    type=ProxyErrorTypes.budget_exceeded,
                    param=None,
                    code=400,
                )
            if isinstance(e, HTTPException):
                raise ProxyException(
                    message=getattr(e, "detail", f"Authentication Error({str(e)})"),
                    type=ProxyErrorTypes.auth_error,
                    param=getattr(e, "param", "None"),
                    code=getattr(e, "status_code", status.HTTP_401_UNAUTHORIZED),
                )
            elif isinstance(e, ProxyException):
                raise e
            raise ProxyException(
                message="Authentication Error, " + str(e),
                type=ProxyErrorTypes.auth_error,
                param=getattr(e, "param", "None"),
                code=status.HTTP_401_UNAUTHORIZED,
            )
