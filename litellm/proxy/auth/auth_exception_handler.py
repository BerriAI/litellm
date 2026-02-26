"""
Handles Authentication Errors
"""

from typing import TYPE_CHECKING, Any, Optional, Union

from fastapi import HTTPException, Request, status

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import ProxyErrorTypes, ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.auth_utils import _get_request_ip_address, abbreviate_api_key
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
            - Original Exception in all other cases
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
                request_route=route,
            )
        else:
            # raise the exception to the caller
            requester_ip = _get_request_ip_address(
                request=request,
                use_x_forwarded_for=general_settings.get("use_x_forwarded_for", False),
            )

            # Build structured context for the log message
            masked_key = abbreviate_api_key(api_key=api_key) if api_key else "None"
            http_method = getattr(request, "method", "UNKNOWN")

            # Extract error category and status code from typed exceptions
            if isinstance(e, ProxyException):
                error_type = e.type
                error_code = e.code
            elif isinstance(e, HTTPException):
                error_type = "http_exception"
                error_code = str(getattr(e, "status_code", "unknown"))
            elif isinstance(e, litellm.BudgetExceededError):
                error_type = "budget_exceeded"
                error_code = "400"
            else:
                error_type = type(e).__name__
                error_code = "401"

            log_extra = {
                "requester_ip": requester_ip,
                "route": route,
                "api_key": masked_key,
                "error_type": error_type,
                "error_code": error_code,
                "http_method": http_method,
            }

            # Use warning level for expected auth failures to avoid noisy ERROR logs
            # and full tracebacks for routine rejected requests (e.g. missing/invalid key).
            # Reserve ERROR + traceback for truly unexpected exceptions and server errors.
            _is_expected_auth_error = isinstance(
                e, (ProxyException, litellm.BudgetExceededError)
            ) or (
                isinstance(e, HTTPException)
                and getattr(e, "status_code", 500) < 500
            )
            if _is_expected_auth_error:
                verbose_proxy_logger.warning(
                    "Auth failed: error_type={}, error_code={}, route={} {}, api_key={}, ip={} - {}".format(
                        error_type,
                        error_code,
                        http_method,
                        route,
                        masked_key,
                        requester_ip,
                        str(e),
                    ),
                    extra=log_extra,
                )
            else:
                verbose_proxy_logger.exception(
                    "Auth exception: error_type={}, error_code={}, route={} {}, api_key={}, ip={} - {}".format(
                        error_type,
                        error_code,
                        http_method,
                        route,
                        masked_key,
                        requester_ip,
                        str(e),
                    ),
                    extra=log_extra,
                )

            # Log this exception to OTEL, Datadog etc
            user_api_key_dict = UserAPIKeyAuth(
                parent_otel_span=parent_otel_span,
                api_key=api_key,
                request_route=route,
            )
            # Allow callbacks to transform the error response
            transformed_exception = await proxy_logging_obj.post_call_failure_hook(
                request_data=request_data,
                original_exception=e,
                user_api_key_dict=user_api_key_dict,
                error_type=ProxyErrorTypes.auth_error,
                route=route,
            )
            # Use transformed exception if callback returned one, otherwise use original
            if transformed_exception is not None:
                e = transformed_exception

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
