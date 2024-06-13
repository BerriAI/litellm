from datetime import datetime
from functools import wraps
from litellm.proxy._types import UserAPIKeyAuth, ManagementEndpointLoggingPayload
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from fastapi import Request


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
