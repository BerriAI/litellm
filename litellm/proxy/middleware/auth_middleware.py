"""
Auth Middleware

Runs user_api_key_auth before FastAPI's dependency injection and stores
the result in request.state.user_api_key_dict.
"""

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import SpecialHeaders
from litellm.proxy.auth.auth_utils import (
    get_end_user_id_from_request_body,
    get_request_route,
    normalize_request_route,
)
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.auth.user_api_key_auth import _user_api_key_auth_builder
from litellm.proxy.common_utils.http_parsing_utils import (
    _read_request_body,
    _safe_get_request_headers,
    populate_request_with_path_params,
)


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            api_key = (
                request.headers.get(SpecialHeaders.openai_authorization.value) or ""
            )
            azure_api_key_header = request.headers.get(
                SpecialHeaders.azure_authorization.value
            )
            anthropic_api_key_header = request.headers.get(
                SpecialHeaders.anthropic_authorization.value
            )
            google_ai_studio_api_key_header = request.headers.get(
                SpecialHeaders.google_ai_studio_authorization.value
            )
            azure_apim_header = request.headers.get(
                SpecialHeaders.azure_apim_authorization.value
            )
            custom_litellm_key_header = request.headers.get(
                SpecialHeaders.custom_litellm_api_key.value
            )

            request_data = await _read_request_body(request=request)
            request_data = populate_request_with_path_params(
                request_data=request_data, request=request
            )
            route = get_request_route(request=request)

            user_api_key_auth_obj = await _user_api_key_auth_builder(
                request=request,
                api_key=api_key,
                azure_api_key_header=azure_api_key_header,
                anthropic_api_key_header=anthropic_api_key_header,
                google_ai_studio_api_key_header=google_ai_studio_api_key_header,
                azure_apim_header=azure_apim_header,
                request_data=request_data,
                custom_litellm_key_header=custom_litellm_key_header,
            )

            RouteChecks.should_call_route(
                route=route, valid_token=user_api_key_auth_obj
            )

            end_user_id = get_end_user_id_from_request_body(
                request_data, _safe_get_request_headers(request)
            )
            if end_user_id is not None:
                user_api_key_auth_obj.end_user_id = end_user_id

            user_api_key_auth_obj.request_route = normalize_request_route(route)

            request.state.user_api_key_dict = user_api_key_auth_obj

        except HTTPException:
            verbose_proxy_logger.debug(
                "AuthMiddleware: auth rejected for %s", request.url.path
            )
        except Exception:
            verbose_proxy_logger.warning(
                "AuthMiddleware: unexpected error for %s",
                request.url.path,
                exc_info=True,
            )

        response = await call_next(request)
        return response
