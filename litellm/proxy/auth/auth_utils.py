import re
import sys
import traceback

from fastapi import Request

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import *


def route_in_additonal_public_routes(current_route: str):
    """
    Helper to check if the user defined public_routes on config.yaml

    Parameters:
    - current_route: str - the route the user is trying to call

    Returns:
    - bool - True if the route is defined in public_routes
    - bool - False if the route is not defined in public_routes


    In order to use this the litellm config.yaml should have the following in general_settings:

    ```yaml
    general_settings:
        master_key: sk-1234
        public_routes: ["LiteLLMRoutes.public_routes", "/spend/calculate"]
    ```
    """

    # check if user is premium_user - if not do nothing
    from litellm.proxy._types import LiteLLMRoutes
    from litellm.proxy.proxy_server import general_settings, premium_user

    try:
        if premium_user is not True:
            return False
        # check if this is defined on the config
        if general_settings is None:
            return False

        routes_defined = general_settings.get("public_routes", [])
        if current_route in routes_defined:
            return True

        return False
    except Exception as e:
        verbose_proxy_logger.error(f"route_in_additonal_public_routes: {str(e)}")
        return False


def is_llm_api_route(route: str) -> bool:
    """
    Helper to checks if provided route is an OpenAI route


    Returns:
        - True: if route is an OpenAI route
        - False: if route is not an OpenAI route
    """

    if route in LiteLLMRoutes.openai_routes.value:
        return True

    if route in LiteLLMRoutes.anthropic_routes.value:
        return True

    # fuzzy match routes like "/v1/threads/thread_49EIN5QF32s4mH20M7GFKdlZ"
    # Check for routes with placeholders
    for openai_route in LiteLLMRoutes.openai_routes.value:
        # Replace placeholders with regex pattern
        # placeholders are written as "/threads/{thread_id}"
        if "{" in openai_route:
            pattern = re.sub(r"\{[^}]+\}", r"[^/]+", openai_route)
            # Anchor the pattern to match the entire string
            pattern = f"^{pattern}$"
            if re.match(pattern, route):
                return True

    return False


def get_request_route(request: Request) -> str:
    """
    Helper to get the route from the request

    remove base url from path if set e.g. `/genai/chat/completions` -> `/chat/completions
    """
    try:
        if hasattr(request, "base_url") and request.url.path.startswith(
            request.base_url.path
        ):
            # remove base_url from path
            return request.url.path[len(request.base_url.path) - 1 :]
        else:
            return request.url.path
    except Exception as e:
        verbose_proxy_logger.debug(
            f"error on get_request_route: {str(e)}, defaulting to request.url.path={request.url.path}"
        )
        return request.url.path


async def check_if_request_size_is_safe(request: Request) -> bool:
    """
    Enterprise Only:
        - Checks if the request size is within the limit

    Args:
        request (Request): The incoming request.

    Returns:
        bool: True if the request size is within the limit

    Raises:
        ProxyException: If the request size is too large

    """
    from litellm.proxy.proxy_server import general_settings, premium_user

    max_request_size_mb = general_settings.get("max_request_size_mb", None)
    if max_request_size_mb is not None:
        # Check if premium user
        if premium_user is not True:
            verbose_proxy_logger.warning(
                f"using max_request_size_mb - not checking -  this is an enterprise only feature. {CommonProxyErrors.not_premium_user.value}"
            )
            return True

        # Get the request body
        content_length = request.headers.get("content-length")

        if content_length:
            header_size = int(content_length)
            header_size_mb = bytes_to_mb(bytes_value=header_size)
            verbose_proxy_logger.debug(
                f"content_length request size in MB={header_size_mb}"
            )

            if header_size_mb > max_request_size_mb:
                raise ProxyException(
                    message=f"Request size is too large. Request size is {header_size_mb} MB. Max size is {max_request_size_mb} MB",
                    type=ProxyErrorTypes.bad_request_error.value,
                    code=400,
                    param="content-length",
                )
        else:
            # If Content-Length is not available, read the body
            body = await request.body()
            body_size = len(body)
            request_size_mb = bytes_to_mb(bytes_value=body_size)

            verbose_proxy_logger.debug(
                f"request body request size in MB={request_size_mb}"
            )
            if request_size_mb > max_request_size_mb:
                raise ProxyException(
                    message=f"Request size is too large. Request size is {request_size_mb} MB. Max size is {max_request_size_mb} MB",
                    type=ProxyErrorTypes.bad_request_error.value,
                    code=400,
                    param="content-length",
                )

    return True


async def check_response_size_is_safe(response: Any) -> bool:
    """
    Enterprise Only:
        - Checks if the response size is within the limit

    Args:
        response (Any): The response to check.

    Returns:
        bool: True if the response size is within the limit

    Raises:
        ProxyException: If the response size is too large

    """

    from litellm.proxy.proxy_server import general_settings, premium_user

    max_response_size_mb = general_settings.get("max_response_size_mb", None)
    if max_response_size_mb is not None:
        # Check if premium user
        if premium_user is not True:
            verbose_proxy_logger.warning(
                f"using max_response_size_mb - not checking -  this is an enterprise only feature. {CommonProxyErrors.not_premium_user.value}"
            )
            return True

        response_size_mb = bytes_to_mb(bytes_value=sys.getsizeof(response))
        verbose_proxy_logger.debug(f"response size in MB={response_size_mb}")
        if response_size_mb > max_response_size_mb:
            raise ProxyException(
                message=f"Response size is too large. Response size is {response_size_mb} MB. Max size is {max_response_size_mb} MB",
                type=ProxyErrorTypes.bad_request_error.value,
                code=400,
                param="content-length",
            )

    return True


def bytes_to_mb(bytes_value: int):
    """
    Helper to convert bytes to MB
    """
    return bytes_value / (1024 * 1024)


# helpers used by parallel request limiter to handle model rpm/tpm limits for a given api key
def get_key_model_rpm_limit(user_api_key_dict: UserAPIKeyAuth) -> Optional[dict]:
    if user_api_key_dict.metadata:
        if "model_rpm_limit" in user_api_key_dict.metadata:
            return user_api_key_dict.metadata["model_rpm_limit"]

    return None


def get_key_model_tpm_limit(user_api_key_dict: UserAPIKeyAuth) -> Optional[dict]:
    if user_api_key_dict.metadata:
        if "model_tpm_limit" in user_api_key_dict.metadata:
            return user_api_key_dict.metadata["model_tpm_limit"]

    return None
