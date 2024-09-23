import re
import sys
import traceback
from typing import List, Optional, Tuple

from fastapi import HTTPException, Request, status

from litellm import Router, provider_list
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import *


def _get_request_ip_address(
    request: Request, use_x_forwarded_for: Optional[bool] = False
) -> Optional[str]:

    client_ip = None
    if use_x_forwarded_for is True and "x-forwarded-for" in request.headers:
        client_ip = request.headers["x-forwarded-for"]
    elif request.client is not None:
        client_ip = request.client.host
    else:
        client_ip = ""

    return client_ip


def _check_valid_ip(
    allowed_ips: Optional[List[str]],
    request: Request,
    use_x_forwarded_for: Optional[bool] = False,
) -> Tuple[bool, Optional[str]]:
    """
    Returns if ip is allowed or not
    """
    if allowed_ips is None:  # if not set, assume true
        return True, None

    # if general_settings.get("use_x_forwarded_for") is True then use x-forwarded-for
    client_ip = _get_request_ip_address(
        request=request, use_x_forwarded_for=use_x_forwarded_for
    )

    # Check if IP address is allowed
    if client_ip not in allowed_ips:
        return False, client_ip

    return True, client_ip


def check_complete_credentials(request_body: dict) -> bool:
    """
    if 'api_base' in request body. Check if complete credentials given. Prevent malicious attacks.
    """
    given_model: Optional[str] = None

    given_model = request_body.get("model")
    if given_model is None:
        return False

    if (
        "sagemaker" in given_model
        or "bedrock" in given_model
        or "vertex_ai" in given_model
        or "vertex_ai_beta" in given_model
    ):
        # complex credentials - easier to make a malicious request
        return False

    if "api_key" in request_body:
        return True

    return False


def _allow_model_level_clientside_configurable_parameters(
    model: str, param: str, llm_router: Optional[Router]
) -> bool:
    """
    Check if model is allowed to use configurable client-side params
    - get matching model
    - check if 'clientside_configurable_parameters' is set for model
    -
    """
    if llm_router is None:
        return False
    # check if model is set
    model_info = llm_router.get_model_group_info(model_group=model)
    if model_info is None:
        # check if wildcard model is set
        if model.split("/", 1)[0] in provider_list:
            model_info = llm_router.get_model_group_info(
                model_group=model.split("/", 1)[0]
            )

    if model_info is None:
        return False

    if model_info is None or model_info.configurable_clientside_auth_params is None:
        return False

    if param in model_info.configurable_clientside_auth_params:
        return True

    return False


def is_request_body_safe(
    request_body: dict, general_settings: dict, llm_router: Optional[Router], model: str
) -> bool:
    """
    Check if the request body is safe.

    A malicious user can set the ﻿api_base to their own domain and invoke POST /chat/completions to intercept and steal the OpenAI API key.
    Relevant issue: https://huntr.com/bounties/4001e1a2-7b7a-4776-a3ae-e6692ec3d997
    """
    banned_params = ["api_base", "base_url"]

    for param in banned_params:
        if (
            param in request_body
            and not check_complete_credentials(  # allow client-credentials to be passed to proxy
                request_body=request_body
            )
        ):
            if general_settings.get("allow_client_side_credentials") is True:
                return True
            elif (
                _allow_model_level_clientside_configurable_parameters(
                    model=model, param=param, llm_router=llm_router
                )
                is True
            ):
                return True
            raise ValueError(
                f"Rejected Request: {param} is not allowed in request body. "
                "Enable with `general_settings::allow_client_side_credentials` on proxy config.yaml. "
                "Relevant Issue: https://huntr.com/bounties/4001e1a2-7b7a-4776-a3ae-e6692ec3d997",
            )

    return True


async def pre_db_read_auth_checks(
    request: Request,
    request_data: dict,
    route: str,
):
    """
    1. Checks if request size is under max_request_size_mb (if set)
    2. Check if request body is safe (example user has not set api_base in request body)
    3. Check if IP address is allowed (if set)
    4. Check if request route is an allowed route on the proxy (if set)

    Returns:
    - True

    Raises:
    - HTTPException if request fails initial auth checks
    """
    from litellm.proxy.proxy_server import general_settings, llm_router, premium_user

    # Check 1. request size
    await check_if_request_size_is_safe(request=request)

    # Check 2. Request body is safe
    is_request_body_safe(
        request_body=request_data,
        general_settings=general_settings,
        llm_router=llm_router,
        model=request_data.get(
            "model", ""
        ),  # [TODO] use model passed in url as well (azure openai routes)
    )

    # Check 3. Check if IP address is allowed
    is_valid_ip, passed_in_ip = _check_valid_ip(
        allowed_ips=general_settings.get("allowed_ips", None),
        use_x_forwarded_for=general_settings.get("use_x_forwarded_for", False),
        request=request,
    )

    if not is_valid_ip:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access forbidden: IP address {passed_in_ip} not allowed.",
        )

    # Check 4. Check if request route is an allowed route on the proxy
    if "allowed_routes" in general_settings:
        _allowed_routes = general_settings["allowed_routes"]
        if premium_user is not True:
            verbose_proxy_logger.error(
                f"Trying to set allowed_routes. This is an Enterprise feature. {CommonProxyErrors.not_premium_user.value}"
            )
        if route not in _allowed_routes:
            verbose_proxy_logger.error(
                f"Route {route} not in allowed_routes={_allowed_routes}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access forbidden: Route {route} not allowed",
            )


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


def is_pass_through_provider_route(route: str) -> bool:
    PROVIDER_SPECIFIC_PASS_THROUGH_ROUTES = [
        "vertex-ai",
    ]

    # check if any of the prefixes are in the route
    for prefix in PROVIDER_SPECIFIC_PASS_THROUGH_ROUTES:
        if prefix in route:
            return True

    return False


def should_run_auth_on_pass_through_provider_route(route: str) -> bool:
    """
    Use this to decide if the rest of the LiteLLM Virtual Key auth checks should run on /vertex-ai/{endpoint} routes
    Use this to decide if the rest of the LiteLLM Virtual Key auth checks should run on provider pass through routes
    ex /vertex-ai/{endpoint} routes
    Run virtual key auth if the following is try:
    - User is premium_user
    - User has enabled litellm_setting.use_client_credentials_pass_through_routes
    """
    from litellm.proxy.proxy_server import general_settings, premium_user

    if premium_user is not True:
        return False

    # premium use has opted into using client credentials
    if (
        general_settings.get("use_client_credentials_pass_through_routes", False)
        is True
    ):
        return False

    # only enabled for LiteLLM Enterprise
    return True


def _has_user_setup_sso():
    """
    Check if the user has set up single sign-on (SSO) by verifying the presence of Microsoft client ID, Google client ID, and UI username environment variables.
    Returns a boolean indicating whether SSO has been set up.
    """
    microsoft_client_id = os.getenv("MICROSOFT_CLIENT_ID", None)
    google_client_id = os.getenv("GOOGLE_CLIENT_ID", None)
    ui_username = os.getenv("UI_USERNAME", None)

    sso_setup = (
        (microsoft_client_id is not None)
        or (google_client_id is not None)
        or (ui_username is not None)
    )

    return sso_setup
