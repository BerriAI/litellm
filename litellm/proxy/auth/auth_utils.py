import os
import re
import sys
from typing import Any, List, Optional, Tuple

from fastapi import HTTPException, Request, status

from litellm import Router, provider_list
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import *
from litellm.types.router import CONFIGURABLE_CLIENTSIDE_AUTH_PARAMS


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


def check_regex_or_str_match(request_body_value: Any, regex_str: str) -> bool:
    """
    Check if request_body_value matches the regex_str or is equal to param
    """
    if re.match(regex_str, request_body_value) or regex_str == request_body_value:
        return True
    return False


def _is_param_allowed(
    param: str,
    request_body_value: Any,
    configurable_clientside_auth_params: CONFIGURABLE_CLIENTSIDE_AUTH_PARAMS,
) -> bool:
    """
    Check if param is a str or dict and if request_body_value is in the list of allowed values
    """
    if configurable_clientside_auth_params is None:
        return False

    for item in configurable_clientside_auth_params:
        if isinstance(item, str) and param == item:
            return True
        elif isinstance(item, Dict):
            if param == "api_base" and check_regex_or_str_match(
                request_body_value=request_body_value,
                regex_str=item["api_base"],
            ):  # assume param is a regex
                return True

    return False


def _allow_model_level_clientside_configurable_parameters(
    model: str, param: str, request_body_value: Any, llm_router: Optional[Router]
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

    return _is_param_allowed(
        param=param,
        request_body_value=request_body_value,
        configurable_clientside_auth_params=model_info.configurable_clientside_auth_params,
    )


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
                    model=model,
                    param=param,
                    request_body_value=request_body[param],
                    llm_router=llm_router,
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

    Supports wildcard patterns (e.g., "/api/*" matches "/api/users", "/api/users/123")

    In order to use this the litellm config.yaml should have the following in general_settings:

    ```yaml
    general_settings:
        master_key: sk-1234
        public_routes: ["LiteLLMRoutes.public_routes", "/spend/calculate", "/api/*"]
    ```
    """
    from litellm.proxy.auth.route_checks import RouteChecks
    from litellm.proxy.proxy_server import general_settings, premium_user

    try:
        if premium_user is not True:
            return False
        if general_settings is None:
            return False

        routes_defined = general_settings.get("public_routes", [])

        # Check exact match first
        if current_route in routes_defined:
            return True

        # Check wildcard patterns
        for route_pattern in routes_defined:
            if RouteChecks._route_matches_wildcard_pattern(
                route=current_route, pattern=route_pattern
            ):
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
def get_key_model_rpm_limit(
    user_api_key_dict: UserAPIKeyAuth,
) -> Optional[Dict[str, int]]:
    """
    Get the model rpm limit for a given api key
    - check key metadata
    - check key model max budget
    - check team metadata
    """
    if user_api_key_dict.metadata:
        if "model_rpm_limit" in user_api_key_dict.metadata:
            return user_api_key_dict.metadata["model_rpm_limit"]
    elif user_api_key_dict.model_max_budget:
        model_rpm_limit: Dict[str, Any] = {}
        for model, budget in user_api_key_dict.model_max_budget.items():
            if "rpm_limit" in budget and budget["rpm_limit"] is not None:
                model_rpm_limit[model] = budget["rpm_limit"]
        return model_rpm_limit
    elif user_api_key_dict.team_metadata:
        if "model_rpm_limit" in user_api_key_dict.team_metadata:
            return user_api_key_dict.team_metadata["model_rpm_limit"]
    return None


def get_key_model_tpm_limit(
    user_api_key_dict: UserAPIKeyAuth,
) -> Optional[Dict[str, int]]:
    if user_api_key_dict.metadata:
        if "model_tpm_limit" in user_api_key_dict.metadata:
            return user_api_key_dict.metadata["model_tpm_limit"]
    elif user_api_key_dict.model_max_budget:
        if "tpm_limit" in user_api_key_dict.model_max_budget:
            return user_api_key_dict.model_max_budget["tpm_limit"]
    elif user_api_key_dict.team_metadata:
        if "model_tpm_limit" in user_api_key_dict.team_metadata:
            return user_api_key_dict.team_metadata["model_tpm_limit"]
    return None


def get_model_rate_limit_from_metadata(
    user_api_key_dict: UserAPIKeyAuth,
    metadata_accessor_key: Literal["team_metadata", "organization_metadata"],
    rate_limit_key: Literal["model_rpm_limit", "model_tpm_limit"],
) -> Optional[Dict[str, int]]:
    if getattr(user_api_key_dict, metadata_accessor_key):
        return getattr(user_api_key_dict, metadata_accessor_key).get(rate_limit_key)
    return None
  
def get_team_model_rpm_limit(
    user_api_key_dict: UserAPIKeyAuth,
) -> Optional[Dict[str, int]]:
    if user_api_key_dict.team_metadata:
        return user_api_key_dict.team_metadata.get("model_rpm_limit")
    return None


def get_team_model_tpm_limit(
    user_api_key_dict: UserAPIKeyAuth,
) -> Optional[Dict[str, int]]:
    if user_api_key_dict.team_metadata:
        return user_api_key_dict.team_metadata.get("model_tpm_limit")
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


def _has_user_setup_sso():
    """
    Check if the user has set up single sign-on (SSO) by verifying the presence of Microsoft client ID, Google client ID or generic client ID and UI username environment variables.
    Returns a boolean indicating whether SSO has been set up.
    """
    microsoft_client_id = os.getenv("MICROSOFT_CLIENT_ID", None)
    google_client_id = os.getenv("GOOGLE_CLIENT_ID", None)
    generic_client_id = os.getenv("GENERIC_CLIENT_ID", None)

    sso_setup = (
        (microsoft_client_id is not None)
        or (google_client_id is not None)
        or (generic_client_id is not None)
    )

    return sso_setup


def get_customer_user_header_from_mapping(user_id_mapping) -> Optional[str]:
    """Return the header_name mapped to CUSTOMER role, if any (dict-based)."""
    if not user_id_mapping:
        return None
    items = user_id_mapping if isinstance(user_id_mapping, list) else [user_id_mapping]
    for item in items:
        if not isinstance(item, dict):
            continue
        role = item.get("litellm_user_role")
        header_name = item.get("header_name")
        if role is None or not header_name:
            continue
        if str(role).lower() == str(LitellmUserRoles.CUSTOMER).lower():
            return header_name
    return None


def get_end_user_id_from_request_body(
    request_body: dict, request_headers: Optional[dict] = None
) -> Optional[str]:
    # Import general_settings here to avoid potential circular import issues at module level
    # and to ensure it's fetched at runtime.
    from litellm.proxy.proxy_server import general_settings

    # Check 1 : Follow the user header mappings feature, if not found, then check for deprecated user_header_name (only if request_headers is provided)
    # User query: "system not respecting user_header_name property"
    # This implies the key in general_settings is 'user_header_name'.
    if request_headers is not None:
        custom_header_name_to_check: Optional[str] = None

        # Prefer user mappings (new behavior)
        user_id_mapping = general_settings.get("user_header_mappings", None)
        if user_id_mapping:
            custom_header_name_to_check = get_customer_user_header_from_mapping(
                user_id_mapping
            )

        # Fallback to deprecated user_header_name if mapping did not specify
        if not custom_header_name_to_check:
            user_id_header_config_key = "user_header_name"
            value = general_settings.get(user_id_header_config_key)
            if isinstance(value, str) and value.strip() != "":
                custom_header_name_to_check = value

        # If we have a header name to check, try to read it from request headers
        if isinstance(custom_header_name_to_check, str):
            for header_name, header_value in request_headers.items():
                if header_name.lower() == custom_header_name_to_check.lower():
                    user_id_from_header = header_value
                    user_id_str = (
                        str(user_id_from_header)
                        if user_id_from_header is not None
                        else ""
                    )
                    if user_id_str.strip():
                        return user_id_str

    # Check 2: 'user' field in request_body (commonly OpenAI)
    if "user" in request_body and request_body["user"] is not None:
        user_from_body_user_field = request_body["user"]
        return str(user_from_body_user_field)

    # Check 3: 'litellm_metadata.user' in request_body (commonly Anthropic)
    litellm_metadata = request_body.get("litellm_metadata")
    if isinstance(litellm_metadata, dict):
        user_from_litellm_metadata = litellm_metadata.get("user")
        if user_from_litellm_metadata is not None:
            return str(user_from_litellm_metadata)

    # Check 4: 'metadata.user_id' in request_body (another common pattern)
    metadata_dict = request_body.get("metadata")
    if isinstance(metadata_dict, dict):
        user_id_from_metadata_field = metadata_dict.get("user_id")
        if user_id_from_metadata_field is not None:
            return str(user_id_from_metadata_field)

    return None


def get_model_from_request(
    request_data: dict, route: str
) -> Optional[Union[str, List[str]]]:
    # First try to get model from request_data
    model = request_data.get("model") or request_data.get("target_model_names")

    if model is not None:
        model_names = model.split(",")
        if len(model_names) == 1:
            model = model_names[0].strip()
        else:
            model = [m.strip() for m in model_names]

    # If model not in request_data, try to extract from route
    if model is None:
        # Parse model from route that follows the pattern /openai/deployments/{model}/*
        match = re.match(r"/openai/deployments/([^/]+)", route)
        if match:
            model = match.group(1)

    return model


def abbreviate_api_key(api_key: str) -> str:
    return f"sk-...{api_key[-4:]}"
