import copy
from typing import TYPE_CHECKING, Any, Dict, Optional

from fastapi import Request

import litellm
from litellm._logging import verbose_logger, verbose_proxy_logger
from litellm.proxy._types import (
    AddTeamCallback,
    CommonProxyErrors,
    TeamCallbackMetadata,
    UserAPIKeyAuth,
)
from litellm.types.utils import SupportedCacheControls

if TYPE_CHECKING:
    from litellm.proxy.proxy_server import ProxyConfig as _ProxyConfig

    ProxyConfig = _ProxyConfig
else:
    ProxyConfig = Any


def parse_cache_control(cache_control):
    cache_dict = {}
    directives = cache_control.split(", ")

    for directive in directives:
        if "=" in directive:
            key, value = directive.split("=")
            cache_dict[key] = value
        else:
            cache_dict[directive] = True

    return cache_dict


def _get_metadata_variable_name(request: Request) -> str:
    """
    Helper to return what the "metadata" field should be called in the request data

    For all /thread or /assistant endpoints we need to call this "litellm_metadata"

    For ALL other endpoints we call this "metadata
    """
    if "thread" in request.url.path or "assistant" in request.url.path:
        return "litellm_metadata"
    if "batches" in request.url.path:
        return "litellm_metadata"
    if "/v1/messages" in request.url.path:
        # anthropic API has a field called metadata
        return "litellm_metadata"
    else:
        return "metadata"


def safe_add_api_version_from_query_params(data: dict, request: Request):
    try:
        if hasattr(request, "query_params"):
            query_params = dict(request.query_params)
            if "api-version" in query_params:
                data["api_version"] = query_params["api-version"]
    except Exception as e:
        verbose_logger.error("error checking api version in query params: %s", str(e))


def convert_key_logging_metadata_to_callback(
    data: AddTeamCallback, team_callback_settings_obj: Optional[TeamCallbackMetadata]
) -> TeamCallbackMetadata:
    if team_callback_settings_obj is None:
        team_callback_settings_obj = TeamCallbackMetadata()
    if data.callback_type == "success":
        if team_callback_settings_obj.success_callback is None:
            team_callback_settings_obj.success_callback = []

        if data.callback_name not in team_callback_settings_obj.success_callback:
            team_callback_settings_obj.success_callback.append(data.callback_name)
    elif data.callback_type == "failure":
        if team_callback_settings_obj.failure_callback is None:
            team_callback_settings_obj.failure_callback = []

        if data.callback_name not in team_callback_settings_obj.failure_callback:
            team_callback_settings_obj.failure_callback.append(data.callback_name)
    elif data.callback_type == "success_and_failure":
        if team_callback_settings_obj.success_callback is None:
            team_callback_settings_obj.success_callback = []
        if team_callback_settings_obj.failure_callback is None:
            team_callback_settings_obj.failure_callback = []

        if data.callback_name not in team_callback_settings_obj.success_callback:
            team_callback_settings_obj.success_callback.append(data.callback_name)

        if data.callback_name not in team_callback_settings_obj.failure_callback:
            team_callback_settings_obj.failure_callback.append(data.callback_name)

    for var, value in data.callback_vars.items():
        if team_callback_settings_obj.callback_vars is None:
            team_callback_settings_obj.callback_vars = {}
        team_callback_settings_obj.callback_vars[var] = str(
            litellm.utils.get_secret(value, default_value=value) or value
        )

    return team_callback_settings_obj


def _get_dynamic_logging_metadata(
    user_api_key_dict: UserAPIKeyAuth,
) -> Optional[TeamCallbackMetadata]:
    callback_settings_obj: Optional[TeamCallbackMetadata] = None
    if (
        user_api_key_dict.metadata is not None
        and "logging" in user_api_key_dict.metadata
    ):
        for item in user_api_key_dict.metadata["logging"]:
            callback_settings_obj = convert_key_logging_metadata_to_callback(
                data=AddTeamCallback(**item),
                team_callback_settings_obj=callback_settings_obj,
            )
    elif user_api_key_dict.team_metadata is not None:
        team_metadata = user_api_key_dict.team_metadata
        if "callback_settings" in team_metadata:
            callback_settings = team_metadata.get("callback_settings", None) or {}
            callback_settings_obj = TeamCallbackMetadata(**callback_settings)
            verbose_proxy_logger.debug(
                "Team callback settings activated: %s", callback_settings_obj
            )
            """
            callback_settings = {
              {
                'callback_vars': {'langfuse_public_key': 'pk', 'langfuse_secret_key': 'sk_'}, 
                'failure_callback': [], 
                'success_callback': ['langfuse', 'langfuse']
            }
            }
            """

    return callback_settings_obj


async def add_litellm_data_to_request(
    data: dict,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth,
    proxy_config: ProxyConfig,
    general_settings: Optional[Dict[str, Any]] = None,
    version: Optional[str] = None,
):
    """
    Adds LiteLLM-specific data to the request.

    Args:
        data (dict): The data dictionary to be modified.
        request (Request): The incoming request.
        user_api_key_dict (UserAPIKeyAuth): The user API key dictionary.
        general_settings (Optional[Dict[str, Any]], optional): General settings. Defaults to None.
        version (Optional[str], optional): Version. Defaults to None.

    Returns:
        dict: The modified data dictionary.

    """
    from litellm.proxy.proxy_server import llm_router, premium_user

    safe_add_api_version_from_query_params(data, request)

    _headers = dict(request.headers)

    # Include original request and headers in the data
    data["proxy_server_request"] = {
        "url": str(request.url),
        "method": request.method,
        "headers": _headers,
        "body": copy.copy(data),  # use copy instead of deepcopy
    }

    ## Dynamic api version (Azure OpenAI endpoints) ##
    try:
        query_params = request.query_params
        # Convert query parameters to a dictionary (optional)
        query_dict = dict(query_params)
    except KeyError:
        query_dict = {}

    ## check for api version in query params
    dynamic_api_version: Optional[str] = query_dict.get("api-version")

    if dynamic_api_version is not None:  # only pass, if set
        data["api_version"] = dynamic_api_version

    ## Forward any LLM API Provider specific headers in extra_headers
    add_provider_specific_headers_to_request(data=data, headers=_headers)

    ## Cache Controls
    headers = request.headers
    verbose_proxy_logger.debug("Request Headers: %s", headers)
    cache_control_header = headers.get("Cache-Control", None)
    if cache_control_header:
        cache_dict = parse_cache_control(cache_control_header)
        data["ttl"] = cache_dict.get("s-maxage")

    verbose_proxy_logger.debug("receiving data: %s", data)

    _metadata_variable_name = _get_metadata_variable_name(request)

    if _metadata_variable_name not in data:
        data[_metadata_variable_name] = {}

    # We want to log the "metadata" from the client side request. Avoid circular reference by not directly assigning metadata to itself
    if "metadata" in data and data["metadata"] is not None:
        data[_metadata_variable_name]["requester_metadata"] = copy.deepcopy(
            data["metadata"]
        )

    data[_metadata_variable_name]["user_api_key"] = user_api_key_dict.api_key
    data[_metadata_variable_name]["user_api_key_alias"] = getattr(
        user_api_key_dict, "key_alias", None
    )
    data[_metadata_variable_name]["user_api_end_user_max_budget"] = getattr(
        user_api_key_dict, "end_user_max_budget", None
    )
    data[_metadata_variable_name]["litellm_api_version"] = version

    if general_settings is not None:
        data[_metadata_variable_name]["global_max_parallel_requests"] = (
            general_settings.get("global_max_parallel_requests", None)
        )

    data[_metadata_variable_name]["user_api_key_user_id"] = user_api_key_dict.user_id
    data[_metadata_variable_name]["user_api_key_org_id"] = user_api_key_dict.org_id
    data[_metadata_variable_name]["user_api_key_team_id"] = getattr(
        user_api_key_dict, "team_id", None
    )
    data[_metadata_variable_name]["user_api_key_team_alias"] = getattr(
        user_api_key_dict, "team_alias", None
    )

    ### KEY-LEVEL Controls
    key_metadata = user_api_key_dict.metadata
    if "cache" in key_metadata:
        data["cache"] = {}
        if isinstance(key_metadata["cache"], dict):
            for k, v in key_metadata["cache"].items():
                if k in SupportedCacheControls:
                    data["cache"][k] = v

    ## KEY-LEVEL SPEND LOGS / TAGS
    if "tags" in key_metadata and key_metadata["tags"] is not None:
        if "tags" in data[_metadata_variable_name] and isinstance(
            data[_metadata_variable_name]["tags"], list
        ):
            data[_metadata_variable_name]["tags"].extend(key_metadata["tags"])
        else:
            data[_metadata_variable_name]["tags"] = key_metadata["tags"]
    if "spend_logs_metadata" in key_metadata and isinstance(
        key_metadata["spend_logs_metadata"], dict
    ):
        if "spend_logs_metadata" in data[_metadata_variable_name] and isinstance(
            data[_metadata_variable_name]["spend_logs_metadata"], dict
        ):
            for key, value in key_metadata["spend_logs_metadata"].items():
                if (
                    key not in data[_metadata_variable_name]["spend_logs_metadata"]
                ):  # don't override k-v pair sent by request (user request)
                    data[_metadata_variable_name]["spend_logs_metadata"][key] = value
        else:
            data[_metadata_variable_name]["spend_logs_metadata"] = key_metadata[
                "spend_logs_metadata"
            ]

    ## TEAM-LEVEL SPEND LOGS/TAGS
    team_metadata = user_api_key_dict.team_metadata or {}
    if "tags" in team_metadata and team_metadata["tags"] is not None:
        if "tags" in data[_metadata_variable_name] and isinstance(
            data[_metadata_variable_name]["tags"], list
        ):
            data[_metadata_variable_name]["tags"].extend(team_metadata["tags"])
        else:
            data[_metadata_variable_name]["tags"] = team_metadata["tags"]
    if "spend_logs_metadata" in team_metadata and isinstance(
        team_metadata["spend_logs_metadata"], dict
    ):
        if "spend_logs_metadata" in data[_metadata_variable_name] and isinstance(
            data[_metadata_variable_name]["spend_logs_metadata"], dict
        ):
            for key, value in team_metadata["spend_logs_metadata"].items():
                if (
                    key not in data[_metadata_variable_name]["spend_logs_metadata"]
                ):  # don't override k-v pair sent by request (user request)
                    data[_metadata_variable_name]["spend_logs_metadata"][key] = value
        else:
            data[_metadata_variable_name]["spend_logs_metadata"] = team_metadata[
                "spend_logs_metadata"
            ]

    # Team spend, budget - used by prometheus.py
    data[_metadata_variable_name][
        "user_api_key_team_max_budget"
    ] = user_api_key_dict.team_max_budget
    data[_metadata_variable_name][
        "user_api_key_team_spend"
    ] = user_api_key_dict.team_spend

    # API Key spend, budget - used by prometheus.py
    data[_metadata_variable_name]["user_api_key_spend"] = user_api_key_dict.spend
    data[_metadata_variable_name][
        "user_api_key_max_budget"
    ] = user_api_key_dict.max_budget

    data[_metadata_variable_name]["user_api_key_metadata"] = user_api_key_dict.metadata
    _headers = dict(request.headers)
    _headers.pop(
        "authorization", None
    )  # do not store the original `sk-..` api key in the db
    data[_metadata_variable_name]["headers"] = _headers
    data[_metadata_variable_name]["endpoint"] = str(request.url)

    # OTEL Controls / Tracing
    # Add the OTEL Parent Trace before sending it LiteLLM
    data[_metadata_variable_name][
        "litellm_parent_otel_span"
    ] = user_api_key_dict.parent_otel_span
    _add_otel_traceparent_to_data(data, request=request)

    ### END-USER SPECIFIC PARAMS ###
    if user_api_key_dict.allowed_model_region is not None:
        data["allowed_model_region"] = user_api_key_dict.allowed_model_region

    ## [Enterprise Only]
    # Add User-IP Address
    requester_ip_address = ""
    if premium_user is True:
        # Only set the IP Address for Enterprise Users

        # logic for tracking IP Address
        if (
            general_settings is not None
            and general_settings.get("use_x_forwarded_for") is True
            and request is not None
            and hasattr(request, "headers")
            and "x-forwarded-for" in request.headers
        ):
            requester_ip_address = request.headers["x-forwarded-for"]
        elif (
            request is not None
            and hasattr(request, "client")
            and hasattr(request.client, "host")
            and request.client is not None
        ):
            requester_ip_address = request.client.host
    data[_metadata_variable_name]["requester_ip_address"] = requester_ip_address

    # Enterprise Only - Check if using tag based routing
    if llm_router and llm_router.enable_tag_filtering is True:
        if premium_user is not True:
            verbose_proxy_logger.warning(
                "router.enable_tag_filtering is on %s \n switched off router.enable_tag_filtering",
                CommonProxyErrors.not_premium_user.value,
            )
            llm_router.enable_tag_filtering = False
        else:
            if "tags" in data:
                data[_metadata_variable_name]["tags"] = data["tags"]

    ### TEAM-SPECIFIC PARAMS ###
    if user_api_key_dict.team_id is not None:
        team_config = await proxy_config.load_team_config(
            team_id=user_api_key_dict.team_id
        )
        if len(team_config) == 0:
            pass
        else:
            team_id = team_config.pop("team_id", None)
            data[_metadata_variable_name]["team_id"] = team_id
            data = {
                **team_config,
                **data,
            }  # add the team-specific configs to the completion call

    # Team Callbacks controls
    callback_settings_obj = _get_dynamic_logging_metadata(
        user_api_key_dict=user_api_key_dict
    )
    if callback_settings_obj is not None:
        data["success_callback"] = callback_settings_obj.success_callback
        data["failure_callback"] = callback_settings_obj.failure_callback

        if callback_settings_obj.callback_vars is not None:
            # unpack callback_vars in data
            for k, v in callback_settings_obj.callback_vars.items():
                data[k] = v

    # Guardrails
    move_guardrails_to_metadata(
        data=data,
        _metadata_variable_name=_metadata_variable_name,
        user_api_key_dict=user_api_key_dict,
    )

    return data


def move_guardrails_to_metadata(
    data: dict,
    _metadata_variable_name: str,
    user_api_key_dict: UserAPIKeyAuth,
):
    """
    Heper to add guardrails from request to metadata

    - If guardrails set on API Key metadata then sets guardrails on request metadata
    - If guardrails not set on API key, then checks request metadata

    """
    if user_api_key_dict.metadata:
        if "guardrails" in user_api_key_dict.metadata:
            from litellm.proxy.proxy_server import premium_user

            if premium_user is not True:
                raise ValueError(
                    f"Using Guardrails on API Key {CommonProxyErrors.not_premium_user}"
                )

            data[_metadata_variable_name]["guardrails"] = user_api_key_dict.metadata[
                "guardrails"
            ]
            return

    if "guardrails" in data:
        data[_metadata_variable_name]["guardrails"] = data["guardrails"]
        del data["guardrails"]

    if "guardrail_config" in data:
        data[_metadata_variable_name]["guardrail_config"] = data["guardrail_config"]
        del data["guardrail_config"]


def add_provider_specific_headers_to_request(
    data: dict,
    headers: dict,
):
    ANTHROPIC_API_HEADERS = [
        "anthropic-version",
        "anthropic-beta",
    ]

    extra_headers = data.get("extra_headers", {}) or {}

    # boolean to indicate if a header was added
    added_header = False
    for header in ANTHROPIC_API_HEADERS:
        if header in headers:
            header_value = headers[header]
            extra_headers.update({header: header_value})
            added_header = True

    if added_header is True:
        data["extra_headers"] = extra_headers

    return


def _add_otel_traceparent_to_data(data: dict, request: Request):
    from litellm.proxy.proxy_server import open_telemetry_logger

    if data is None:
        return
    if open_telemetry_logger is None:
        # if user is not use OTEL don't send extra_headers
        # relevant issue: https://github.com/BerriAI/litellm/issues/4448
        return

    if litellm.forward_traceparent_to_llm_provider is True:
        if request.headers:
            if "traceparent" in request.headers:
                # we want to forward this to the LLM Provider
                # Relevant issue: https://github.com/BerriAI/litellm/issues/4419
                # pass this in extra_headers
                if "extra_headers" not in data:
                    data["extra_headers"] = {}
                _exra_headers = data["extra_headers"]
                if "traceparent" not in _exra_headers:
                    _exra_headers["traceparent"] = request.headers["traceparent"]
