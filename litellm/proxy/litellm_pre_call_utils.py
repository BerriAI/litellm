import copy
from typing import TYPE_CHECKING, Any, Dict, Optional

from fastapi import Request

from litellm._logging import verbose_logger, verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
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

    For all /thread endpoints we need to call this "litellm_metadata"

    For ALL other endpoints we call this "metadata
    """
    if "thread" in request.url.path:
        return "litellm_metadata"
    else:
        return "metadata"


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
    query_params = dict(request.query_params)
    if "api-version" in query_params:
        data["api_version"] = query_params["api-version"]

    # Include original request and headers in the data
    data["proxy_server_request"] = {
        "url": str(request.url),
        "method": request.method,
        "headers": dict(request.headers),
        "body": copy.copy(data),  # use copy instead of deepcopy
    }

    ## Cache Controls
    headers = request.headers
    verbose_proxy_logger.debug("Request Headers: %s", headers)
    cache_control_header = headers.get("Cache-Control", None)
    if cache_control_header:
        cache_dict = parse_cache_control(cache_control_header)
        data["ttl"] = cache_dict.get("s-maxage")

    ### KEY-LEVEL CACHNG
    key_metadata = user_api_key_dict.metadata
    if "cache" in key_metadata:
        data["cache"] = {}
        if isinstance(key_metadata["cache"], dict):
            for k, v in key_metadata["cache"].items():
                if k in SupportedCacheControls:
                    data["cache"][k] = v

    verbose_proxy_logger.debug("receiving data: %s", data)

    _metadata_variable_name = _get_metadata_variable_name(request)

    if _metadata_variable_name not in data:
        data[_metadata_variable_name] = {}
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
    # Add the OTEL Parent Trace before sending it LiteLLM
    data[_metadata_variable_name][
        "litellm_parent_otel_span"
    ] = user_api_key_dict.parent_otel_span

    ### END-USER SPECIFIC PARAMS ###
    if user_api_key_dict.allowed_model_region is not None:
        data["allowed_model_region"] = user_api_key_dict.allowed_model_region

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

    return data
