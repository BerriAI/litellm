import asyncio
import copy
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from fastapi import Request
from starlette.datastructures import Headers

import litellm
from litellm._logging import verbose_logger, verbose_proxy_logger
from litellm._service_logger import ServiceLogging
from litellm.litellm_core_utils.safe_json_loads import safe_json_loads
from litellm.proxy._types import (
    AddTeamCallback,
    CommonProxyErrors,
    LitellmDataForBackendLLMCall,
    SpecialHeaders,
    TeamCallbackMetadata,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.router import Router
from litellm.types.llms.anthropic import ANTHROPIC_API_HEADERS
from litellm.types.services import ServiceTypes
from litellm.types.utils import (
    ProviderSpecificHeader,
    StandardLoggingUserAPIKeyMetadata,
    SupportedCacheControls,
)

service_logger_obj = ServiceLogging()  # used for tracking latency on OTEL


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
    if RouteChecks._is_assistants_api_request(request):
        return "litellm_metadata"

    LITELLM_METADATA_ROUTES = [
        "batches",
        "/v1/messages",
        "responses",
        "files",
    ]

    if any(
        [
            litellm_metadata_route in request.url.path
            for litellm_metadata_route in LITELLM_METADATA_ROUTES
        ]
    ):
        return "litellm_metadata"
    else:
        return "metadata"


def safe_add_api_version_from_query_params(data: dict, request: Request):
    try:
        if hasattr(request, "query_params"):
            query_params = dict(request.query_params)
            if "api-version" in query_params:
                data["api_version"] = query_params["api-version"]
    except KeyError:
        pass
    except Exception as e:
        verbose_logger.exception(
            "error checking api version in query params: %s", str(e)
        )


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
    elif (
        not data.callback_type or data.callback_type == "success_and_failure"
    ):  # assume 'success_and_failure' = litellm.callbacks
        if team_callback_settings_obj.success_callback is None:
            team_callback_settings_obj.success_callback = []
        if team_callback_settings_obj.failure_callback is None:
            team_callback_settings_obj.failure_callback = []
        if team_callback_settings_obj.callbacks is None:
            team_callback_settings_obj.callbacks = []

        if data.callback_name not in team_callback_settings_obj.success_callback:
            team_callback_settings_obj.success_callback.append(data.callback_name)

        if data.callback_name not in team_callback_settings_obj.failure_callback:
            team_callback_settings_obj.failure_callback.append(data.callback_name)

        if data.callback_name not in team_callback_settings_obj.callbacks:
            team_callback_settings_obj.callbacks.append(data.callback_name)

    for var, value in data.callback_vars.items():
        if team_callback_settings_obj.callback_vars is None:
            team_callback_settings_obj.callback_vars = {}
        team_callback_settings_obj.callback_vars[var] = str(
            litellm.utils.get_secret(value, default_value=value) or value
        )

    return team_callback_settings_obj


class KeyAndTeamLoggingSettings:
    """
    Helper class to get the dynamic logging settings for the key and team
    """

    @staticmethod
    def get_key_dynamic_logging_settings(user_api_key_dict: UserAPIKeyAuth):
        if (
            user_api_key_dict.metadata is not None
            and "logging" in user_api_key_dict.metadata
        ):
            return user_api_key_dict.metadata["logging"]
        return None

    @staticmethod
    def get_team_dynamic_logging_settings(user_api_key_dict: UserAPIKeyAuth):
        if (
            user_api_key_dict.team_metadata is not None
            and "logging" in user_api_key_dict.team_metadata
        ):
            return user_api_key_dict.team_metadata["logging"]
        return None


def _get_dynamic_logging_metadata(
    user_api_key_dict: UserAPIKeyAuth, proxy_config: ProxyConfig
) -> Optional[TeamCallbackMetadata]:
    callback_settings_obj: Optional[TeamCallbackMetadata] = None
    key_dynamic_logging_settings: Optional[dict] = (
        KeyAndTeamLoggingSettings.get_key_dynamic_logging_settings(user_api_key_dict)
    )
    team_dynamic_logging_settings: Optional[dict] = (
        KeyAndTeamLoggingSettings.get_team_dynamic_logging_settings(user_api_key_dict)
    )
    #########################################################################################
    # Key-based callbacks
    #########################################################################################
    if key_dynamic_logging_settings is not None:
        for item in key_dynamic_logging_settings:
            callback_settings_obj = convert_key_logging_metadata_to_callback(
                data=AddTeamCallback(**item),
                team_callback_settings_obj=callback_settings_obj,
            )
    #########################################################################################
    # Team-based callbacks
    #########################################################################################
    elif team_dynamic_logging_settings is not None:
        for item in team_dynamic_logging_settings:
            callback_settings_obj = convert_key_logging_metadata_to_callback(
                data=AddTeamCallback(**item),
                team_callback_settings_obj=callback_settings_obj,
            )
    #########################################################################################
    # Deprecated format - maintained for backwards compatibility
    #########################################################################################
    elif (
        user_api_key_dict.team_metadata is not None
        and "callback_settings" in user_api_key_dict.team_metadata
    ):
        """
        callback_settings = {
            {
            'callback_vars': {'langfuse_public_key': 'pk', 'langfuse_secret_key': 'sk_'},
            'failure_callback': [],
            'success_callback': ['langfuse', 'langfuse']
        }
        }
        """
        team_metadata = user_api_key_dict.team_metadata
        callback_settings = team_metadata.get("callback_settings", None) or {}
        callback_settings_obj = TeamCallbackMetadata(**callback_settings)
        verbose_proxy_logger.debug(
            "Team callback settings activated: %s", callback_settings_obj
        )
    #########################################################################################
    # Enter here when configured on the config.yaml file.
    #########################################################################################
    elif user_api_key_dict.team_id is not None:
        callback_settings_obj = (
            LiteLLMProxyRequestSetup.add_team_based_callbacks_from_config(
                team_id=user_api_key_dict.team_id, proxy_config=proxy_config
            )
        )
    return callback_settings_obj


def clean_headers(
    headers: Headers, litellm_key_header_name: Optional[str] = None
) -> dict:
    """
    Removes litellm api key from headers
    """
    special_headers = [v.value.lower() for v in SpecialHeaders._member_map_.values()]
    special_headers = special_headers
    if litellm_key_header_name is not None:
        special_headers.append(litellm_key_header_name.lower())
    clean_headers = {}

    for header, value in headers.items():
        if header.lower() not in special_headers:
            clean_headers[header] = value
    return clean_headers


class LiteLLMProxyRequestSetup:
    @staticmethod
    def _get_timeout_from_request(headers: dict) -> Optional[float]:
        """
        Workaround for client request from Vercel's AI SDK.

        Allow's user to set a timeout in the request headers.

        Example:

        ```js
        const openaiProvider = createOpenAI({
            baseURL: liteLLM.baseURL,
            apiKey: liteLLM.apiKey,
            compatibility: "compatible",
            headers: {
                "x-litellm-timeout": "90"
            },
        });
        ```
        """
        timeout_header = headers.get("x-litellm-timeout", None)
        if timeout_header is not None:
            return float(timeout_header)
        return None

    @staticmethod
    def _get_num_retries_from_request(headers: dict) -> Optional[int]:
        """
        Workaround for client request from Vercel's AI SDK.
        """
        num_retries_header = headers.get("x-litellm-num-retries", None)
        if num_retries_header is not None:
            return int(num_retries_header)
        return None

    @staticmethod
    def _get_forwardable_headers(
        headers: Union[Headers, dict],
    ):
        """
        Get the headers that should be forwarded to the LLM Provider.

        Looks for any `x-` headers and sends them to the LLM Provider.

        [07/09/2025] - Support 'anthropic-beta' header as well.
        """
        forwarded_headers = {}
        for header, value in headers.items():
            if header.lower().startswith("x-") and not header.lower().startswith(
                "x-stainless"
            ):  # causes openai sdk to fail
                forwarded_headers[header] = value
            elif header.lower().startswith("anthropic-beta"):
                forwarded_headers[header] = value

        return forwarded_headers

    @staticmethod
    def _get_case_insensitive_header(headers: dict, key: str) -> Optional[str]:
        """
        Get a case-insensitive header from the headers dictionary.
        """
        for header, value in headers.items():
            if header.lower() == key.lower():
                return value
        return None

    @staticmethod
    def get_user_from_headers(
        headers: dict, general_settings: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Get the user from the specified header if `general_settings.user_header_name` is set.
        """
        if general_settings is None:
            return None

        header_name = general_settings.get("user_header_name")
        if header_name is None or header_name == "":
            return None

        if not isinstance(header_name, str):
            raise TypeError(
                f"Expected user_header_name to be a str but got {type(header_name)}"
            )

        user = LiteLLMProxyRequestSetup._get_case_insensitive_header(
            headers, header_name
        )
        if user is not None:
            verbose_logger.info(f'found user "{user}" in header "{header_name}"')

        return user

    @staticmethod
    def get_openai_org_id_from_headers(
        headers: dict, general_settings: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Get the OpenAI Org ID from the headers.
        """
        if (
            general_settings is not None
            and general_settings.get("forward_openai_org_id") is not True
        ):
            return None
        for header, value in headers.items():
            if header.lower() == "openai-organization":
                verbose_logger.info(f"found openai org id: {value}, sending to llm")
                return value
        return None

    @staticmethod
    def add_headers_to_llm_call(
        headers: dict, user_api_key_dict: UserAPIKeyAuth
    ) -> dict:
        """
        Add headers to the LLM call

        - Checks request headers for forwardable headers
        - Checks if user information should be added to the headers
        """

        returned_headers = LiteLLMProxyRequestSetup._get_forwardable_headers(headers)

        if litellm.add_user_information_to_llm_headers is True:
            litellm_logging_metadata_headers = (
                LiteLLMProxyRequestSetup.get_sanitized_user_information_from_key(
                    user_api_key_dict=user_api_key_dict
                )
            )
            for k, v in litellm_logging_metadata_headers.items():
                if v is not None:
                    returned_headers["x-litellm-{}".format(k)] = v

        return returned_headers

    @staticmethod
    def add_headers_to_llm_call_by_model_group(
        data: dict, headers: dict, user_api_key_dict: UserAPIKeyAuth
    ) -> dict:
        """
        Add headers to the LLM call by model group
        """
        data_model = data.get("model")
        if (
            data_model is not None
            and litellm.model_group_settings is not None
            and litellm.model_group_settings.forward_client_headers_to_llm_api
            is not None
            and data_model
            in litellm.model_group_settings.forward_client_headers_to_llm_api
        ):
            _headers = LiteLLMProxyRequestSetup.add_headers_to_llm_call(
                headers, user_api_key_dict
            )
            if _headers != {}:
                data["headers"] = _headers
        return data

    @staticmethod
    def add_litellm_data_for_backend_llm_call(
        *,
        headers: dict,
        user_api_key_dict: UserAPIKeyAuth,
        general_settings: Optional[Dict[str, Any]] = None,
    ) -> LitellmDataForBackendLLMCall:
        """
        - Adds user from headers
        - Adds forwardable headers
        - Adds org id
        """
        data = LitellmDataForBackendLLMCall()

        if (
            general_settings
            and general_settings.get("forward_client_headers_to_llm_api") is True
        ):
            _headers = LiteLLMProxyRequestSetup.add_headers_to_llm_call(
                headers, user_api_key_dict
            )
            if _headers != {}:
                data["headers"] = _headers
        _organization = LiteLLMProxyRequestSetup.get_openai_org_id_from_headers(
            headers, general_settings
        )
        if _organization is not None:
            data["organization"] = _organization

        timeout = LiteLLMProxyRequestSetup._get_timeout_from_request(headers)
        if timeout is not None:
            data["timeout"] = timeout

        num_retries = LiteLLMProxyRequestSetup._get_num_retries_from_request(headers)
        if num_retries is not None:
            data["num_retries"] = num_retries

        return data

    @staticmethod
    def get_sanitized_user_information_from_key(
        user_api_key_dict: UserAPIKeyAuth,
    ) -> StandardLoggingUserAPIKeyMetadata:
        user_api_key_logged_metadata = StandardLoggingUserAPIKeyMetadata(
            user_api_key_hash=user_api_key_dict.api_key,  # just the hashed token
            user_api_key_alias=user_api_key_dict.key_alias,
            user_api_key_team_id=user_api_key_dict.team_id,
            user_api_key_user_id=user_api_key_dict.user_id,
            user_api_key_org_id=user_api_key_dict.org_id,
            user_api_key_team_alias=user_api_key_dict.team_alias,
            user_api_key_end_user_id=user_api_key_dict.end_user_id,
            user_api_key_user_email=user_api_key_dict.user_email,
            user_api_key_request_route=user_api_key_dict.request_route,
        )
        return user_api_key_logged_metadata

    @staticmethod
    def add_user_api_key_auth_to_request_metadata(
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        _metadata_variable_name: str,
    ) -> dict:
        """
        Adds the `UserAPIKeyAuth` object to the request metadata.
        """
        user_api_key_logged_metadata = (
            LiteLLMProxyRequestSetup.get_sanitized_user_information_from_key(
                user_api_key_dict=user_api_key_dict
            )
        )
        data[_metadata_variable_name].update(user_api_key_logged_metadata)
        data[_metadata_variable_name][
            "user_api_key"
        ] = user_api_key_dict.api_key  # this is just the hashed token

        data[_metadata_variable_name]["user_api_end_user_max_budget"] = getattr(
            user_api_key_dict, "end_user_max_budget", None
        )
        return data

    @staticmethod
    def add_key_level_controls(
        key_metadata: dict, data: dict, _metadata_variable_name: str
    ):
        if "cache" in key_metadata:
            data["cache"] = {}
            if isinstance(key_metadata["cache"], dict):
                for k, v in key_metadata["cache"].items():
                    if k in SupportedCacheControls:
                        data["cache"][k] = v

        ## KEY-LEVEL SPEND LOGS / TAGS
        if "tags" in key_metadata and key_metadata["tags"] is not None:
            data[_metadata_variable_name]["tags"] = (
                LiteLLMProxyRequestSetup._merge_tags(
                    request_tags=data[_metadata_variable_name].get("tags"),
                    tags_to_add=key_metadata["tags"],
                )
            )
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
                        data[_metadata_variable_name]["spend_logs_metadata"][
                            key
                        ] = value
            else:
                data[_metadata_variable_name]["spend_logs_metadata"] = key_metadata[
                    "spend_logs_metadata"
                ]

        ## KEY-LEVEL DISABLE FALLBACKS
        if "disable_fallbacks" in key_metadata and isinstance(
            key_metadata["disable_fallbacks"], bool
        ):
            data["disable_fallbacks"] = key_metadata["disable_fallbacks"]
        return data

    @staticmethod
    def _merge_tags(request_tags: Optional[list], tags_to_add: Optional[list]) -> list:
        """
        Helper function to merge two lists of tags, ensuring no duplicates.

        Args:
            request_tags (Optional[list]): List of tags from the original request
            tags_to_add (Optional[list]): List of tags to add

        Returns:
            list: Combined list of unique tags
        """
        final_tags = []

        if request_tags and isinstance(request_tags, list):
            final_tags.extend(request_tags)

        if tags_to_add and isinstance(tags_to_add, list):
            for tag in tags_to_add:
                if tag not in final_tags:
                    final_tags.append(tag)

        return final_tags

    @staticmethod
    def add_team_based_callbacks_from_config(
        team_id: str,
        proxy_config: ProxyConfig,
    ) -> Optional[TeamCallbackMetadata]:
        """
        Add team-based callbacks from the config
        """
        team_config = proxy_config.load_team_config(team_id=team_id)
        if len(team_config.keys()) == 0:
            return None

        callback_vars_dict = {**team_config.get("callback_vars", team_config)}
        callback_vars_dict.pop("team_id", None)
        callback_vars_dict.pop("success_callback", None)
        callback_vars_dict.pop("failure_callback", None)

        return TeamCallbackMetadata(
            success_callback=team_config.get("success_callback", None),
            failure_callback=team_config.get("failure_callback", None),
            callback_vars=callback_vars_dict,
        )

    @staticmethod
    def add_request_tag_to_metadata(
        llm_router: Optional[Router],
        headers: dict,
        data: dict,
    ) -> Optional[List[str]]:
        tags = None

        # Check request headers for tags
        if "x-litellm-tags" in headers:
            if isinstance(headers["x-litellm-tags"], str):
                _tags = headers["x-litellm-tags"].split(",")
                tags = [tag.strip() for tag in _tags]
            elif isinstance(headers["x-litellm-tags"], list):
                tags = headers["x-litellm-tags"]
        # Check request body for tags
        if "tags" in data and isinstance(data["tags"], list):
            tags = data["tags"]

        return tags


async def add_litellm_data_to_request(  # noqa: PLR0915
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
    from litellm.types.proxy.litellm_pre_call_utils import SecretFields

    safe_add_api_version_from_query_params(data, request)

    _headers = clean_headers(
        request.headers,
        litellm_key_header_name=(
            general_settings.get("litellm_key_header_name")
            if general_settings is not None
            else None
        ),
    )

    data.update(
        LiteLLMProxyRequestSetup.add_litellm_data_for_backend_llm_call(
            headers=_headers,
            user_api_key_dict=user_api_key_dict,
            general_settings=general_settings,
        )
    )

    # check for forwardable headers
    data = LiteLLMProxyRequestSetup.add_headers_to_llm_call_by_model_group(
        data=data, headers=_headers, user_api_key_dict=user_api_key_dict
    )

    # Parse user info from headers
    user = LiteLLMProxyRequestSetup.get_user_from_headers(_headers, general_settings)
    if user is not None:
        if user_api_key_dict.end_user_id is None:
            user_api_key_dict.end_user_id = user
        if "user" not in data:
            data["user"] = user

    # Include original request and headers in the data
    data["proxy_server_request"] = {
        "url": str(request.url),
        "method": request.method,
        "headers": _headers,
        "body": copy.copy(data),  # use copy instead of deepcopy
    }

    data["secret_fields"] = SecretFields(raw_headers=dict(request.headers))

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

    if data.get(_metadata_variable_name, None) is None:
        data[_metadata_variable_name] = {}

    # Parse metadata if it's a string (e.g., from multipart/form-data)
    if "metadata" in data and data["metadata"] is not None:
        if isinstance(data["metadata"], str):
            data["metadata"] = safe_json_loads(data["metadata"])
            if not isinstance(data["metadata"], dict):
                verbose_proxy_logger.warning(
                    f"Failed to parse 'metadata' as JSON dict. Received value: {data['metadata']}"
                )
        data[_metadata_variable_name]["requester_metadata"] = copy.deepcopy(
            data["metadata"]
        )

    data = LiteLLMProxyRequestSetup.add_user_api_key_auth_to_request_metadata(
        data=data,
        user_api_key_dict=user_api_key_dict,
        _metadata_variable_name=_metadata_variable_name,
    )
    data[_metadata_variable_name]["litellm_api_version"] = version

    if general_settings is not None:
        data[_metadata_variable_name]["global_max_parallel_requests"] = (
            general_settings.get("global_max_parallel_requests", None)
        )

    ### KEY-LEVEL Controls
    key_metadata = user_api_key_dict.metadata
    data = LiteLLMProxyRequestSetup.add_key_level_controls(
        key_metadata=key_metadata,
        data=data,
        _metadata_variable_name=_metadata_variable_name,
    )
    ## TEAM-LEVEL SPEND LOGS/TAGS
    team_metadata = user_api_key_dict.team_metadata or {}
    if "tags" in team_metadata and team_metadata["tags"] is not None:
        data[_metadata_variable_name]["tags"] = LiteLLMProxyRequestSetup._merge_tags(
            request_tags=data[_metadata_variable_name].get("tags"),
            tags_to_add=team_metadata["tags"],
        )
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
    data[_metadata_variable_name][
        "user_api_key_request_route"
    ] = user_api_key_dict.request_route

    # API Key spend, budget - used by prometheus.py
    data[_metadata_variable_name]["user_api_key_spend"] = user_api_key_dict.spend
    data[_metadata_variable_name][
        "user_api_key_max_budget"
    ] = user_api_key_dict.max_budget
    data[_metadata_variable_name][
        "user_api_key_model_max_budget"
    ] = user_api_key_dict.model_max_budget

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
    start_time = time.time()
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

    # Check if using tag based routing
    tags = LiteLLMProxyRequestSetup.add_request_tag_to_metadata(
        llm_router=llm_router,
        headers=dict(request.headers),
        data=data,
    )

    if tags is not None:
        data[_metadata_variable_name]["tags"] = tags

    # Team Callbacks controls
    callback_settings_obj = _get_dynamic_logging_metadata(
        user_api_key_dict=user_api_key_dict, proxy_config=proxy_config
    )
    if callback_settings_obj is not None:
        data["success_callback"] = callback_settings_obj.success_callback
        data["failure_callback"] = callback_settings_obj.failure_callback

        if callback_settings_obj.callback_vars is not None:
            # unpack callback_vars in data
            for k, v in callback_settings_obj.callback_vars.items():
                data[k] = v

    # Add disabled callbacks from key metadata
    if (
        user_api_key_dict.metadata
        and "litellm_disabled_callbacks" in user_api_key_dict.metadata
    ):
        disabled_callbacks = user_api_key_dict.metadata["litellm_disabled_callbacks"]
        if disabled_callbacks and isinstance(disabled_callbacks, list):
            data["litellm_disabled_callbacks"] = disabled_callbacks

    # Guardrails
    move_guardrails_to_metadata(
        data=data,
        _metadata_variable_name=_metadata_variable_name,
        user_api_key_dict=user_api_key_dict,
    )

    # Team Model Aliases
    _update_model_if_team_alias_exists(
        data=data,
        user_api_key_dict=user_api_key_dict,
    )

    verbose_proxy_logger.debug(
        "[PROXY] returned data from litellm_pre_call_utils: %s", data
    )

    ## ENFORCED PARAMS CHECK
    # loop through each enforced param
    # example enforced_params ['user', 'metadata', 'metadata.generation_name']
    _enforced_params_check(
        request_body=data,
        general_settings=general_settings,
        user_api_key_dict=user_api_key_dict,
        premium_user=premium_user,
    )

    end_time = time.time()
    asyncio.create_task(
        service_logger_obj.async_service_success_hook(
            service=ServiceTypes.PROXY_PRE_CALL,
            duration=end_time - start_time,
            call_type="add_litellm_data_to_request",
            start_time=start_time,
            end_time=end_time,
            parent_otel_span=user_api_key_dict.parent_otel_span,
        )
    )

    return data


def _update_model_if_team_alias_exists(
    data: dict,
    user_api_key_dict: UserAPIKeyAuth,
) -> None:
    """
    Update the model if the team alias exists

    If a alias map has been set on a team, then we want to make the request with the model the team alias is pointing to

    eg.
        - user calls `gpt-4o`
        - team.model_alias_map = {
            "gpt-4o": "gpt-4o-team-1"
        }
        - requested_model = "gpt-4o-team-1"
    """
    _model = data.get("model")
    if (
        _model
        and user_api_key_dict.team_model_aliases
        and _model in user_api_key_dict.team_model_aliases
    ):
        data["model"] = user_api_key_dict.team_model_aliases[_model]
    return


def _get_enforced_params(
    general_settings: Optional[dict], user_api_key_dict: UserAPIKeyAuth
) -> Optional[list]:
    enforced_params: Optional[list] = None
    if general_settings is not None:
        enforced_params = general_settings.get("enforced_params")
        if (
            "service_account_settings" in general_settings
            and check_if_token_is_service_account(user_api_key_dict) is True
        ):
            service_account_settings = general_settings["service_account_settings"]
            if "enforced_params" in service_account_settings:
                if enforced_params is None:
                    enforced_params = []
                enforced_params.extend(service_account_settings["enforced_params"])
    if user_api_key_dict.metadata.get("enforced_params", None) is not None:
        if enforced_params is None:
            enforced_params = []
        enforced_params.extend(user_api_key_dict.metadata["enforced_params"])
    return enforced_params


def check_if_token_is_service_account(valid_token: UserAPIKeyAuth) -> bool:
    """
    Checks if the token is a service account

    Returns:
        bool: True if token is a service account

    """
    if valid_token.metadata:
        if "service_account_id" in valid_token.metadata:
            return True
    return False


def _enforced_params_check(
    request_body: dict,
    general_settings: Optional[dict],
    user_api_key_dict: UserAPIKeyAuth,
    premium_user: bool,
) -> bool:
    """
    If enforced params are set, check if the request body contains the enforced params.
    """
    enforced_params: Optional[list] = _get_enforced_params(
        general_settings=general_settings, user_api_key_dict=user_api_key_dict
    )
    if enforced_params is None:
        return True
    if enforced_params is not None and premium_user is not True:
        raise ValueError(
            f"Enforced Params is an Enterprise feature. Enforced Params: {enforced_params}. {CommonProxyErrors.not_premium_user.value}"
        )

    for enforced_param in enforced_params:
        _enforced_params = enforced_param.split(".")
        if len(_enforced_params) == 1:
            if _enforced_params[0] not in request_body:
                raise ValueError(
                    f"BadRequest please pass param={_enforced_params[0]} in request body. This is a required param"
                )
        elif len(_enforced_params) == 2:
            # this is a scenario where user requires request['metadata']['generation_name'] to exist
            if _enforced_params[0] not in request_body:
                raise ValueError(
                    f"BadRequest please pass param={_enforced_params[0]} in request body. This is a required param"
                )
            if _enforced_params[1] not in request_body[_enforced_params[0]]:
                raise ValueError(
                    f"BadRequest please pass param=[{_enforced_params[0]}][{_enforced_params[1]}] in request body. This is a required param"
                )
    return True


def _add_guardrails_from_key_or_team_metadata(
    key_metadata: Optional[dict],
    team_metadata: Optional[dict],
    data: dict,
    metadata_variable_name: str,
) -> None:
    """
    Helper add guardrails from key or team metadata to request data

    Args:
        key_metadata: The key metadata dictionary to check for guardrails
        team_metadata: The team metadata dictionary to check for guardrails
        data: The request data to update
        metadata_variable_name: The name of the metadata field in data

    """
    from litellm.proxy.utils import _premium_user_check

    for _management_object_metadata in [key_metadata, team_metadata]:
        if _management_object_metadata and "guardrails" in _management_object_metadata:
            if len(_management_object_metadata["guardrails"]) > 0:
                _premium_user_check()

            data[metadata_variable_name]["guardrails"] = _management_object_metadata[
                "guardrails"
            ]


def move_guardrails_to_metadata(
    data: dict,
    _metadata_variable_name: str,
    user_api_key_dict: UserAPIKeyAuth,
):
    """
    Helper to add guardrails from request to metadata

    - If guardrails set on API Key metadata then sets guardrails on request metadata
    - If guardrails not set on API key, then checks request metadata
    """
    # Check key-level guardrails
    _add_guardrails_from_key_or_team_metadata(
        key_metadata=user_api_key_dict.metadata,
        team_metadata=user_api_key_dict.team_metadata,
        data=data,
        metadata_variable_name=_metadata_variable_name,
    )

    # Check request-level guardrails
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
    anthropic_headers = {}
    # boolean to indicate if a header was added
    added_header = False
    for header in ANTHROPIC_API_HEADERS:
        if header in headers:
            header_value = headers[header]
            anthropic_headers[header] = header_value
            added_header = True

    if added_header is True:
        data["provider_specific_header"] = ProviderSpecificHeader(
            custom_llm_provider="anthropic",
            extra_headers=anthropic_headers,
        )

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
