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
    LitellmUserRoles,
    SpecialHeaders,
    TeamCallbackMetadata,
    UserAPIKeyAuth,
)

# Cache special headers as a frozenset for O(1) lookup performance
_SPECIAL_HEADERS_CACHE = frozenset(
    v.value.lower() for v in SpecialHeaders._member_map_.values()
)
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.router import Router
from litellm.types.llms.anthropic import ANTHROPIC_API_HEADERS
from litellm.types.services import ServiceTypes
from litellm.types.utils import (
    LlmProviders,
    ProviderSpecificHeader,
    StandardLoggingUserAPIKeyMetadata,
    SupportedCacheControls,
)

service_logger_obj = ServiceLogging()  # used for tracking latency on OTEL


if TYPE_CHECKING:
    from litellm.proxy.proxy_server import ProxyConfig as _ProxyConfig
    from litellm.types.proxy.policy_engine import PolicyMatchContext

    ProxyConfig = _ProxyConfig
else:
    ProxyConfig = Any
    PolicyMatchContext = Any


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


LITELLM_METADATA_ROUTES = (
    "batches",
    "/v1/messages",
    "responses",
    "files",
)


def _get_metadata_variable_name(request: Request) -> str:
    """
    Helper to return what the "metadata" field should be called in the request data

    For all /thread or /assistant endpoints we need to call this "litellm_metadata"

    For ALL other endpoints we call this "metadata
    """
    if RouteChecks._is_assistants_api_request(request):
        return "litellm_metadata"

    if any(route in request.url.path for route in LITELLM_METADATA_ROUTES):
        return "litellm_metadata"

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
    key_dynamic_logging_settings: Optional[
        dict
    ] = KeyAndTeamLoggingSettings.get_key_dynamic_logging_settings(user_api_key_dict)
    team_dynamic_logging_settings: Optional[
        dict
    ] = KeyAndTeamLoggingSettings.get_team_dynamic_logging_settings(user_api_key_dict)
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
    clean_headers = {}
    litellm_key_lower = (
        litellm_key_header_name.lower() if litellm_key_header_name is not None else None
    )

    for header, value in headers.items():
        header_lower = header.lower()
        # Check if header should be excluded: either in special headers cache or matches custom litellm key
        if header_lower not in _SPECIAL_HEADERS_CACHE and (
            litellm_key_lower is None or header_lower != litellm_key_lower
        ):
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
    def _get_stream_timeout_from_request(headers: dict) -> Optional[float]:
        """
        Get the `stream_timeout` from the request headers.
        """
        stream_timeout_header = headers.get("x-litellm-stream-timeout", None)
        if stream_timeout_header is not None:
            return float(stream_timeout_header)
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
    def _get_spend_logs_metadata_from_request_headers(headers: dict) -> Optional[dict]:
        """
        Get the `spend_logs_metadata` from the request headers.
        """
        from litellm.litellm_core_utils.safe_json_loads import safe_json_loads

        spend_logs_metadata_header = headers.get("x-litellm-spend-logs-metadata", None)
        if spend_logs_metadata_header is not None:
            return safe_json_loads(spend_logs_metadata_header)
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
    def add_internal_user_from_user_mapping(
        general_settings: Optional[Dict],
        user_api_key_dict: UserAPIKeyAuth,
        headers: dict,
    ) -> UserAPIKeyAuth:
        if general_settings is None:
            return user_api_key_dict
        user_header_mapping = general_settings.get("user_header_mappings")
        if not user_header_mapping:
            return user_api_key_dict
        header_name = LiteLLMProxyRequestSetup.get_internal_user_header_from_mapping(
            user_header_mapping
        )
        if not header_name:
            return user_api_key_dict
        header_value = LiteLLMProxyRequestSetup._get_case_insensitive_header(
            headers, header_name
        )
        if header_value:
            user_api_key_dict.user_id = header_value
            return user_api_key_dict
        return user_api_key_dict

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
        from litellm.proxy.auth.auth_checks import _check_model_access_helper
        from litellm.proxy.proxy_server import llm_router

        data_model = data.get("model")

        if (
            data_model is not None
            and litellm.model_group_settings is not None
            and litellm.model_group_settings.forward_client_headers_to_llm_api
            is not None
            and _check_model_access_helper(
                model=data_model,
                llm_router=llm_router,
                models=litellm.model_group_settings.forward_client_headers_to_llm_api,
                team_model_aliases=user_api_key_dict.team_model_aliases,
                team_id=user_api_key_dict.team_id,
            )  # handles aliases, wildcards, etc.
        ):
            _headers = LiteLLMProxyRequestSetup.add_headers_to_llm_call(
                headers, user_api_key_dict
            )
            if _headers != {}:
                data["headers"] = _headers
        return data

    @staticmethod
    def get_internal_user_header_from_mapping(user_header_mapping) -> Optional[str]:
        if not user_header_mapping:
            return None
        items = (
            user_header_mapping
            if isinstance(user_header_mapping, list)
            else [user_header_mapping]
        )
        for item in items:
            if not isinstance(item, dict):
                continue
            role = item.get("litellm_user_role")
            header_name = item.get("header_name")
            if role is None or not header_name:
                continue
            if str(role).lower() == str(LitellmUserRoles.INTERNAL_USER).lower():
                return header_name
        return None

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

        stream_timeout = LiteLLMProxyRequestSetup._get_stream_timeout_from_request(
            headers
        )
        if stream_timeout is not None:
            data["stream_timeout"] = stream_timeout

        num_retries = LiteLLMProxyRequestSetup._get_num_retries_from_request(headers)
        if num_retries is not None:
            data["num_retries"] = num_retries

        return data

    @staticmethod
    def add_litellm_metadata_from_request_headers(
        headers: dict,
        data: dict,
        _metadata_variable_name: str,
    ) -> dict:
        """
        Add litellm metadata from request headers

        Relevant issue: https://github.com/BerriAI/litellm/issues/14008
        """
        from litellm.proxy._types import LitellmMetadataFromRequestHeaders

        metadata_from_headers = LitellmMetadataFromRequestHeaders()
        spend_logs_metadata = (
            LiteLLMProxyRequestSetup._get_spend_logs_metadata_from_request_headers(
                headers
            )
        )
        if spend_logs_metadata is not None:
            metadata_from_headers["spend_logs_metadata"] = spend_logs_metadata

        #########################################################################################
        # Finally update the requests metadata with the `metadata_from_headers`
        #########################################################################################
        agent_id_from_header = headers.get("x-litellm-agent-id")
        trace_id_from_header = headers.get("x-litellm-trace-id")
        if agent_id_from_header:
            metadata_from_headers["agent_id"] = agent_id_from_header
            verbose_proxy_logger.debug(f"Extracted agent_id from header: {agent_id_from_header}")
        
        if trace_id_from_header:
            metadata_from_headers["trace_id"] = trace_id_from_header
            verbose_proxy_logger.debug(f"Extracted trace_id from header: {trace_id_from_header}")

        if isinstance(data[_metadata_variable_name], dict):
            data[_metadata_variable_name].update(metadata_from_headers)
        return data

    @staticmethod
    def get_sanitized_user_information_from_key(
        user_api_key_dict: UserAPIKeyAuth,
    ) -> StandardLoggingUserAPIKeyMetadata:
        user_api_key_logged_metadata = StandardLoggingUserAPIKeyMetadata(
            user_api_key_hash=user_api_key_dict.api_key,  # just the hashed token
            user_api_key_alias=user_api_key_dict.key_alias,
            user_api_key_spend=user_api_key_dict.spend,
            user_api_key_max_budget=user_api_key_dict.max_budget,
            user_api_key_team_id=user_api_key_dict.team_id,
            user_api_key_user_id=user_api_key_dict.user_id,
            user_api_key_org_id=user_api_key_dict.org_id,
            user_api_key_team_alias=user_api_key_dict.team_alias,
            user_api_key_end_user_id=user_api_key_dict.end_user_id,
            user_api_key_user_email=user_api_key_dict.user_email,
            user_api_key_request_route=user_api_key_dict.request_route,
            user_api_key_budget_reset_at=(
                user_api_key_dict.budget_reset_at.isoformat()
                if user_api_key_dict.budget_reset_at
                else None
            ),
            user_api_key_auth_metadata=user_api_key_dict.metadata,
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
        # Add the full UserAPIKeyAuth object for MCP server access control
        data[_metadata_variable_name]["user_api_key_auth"] = user_api_key_dict
        return data

    @staticmethod
    def add_management_endpoint_metadata_to_request_metadata(
        data: dict,
        management_endpoint_metadata: dict,
        _metadata_variable_name: str,
    ) -> dict:
        """
        Adds the `UserAPIKeyAuth` metadata to the request metadata.

        ignore any sensitive fields like logging, api_key, etc.
        """
        if _metadata_variable_name not in data:
            return data
        from litellm.proxy._types import (
            LiteLLM_ManagementEndpoint_MetadataFields,
            LiteLLM_ManagementEndpoint_MetadataFields_Premium,
        )

        # ignore any special fields
        added_metadata = {}
        for k, v in management_endpoint_metadata.items():
            if k not in (
                LiteLLM_ManagementEndpoint_MetadataFields_Premium
                + LiteLLM_ManagementEndpoint_MetadataFields
            ):
                added_metadata[k] = v
        if data[_metadata_variable_name].get("user_api_key_auth_metadata") is None:
            data[_metadata_variable_name]["user_api_key_auth_metadata"] = {}
        data[_metadata_variable_name]["user_api_key_auth_metadata"].update(
            added_metadata
        )
        return data

    @staticmethod
    def add_key_level_controls(
        key_metadata: Optional[dict], data: dict, _metadata_variable_name: str
    ):
        if key_metadata is None:
            return data
        if "cache" in key_metadata:
            data["cache"] = {}
            if isinstance(key_metadata["cache"], dict):
                for k, v in key_metadata["cache"].items():
                    if k in SupportedCacheControls:
                        data["cache"][k] = v

        ## KEY-LEVEL SPEND LOGS / TAGS
        if "tags" in key_metadata and key_metadata["tags"] is not None:
            data[_metadata_variable_name][
                "tags"
            ] = LiteLLMProxyRequestSetup._merge_tags(
                request_tags=data[_metadata_variable_name].get("tags"),
                tags_to_add=key_metadata["tags"],
            )
        if "disable_global_guardrails" in key_metadata and isinstance(
            key_metadata["disable_global_guardrails"], bool
        ):
            data[_metadata_variable_name]["disable_global_guardrails"] = key_metadata[
                "disable_global_guardrails"
            ]
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

        ## KEY-LEVEL METADATA
        data = LiteLLMProxyRequestSetup.add_management_endpoint_metadata_to_request_metadata(
            data=data,
            management_endpoint_metadata=key_metadata,
            _metadata_variable_name=_metadata_variable_name,
        )
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

    _headers = clean_headers(
        request.headers,
        litellm_key_header_name=(
            general_settings.get("litellm_key_header_name")
            if general_settings is not None
            else None
        ),
    )

    ##########################################################
    # Init - Proxy Server Request
    # we do this as soon as entering so we track the original request
    ##########################################################
    # Track arrival time for queue time metric
    arrival_time = time.time()
    data["proxy_server_request"] = {
        "url": str(request.url),
        "method": request.method,
        "headers": _headers,
        "body": copy.copy(data),  # use copy instead of deepcopy
        "arrival_time": arrival_time,  # Track when request arrived at proxy
    }

    safe_add_api_version_from_query_params(data, request)
    _metadata_variable_name = _get_metadata_variable_name(request)
    if data.get(_metadata_variable_name, None) is None:
        data[_metadata_variable_name] = {}

    data.update(
        LiteLLMProxyRequestSetup.add_litellm_data_for_backend_llm_call(
            headers=_headers,
            user_api_key_dict=user_api_key_dict,
            general_settings=general_settings,
        )
    )

    data.update(
        LiteLLMProxyRequestSetup.add_litellm_metadata_from_request_headers(
            headers=_headers,
            data=data,
            _metadata_variable_name=_metadata_variable_name,
        )
    )

    # Add headers to metadata for guardrails to access (fixes #17477)
    # Guardrails use metadata["headers"] to access request headers (e.g., User-Agent)
    if _metadata_variable_name in data and isinstance(
        data[_metadata_variable_name], dict
    ):
        data[_metadata_variable_name]["headers"] = _headers

    # check for forwardable headers
    data = LiteLLMProxyRequestSetup.add_headers_to_llm_call_by_model_group(
        data=data, headers=_headers, user_api_key_dict=user_api_key_dict
    )

    user_api_key_dict = LiteLLMProxyRequestSetup.add_internal_user_from_user_mapping(
        general_settings, user_api_key_dict, _headers
    )

    # Parse user info from headers
    user = LiteLLMProxyRequestSetup.get_user_from_headers(_headers, general_settings)
    if user is not None:
        if user_api_key_dict.end_user_id is None:
            user_api_key_dict.end_user_id = user
        if "user" not in data:
            data["user"] = user

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

    # Parse litellm_metadata if it's a string (e.g., from multipart/form-data or extra_body)
    if "litellm_metadata" in data and data["litellm_metadata"] is not None:
        if isinstance(data["litellm_metadata"], str):
            parsed_litellm_metadata = safe_json_loads(data["litellm_metadata"])
            if not isinstance(parsed_litellm_metadata, dict):
                verbose_proxy_logger.warning(
                    f"Failed to parse 'litellm_metadata' as JSON dict. Received value: {data['litellm_metadata']}"
                )
            else:
                data["litellm_metadata"] = parsed_litellm_metadata
        # Merge litellm_metadata into the metadata variable (preserving existing values)
        if isinstance(data["litellm_metadata"], dict):
            for key, value in data["litellm_metadata"].items():
                if key not in data[_metadata_variable_name]:
                    data[_metadata_variable_name][key] = value

    data = LiteLLMProxyRequestSetup.add_user_api_key_auth_to_request_metadata(
        data=data,
        user_api_key_dict=user_api_key_dict,
        _metadata_variable_name=_metadata_variable_name,
    )
    data[_metadata_variable_name]["litellm_api_version"] = version

    if general_settings is not None:
        data[_metadata_variable_name][
            "global_max_parallel_requests"
        ] = general_settings.get("global_max_parallel_requests", None)

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
    if "disable_global_guardrails" in team_metadata and isinstance(
        team_metadata["disable_global_guardrails"], bool
    ):
        data[_metadata_variable_name]["disable_global_guardrails"] = team_metadata[
            "disable_global_guardrails"
        ]
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

    ## TEAM-LEVEL METADATA
    data = (
        LiteLLMProxyRequestSetup.add_management_endpoint_metadata_to_request_metadata(
            data=data,
            management_endpoint_metadata=team_metadata,
            _metadata_variable_name=_metadata_variable_name,
        )
    )

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

    # User spend, budget - used by prometheus.py
    # Follow same pattern as team and API key budgets
    data[_metadata_variable_name][
        "user_api_key_user_spend"
    ] = user_api_key_dict.user_spend
    data[_metadata_variable_name][
        "user_api_key_user_max_budget"
    ] = user_api_key_dict.user_max_budget

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
    if True:  # Always set the IP Address if available
        # logic for tracking IP Address

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

    # Add User-Agent
    user_agent = ""
    if (
        request is not None
        and hasattr(request, "headers")
        and "user-agent" in request.headers
    ):
        user_agent = request.headers["user-agent"]
    data[_metadata_variable_name]["user_agent"] = user_agent

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

    # Guardrails from key/team metadata
    move_guardrails_to_metadata(
        data=data,
        _metadata_variable_name=_metadata_variable_name,
        user_api_key_dict=user_api_key_dict,
    )

    # Guardrails from policy engine
    add_guardrails_from_policy_engine(
        data=data,
        metadata_variable_name=_metadata_variable_name,
        user_api_key_dict=user_api_key_dict,
    )

    # Team Model Aliases
    _update_model_if_team_alias_exists(
        data=data,
        user_api_key_dict=user_api_key_dict,
    )

    # Key Model Aliases
    _update_model_if_key_alias_exists(
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


def _update_model_if_key_alias_exists(
    data: dict,
    user_api_key_dict: UserAPIKeyAuth,
) -> None:
    """
    Update the model if the key alias exists

    If an alias map has been set on a key, then we want to make the request with the model the key alias is pointing to

    eg.
        - user calls `modelAlias`
        - key.aliases = {
            "modelAlias": "xai/grok-4-fast-non-reasoning"
        }
        - requested_model = "xai/grok-4-fast-non-reasoning"
    """
    _model = data.get("model")
    if (
        _model
        and user_api_key_dict.aliases
        and isinstance(user_api_key_dict.aliases, dict)
        and _model in user_api_key_dict.aliases
    ):
        data["model"] = user_api_key_dict.aliases[_model]
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
    if enforced_params and premium_user is not True:
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

    Key guardrails are set first, then team guardrails are appended (without duplicates).

    Args:
        key_metadata: The key metadata dictionary to check for guardrails
        team_metadata: The team metadata dictionary to check for guardrails
        data: The request data to update
        metadata_variable_name: The name of the metadata field in data

    """
    from litellm.proxy.utils import _premium_user_check

    # Initialize guardrails set (avoiding duplicates)
    combined_guardrails = set()

    # Add key-level guardrails first
    if key_metadata and "guardrails" in key_metadata:
        if (
            isinstance(key_metadata["guardrails"], list)
            and len(key_metadata["guardrails"]) > 0
        ):
            _premium_user_check()
            combined_guardrails.update(key_metadata["guardrails"])

    # Add team-level guardrails (set automatically handles duplicates)
    if team_metadata and "guardrails" in team_metadata:
        if (
            isinstance(team_metadata["guardrails"], list)
            and len(team_metadata["guardrails"]) > 0
        ):
            _premium_user_check()
            combined_guardrails.update(team_metadata["guardrails"])

    # Set combined guardrails in metadata as list
    if combined_guardrails:
        data[metadata_variable_name]["guardrails"] = list(combined_guardrails)


def _add_guardrails_from_policies_in_metadata(
    key_metadata: Optional[dict],
    team_metadata: Optional[dict],
    data: dict,
    metadata_variable_name: str,
) -> None:
    """
    Helper to resolve guardrails from policies attached to key/team metadata.

    This function:
    1. Gets policy names from key and team metadata
    2. Resolves guardrails from those policies (including inheritance)
    3. Adds resolved guardrails to request metadata

    Args:
        key_metadata: The key metadata dictionary to check for policies
        team_metadata: The team metadata dictionary to check for policies
        data: The request data to update
        metadata_variable_name: The name of the metadata field in data
    """
    from litellm._logging import verbose_proxy_logger
    from litellm.proxy.policy_engine.policy_registry import get_policy_registry
    from litellm.proxy.policy_engine.policy_resolver import PolicyResolver
    from litellm.proxy.utils import _premium_user_check
    from litellm.types.proxy.policy_engine import PolicyMatchContext

    # Collect policy names from key and team metadata
    policy_names: set = set()

    # Add key-level policies first
    if key_metadata and "policies" in key_metadata:
        if (
            isinstance(key_metadata["policies"], list)
            and len(key_metadata["policies"]) > 0
        ):
            _premium_user_check()
            policy_names.update(key_metadata["policies"])

    # Add team-level policies
    if team_metadata and "policies" in team_metadata:
        if (
            isinstance(team_metadata["policies"], list)
            and len(team_metadata["policies"]) > 0
        ):
            _premium_user_check()
            policy_names.update(team_metadata["policies"])

    if not policy_names:
        return

    verbose_proxy_logger.debug(
        f"Policy engine: resolving guardrails from key/team policies: {policy_names}"
    )

    # Check if policy registry is initialized
    registry = get_policy_registry()
    if not registry.is_initialized():
        verbose_proxy_logger.debug(
            "Policy engine not initialized, skipping policy resolution from metadata"
        )
        return

    # Build context for policy resolution (model from request data)
    context = PolicyMatchContext(model=data.get("model"))

    # Get all policies from registry
    all_policies = registry.get_all_policies()

    # Resolve guardrails from the specified policies
    resolved_guardrails: set = set()
    for policy_name in policy_names:
        if registry.has_policy(policy_name):
            resolved_policy = PolicyResolver.resolve_policy_guardrails(
                policy_name=policy_name,
                policies=all_policies,
                context=context,
            )
            resolved_guardrails.update(resolved_policy.guardrails)
            verbose_proxy_logger.debug(
                f"Policy engine: resolved guardrails from policy '{policy_name}': {resolved_policy.guardrails}"
            )
        else:
            verbose_proxy_logger.warning(
                f"Policy engine: policy '{policy_name}' not found in registry"
            )

    if not resolved_guardrails:
        return

    # Add resolved guardrails to request metadata
    if metadata_variable_name not in data:
        data[metadata_variable_name] = {}

    existing_guardrails = data[metadata_variable_name].get("guardrails", [])
    if not isinstance(existing_guardrails, list):
        existing_guardrails = []

    # Combine existing guardrails with policy-resolved guardrails (no duplicates)
    combined = set(existing_guardrails)
    combined.update(resolved_guardrails)
    data[metadata_variable_name]["guardrails"] = list(combined)

    # Store applied policies in metadata for tracking
    if "applied_policies" not in data[metadata_variable_name]:
        data[metadata_variable_name]["applied_policies"] = []
    data[metadata_variable_name]["applied_policies"].extend(list(policy_names))

    verbose_proxy_logger.debug(
        f"Policy engine: added guardrails from key/team policies to request metadata: {list(resolved_guardrails)}"
    )


def move_guardrails_to_metadata(
    data: dict,
    _metadata_variable_name: str,
    user_api_key_dict: UserAPIKeyAuth,
):
    """
    Helper to add guardrails from request to metadata

    - If guardrails set on API Key metadata then sets guardrails on request metadata
    - If guardrails not set on API key, then checks request metadata
    - Adds guardrails from policies attached to key/team metadata
    - Adds guardrails from policy engine based on team/key/model context
    """
    # Check key-level guardrails
    _add_guardrails_from_key_or_team_metadata(
        key_metadata=user_api_key_dict.metadata,
        team_metadata=user_api_key_dict.team_metadata,
        data=data,
        metadata_variable_name=_metadata_variable_name,
    )

    #########################################################################################
    # Add guardrails from policies attached to key/team metadata
    #########################################################################################
    _add_guardrails_from_policies_in_metadata(
        key_metadata=user_api_key_dict.metadata,
        team_metadata=user_api_key_dict.team_metadata,
        data=data,
        metadata_variable_name=_metadata_variable_name,
    )

    #########################################################################################
    # Add guardrails from policy engine based on team/key/model context
    #########################################################################################
    add_guardrails_from_policy_engine(
        data=data,
        metadata_variable_name=_metadata_variable_name,
        user_api_key_dict=user_api_key_dict,
    )

    #########################################################################################
    # User's might send "guardrails" in the request body, we need to add them to the request metadata.
    # Since downstream logic requires "guardrails" to be in the request metadata
    #########################################################################################
    if "guardrails" in data:
        request_body_guardrails = data.pop("guardrails")
        if "guardrails" in data[_metadata_variable_name] and isinstance(
            data[_metadata_variable_name]["guardrails"], list
        ):
            data[_metadata_variable_name]["guardrails"].extend(request_body_guardrails)
        else:
            data[_metadata_variable_name]["guardrails"] = request_body_guardrails

    #########################################################################################
    if "guardrail_config" in data:
        request_body_guardrail_config = data.pop("guardrail_config")
        if "guardrail_config" in data[_metadata_variable_name] and isinstance(
            data[_metadata_variable_name]["guardrail_config"], dict
        ):
            data[_metadata_variable_name]["guardrail_config"].update(
                request_body_guardrail_config
            )
        else:
            data[_metadata_variable_name][
                "guardrail_config"
            ] = request_body_guardrail_config


def _match_and_track_policies(
    data: dict,
    context: "PolicyMatchContext",
    request_body_policies: Any,
) -> tuple[list[str], dict[str, str]]:
    """
    Match policies via attachments and request body, track them in metadata.

    Returns:
        Tuple of (applied_policy_names, policy_reasons)
    """
    from litellm._logging import verbose_proxy_logger
    from litellm.proxy.common_utils.callback_utils import (
        add_policy_sources_to_metadata,
        add_policy_to_applied_policies_header,
    )
    from litellm.proxy.policy_engine.attachment_registry import (
        get_attachment_registry,
    )
    from litellm.proxy.policy_engine.policy_matcher import PolicyMatcher

    # Get matching policies via attachments (with match reasons for attribution)
    attachment_registry = get_attachment_registry()
    matches_with_reasons = attachment_registry.get_attached_policies_with_reasons(
        context
    )
    matching_policy_names = [m["policy_name"] for m in matches_with_reasons]
    policy_reasons = {m["policy_name"]: m["matched_via"] for m in matches_with_reasons}

    verbose_proxy_logger.debug(
        f"Policy engine: matched policies via attachments: {matching_policy_names}"
    )

    # Combine attachment-based policies with dynamic request body policies
    all_policy_names = set(matching_policy_names)
    if request_body_policies and isinstance(request_body_policies, list):
        all_policy_names.update(request_body_policies)
        verbose_proxy_logger.debug(
            f"Policy engine: added dynamic policies from request body: {request_body_policies}"
        )

    if not all_policy_names:
        return [], {}

    # Filter to only policies whose conditions match the context
    applied_policy_names = PolicyMatcher.get_policies_with_matching_conditions(
        policy_names=list(all_policy_names),
        context=context,
    )

    verbose_proxy_logger.debug(
        f"Policy engine: applied policies (conditions matched): {applied_policy_names}"
    )

    # Track applied policies in metadata for response headers
    for policy_name in applied_policy_names:
        add_policy_to_applied_policies_header(
            request_data=data, policy_name=policy_name
        )

    # Track policy attribution sources for x-litellm-policy-sources header
    applied_reasons = {
        name: policy_reasons[name]
        for name in applied_policy_names
        if name in policy_reasons
    }
    add_policy_sources_to_metadata(
        request_data=data, policy_sources=applied_reasons
    )

    return applied_policy_names, policy_reasons


def _apply_resolved_guardrails_to_metadata(
    data: dict,
    metadata_variable_name: str,
    context: "PolicyMatchContext",
) -> None:
    """Apply resolved guardrails and pipelines to request metadata."""
    from litellm._logging import verbose_proxy_logger
    from litellm.proxy.policy_engine.policy_resolver import PolicyResolver

    # Resolve guardrails from matching policies
    resolved_guardrails = PolicyResolver.resolve_guardrails_for_context(context=context)

    verbose_proxy_logger.debug(
        f"Policy engine: resolved guardrails: {resolved_guardrails}"
    )

    # Resolve pipelines from matching policies
    pipelines = PolicyResolver.resolve_pipelines_for_context(context=context)

    # Add resolved guardrails to request metadata
    if metadata_variable_name not in data:
        data[metadata_variable_name] = {}

    # Track pipeline-managed guardrails to exclude from independent execution
    pipeline_managed_guardrails: set = set()
    if pipelines:
        pipeline_managed_guardrails = PolicyResolver.get_pipeline_managed_guardrails(
            pipelines
        )
        data[metadata_variable_name]["_guardrail_pipelines"] = pipelines
        data[metadata_variable_name]["_pipeline_managed_guardrails"] = (
            pipeline_managed_guardrails
        )
        verbose_proxy_logger.debug(
            f"Policy engine: resolved {len(pipelines)} pipeline(s), "
            f"managed guardrails: {pipeline_managed_guardrails}"
        )

    if not resolved_guardrails and not pipelines:
        return

    existing_guardrails = data[metadata_variable_name].get("guardrails", [])
    if not isinstance(existing_guardrails, list):
        existing_guardrails = []

    # Combine existing guardrails with policy-resolved guardrails (no duplicates)
    # Exclude pipeline-managed guardrails from the flat list
    combined = set(existing_guardrails)
    combined.update(resolved_guardrails)
    combined -= pipeline_managed_guardrails
    data[metadata_variable_name]["guardrails"] = list(combined)

    verbose_proxy_logger.debug(
        f"Policy engine: added guardrails to request metadata: {list(combined)}"
    )


def add_guardrails_from_policy_engine(
    data: dict,
    metadata_variable_name: str,
    user_api_key_dict: UserAPIKeyAuth,
) -> None:
    """
    Add guardrails from the policy engine based on request context.

    This function:
    1. Extracts "policies" from request body (if present) for dynamic policy application
    2. Gets matching policies based on team_alias, key_alias, and model (via attachments)
    3. Combines dynamic policies with attachment-based policies
    4. Resolves guardrails from all policies (including inheritance)
    5. Adds guardrails to request metadata
    6. Tracks applied policies in metadata for response headers
    7. Removes "policies" from request body so it's not forwarded to LLM provider

    Args:
        data: The request data to update
        metadata_variable_name: The name of the metadata field in data
        user_api_key_dict: The user's API key authentication info
    """
    from litellm._logging import verbose_proxy_logger
    from litellm.proxy.common_utils.http_parsing_utils import (
        get_tags_from_request_body,
    )
    from litellm.proxy.policy_engine.policy_registry import get_policy_registry
    from litellm.types.proxy.policy_engine import PolicyMatchContext

    # Extract dynamic policies from request body (if present)
    request_body_policies = data.pop("policies", None)

    registry = get_policy_registry()
    verbose_proxy_logger.debug(
        f"Policy engine: registry initialized={registry.is_initialized()}, "
        f"policy_count={len(registry.get_all_policies())}"
    )
    if not registry.is_initialized():
        verbose_proxy_logger.debug(
            "Policy engine not initialized, skipping policy matching"
        )
        return

    # Extract tags and build context
    all_tags = get_tags_from_request_body(data) or None
    context = PolicyMatchContext(
        team_alias=user_api_key_dict.team_alias,
        key_alias=user_api_key_dict.key_alias,
        model=data.get("model"),
        tags=all_tags,
    )

    verbose_proxy_logger.debug(
        f"Policy engine: matching policies for context team_alias={context.team_alias}, "
        f"key_alias={context.key_alias}, model={context.model}, tags={context.tags}"
    )

    # Match and track policies based on attachments and request body
    _match_and_track_policies(data, context, request_body_policies)

    # Always resolve and apply guardrails, even if no policies matched above.
    # PolicyResolver does its own independent matching and inheritance resolution,
    # so guardrails can still be applied via inherited parent policies.
    _apply_resolved_guardrails_to_metadata(data, metadata_variable_name, context)


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
        # Anthropic headers work across multiple providers
        # Store as comma-separated list so retrieval can match any of them
        data["provider_specific_header"] = ProviderSpecificHeader(
            custom_llm_provider=f"{LlmProviders.ANTHROPIC.value},{LlmProviders.BEDROCK.value},{LlmProviders.VERTEX_AI.value}",
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
