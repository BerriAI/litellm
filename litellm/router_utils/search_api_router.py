"""
Router utilities for Search API integration.

Handles search tool selection, load balancing, and fallback logic for search requests.
"""

import asyncio
import random
import traceback
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple

import litellm
from litellm._logging import verbose_router_logger


class SearchAPIRouter:
    """
    Static utility class for routing search API calls through the LiteLLM router.

    Provides methods for search tool selection, load balancing, and fallback handling.
    """

    @staticmethod
    def _get_team_config_from_default_settings(
        team_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve team config from litellm.default_team_settings.

        This allows search requests to read per-team settings from proxy config
        (YAML) similar to completion paths that use ProxyConfig.load_team_config().
        """
        if not team_id:
            return None

        default_team_settings = getattr(litellm, "default_team_settings", None)
        if not isinstance(default_team_settings, list):
            return None

        for team_setting in default_team_settings:
            if (
                isinstance(team_setting, dict)
                and team_setting.get("team_id") == team_id
            ):
                return team_setting
        return None

    @staticmethod
    def _resolve_search_provider_credentials(
        *,
        search_provider: str,
        tool_litellm_params: Dict[str, Any],
        request_metadata: Optional[Dict[str, Any]] = None,
        team_metadata: Optional[Dict[str, Any]] = None,
        team_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Resolve search provider credentials with precedence:
        1. request metadata.search_provider_config.{provider}
        2. team metadata.search_provider_config.{provider}
        3. default_team_settings.search_provider_config.{provider}
        4. search_tool.litellm_params
        5. env fallback in provider validate_environment()
        """
        resolved_api_key: Optional[str] = None
        resolved_api_base: Optional[str] = None

        request_provider_config = {}
        if isinstance(request_metadata, dict):
            search_provider_config = request_metadata.get("search_provider_config")
            if isinstance(search_provider_config, dict):
                request_provider_config = search_provider_config.get(
                    search_provider, {}
                )

        team_provider_config = {}
        if isinstance(team_metadata, dict):
            search_provider_config = team_metadata.get("search_provider_config")
            if isinstance(search_provider_config, dict):
                team_provider_config = search_provider_config.get(search_provider, {})

        team_settings_provider_config = {}
        if isinstance(team_config, dict):
            search_provider_config = team_config.get("search_provider_config")
            if isinstance(search_provider_config, dict):
                team_settings_provider_config = search_provider_config.get(
                    search_provider, {}
                )

        if isinstance(request_provider_config, dict):
            resolved_api_key = request_provider_config.get("api_key")
            resolved_api_base = request_provider_config.get("api_base")

        if resolved_api_key is None and isinstance(team_provider_config, dict):
            resolved_api_key = team_provider_config.get("api_key")
        if resolved_api_base is None and isinstance(team_provider_config, dict):
            resolved_api_base = team_provider_config.get("api_base")

        if resolved_api_key is None and isinstance(team_settings_provider_config, dict):
            resolved_api_key = team_settings_provider_config.get("api_key")
        if resolved_api_base is None and isinstance(
            team_settings_provider_config, dict
        ):
            resolved_api_base = team_settings_provider_config.get("api_base")

        if resolved_api_key is None:
            resolved_api_key = tool_litellm_params.get("api_key")
        if resolved_api_base is None:
            resolved_api_base = tool_litellm_params.get("api_base")

        return resolved_api_key, resolved_api_base

    @staticmethod
    async def update_router_search_tools(router_instance: Any, search_tools: list):
        """
        Update the router with search tools from the database.

        This method is called by a cron job to sync search tools from DB to router.

        Args:
            router_instance: The Router instance to update
            search_tools: List of search tool configurations from the database
        """
        try:
            from litellm.types.router import SearchToolTypedDict

            verbose_router_logger.debug(
                f"Adding {len(search_tools)} search tools to router"
            )

            # Convert search tools to the format expected by the router
            router_search_tools: list = []
            for tool in search_tools:
                # Create dict that matches SearchToolTypedDict structure
                router_search_tool: SearchToolTypedDict = {  # type: ignore
                    "search_tool_id": tool.get("search_tool_id"),
                    "search_tool_name": tool.get("search_tool_name"),
                    "litellm_params": tool.get("litellm_params", {}),
                    "search_tool_info": tool.get("search_tool_info"),
                }
                router_search_tools.append(router_search_tool)

            # Update the router's search_tools list
            router_instance.search_tools = router_search_tools

            verbose_router_logger.info(
                f"Successfully updated router with {len(router_search_tools)} search tool(s)"
            )

        except Exception as e:
            verbose_router_logger.exception(
                f"Error updating router with search tools: {str(e)}"
            )
            raise e

    @staticmethod
    def get_matching_search_tools(
        router_instance: Any,
        search_tool_name: str,
    ) -> list:
        """
        Get all search tools matching the given name.

        Args:
            router_instance: The Router instance
            search_tool_name: Name of the search tool to find

        Returns:
            List of matching search tool configurations

        Raises:
            ValueError: If no matching search tools are found
        """
        matching_tools = [
            tool
            for tool in router_instance.search_tools
            if tool.get("search_tool_name") == search_tool_name
        ]

        if not matching_tools:
            raise ValueError(
                f"Search tool '{search_tool_name}' not found in router.search_tools"
            )

        return matching_tools

    @staticmethod
    async def async_search_with_fallbacks(
        router_instance: Any,
        original_function: Callable,
        **kwargs,
    ):
        """
        Helper function to make a search API call through the router with load balancing and fallbacks.
        Reuses the router's retry/fallback infrastructure.

        Args:
            router_instance: The Router instance
            original_function: The original litellm.asearch function
            **kwargs: Search parameters including search_tool_name, query, etc.

        Returns:
            SearchResponse from the search API
        """
        try:
            search_tool_name = kwargs.get("search_tool_name", kwargs.get("model"))

            if not search_tool_name:
                raise ValueError(
                    "search_tool_name or model parameter is required for search"
                )

            # Set up kwargs for the fallback system
            kwargs["model"] = (
                search_tool_name  # Use model field for compatibility with fallback system
            )
            kwargs["original_generic_function"] = original_function
            # Bind router_instance to the helper method using partial
            kwargs["original_function"] = partial(
                SearchAPIRouter.async_search_with_fallbacks_helper,
                router_instance=router_instance,
            )

            # Update kwargs before fallbacks (for logging, metadata, etc)
            router_instance._update_kwargs_before_fallbacks(
                model=search_tool_name,
                kwargs=kwargs,
                metadata_variable_name="litellm_metadata",
            )

            available_search_tool_names = [
                tool.get("search_tool_name") for tool in router_instance.search_tools
            ]
            verbose_router_logger.debug(
                f"Inside SearchAPIRouter.async_search_with_fallbacks() - search_tool_name: {search_tool_name}, Available Search Tools: {available_search_tool_names}, kwargs: {kwargs}"
            )

            # Use the existing retry/fallback infrastructure
            response = await router_instance.async_function_with_fallbacks(**kwargs)
            return response

        except Exception as e:
            from litellm.router_utils.handle_error import send_llm_exception_alert

            asyncio.create_task(
                send_llm_exception_alert(
                    litellm_router_instance=router_instance,
                    request_kwargs=kwargs,
                    error_traceback_str=traceback.format_exc(),
                    original_exception=e,
                )
            )
            raise e

    @staticmethod
    async def async_search_with_fallbacks_helper(
        router_instance: Any,
        model: str,
        original_generic_function: Callable,
        **kwargs,
    ):
        """
        Helper function for search API calls - selects a search tool and calls the original function.
        Called by async_function_with_fallbacks for each retry attempt.

        Args:
            router_instance: The Router instance
            model: The search tool name (passed as model for compatibility)
            original_generic_function: The original litellm.asearch function
            **kwargs: Search parameters

        Returns:
            SearchResponse from the selected search provider
        """
        search_tool_name = model  # model field contains the search_tool_name

        try:
            # Find matching search tools
            matching_tools = SearchAPIRouter.get_matching_search_tools(
                router_instance=router_instance,
                search_tool_name=search_tool_name,
            )

            # Simple random selection for load balancing across multiple providers with same name
            # For search tools, we use simple random choice since they don't have TPM/RPM constraints
            selected_tool = random.choice(matching_tools)

            # Extract search provider and other params from litellm_params
            litellm_params = selected_tool.get("litellm_params", {})
            search_provider = litellm_params.get("search_provider")
            if not search_provider:
                raise ValueError(
                    f"search_provider not found in litellm_params for search tool '{search_tool_name}'"
                )

            request_metadata = kwargs.get("metadata")
            litellm_metadata = kwargs.get("litellm_metadata")
            if not isinstance(request_metadata, dict) and isinstance(
                litellm_metadata, dict
            ):
                request_metadata = litellm_metadata

            team_metadata = {}
            team_id: Optional[str] = None
            if isinstance(request_metadata, dict):
                _team_metadata = request_metadata.get("user_api_key_team_metadata")
                if isinstance(_team_metadata, dict):
                    team_metadata = _team_metadata
                _team_id = request_metadata.get("user_api_key_team_id")
                if isinstance(_team_id, str):
                    team_id = _team_id

            team_config = SearchAPIRouter._get_team_config_from_default_settings(
                team_id=team_id
            )

            api_key, api_base = SearchAPIRouter._resolve_search_provider_credentials(
                search_provider=search_provider,
                tool_litellm_params=litellm_params,
                request_metadata=request_metadata,
                team_metadata=team_metadata,
                team_config=team_config,
            )

            verbose_router_logger.debug(
                f"Selected search tool with provider: {search_provider}, team_id={team_id}"
            )

            # Call the original search function with the provider config
            response = await original_generic_function(
                search_provider=search_provider,
                api_key=api_key,
                api_base=api_base,
                **kwargs,
            )

            return response

        except Exception as e:
            verbose_router_logger.error(
                f"Error in SearchAPIRouter.async_search_with_fallbacks_helper for {search_tool_name}: {str(e)}"
            )
            raise e
