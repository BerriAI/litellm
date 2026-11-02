"""
Router utilities for Search API integration.

Handles search tool selection, load balancing, and fallback logic for search requests.
"""

import asyncio
import traceback
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_router_logger
from litellm.router_api import ChangeSet
from litellm.router_deprecated import renamed_router_param
from litellm.router_endpoint_table import search_tool_call
from litellm.types.router import SearchToolLiteLLMParams, SearchToolTypedDict

if TYPE_CHECKING:
    from litellm.router_api import RoutingContext


class SearchAPIRouter:
    """
    Static utility class for routing search API calls through the LiteLLM router.

    Provides methods for search tool selection, load balancing, and fallback handling.
    """

    @staticmethod
    def _resolve_search_provider_credentials(
        *,
        tool_litellm_params: SearchToolLiteLLMParams,
    ) -> tuple[str | None, str | None]:
        """
        Resolve search provider credentials from tool configuration ONLY.

        Credentials are stored only in search_tool.litellm_params, never in team/key metadata.
        This ensures secrets are not exposed in team/key API responses.

        Args:
            tool_litellm_params: Search tool litellm_params with credentials

        Returns:
            Tuple of (api_key, api_base) from tool configuration
        """
        resolved_api_key: Final[str | None] = tool_litellm_params.get("api_key")
        resolved_api_base: Final[str | None] = tool_litellm_params.get("api_base")

        return resolved_api_key, resolved_api_base

    @staticmethod
    @renamed_router_param("router_instance")
    async def update_router_search_tools(ctx: "RoutingContext", search_tools: list):
        """
        Update the router with search tools from the database.

        This method is called by a cron job to sync search tools from DB to router.

        Args:
            ctx: The routing context of the router to update
            search_tools: List of search tool configurations from the database
        """
        try:
            verbose_router_logger.debug("Adding %s search tools to router", len(search_tools))

            # Convert search tools to the format expected by the router
            router_search_tools: Final[list] = []
            for tool in search_tools:
                # Create dict that matches SearchToolTypedDict structure
                router_search_tool: SearchToolTypedDict = {
                    "search_tool_id": tool.get("search_tool_id"),
                    "search_tool_name": tool.get("search_tool_name"),
                    "litellm_params": tool.get("litellm_params", {}),
                    "search_tool_info": tool.get("search_tool_info"),
                }
                router_search_tools.append(router_search_tool)

            # Update the router's search_tools list
            ctx.apply(ChangeSet(assign={"search_tools": router_search_tools}))

            verbose_router_logger.info("Successfully updated router with %s search tool(s)", len(router_search_tools))

        except Exception as e:
            verbose_router_logger.exception("Error updating router with search tools: %s", e)
            raise e

    @staticmethod
    @renamed_router_param("router_instance")
    def get_matching_search_tools(
        ctx: "RoutingContext",
        search_tool_name: str,
    ) -> list:
        """
        Get all search tools matching the given name.

        Args:
            ctx: The routing context of the router
            search_tool_name: Name of the search tool to find

        Returns:
            List of matching search tool configurations

        Raises:
            ValueError: If no matching search tools are found
        """
        matching_tools: Final = [
            tool for tool in ctx.settings().search_tools if tool.get("search_tool_name") == search_tool_name
        ]

        if not matching_tools:
            raise ValueError(f"Search tool '{search_tool_name}' not found in router.search_tools")

        return matching_tools

    @staticmethod
    @renamed_router_param("router_instance", new="litellm_routing_ctx")
    async def async_search_with_fallbacks(
        litellm_routing_ctx: "RoutingContext",
        original_function: Callable,
        **kwargs,
    ):
        """
        Helper function to make a search API call through the router with load balancing and fallbacks.
        Reuses the router's retry/fallback infrastructure.

        Args:
            litellm_routing_ctx: The routing context of the router
            original_function: The original litellm.asearch function
            **kwargs: Search parameters including search_tool_name, query, etc.

        Returns:
            SearchResponse from the search API
        """
        ctx: Final = litellm_routing_ctx
        try:
            search_tool_name: Final = kwargs.get("search_tool_name", kwargs.get("model"))

            if not search_tool_name:
                raise ValueError("search_tool_name or model parameter is required for search")

            # Set up kwargs for the fallback system
            kwargs["model"] = search_tool_name  # Use model field for compatibility with fallback system
            kwargs["original_generic_function"] = original_function
            # Bind ctx to the helper method using partial
            kwargs["original_function"] = partial(
                SearchAPIRouter.async_search_with_fallbacks_helper,
                litellm_routing_ctx=ctx,
            )

            # Update kwargs before fallbacks (for logging, metadata, etc)
            ctx.update_kwargs_before_fallbacks(
                model=search_tool_name,
                kwargs=kwargs,
                metadata_variable_name="litellm_metadata",
            )

            available_search_tool_names: Final = [tool.get("search_tool_name") for tool in ctx.settings().search_tools]
            verbose_router_logger.debug(
                "Inside SearchAPIRouter.async_search_with_fallbacks() - search_tool_name: %s, Available Search Tools: %s",
                search_tool_name,
                available_search_tool_names,
            )

            # Use the existing retry/fallback infrastructure
            response: Final = await ctx.with_fallbacks(**kwargs)
            return response

        except Exception as e:
            from litellm.router_utils.handle_error import send_llm_exception_alert

            asyncio.create_task(
                send_llm_exception_alert(
                    ctx=ctx,
                    request_kwargs=kwargs,
                    error_traceback_str=traceback.format_exc(),
                    original_exception=e,
                )
            )
            raise e

    @staticmethod
    @renamed_router_param("router_instance", new="litellm_routing_ctx")
    async def async_search_with_fallbacks_helper(
        litellm_routing_ctx: "RoutingContext",
        model: str,
        original_generic_function: Callable,
        **kwargs,
    ):
        """
        Helper function for search API calls - selects a search tool and calls the original function.
        Called by async_function_with_fallbacks for each retry attempt.

        Args:
            litellm_routing_ctx: The routing context of the router
            model: The search tool name (passed as model for compatibility)
            original_generic_function: The original litellm.asearch function
            **kwargs: Search parameters

        Returns:
            SearchResponse from the selected search provider
        """
        ctx: Final = litellm_routing_ctx
        search_tool_name: Final = model  # model field contains the search_tool_name

        try:
            call: Final = search_tool_call(ctx.settings().search_tools, search_tool_name, kwargs)
            search_provider: Final = call["search_provider"]

            verbose_router_logger.debug("Selected search tool with provider: %s", search_provider)

            # Call the original search function with the provider config
            response: Final = await original_generic_function(**call)

            return response

        except Exception as e:
            verbose_router_logger.error(
                "Error in SearchAPIRouter.async_search_with_fallbacks_helper for %s: %s", search_tool_name, e
            )
            raise e
