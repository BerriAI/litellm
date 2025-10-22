"""
Router utilities for Search API integration.

Handles search tool selection, load balancing, and fallback logic for search requests.
"""

import asyncio
import random
import traceback
from functools import partial
from typing import Any, Callable

from litellm._logging import verbose_router_logger


class SearchAPIRouter:
    """
    Static utility class for routing search API calls through the LiteLLM router.
    
    Provides methods for search tool selection, load balancing, and fallback handling.
    """

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
            tool for tool in router_instance.search_tools
            if tool.get("search_tool_name") == search_tool_name
        ]
        
        if not matching_tools:
            raise ValueError(f"Search tool '{search_tool_name}' not found in router.search_tools")
        
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
                raise ValueError("search_tool_name or model parameter is required for search")
            
            # Set up kwargs for the fallback system
            kwargs["model"] = search_tool_name  # Use model field for compatibility with fallback system
            kwargs["original_generic_function"] = original_function
            # Bind router_instance to the helper method using partial
            kwargs["original_function"] = partial(
                SearchAPIRouter.async_search_with_fallbacks_helper,
                router_instance=router_instance,
            )
            
            # Update kwargs before fallbacks (for logging, metadata, etc)
            router_instance._update_kwargs_before_fallbacks(
                model=search_tool_name, kwargs=kwargs, metadata_variable_name="litellm_metadata"
            )
            
            verbose_router_logger.debug(
                f"Inside SearchAPIRouter.async_search_with_fallbacks() - search_tool_name: {search_tool_name}; kwargs: {kwargs}"
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
            api_key = litellm_params.get("api_key")
            api_base = litellm_params.get("api_base")
            
            if not search_provider:
                raise ValueError(f"search_provider not found in litellm_params for search tool '{search_tool_name}'")
            
            verbose_router_logger.debug(
                f"Selected search tool with provider: {search_provider}"
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

