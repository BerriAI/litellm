"""
Hook to intercept Anthropic web_search tool calls and route them through LiteLLM's search API.

When Claude Code or other Anthropic clients use the web_search tool on the /v1/messages endpoint,
this hook intercepts the request and uses litellm.search (via router.search) to perform the search,
then injects the results back into the request similar to prompt caching injection.

This allows users to:
1. Use their own search providers (Perplexity, Google PSE, etc.) instead of Anthropic's web search
2. Route web search through LiteLLM's load balancing and fallback logic
3. Control costs and providers for web search independently
"""

import copy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.llms.anthropic import (
    ANTHROPIC_HOSTED_TOOLS,
    AnthropicWebSearchTool,
)
from litellm.types.utils import CallTypes

if TYPE_CHECKING:
    from litellm.router import Router
else:
    Router = Any


class AnthropicWebSearchHook(CustomLogger):
    """
    Hook to intercept Anthropic web_search tool and route through LiteLLM search API.
    
    This hook:
    1. Detects when web_search tool is present in the tools list
    2. Removes the web_search tool from the request (since we'll handle it ourselves)
    3. When the model requests a web search, performs the search via router.search
    4. Injects the search results into the messages
    
    Usage:
        Set `litellm_settings.web_search_routing: true` in your proxy config to enable.
        Configure search tools in the router to specify which search provider to use.
    """

    def __init__(
        self,
        llm_router: Optional[Router] = None,
        search_tool_name: Optional[str] = None,
    ):
        """
        Initialize the hook.
        
        Args:
            llm_router: The LiteLLM router instance for making search calls
            search_tool_name: Name of the search tool configured in the router
        """
        super().__init__()
        self.llm_router = llm_router
        self.search_tool_name = search_tool_name or "default_search"

    def set_router(self, llm_router: Router):
        """Set the router instance."""
        self.llm_router = llm_router

    def set_search_tool_name(self, search_tool_name: str):
        """Set the search tool name to use."""
        self.search_tool_name = search_tool_name

    @staticmethod
    def _detect_web_search_tool(
        tools: Optional[List[Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        """
        Detect if web_search tool is present in the tools list.
        
        Args:
            tools: List of tools from the request
            
        Returns:
            The web_search tool config if found, None otherwise
        """
        if not tools:
            return None

        for tool in tools:
            if isinstance(tool, dict):
                tool_type = tool.get("type", "")
                # Check for Anthropic web_search tool types
                if tool_type.startswith(ANTHROPIC_HOSTED_TOOLS.WEB_SEARCH.value):
                    return tool
        return None

    @staticmethod
    def _remove_web_search_tool(
        tools: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Remove web_search tool from tools list and return both.
        
        Args:
            tools: List of tools from the request
            
        Returns:
            Tuple of (tools without web_search, web_search tool config)
        """
        web_search_tool = None
        filtered_tools = []

        for tool in tools:
            if isinstance(tool, dict):
                tool_type = tool.get("type", "")
                if tool_type.startswith(ANTHROPIC_HOSTED_TOOLS.WEB_SEARCH.value):
                    web_search_tool = tool
                else:
                    filtered_tools.append(tool)
            else:
                filtered_tools.append(tool)

        return filtered_tools, web_search_tool

    @staticmethod
    def _extract_search_params_from_tool(
        web_search_tool: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract search parameters from the web_search tool configuration.
        
        Args:
            web_search_tool: The web_search tool configuration
            
        Returns:
            Dict with search parameters (max_results, user_location, etc.)
        """
        params: Dict[str, Any] = {}

        # Extract max_uses -> max_results
        max_uses = web_search_tool.get("max_uses")
        if max_uses is not None:
            params["max_results"] = max_uses

        # Extract user_location for potential country filtering
        user_location = web_search_tool.get("user_location")
        if user_location:
            country = user_location.get("country")
            if country:
                params["country"] = country

        return params

    async def _perform_search(
        self,
        query: str,
        search_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Perform a search using the router's search functionality.
        
        Args:
            query: The search query
            search_params: Optional search parameters
            
        Returns:
            Search results in a format suitable for injection, or None on failure
        """
        if self.llm_router is None:
            verbose_logger.warning(
                "AnthropicWebSearchHook: No router configured, cannot perform search"
            )
            return None

        try:
            # Use the router's asearch method
            search_response = await self.llm_router.asearch(
                query=query,
                search_tool_name=self.search_tool_name,
                **(search_params or {}),
            )

            # Convert SearchResponse to a format suitable for message injection
            results = []
            for result in search_response.results:
                result_dict = {
                    "title": result.title,
                    "url": result.url,
                    "snippet": result.snippet,
                }
                if result.date:
                    result_dict["date"] = result.date
                results.append(result_dict)

            return {
                "type": "web_search_results",
                "results": results,
            }

        except Exception as e:
            verbose_logger.error(
                f"AnthropicWebSearchHook: Error performing search: {str(e)}"
            )
            return None

    @staticmethod
    def _format_search_results_for_injection(
        search_results: Dict[str, Any]
    ) -> str:
        """
        Format search results as a string for injection into messages.
        
        Args:
            search_results: The search results dict
            
        Returns:
            Formatted string with search results
        """
        results = search_results.get("results", [])
        if not results:
            return "No search results found."

        formatted_parts = ["Web Search Results:\n"]
        for i, result in enumerate(results, 1):
            formatted_parts.append(f"\n{i}. {result.get('title', 'Untitled')}")
            formatted_parts.append(f"   URL: {result.get('url', '')}")
            formatted_parts.append(f"   {result.get('snippet', '')}")
            if result.get("date"):
                formatted_parts.append(f"   Date: {result.get('date')}")

        return "\n".join(formatted_parts)

    async def async_pre_call_deployment_hook(
        self, kwargs: Dict[str, Any], call_type: Optional[CallTypes]
    ) -> Optional[dict]:
        """
        Pre-call hook to detect and handle web_search tool routing.
        
        This hook:
        1. Only runs for anthropic_messages call type
        2. Detects web_search tool in the request
        3. Stores the web_search config for later use
        4. Removes web_search from tools (we'll handle it via router.search)
        
        Args:
            kwargs: Request kwargs
            call_type: The type of call being made
            
        Returns:
            Modified kwargs if web_search was detected, None otherwise
        """
        # Only process anthropic_messages calls
        if call_type != CallTypes.anthropic_messages:
            return None

        tools = kwargs.get("tools")
        if not tools:
            return None

        # Check if web_search tool is present
        web_search_tool = self._detect_web_search_tool(tools)
        if web_search_tool is None:
            return None

        # Check if we should route web search (only for non-anthropic providers)
        # For native Anthropic, let the server handle web search
        custom_llm_provider = kwargs.get("custom_llm_provider", "")
        if custom_llm_provider == "anthropic":
            verbose_logger.debug(
                "AnthropicWebSearchHook: Skipping for native Anthropic provider"
            )
            return None

        verbose_logger.debug(
            f"AnthropicWebSearchHook: Detected web_search tool for provider {custom_llm_provider}"
        )

        # Create a copy of kwargs to modify
        modified_kwargs = copy.deepcopy(kwargs)

        # Remove web_search tool and store config
        filtered_tools, web_search_config = self._remove_web_search_tool(
            modified_kwargs["tools"]
        )
        modified_kwargs["tools"] = filtered_tools if filtered_tools else None

        # Store web_search config for potential use in response handling
        modified_kwargs["_web_search_config"] = web_search_config
        modified_kwargs["_web_search_params"] = self._extract_search_params_from_tool(
            web_search_config
        )

        verbose_logger.debug(
            f"AnthropicWebSearchHook: Removed web_search tool, {len(filtered_tools)} tools remaining"
        )

        return modified_kwargs

    @staticmethod
    def should_use_web_search_hook(
        kwargs: Dict[str, Any],
        llm_router: Optional[Router] = None,
    ) -> bool:
        """
        Check if the web search hook should be used for this request.
        
        Args:
            kwargs: Request kwargs
            llm_router: The router instance
            
        Returns:
            True if web search hook should be used
        """
        # Check if router has search tools configured
        if llm_router is None:
            return False

        search_tools = getattr(llm_router, "search_tools", None)
        if not search_tools:
            return False

        # Check if web_search tool is in the request
        tools = kwargs.get("tools")
        if not tools:
            return False

        web_search_tool = AnthropicWebSearchHook._detect_web_search_tool(tools)
        if web_search_tool is None:
            return False

        # Check if this is a non-anthropic provider
        custom_llm_provider = kwargs.get("custom_llm_provider", "")
        if custom_llm_provider == "anthropic":
            return False

        return True

    @staticmethod
    def get_web_search_hook_instance(
        llm_router: Optional[Router] = None,
        search_tool_name: Optional[str] = None,
    ) -> Optional["AnthropicWebSearchHook"]:
        """
        Get a configured web search hook instance.
        
        Args:
            llm_router: The router instance
            search_tool_name: Name of the search tool to use
            
        Returns:
            Configured hook instance or None
        """
        from litellm.litellm_core_utils.litellm_logging import (
            _init_custom_logger_compatible_class,
        )

        hook = _init_custom_logger_compatible_class(
            logging_integration="anthropic_web_search_hook",
            internal_usage_cache=None,
            llm_router=llm_router,
        )

        if hook and isinstance(hook, AnthropicWebSearchHook):
            if llm_router:
                hook.set_router(llm_router)
            if search_tool_name:
                hook.set_search_tool_name(search_tool_name)
            return hook

        return None
