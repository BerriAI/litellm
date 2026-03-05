"""
WebSearch Interception Handler

CustomLogger that intercepts WebSearch tool calls for models that don't
natively support web search (e.g., Bedrock/Claude) and executes them
server-side using litellm router's search tools.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union

import litellm
from litellm._logging import verbose_logger
from litellm.constants import LITELLM_WEB_SEARCH_TOOL_NAME
from litellm.integrations.custom_logger import CustomLogger, ToolCallResult
from litellm.integrations.websearch_interception.tools import (
    get_litellm_web_search_tool,
    get_litellm_web_search_tool_openai,
    is_web_search_tool,
)
from litellm.integrations.websearch_interception.transformation import (
    WebSearchTransformation,
)
from litellm.types.integrations.websearch_interception import (
    WebSearchInterceptionConfig,
)
from litellm.types.utils import LlmProviders

# Tool names that indicate a web search tool_use block
WEBSEARCH_NAMES = frozenset({LITELLM_WEB_SEARCH_TOOL_NAME, "WebSearch", "web_search"})


class WebSearchInterceptionLogger(CustomLogger):
    """
    CustomLogger that intercepts WebSearch tool calls for models that don't
    natively support web search.

    Uses the simplified async_execute_tool_calls hook — the framework handles
    message construction, thinking block preservation, and follow-up requests.
    """

    def __init__(
        self,
        enabled_providers: Optional[List[Union[LlmProviders, str]]] = None,
        search_tool_name: Optional[str] = None,
    ):
        """
        Args:
            enabled_providers: List of LLM providers to enable interception for.
                              Use LlmProviders enum values (e.g., [LlmProviders.BEDROCK])
                              If None or empty list, enables for ALL providers.
                              Default: None (all providers enabled)
            search_tool_name: Name of search tool configured in router's search_tools.
                             If None, will attempt to use first available search tool.
        """
        super().__init__()
        # Convert enum values to strings for comparison
        if enabled_providers is None:
            self.enabled_providers = [LlmProviders.BEDROCK.value]
        else:
            self.enabled_providers = [p.value if isinstance(p, LlmProviders) else p for p in enabled_providers]
        self.search_tool_name = search_tool_name

    # -----------------------------------------------------------------
    # Pre-call hooks (tool conversion + stream handling)
    # -----------------------------------------------------------------

    async def async_pre_call_deployment_hook(self, kwargs: Dict[str, Any], call_type: Optional[Any]) -> Optional[dict]:
        """
        Pre-call hook to convert native Anthropic web_search tools to regular tools.

        This prevents Bedrock from trying to execute web search server-side (which fails).
        Instead, we convert it to a regular tool so the model returns tool_use blocks
        that we can intercept and execute ourselves.
        """
        # Get provider from litellm_params (set by router in _add_deployment)
        custom_llm_provider = kwargs.get("litellm_params", {}).get("custom_llm_provider", "")

        if custom_llm_provider not in self.enabled_providers:
            return None

        # Check if request has tools with native web_search
        tools = kwargs.get("tools")
        if not tools:
            return None

        # Check if any tool is a web search tool (native or already LiteLLM standard)
        has_websearch = any(is_web_search_tool(t) for t in tools)

        if not has_websearch:
            return None

        verbose_logger.debug("WebSearchInterception: Converting native web_search tools to LiteLLM standard")

        # Convert native/custom web_search tools to LiteLLM standard
        converted_tools = []
        for tool in tools:
            if is_web_search_tool(tool):
                converted_tool = get_litellm_web_search_tool()
                converted_tools.append(converted_tool)
                verbose_logger.debug(
                    f"WebSearchInterception: Converted {tool.get('name', 'unknown')} "
                    f"(type={tool.get('type', 'none')}) to {LITELLM_WEB_SEARCH_TOOL_NAME}"
                )
            else:
                converted_tools.append(tool)

        return {**kwargs, "tools": converted_tools}

    @classmethod
    def from_config_yaml(cls, config: WebSearchInterceptionConfig) -> "WebSearchInterceptionLogger":
        """
        Initialize WebSearchInterceptionLogger from proxy config.yaml parameters.

        Args:
            config: Configuration dictionary from litellm_settings.websearch_interception_params
        """
        enabled_providers_str = config.get("enabled_providers", None)
        search_tool_name = config.get("search_tool_name", None)

        enabled_providers: Optional[List[Union[LlmProviders, str]]] = None
        if enabled_providers_str is not None:
            enabled_providers = []
            for provider in enabled_providers_str:
                try:
                    provider_enum = LlmProviders(provider)
                    enabled_providers.append(provider_enum)
                except ValueError:
                    enabled_providers.append(provider)

        return cls(
            enabled_providers=enabled_providers,
            search_tool_name=search_tool_name,
        )

    async def async_pre_request_hook(self, model: str, messages: List[Dict], kwargs: Dict) -> Optional[Dict]:
        """
        Pre-request hook to convert native web search tools to LiteLLM standard
        and convert stream=True to stream=False for interception.
        """
        custom_llm_provider = kwargs.get("litellm_params", {}).get("custom_llm_provider", "")

        verbose_logger.debug(
            f"WebSearchInterception: Pre-request hook called"
            f" - custom_llm_provider={custom_llm_provider}"
            f" - enabled_providers={self.enabled_providers or 'ALL'}"
        )

        if self.enabled_providers is not None and custom_llm_provider not in self.enabled_providers:
            return None

        tools = kwargs.get("tools")
        if not tools:
            return None

        has_websearch = any(is_web_search_tool(t) for t in tools)
        if not has_websearch:
            return None

        verbose_logger.debug(f"WebSearchInterception: Pre-request hook triggered for provider={custom_llm_provider}")

        # Convert native web search tools to LiteLLM standard
        converted_tools = []
        for tool in tools:
            if is_web_search_tool(tool):
                standard_tool = get_litellm_web_search_tool()
                converted_tools.append(standard_tool)
            else:
                converted_tools.append(tool)

        kwargs["tools"] = converted_tools

        # Convert stream=True to stream=False for WebSearch interception
        if kwargs.get("stream"):
            verbose_logger.debug("WebSearchInterception: Converting stream=True to stream=False")
            kwargs["stream"] = False
            kwargs["_websearch_interception_converted_stream"] = True

        return kwargs

    # -----------------------------------------------------------------
    # Simplified tool execution hook
    # -----------------------------------------------------------------

    async def async_execute_tool_calls(self, response, kwargs):
        """Detect and execute websearch tool calls."""
        provider = kwargs.get("custom_llm_provider", "")
        if self.enabled_providers is not None and provider not in self.enabled_providers:
            return []

        # Get content blocks from Anthropic-style response
        if isinstance(response, dict):
            content = response.get("content", [])
        else:
            content = getattr(response, "content", None) or []

        if not content:
            return []

        # Find websearch tool_use blocks and execute searches
        search_tasks = []
        tool_call_ids = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type")
                bname = block.get("name")
                bid = block.get("id")
                binput = block.get("input", {})
            else:
                btype = getattr(block, "type", None)
                bname = getattr(block, "name", None)
                bid = getattr(block, "id", None)
                binput = getattr(block, "input", {})

            if btype == "tool_use" and bname in WEBSEARCH_NAMES:
                query = binput.get("query", "") if isinstance(binput, dict) else ""
                if query:
                    search_tasks.append(self._execute_search(query))
                    tool_call_ids.append(bid)
                else:
                    verbose_logger.warning(f"WebSearchInterception: Tool call {bid} has no query")

        if not search_tasks:
            return []

        verbose_logger.debug(f"WebSearchInterception: Executing {len(search_tasks)} search(es) in parallel")

        # Execute searches in parallel
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Build ToolCallResults
        results = []
        for i, (tc_id, result) in enumerate(zip(tool_call_ids, search_results)):
            if isinstance(result, Exception):
                verbose_logger.error(f"WebSearchInterception: Search {i} failed: {result}")
                results.append(ToolCallResult(
                    tool_call_id=tc_id,
                    content=f"Search failed: {result}",
                    is_error=True,
                ))
            else:
                results.append(ToolCallResult(
                    tool_call_id=tc_id,
                    content=str(result),
                    is_error=False,
                ))

        return results

    # -----------------------------------------------------------------
    # Search execution
    # -----------------------------------------------------------------

    async def _execute_search(self, query: str) -> str:
        """Execute a single web search using router's search tools."""
        try:
            try:
                from litellm.proxy.proxy_server import llm_router
            except ImportError:
                llm_router = None

            # Determine search provider from router's search_tools
            search_provider: Optional[str] = None
            if llm_router is not None and hasattr(llm_router, "search_tools"):
                if self.search_tool_name:
                    matching_tools = [
                        tool
                        for tool in llm_router.search_tools
                        if tool.get("search_tool_name") == self.search_tool_name
                    ]
                    if matching_tools:
                        search_tool = matching_tools[0]
                        search_provider = search_tool.get("litellm_params", {}).get("search_provider")

                if not search_provider and llm_router.search_tools:
                    first_tool = llm_router.search_tools[0]
                    search_provider = first_tool.get("litellm_params", {}).get("search_provider")

            if not search_provider:
                search_provider = "perplexity"

            verbose_logger.debug(
                f"WebSearchInterception: Executing search for '{query}' using provider '{search_provider}'"
            )
            result = await litellm.asearch(query=query, search_provider=search_provider)

            search_result_text = WebSearchTransformation.format_search_response(result)
            verbose_logger.debug(
                f"WebSearchInterception: Search completed for '{query}', got {len(search_result_text)} chars"
            )
            return search_result_text
        except Exception as e:
            verbose_logger.error(f"WebSearchInterception: Search failed for '{query}': {str(e)}")
            raise

    # -----------------------------------------------------------------
    # Legacy agentic loop hooks (kept for backward compatibility)
    # -----------------------------------------------------------------
    # NOTE: These are no longer used when async_execute_tool_calls is
    # implemented. They remain so older framework versions that only
    # call the two-step pattern still work.

    async def async_should_run_agentic_loop(self, response, model, messages, tools, stream, custom_llm_provider, kwargs):
        return False, {}

    async def async_should_run_chat_completion_agentic_loop(self, response, model, messages, tools, stream, custom_llm_provider, kwargs):
        return False, {}

    # -----------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------

    @staticmethod
    def initialize_from_proxy_config(
        litellm_settings: Dict[str, Any],
        callback_specific_params: Dict[str, Any],
    ) -> "WebSearchInterceptionLogger":
        """
        Static method to initialize WebSearchInterceptionLogger from proxy config.

        Used in callback_utils.py to simplify initialization logic.
        """
        websearch_params: WebSearchInterceptionConfig = {}
        if "websearch_interception_params" in litellm_settings:
            websearch_params = litellm_settings["websearch_interception_params"]
        elif "websearch_interception" in callback_specific_params:
            websearch_params = callback_specific_params["websearch_interception"]

        return WebSearchInterceptionLogger.from_config_yaml(websearch_params)
