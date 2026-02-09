"""
WebSearch Interception Handler

CustomLogger that intercepts WebSearch tool calls for models that don't
natively support web search (e.g., Bedrock/Claude) and executes them
server-side using litellm router's search tools.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import litellm
from litellm._logging import verbose_logger
from litellm.anthropic_interface import messages as anthropic_messages
from litellm.constants import LITELLM_WEB_SEARCH_TOOL_NAME
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.websearch_interception.tools import (
    get_litellm_web_search_tool,
    is_web_search_tool,
    is_web_search_tool_chat_completion,
)
from litellm.integrations.websearch_interception.transformation import (
    WebSearchTransformation,
)
from litellm.types.integrations.websearch_interception import (
    WebSearchInterceptionConfig,
)
from litellm.types.utils import LlmProviders


class WebSearchInterceptionLogger(CustomLogger):
    """
    CustomLogger that intercepts WebSearch tool calls for models that don't
    natively support web search.

    Implements agentic loop:
    1. Detects WebSearch tool_use in model response
    2. Executes litellm.asearch() for each query using router's search tools
    3. Makes follow-up request with search results
    4. Returns final response
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
            self.enabled_providers = [
                p.value if isinstance(p, LlmProviders) else p
                for p in enabled_providers
            ]
        self.search_tool_name = search_tool_name
        self._request_has_websearch = False  # Track if current request has web search

    async def async_pre_call_deployment_hook(
        self, kwargs: Dict[str, Any], call_type: Optional[Any]
    ) -> Optional[dict]:
        """
        Pre-call hook to convert native Anthropic web_search tools to regular tools.

        This prevents Bedrock from trying to execute web search server-side (which fails).
        Instead, we convert it to a regular tool so the model returns tool_use blocks
        that we can intercept and execute ourselves.
        """
        # Check if this is for an enabled provider
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

        verbose_logger.debug(
            "WebSearchInterception: Converting native web_search tools to LiteLLM standard"
        )

        # Convert native/custom web_search tools to LiteLLM standard
        converted_tools = []
        for tool in tools:
            if is_web_search_tool(tool):
                # Convert to LiteLLM standard web search tool
                converted_tool = get_litellm_web_search_tool()
                converted_tools.append(converted_tool)
                verbose_logger.debug(
                    f"WebSearchInterception: Converted {tool.get('name', 'unknown')} "
                    f"(type={tool.get('type', 'none')}) to {LITELLM_WEB_SEARCH_TOOL_NAME}"
                )
            else:
                # Keep other tools as-is
                converted_tools.append(tool)

        # Return modified kwargs with converted tools
        return {"tools": converted_tools}

    @classmethod
    def from_config_yaml(
        cls, config: WebSearchInterceptionConfig
    ) -> "WebSearchInterceptionLogger":
        """
        Initialize WebSearchInterceptionLogger from proxy config.yaml parameters.

        Args:
            config: Configuration dictionary from litellm_settings.websearch_interception_params

        Returns:
            Configured WebSearchInterceptionLogger instance

        Example:
            From proxy_config.yaml:
                litellm_settings:
                  websearch_interception_params:
                    enabled_providers: ["bedrock"]
                    search_tool_name: "my-perplexity-search"

            Usage:
                config = litellm_settings.get("websearch_interception_params", {})
                logger = WebSearchInterceptionLogger.from_config_yaml(config)
        """
        # Extract parameters from config
        enabled_providers_str = config.get("enabled_providers", None)
        search_tool_name = config.get("search_tool_name", None)

        # Convert string provider names to LlmProviders enum values
        enabled_providers: Optional[List[Union[LlmProviders, str]]] = None
        if enabled_providers_str is not None:
            enabled_providers = []
            for provider in enabled_providers_str:
                try:
                    # Try to convert string to LlmProviders enum
                    provider_enum = LlmProviders(provider)
                    enabled_providers.append(provider_enum)
                except ValueError:
                    # If conversion fails, keep as string
                    enabled_providers.append(provider)

        return cls(
            enabled_providers=enabled_providers,
            search_tool_name=search_tool_name,
        )

    async def async_pre_request_hook(
        self, model: str, messages: List[Dict], kwargs: Dict
    ) -> Optional[Dict]:
        """
        Pre-request hook to convert native web search tools to LiteLLM standard.

        This hook is called before the API request is made, allowing us to:
        1. Detect native web search tools (web_search_20250305, etc.)
        2. Convert them to LiteLLM standard format (litellm_web_search)
        3. Convert stream=True to stream=False for interception

        This prevents providers like Bedrock from trying to execute web search
        natively (which fails), and ensures our agentic loop can intercept tool_use.

        Returns:
            Modified kwargs dict with converted tools, or None if no modifications needed
        """
        # Check if this request is for an enabled provider
        custom_llm_provider = kwargs.get("litellm_params", {}).get(
            "custom_llm_provider", ""
        )

        verbose_logger.debug(
            f"WebSearchInterception: Pre-request hook called"
            f" - custom_llm_provider={custom_llm_provider}"
            f" - enabled_providers={self.enabled_providers or 'ALL'}"
        )

        if self.enabled_providers is not None and custom_llm_provider not in self.enabled_providers:
            verbose_logger.debug(
                f"WebSearchInterception: Skipping - provider {custom_llm_provider} not in {self.enabled_providers}"
            )
            return None

        # Check if request has tools
        tools = kwargs.get("tools")
        if not tools:
            return None

        # Check if any tool is a web search tool
        has_websearch = any(is_web_search_tool(t) for t in tools)
        if not has_websearch:
            return None

        verbose_logger.debug(
            f"WebSearchInterception: Pre-request hook triggered for provider={custom_llm_provider}"
        )

        # Convert native web search tools to LiteLLM standard
        converted_tools = []
        for tool in tools:
            if is_web_search_tool(tool):
                standard_tool = get_litellm_web_search_tool()
                converted_tools.append(standard_tool)
                verbose_logger.debug(
                    f"WebSearchInterception: Converted {tool.get('name', 'unknown')} "
                    f"(type={tool.get('type', 'none')}) to {LITELLM_WEB_SEARCH_TOOL_NAME}"
                )
            else:
                converted_tools.append(tool)

        # Update kwargs with converted tools
        kwargs["tools"] = converted_tools
        verbose_logger.debug(
            f"WebSearchInterception: Tools after conversion: {[t.get('name') for t in converted_tools]}"
        )

        # Convert stream=True to stream=False for WebSearch interception
        if kwargs.get("stream"):
            verbose_logger.debug(
                "WebSearchInterception: Converting stream=True to stream=False"
            )
            kwargs["stream"] = False
            kwargs["_websearch_interception_converted_stream"] = True

        return kwargs

    async def async_should_run_agentic_loop(
        self,
        response: Any,
        model: str,
        messages: List[Dict],
        tools: Optional[List[Dict]],
        stream: bool,
        custom_llm_provider: str,
        kwargs: Dict,
    ) -> Tuple[bool, Dict]:
        """
        Check if WebSearch tool interception is needed for Anthropic Messages API.
        
        This is the legacy method for Anthropic-style responses.
        For chat completions, use async_should_run_chat_completion_agentic_loop instead.
        """

        verbose_logger.debug(f"WebSearchInterception: Hook called! provider={custom_llm_provider}, stream={stream}")
        verbose_logger.debug(f"WebSearchInterception: Response type: {type(response)}")

        # Check if provider should be intercepted
        # Note: custom_llm_provider is already normalized by get_llm_provider()
        # (e.g., "bedrock/invoke/..." -> "bedrock")
        if self.enabled_providers is not None and custom_llm_provider not in self.enabled_providers:
            verbose_logger.debug(
                f"WebSearchInterception: Skipping provider {custom_llm_provider} (not in enabled list: {self.enabled_providers})"
            )
            return False, {}

        # Check if tools include any web search tool (LiteLLM standard or native)
        has_websearch_tool = any(is_web_search_tool(t) for t in (tools or []))
        if not has_websearch_tool:
            verbose_logger.debug(
                "WebSearchInterception: No web search tool in request"
            )
            return False, {}

        # Detect WebSearch tool_use in response (Anthropic format)
        should_intercept, tool_calls = WebSearchTransformation.transform_request(
            response=response,
            stream=stream,
            response_format="anthropic",
        )

        if not should_intercept:
            verbose_logger.debug(
                "WebSearchInterception: No WebSearch tool_use detected in response"
            )
            return False, {}

        verbose_logger.debug(
            f"WebSearchInterception: Detected {len(tool_calls)} WebSearch tool call(s), executing agentic loop"
        )

        # Return tools dict with tool calls
        tools_dict = {
            "tool_calls": tool_calls,
            "tool_type": "websearch",
            "provider": custom_llm_provider,
            "response_format": "anthropic",
        }
        return True, tools_dict

    async def async_should_run_chat_completion_agentic_loop(
        self,
        response: Any,
        model: str,
        messages: List[Dict],
        tools: Optional[List[Dict]],
        stream: bool,
        custom_llm_provider: str,
        kwargs: Dict,
    ) -> Tuple[bool, Dict]:
        """
        Check if WebSearch tool interception is needed for Chat Completions API.
        
        Similar to async_should_run_agentic_loop but for OpenAI-style chat completions.
        """

        verbose_logger.debug(f"WebSearchInterception: Chat completion hook called! provider={custom_llm_provider}, stream={stream}")
        verbose_logger.debug(f"WebSearchInterception: Response type: {type(response)}")

        # Check if provider should be intercepted
        if self.enabled_providers is not None and custom_llm_provider not in self.enabled_providers:
            verbose_logger.debug(
                f"WebSearchInterception: Skipping provider {custom_llm_provider} (not in enabled list: {self.enabled_providers})"
            )
            return False, {}

        # Check if tools include any web search tool (strict check for chat completions)
        has_websearch_tool = any(is_web_search_tool_chat_completion(t) for t in (tools or []))
        if not has_websearch_tool:
            verbose_logger.debug(
                "WebSearchInterception: No litellm_web_search tool in request"
            )
            return False, {}

        # Detect WebSearch tool_calls in response (OpenAI format)
        should_intercept, tool_calls = WebSearchTransformation.transform_request(
            response=response,
            stream=stream,
            response_format="openai",
        )

        if not should_intercept:
            verbose_logger.debug(
                "WebSearchInterception: No WebSearch tool_calls detected in response"
            )
            return False, {}

        verbose_logger.debug(
            f"WebSearchInterception: Detected {len(tool_calls)} WebSearch tool call(s), executing agentic loop"
        )

        # Return tools dict with tool calls
        tools_dict = {
            "tool_calls": tool_calls,
            "tool_type": "websearch",
            "provider": custom_llm_provider,
            "response_format": "openai",
        }
        return True, tools_dict

    async def async_run_agentic_loop(
        self,
        tools: Dict,
        model: str,
        messages: List[Dict],
        response: Any,
        anthropic_messages_provider_config: Any,
        anthropic_messages_optional_request_params: Dict,
        logging_obj: Any,
        stream: bool,
        kwargs: Dict,
    ) -> Any:
        """
        Execute agentic loop with WebSearch execution for Anthropic Messages API.
        
        This is the legacy method for Anthropic-style responses.
        """

        tool_calls = tools["tool_calls"]

        verbose_logger.debug(
            f"WebSearchInterception: Executing agentic loop for {len(tool_calls)} search(es)"
        )

        return await self._execute_agentic_loop(
            model=model,
            messages=messages,
            tool_calls=tool_calls,
            anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
            logging_obj=logging_obj,
            stream=stream,
            kwargs=kwargs,
        )

    async def async_run_chat_completion_agentic_loop(
        self,
        tools: Dict,
        model: str,
        messages: List[Dict],
        response: Any,
        optional_params: Dict,
        logging_obj: Any,
        stream: bool,
        kwargs: Dict,
    ) -> Any:
        """
        Execute agentic loop with WebSearch execution for Chat Completions API.
        
        Similar to async_run_agentic_loop but for OpenAI-style chat completions.
        """

        tool_calls = tools["tool_calls"]
        response_format = tools.get("response_format", "openai")

        verbose_logger.debug(
            f"WebSearchInterception: Executing chat completion agentic loop for {len(tool_calls)} search(es)"
        )

        return await self._execute_chat_completion_agentic_loop(
            model=model,
            messages=messages,
            tool_calls=tool_calls,
            optional_params=optional_params,
            logging_obj=logging_obj,
            stream=stream,
            kwargs=kwargs,
            response_format=response_format,
        )

    async def _execute_agentic_loop(
        self,
        model: str,
        messages: List[Dict],
        tool_calls: List[Dict],
        anthropic_messages_optional_request_params: Dict,
        logging_obj: Any,
        stream: bool,
        kwargs: Dict,
    ) -> Any:
        """Execute litellm.search() and make follow-up request"""

        # Extract search queries from tool_use blocks
        search_tasks = []
        for tool_call in tool_calls:
            query = tool_call["input"].get("query")
            if query:
                verbose_logger.debug(
                    f"WebSearchInterception: Queuing search for query='{query}'"
                )
                search_tasks.append(self._execute_search(query))
            else:
                verbose_logger.warning(
                    f"WebSearchInterception: Tool call {tool_call['id']} has no query"
                )
                # Add empty result for tools without query
                search_tasks.append(self._create_empty_search_result())

        # Execute searches in parallel
        verbose_logger.debug(
            f"WebSearchInterception: Executing {len(search_tasks)} search(es) in parallel"
        )
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Handle any exceptions in search results
        final_search_results: List[str] = []
        for i, result in enumerate(search_results):
            if isinstance(result, Exception):
                verbose_logger.error(
                    f"WebSearchInterception: Search {i} failed with error: {str(result)}"
                )
                final_search_results.append(
                    f"Search failed: {str(result)}"
                )
            elif isinstance(result, str):
                # Explicitly cast to str for type checker
                final_search_results.append(cast(str, result))
            else:
                # Should never happen, but handle for type safety
                verbose_logger.warning(
                    f"WebSearchInterception: Unexpected result type {type(result)} at index {i}"
                )
                final_search_results.append(str(result))

        # Build assistant and user messages using transformation
        assistant_message, user_message = WebSearchTransformation.transform_response(
            tool_calls=tool_calls,
            search_results=final_search_results,
        )

        # Make follow-up request with search results
        # Type cast: user_message is a Dict for Anthropic format (default response_format)
        follow_up_messages = messages + [assistant_message, cast(Dict, user_message)]

        verbose_logger.debug(
            "WebSearchInterception: Making follow-up request with search results"
        )
        verbose_logger.debug(
            f"WebSearchInterception: Follow-up messages count: {len(follow_up_messages)}"
        )
        verbose_logger.debug(
            f"WebSearchInterception: Last message (tool_result): {user_message}"
        )

        # Use anthropic_messages.acreate for follow-up request
        try:
            # Extract max_tokens from optional params or kwargs
            # max_tokens is a required parameter for anthropic_messages.acreate()
            max_tokens = anthropic_messages_optional_request_params.get(
                "max_tokens",
                kwargs.get("max_tokens", 1024)  # Default to 1024 if not found
            )

            verbose_logger.debug(
                f"WebSearchInterception: Using max_tokens={max_tokens} for follow-up request"
            )

            # Create a copy of optional params without max_tokens (since we pass it explicitly)
            optional_params_without_max_tokens = {
                k: v for k, v in anthropic_messages_optional_request_params.items()
                if k != 'max_tokens'
            }

            # Remove internal websearch interception flags from kwargs before follow-up request
            # These flags are used internally and should not be passed to the LLM provider
            kwargs_for_followup = {
                k: v for k, v in kwargs.items()
                if not k.startswith('_websearch_interception')
            }

            # Get model from logging_obj.model_call_details["agentic_loop_params"]
            # This preserves the full model name with provider prefix (e.g., "bedrock/invoke/...")
            full_model_name = model
            if logging_obj is not None:
                agentic_params = logging_obj.model_call_details.get("agentic_loop_params", {})
                full_model_name = agentic_params.get("model", model)
            verbose_logger.debug(
                f"WebSearchInterception: Using model name: {full_model_name}"
            )
            
            final_response = await anthropic_messages.acreate(
                max_tokens=max_tokens,
                messages=follow_up_messages,
                model=full_model_name,
                **optional_params_without_max_tokens,
                **kwargs_for_followup,
            )
            verbose_logger.debug(
                f"WebSearchInterception: Follow-up request completed, response type: {type(final_response)}"
            )
            verbose_logger.debug(
                f"WebSearchInterception: Final response: {final_response}"
            )
            return final_response
        except Exception as e:
            verbose_logger.exception(
                f"WebSearchInterception: Follow-up request failed: {str(e)}"
            )
            raise

    async def _execute_search(self, query: str) -> str:
        """Execute a single web search using router's search tools"""
        try:
            # Import router from proxy_server
            try:
                from litellm.proxy.proxy_server import llm_router
            except ImportError:
                verbose_logger.warning(
                    "WebSearchInterception: Could not import llm_router from proxy_server, "
                    "falling back to direct litellm.asearch() with perplexity"
                )
                llm_router = None

            # Determine search provider from router's search_tools
            search_provider: Optional[str] = None
            if llm_router is not None and hasattr(llm_router, "search_tools"):
                if self.search_tool_name:
                    # Find specific search tool by name
                    matching_tools = [
                        tool for tool in llm_router.search_tools
                        if tool.get("search_tool_name") == self.search_tool_name
                    ]
                    if matching_tools:
                        search_tool = matching_tools[0]
                        search_provider = search_tool.get("litellm_params", {}).get("search_provider")
                        verbose_logger.debug(
                            f"WebSearchInterception: Found search tool '{self.search_tool_name}' "
                            f"with provider '{search_provider}'"
                        )
                    else:
                        verbose_logger.warning(
                            f"WebSearchInterception: Search tool '{self.search_tool_name}' not found in router, "
                            "falling back to first available or perplexity"
                        )

                # If no specific tool or not found, use first available
                if not search_provider and llm_router.search_tools:
                    first_tool = llm_router.search_tools[0]
                    search_provider = first_tool.get("litellm_params", {}).get("search_provider")
                    verbose_logger.debug(
                        f"WebSearchInterception: Using first available search tool with provider '{search_provider}'"
                    )

            # Fallback to perplexity if no router or no search tools configured
            if not search_provider:
                search_provider = "perplexity"
                verbose_logger.debug(
                    "WebSearchInterception: No search tools configured in router, "
                    f"using default provider '{search_provider}'"
                )

            verbose_logger.debug(
                f"WebSearchInterception: Executing search for '{query}' using provider '{search_provider}'"
            )
            result = await litellm.asearch(
                query=query, search_provider=search_provider
            )

            # Format using transformation function
            search_result_text = WebSearchTransformation.format_search_response(result)

            verbose_logger.debug(
                f"WebSearchInterception: Search completed for '{query}', got {len(search_result_text)} chars"
            )
            return search_result_text
        except Exception as e:
            verbose_logger.error(
                f"WebSearchInterception: Search failed for '{query}': {str(e)}"
            )
            raise

    async def _execute_chat_completion_agentic_loop( # noqa: PLR0915
        self,
        model: str,
        messages: List[Dict],
        tool_calls: List[Dict],
        optional_params: Dict,
        logging_obj: Any,
        stream: bool,
        kwargs: Dict,
        response_format: str = "openai",
    ) -> Any:
        """Execute litellm.search() and make follow-up chat completion request"""

        # Extract search queries from tool_calls
        search_tasks = []
        for tool_call in tool_calls:
            # Handle both Anthropic-style input and OpenAI-style function.arguments
            query = None
            if "input" in tool_call and isinstance(tool_call["input"], dict):
                query = tool_call["input"].get("query")
            elif "function" in tool_call:
                func = tool_call["function"]
                if isinstance(func, dict):
                    args = func.get("arguments", {})
                    if isinstance(args, dict):
                        query = args.get("query")
                    
            if query:
                verbose_logger.debug(
                    f"WebSearchInterception: Queuing search for query='{query}'"
                )
                search_tasks.append(self._execute_search(query))
            else:
                verbose_logger.warning(
                    f"WebSearchInterception: Tool call {tool_call.get('id')} has no query"
                )
                # Add empty result for tools without query
                search_tasks.append(self._create_empty_search_result())

        # Execute searches in parallel
        verbose_logger.debug(
            f"WebSearchInterception: Executing {len(search_tasks)} search(es) in parallel"
        )
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Handle any exceptions in search results
        final_search_results: List[str] = []
        for i, result in enumerate(search_results):
            if isinstance(result, Exception):
                verbose_logger.error(
                    f"WebSearchInterception: Search {i} failed with error: {str(result)}"
                )
                final_search_results.append(
                    f"Search failed: {str(result)}"
                )
            elif isinstance(result, str):
                final_search_results.append(cast(str, result))
            else:
                verbose_logger.warning(
                    f"WebSearchInterception: Unexpected result type {type(result)} at index {i}"
                )
                final_search_results.append(str(result))

        # Build assistant and tool messages using transformation
        assistant_message, tool_messages_or_user = WebSearchTransformation.transform_response(
            tool_calls=tool_calls,
            search_results=final_search_results,
            response_format=response_format,
        )

        # Make follow-up request with search results
        # For OpenAI format, tool_messages_or_user is a list of tool messages
        if response_format == "openai":
            follow_up_messages = messages + [assistant_message] + cast(List[Dict], tool_messages_or_user)
        else:
            # For Anthropic format (shouldn't happen in this method, but handle it)
            follow_up_messages = messages + [assistant_message, cast(Dict, tool_messages_or_user)]

        verbose_logger.debug(
            "WebSearchInterception: Making follow-up chat completion request with search results"
        )
        verbose_logger.debug(
            f"WebSearchInterception: Follow-up messages count: {len(follow_up_messages)}"
        )

        # Use litellm.acompletion for follow-up request
        try:
            # Remove internal parameters that shouldn't be passed to follow-up request
            internal_params = {
                '_websearch_interception',
                'acompletion',
                'litellm_logging_obj',
                'custom_llm_provider',
                'model_alias_map',
                'stream_response',
                'custom_prompt_dict',
            }
            kwargs_for_followup = {
                k: v for k, v in kwargs.items()
                if not k.startswith('_websearch_interception') and k not in internal_params
            }

            # Get full model name from kwargs
            full_model_name = model
            if "custom_llm_provider" in kwargs:
                custom_llm_provider = kwargs["custom_llm_provider"]
                # Reconstruct full model name with provider prefix if needed
                if not model.startswith(custom_llm_provider):
                    # Check if model already has a provider prefix
                    if "/" not in model:
                        full_model_name = f"{custom_llm_provider}/{model}"
            
            verbose_logger.debug(
                f"WebSearchInterception: Using model name: {full_model_name}"
            )

            # Prepare tools for follow-up request (same as original)
            tools_param = optional_params.get("tools")
            
            # Remove tools and extra_body from optional_params to avoid issues
            # extra_body often contains internal LiteLLM params that shouldn't be forwarded
            optional_params_clean = {
                k: v for k, v in optional_params.items() 
                if k not in {"tools", "extra_body", "model_alias_map","stream_response",  "custom_prompt_dict"   }
            }
            
            final_response = await litellm.acompletion(
                model=full_model_name,
                messages=follow_up_messages,
                tools=tools_param,
                **optional_params_clean,
                **kwargs_for_followup,
            )
            
            verbose_logger.debug(
                f"WebSearchInterception: Follow-up request completed, response type: {type(final_response)}"
            )
            return final_response
        except Exception as e:
            verbose_logger.exception(
                f"WebSearchInterception: Follow-up request failed: {str(e)}"
            )
            raise

    async def _create_empty_search_result(self) -> str:
        """Create an empty search result for tool calls without queries"""
        return "No search query provided"

    @staticmethod
    def initialize_from_proxy_config(
        litellm_settings: Dict[str, Any],
        callback_specific_params: Dict[str, Any],
    ) -> "WebSearchInterceptionLogger":
        """
        Static method to initialize WebSearchInterceptionLogger from proxy config.

        Used in callback_utils.py to simplify initialization logic.

        Args:
            litellm_settings: Dictionary containing litellm_settings from proxy_config.yaml
            callback_specific_params: Dictionary containing callback-specific parameters

        Returns:
            Configured WebSearchInterceptionLogger instance

        Example:
            From callback_utils.py:
                websearch_obj = WebSearchInterceptionLogger.initialize_from_proxy_config(
                    litellm_settings=litellm_settings,
                    callback_specific_params=callback_specific_params
                )
        """
        # Get websearch_interception_params from litellm_settings or callback_specific_params
        websearch_params: WebSearchInterceptionConfig = {}
        if "websearch_interception_params" in litellm_settings:
            websearch_params = litellm_settings["websearch_interception_params"]
        elif "websearch_interception" in callback_specific_params:
            websearch_params = callback_specific_params["websearch_interception"]

        # Use classmethod to initialize from config
        return WebSearchInterceptionLogger.from_config_yaml(websearch_params)
