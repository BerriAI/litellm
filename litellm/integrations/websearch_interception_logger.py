"""
WebSearch Interception Logger

Intercepts WebSearch tool calls for models that don't natively support web search
(e.g., Bedrock/Claude) and executes them server-side using litellm.search().

Usage:
    import litellm
    from litellm.integrations.websearch_interception_logger import WebSearchInterceptionLogger

    # Enable interception for Bedrock
    litellm.callbacks = [WebSearchInterceptionLogger(enabled_providers=["bedrock"])]

    # Make request with WebSearch tool
    response = await litellm.messages.acreate(
        model="bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=[{"role": "user", "content": "Search for litellm"}],
        tools=[{"name": "WebSearch", ...}],
        max_tokens=1024,
    )
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple

import litellm
from litellm._logging import verbose_logger
from litellm.anthropic_interface import messages as anthropic_messages
from litellm.integrations.custom_logger import CustomLogger


class WebSearchInterceptionLogger(CustomLogger):
    """
    CustomLogger that intercepts WebSearch tool calls for models that don't
    natively support web search.

    Implements agentic loop:
    1. Detects WebSearch tool_use in model response
    2. Executes litellm.search() for each query
    3. Makes follow-up request with search results
    4. Returns final response
    """

    def __init__(
        self,
        enabled_providers: Optional[List[str]] = None,
        search_provider: str = "perplexity",
    ):
        """
        Args:
            enabled_providers: List of providers to enable interception for.
                              Default: ["bedrock"] (all except native Anthropic)
            search_provider: Search provider to use for litellm.search().
                            Default: "perplexity"
        """
        super().__init__()
        self.enabled_providers = enabled_providers or ["bedrock"]
        self.search_provider = search_provider

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
        """Check if WebSearch tool interception is needed"""

        verbose_logger.info(f"WebSearchInterception: Hook called! provider={custom_llm_provider}, stream={stream}")
        verbose_logger.info(f"WebSearchInterception: Response type: {type(response)}")

        # Check if provider should be intercepted
        if custom_llm_provider not in self.enabled_providers:
            verbose_logger.info(
                f"WebSearchInterception: Skipping provider {custom_llm_provider} (not in enabled list: {self.enabled_providers})"
            )
            return False, {}

        # Check if tools include WebSearch
        has_websearch_tool = any(t.get("name") == "WebSearch" for t in (tools or []))
        if not has_websearch_tool:
            verbose_logger.debug(
                "WebSearchInterception: No WebSearch tool in request"
            )
            return False, {}

        # Detect WebSearch tool_use in response
        should_intercept, tool_calls = await self._detect_websearch_tool_use(
            response=response,
            stream=stream,
        )

        if not should_intercept:
            verbose_logger.debug(
                "WebSearchInterception: No WebSearch tool_use detected in response"
            )
            return False, {}

        verbose_logger.info(
            f"WebSearchInterception: Detected {len(tool_calls)} WebSearch tool call(s), executing agentic loop"
        )

        # Return agentic context with tool calls
        agentic_context = {
            "tool_calls": tool_calls,
            "tool_type": "websearch",
            "provider": custom_llm_provider,
        }
        return True, agentic_context

    async def async_run_agentic_loop(
        self,
        agentic_context: Dict,
        model: str,
        messages: List[Dict],
        response: Any,
        anthropic_messages_provider_config: Any,
        anthropic_messages_optional_request_params: Dict,
        logging_obj: Any,
        stream: bool,
        kwargs: Dict,
    ) -> Any:
        """Execute agentic loop with WebSearch execution"""

        tool_calls = agentic_context["tool_calls"]

        verbose_logger.info(
            f"WebSearchInterception: Executing agentic loop for {len(tool_calls)} search(es)"
        )

        return await self._execute_agentic_loop(
            model=model,
            messages=messages,
            tool_calls=tool_calls,
            anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
            stream=stream,
            kwargs=kwargs,
        )

    async def _detect_websearch_tool_use(
        self,
        response: Any,
        stream: bool,
    ) -> Tuple[bool, List[Dict]]:
        """Detect if response contains WebSearch tool_use blocks"""

        if stream:
            # For streaming: need to consume and buffer the stream
            # TODO: Implement streaming detection in phase 2
            verbose_logger.warning(
                "WebSearchInterception: Streaming not yet supported, skipping"
            )
            return False, []
        else:
            # For non-streaming: parse response directly
            return self._detect_from_non_streaming_response(response)

    def _detect_from_non_streaming_response(
        self,
        response: Any,
    ) -> Tuple[bool, List[Dict]]:
        """Parse non-streaming response for WebSearch tool_use"""

        # Handle both dict and object responses
        if isinstance(response, dict):
            content = response.get("content", [])
        else:
            if not hasattr(response, "content"):
                verbose_logger.debug(
                    "WebSearchInterception: Response has no content attribute"
                )
                return False, []
            content = response.content or []

        if not content:
            verbose_logger.debug(
                "WebSearchInterception: Response has empty content"
            )
            return False, []

        # Find all WebSearch tool_use blocks
        tool_calls = []
        for block in content:
            # Handle both dict and object blocks
            if isinstance(block, dict):
                block_type = block.get("type")
                block_name = block.get("name")
                block_id = block.get("id")
                block_input = block.get("input", {})
            else:
                block_type = getattr(block, "type", None)
                block_name = getattr(block, "name", None)
                block_id = getattr(block, "id", None)
                block_input = getattr(block, "input", {})

            if block_type == "tool_use" and block_name == "WebSearch":
                # Convert to dict for easier handling
                tool_call = {
                    "id": block_id,
                    "type": "tool_use",
                    "name": "WebSearch",
                    "input": block_input,
                }
                tool_calls.append(tool_call)
                verbose_logger.debug(
                    f"WebSearchInterception: Found WebSearch tool_use with id={tool_call['id']}"
                )

        return len(tool_calls) > 0, tool_calls

    async def _execute_agentic_loop(
        self,
        model: str,
        messages: List[Dict],
        tool_calls: List[Dict],
        anthropic_messages_optional_request_params: Dict,
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
        verbose_logger.info(
            f"WebSearchInterception: Executing {len(search_tasks)} search(es) in parallel"
        )
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Handle any exceptions in search results
        final_search_results = []
        for i, result in enumerate(search_results):
            if isinstance(result, Exception):
                verbose_logger.error(
                    f"WebSearchInterception: Search {i} failed with error: {str(result)}"
                )
                final_search_results.append(
                    f"Search failed: {str(result)}"
                )
            else:
                final_search_results.append(result)

        # Build assistant message with tool_use blocks
        assistant_message = {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": tc["input"],
                }
                for tc in tool_calls
            ],
        }

        # Build user message with tool_result blocks
        user_message = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_calls[i]["id"],
                    "content": final_search_results[i],
                }
                for i in range(len(tool_calls))
            ],
        }

        # Make follow-up request with search results
        follow_up_messages = messages + [assistant_message, user_message]

        verbose_logger.info(
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

            final_response = await anthropic_messages.acreate(
                max_tokens=max_tokens,
                messages=follow_up_messages,
                model=model,
                **optional_params_without_max_tokens,
                **kwargs,
            )
            verbose_logger.info(
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
        """Execute a single web search using litellm.search()"""
        try:
            verbose_logger.debug(
                f"WebSearchInterception: Executing search for '{query}' using provider '{self.search_provider}'"
            )
            result = await litellm.asearch(
                query=query, search_provider=self.search_provider
            )

            # Convert SearchResponse to string
            if hasattr(result, "results") and result.results:
                # Format results as text
                search_result_text = "\n\n".join(
                    [
                        f"Title: {r.title}\nURL: {r.url}\nSnippet: {r.snippet}"
                        for r in result.results
                    ]
                )
            else:
                search_result_text = str(result)

            verbose_logger.debug(
                f"WebSearchInterception: Search completed for '{query}', got {len(search_result_text)} chars"
            )
            return search_result_text
        except Exception as e:
            verbose_logger.error(
                f"WebSearchInterception: Search failed for '{query}': {str(e)}"
            )
            raise

    async def _create_empty_search_result(self) -> str:
        """Create an empty search result for tool calls without queries"""
        return "No search query provided"
