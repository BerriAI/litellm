"""
WebSearch Tool Transformation

Transforms between Anthropic tool_use format and LiteLLM search format.
"""

from typing import Any, Dict, List, Tuple

from litellm._logging import verbose_logger
from litellm.constants import LITELLM_WEB_SEARCH_TOOL_NAME
from litellm.llms.base_llm.search.transformation import SearchResponse


class WebSearchTransformation:
    """
    Transformation class for WebSearch tool interception.

    Handles transformation between:
    - Anthropic tool_use format → LiteLLM search requests
    - LiteLLM SearchResponse → Anthropic tool_result format
    """

    @staticmethod
    def transform_request(
        response: Any,
        stream: bool,
    ) -> Tuple[bool, List[Dict]]:
        """
        Transform Anthropic response to extract WebSearch tool calls.

        Detects if response contains WebSearch tool_use blocks and extracts
        the search queries for execution.

        Args:
            response: Model response (dict or AnthropicMessagesResponse)
            stream: Whether response is streaming

        Returns:
            (has_websearch, tool_calls):
                has_websearch: True if WebSearch tool_use found
                tool_calls: List of tool_use dicts with id, name, input

        Note:
            Streaming requests are handled by converting stream=True to stream=False
            in the WebSearchInterceptionLogger.async_log_pre_api_call hook before
            the API request is made. This means by the time this method is called,
            streaming requests have already been converted to non-streaming.
        """
        if stream:
            # This should not happen in practice since we convert streaming to non-streaming
            # in async_log_pre_api_call, but keep this check for safety
            verbose_logger.warning(
                "WebSearchInterception: Unexpected streaming response, skipping interception"
            )
            return False, []

        # Parse non-streaming response
        return WebSearchTransformation._detect_from_non_streaming_response(response)

    @staticmethod
    def _detect_from_non_streaming_response(
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

            # Check for LiteLLM standard or legacy web search tools
            # Handles: litellm_web_search, WebSearch, web_search
            if block_type == "tool_use" and block_name in (
                LITELLM_WEB_SEARCH_TOOL_NAME, "WebSearch", "web_search"
            ):
                # Convert to dict for easier handling
                tool_call = {
                    "id": block_id,
                    "type": "tool_use",
                    "name": block_name,  # Preserve original name
                    "input": block_input,
                }
                tool_calls.append(tool_call)
                verbose_logger.debug(
                    f"WebSearchInterception: Found {block_name} tool_use with id={tool_call['id']}"
                )

        return len(tool_calls) > 0, tool_calls

    @staticmethod
    def transform_response(
        tool_calls: List[Dict],
        search_results: List[str],
    ) -> Tuple[Dict, Dict]:
        """
        Transform LiteLLM search results to Anthropic tool_result format.

        Builds the assistant and user messages needed for the agentic loop
        follow-up request.

        Args:
            tool_calls: List of tool_use dicts from transform_request
            search_results: List of search result strings (one per tool_call)

        Returns:
            (assistant_message, user_message):
                assistant_message: Message with tool_use blocks
                user_message: Message with tool_result blocks
        """
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
                    "content": search_results[i],
                }
                for i in range(len(tool_calls))
            ],
        }

        return assistant_message, user_message

    @staticmethod
    def format_search_response(result: SearchResponse) -> str:
        """
        Format SearchResponse as text for tool_result content.

        Args:
            result: SearchResponse from litellm.asearch()

        Returns:
            Formatted text with Title, URL, Snippet for each result
        """
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

        return search_result_text
