"""
WebSearch Tool Transformation

Transforms between Anthropic tool_use format and LiteLLM search format.
"""

from typing import Any, Dict, List, NamedTuple, Tuple

from litellm._logging import verbose_logger
from litellm.constants import LITELLM_WEB_SEARCH_TOOL_NAME
from litellm.llms.base_llm.search.transformation import SearchResponse


class TransformRequestResult(NamedTuple):
    """Result of transform_request() for WebSearch tool detection."""

    has_websearch: bool
    """True if WebSearch tool_use was found in the response."""

    tool_calls: List[Dict]
    """List of tool_use dicts with id, name, input."""

    thinking_blocks: List[Dict]
    """List of thinking/redacted_thinking blocks to preserve."""


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
    ) -> TransformRequestResult:
        """
        Transform Anthropic response to extract WebSearch tool calls.

        Detects if response contains WebSearch tool_use blocks and extracts
        the search queries for execution. Also captures thinking blocks for
        proper follow-up message construction.

        Args:
            response: Model response (dict or AnthropicMessagesResponse)
            stream: Whether response is streaming

        Returns:
            TransformRequestResult with has_websearch, tool_calls, and thinking_blocks

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
            return TransformRequestResult(False, [], [])

        # Parse non-streaming response
        return WebSearchTransformation._detect_from_non_streaming_response(response)

    @staticmethod
    def _detect_from_non_streaming_response(
        response: Any,
    ) -> TransformRequestResult:
        """Parse non-streaming response for WebSearch tool_use and thinking blocks"""

        # Handle both dict and object responses
        if isinstance(response, dict):
            content = response.get("content", [])
        else:
            if not hasattr(response, "content"):
                verbose_logger.debug(
                    "WebSearchInterception: Response has no content attribute"
                )
                return TransformRequestResult(False, [], [])
            content = response.content or []

        if not content:
            verbose_logger.debug("WebSearchInterception: Response has empty content")
            return TransformRequestResult(False, [], [])

        # Find all WebSearch tool_use blocks and thinking blocks
        tool_calls = []
        thinking_blocks = []
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

            # Capture thinking and redacted_thinking blocks for follow-up messages
            # Normalize to dict to ensure JSON serialization works
            if block_type in ("thinking", "redacted_thinking"):
                if isinstance(block, dict):
                    thinking_blocks.append(block)
                else:
                    # Normalize SDK objects to dicts for safe serialization in follow-up requests
                    normalized = {"type": block_type}
                    for attr in ("thinking", "data", "signature"):
                        if hasattr(block, attr):
                            normalized[attr] = getattr(block, attr)
                    thinking_blocks.append(normalized)
                verbose_logger.debug(
                    f"WebSearchInterception: Captured {block_type} block for follow-up"
                )

            # Check for LiteLLM standard or legacy web search tools
            # Handles: litellm_web_search, WebSearch, web_search
            if block_type == "tool_use" and block_name in (
                LITELLM_WEB_SEARCH_TOOL_NAME,
                "WebSearch",
                "web_search",
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
                    f"WebSearchInterception: Found {block_name} tool_use with id={block_id}"
                )

        return TransformRequestResult(len(tool_calls) > 0, tool_calls, thinking_blocks)

    @staticmethod
    def transform_response(
        tool_calls: List[Dict],
        search_results: List[str],
        thinking_blocks: List[Dict],
    ) -> Tuple[Dict, Dict]:
        """
        Transform LiteLLM search results to Anthropic tool_result format.

        Builds the assistant and user messages needed for the agentic loop
        follow-up request.

        Args:
            tool_calls: List of tool_use dicts from transform_request
            search_results: List of search result strings (one per tool_call)
            thinking_blocks: List of thinking/redacted_thinking blocks to include at the start of
                             assistant message

        Returns:
            (assistant_message, user_message):
                assistant_message: Message with thinking blocks (if any) then tool_use blocks
                user_message: Message with tool_result blocks
        """
        # Build assistant message content - thinking blocks first, then tool_use
        assistant_content = []

        # Add thinking blocks at the start (required when thinking is enabled)
        if thinking_blocks:
            assistant_content.extend(thinking_blocks)

        # Add tool_use blocks
        assistant_content.extend(
            [
                {
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": tc["input"],
                }
                for tc in tool_calls
            ]
        )

        assistant_message = {
            "role": "assistant",
            "content": assistant_content,
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
