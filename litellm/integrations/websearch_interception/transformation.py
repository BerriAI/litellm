"""
WebSearch Tool Transformation

Transforms between Anthropic/OpenAI tool_use format and LiteLLM search format.
"""
import json
from typing import Any, Dict, List, Tuple, Union

from litellm._logging import verbose_logger
from litellm.constants import LITELLM_WEB_SEARCH_TOOL_NAME
from litellm.llms.base_llm.search.transformation import SearchResponse


class WebSearchTransformation:
    """
    Transformation class for WebSearch tool interception.

    Handles transformation between:
    - Anthropic tool_use format → LiteLLM search requests
    - OpenAI tool_calls format → LiteLLM search requests
    - LiteLLM SearchResponse → Anthropic/OpenAI tool_result format
    """

    @staticmethod
    def transform_request(
        response: Any,
        stream: bool,
        response_format: str = "anthropic",
    ) -> Tuple[bool, List[Dict]]:
        """
        Transform model response to extract WebSearch tool calls.

        Detects if response contains WebSearch tool_use/tool_calls blocks and extracts
        the search queries for execution.

        Args:
            response: Model response (dict, AnthropicMessagesResponse, or ModelResponse)
            stream: Whether response is streaming
            response_format: Response format - "anthropic" or "openai" (default: "anthropic")

        Returns:
            (has_websearch, tool_calls):
                has_websearch: True if WebSearch tool_use found
                tool_calls: List of tool_use/tool_calls dicts with id, name, input/function

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

        # Parse non-streaming response based on format
        if response_format == "openai":
            return WebSearchTransformation._detect_from_openai_response(response)
        else:
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
    def _detect_from_openai_response(
        response: Any,
    ) -> Tuple[bool, List[Dict]]:
        """Parse OpenAI-style response for WebSearch tool_calls"""
        
        # Handle both dict and ModelResponse objects
        if isinstance(response, dict):
            choices = response.get("choices", [])
        else:
            if not hasattr(response, "choices"):
                verbose_logger.debug(
                    "WebSearchInterception: Response has no choices attribute"
                )
                return False, []
            choices = response.choices or []

        if not choices:
            verbose_logger.debug(
                "WebSearchInterception: Response has empty choices"
            )
            return False, []

        # Get first choice's message
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message", {})
        else:
            message = getattr(first_choice, "message", None)
        
        if not message:
            verbose_logger.debug(
                "WebSearchInterception: First choice has no message"
            )
            return False, []

        # Get tool_calls from message
        if isinstance(message, dict):
            openai_tool_calls = message.get("tool_calls", [])
        else:
            openai_tool_calls = getattr(message, "tool_calls", None) or []

        if not openai_tool_calls:
            verbose_logger.debug(
                "WebSearchInterception: Message has no tool_calls"
            )
            return False, []

        # Find all WebSearch tool calls
        tool_calls = []
        for tool_call in openai_tool_calls:
            # Handle both dict and object tool calls
            if isinstance(tool_call, dict):
                tool_id = tool_call.get("id")
                tool_type = tool_call.get("type")
                function = tool_call.get("function", {})
                function_name = function.get("name") if isinstance(function, dict) else getattr(function, "name", None)
                function_arguments = function.get("arguments") if isinstance(function, dict) else getattr(function, "arguments", None)
            else:
                tool_id = getattr(tool_call, "id", None)
                tool_type = getattr(tool_call, "type", None)
                function = getattr(tool_call, "function", None)
                function_name = getattr(function, "name", None) if function else None
                function_arguments = getattr(function, "arguments", None) if function else None

            # Check for LiteLLM standard or legacy web search tools
            if tool_type == "function" and function_name in (
                LITELLM_WEB_SEARCH_TOOL_NAME, "WebSearch", "web_search"
            ):
                # Parse arguments (might be JSON string)
                if isinstance(function_arguments, str):
                    try:
                        arguments = json.loads(function_arguments)
                    except json.JSONDecodeError:
                        verbose_logger.warning(
                            f"WebSearchInterception: Failed to parse function arguments: {function_arguments}"
                        )
                        arguments = {}
                else:
                    arguments = function_arguments or {}

                # Convert to internal format (similar to Anthropic)
                tool_call_dict = {
                    "id": tool_id,
                    "type": "function",
                    "name": function_name,
                    "function": {
                        "name": function_name,
                        "arguments": arguments,
                    },
                    "input": arguments,  # For compatibility with Anthropic format
                }
                tool_calls.append(tool_call_dict)
                verbose_logger.debug(
                    f"WebSearchInterception: Found {function_name} tool_call with id={tool_id}"
                )

        return len(tool_calls) > 0, tool_calls

    @staticmethod
    def transform_response(
        tool_calls: List[Dict],
        search_results: List[str],
        response_format: str = "anthropic",
    ) -> Tuple[Dict, Union[Dict, List[Dict]]]:
        """
        Transform LiteLLM search results to Anthropic/OpenAI tool_result format.

        Builds the assistant and user/tool messages needed for the agentic loop
        follow-up request.

        Args:
            tool_calls: List of tool_use/tool_calls dicts from transform_request
            search_results: List of search result strings (one per tool_call)
            response_format: Response format - "anthropic" or "openai" (default: "anthropic")

        Returns:
            (assistant_message, user_or_tool_messages):
                For Anthropic: assistant_message with tool_use blocks, user_message with tool_result blocks
                For OpenAI: assistant_message with tool_calls, tool_messages list with tool results
        """
        if response_format == "openai":
            return WebSearchTransformation._transform_response_openai(
                tool_calls, search_results
            )
        else:
            return WebSearchTransformation._transform_response_anthropic(
                tool_calls, search_results
            )

    @staticmethod
    def _transform_response_anthropic(
        tool_calls: List[Dict],
        search_results: List[str],
    ) -> Tuple[Dict, Dict]:
        """Transform to Anthropic format (single user message with tool_result blocks)"""
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
    def _transform_response_openai(
        tool_calls: List[Dict],
        search_results: List[str],
    ) -> Tuple[Dict, List[Dict]]:
        """Transform to OpenAI format (assistant with tool_calls, separate tool messages)"""
        # Build assistant message with tool_calls
        assistant_message = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["input"]) if isinstance(tc["input"], dict) else str(tc["input"]),
                    },
                }
                for tc in tool_calls
            ],
        }

        # Build separate tool messages (one per tool call)
        tool_messages = [
            {
                "role": "tool",
                "tool_call_id": tool_calls[i]["id"],
                "content": search_results[i],
            }
            for i in range(len(tool_calls))
        ]

        return assistant_message, tool_messages

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
