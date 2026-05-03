"""
WebFetch Tool Transformation

Transforms between Anthropic/OpenAI tool_use format and LiteLLM fetch format.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from litellm._logging import verbose_logger
from litellm.constants import LITELLM_WEB_FETCH_TOOL_NAME
from litellm.llms.base_llm.fetch.transformation import WebFetchResponse


class WebFetchTransformation:
    """
    Transformation class for WebFetch tool interception.

    Handles transformation between:
    - Anthropic tool_use format → LiteLLM fetch requests
    - OpenAI tool_calls format → LiteLLM fetch requests
    - LiteLLM WebFetchResponse → Anthropic/OpenAI tool_result format
    """

    @staticmethod
    def transform_request(
        response: Any,
        stream: bool,
        response_format: str = "anthropic",
    ) -> Tuple[bool, List[Dict]]:
        """
        Transform model response to extract WebFetch tool calls.

        Detects if response contains WebFetch tool_use/tool_calls blocks and extracts
        the URLs for execution.

        Args:
            response: Model response (dict, AnthropicMessagesResponse, or ModelResponse)
            stream: Whether response is streaming
            response_format: Response format - "anthropic" or "openai" (default: "anthropic")

        Returns:
            (has_webfetch, tool_calls):
                has_webfetch: True if WebFetch tool_use found
                tool_calls: List of tool_use/tool_calls dicts with id, name, input/function
        """
        if stream:
            verbose_logger.warning(
                "WebFetchInterception: Unexpected streaming response, skipping interception"
            )
            return False, []

        # Parse non-streaming response based on format
        if response_format == "openai":
            return WebFetchTransformation._detect_from_openai_response(response)
        else:
            return WebFetchTransformation._detect_from_non_streaming_response(response)

    @staticmethod
    def _detect_from_non_streaming_response(
        response: Any,
    ) -> Tuple[bool, List[Dict]]:
        """Parse non-streaming response for WebFetch tool_use"""

        # Handle both dict and object responses
        if isinstance(response, dict):
            content = response.get("content", [])
        else:
            if not hasattr(response, "content"):
                verbose_logger.debug(
                    "WebFetchInterception: Response has no content attribute"
                )
                return False, []
            content = response.content or []

        if not content:
            verbose_logger.debug("WebFetchInterception: Response has empty content")
            return False, []

        # Find all WebFetch tool_use blocks
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
                block_input = getattr(block, "input", {}) or {}

            # Check for web fetch tool_use blocks
            if block_type == "tool_use" and block_name == LITELLM_WEB_FETCH_TOOL_NAME:
                tool_calls.append(
                    {
                        "id": block_id,
                        "name": block_name,
                        "input": block_input,
                    }
                )
                verbose_logger.debug(
                    f"WebFetchInterception: Detected tool_use for fetch: "
                    f"id={block_id}, url={block_input.get('url')}"
                )

        if tool_calls:
            verbose_logger.debug(
                f"WebFetchInterception: Found {len(tool_calls)} fetch tool call(s)"
            )
            return True, tool_calls

        return False, []

    @staticmethod
    def _detect_from_openai_response(
        response: Any,
    ) -> Tuple[bool, List[Dict]]:
        """Parse OpenAI chat completion response for WebFetch tool_calls"""

        # Handle ModelResponse (litellm type)
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            message = getattr(choice, "message", None)
            if message and hasattr(message, "tool_calls"):
                tool_calls_raw = message.tool_calls
            else:
                tool_calls_raw = None
        elif isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                tool_calls_raw = message.get("tool_calls")
            else:
                tool_calls_raw = None
        else:
            tool_calls_raw = None

        if not tool_calls_raw:
            return False, []

        # Filter for WebFetch tool_calls
        tool_calls = []
        for tool_call in tool_calls_raw:
            if isinstance(tool_call, dict):
                function = tool_call.get("function", {})
                tool_name = function.get("name", "")
                tool_id = tool_call.get("id", "")
                tool_input = function.get("arguments", "")
            else:
                function = getattr(tool_call, "function", None)
                if function:
                    tool_name = getattr(function, "name", "")
                    tool_id = getattr(tool_call, "id", "")
                    tool_input = getattr(function, "arguments", "")
                else:
                    continue

            if tool_name == LITELLM_WEB_FETCH_TOOL_NAME:
                # Parse arguments from JSON string if needed
                if isinstance(tool_input, str):
                    try:
                        import json

                        tool_input = json.loads(tool_input)
                    except (json.JSONDecodeError, TypeError):
                        tool_input = {}

                tool_calls.append(
                    {
                        "id": tool_id,
                        "function": {
                            "name": tool_name,
                            "arguments": tool_input,
                        },
                    }
                )
                verbose_logger.debug(
                    f"WebFetchInterception: Detected tool_call for fetch: "
                    f"id={tool_id}, url={tool_input.get('url') if isinstance(tool_input, dict) else 'unknown'}"
                )

        if tool_calls:
            verbose_logger.debug(
                f"WebFetchInterception: Found {len(tool_calls)} fetch tool call(s)"
            )
            return True, tool_calls

        return False, []

    @staticmethod
    def transform_response(
        tool_calls: List[Dict],
        fetch_results: List[str],
        response_format: str = "anthropic",
        thinking_blocks: Optional[List[Dict]] = None,
    ) -> Tuple[Dict, Union[Dict, List[Dict]]]:
        """
        Transform fetch results into tool_result format for follow-up request.

        Args:
            tool_calls: List of tool_use/tool_calls dicts
            fetch_results: List of fetch result strings (markdown content)
            response_format: "anthropic" or "openai"
            thinking_blocks: Optional thinking blocks to preserve

        Returns:
            (assistant_message, tool_result_message(s)):
                assistant_message: Message with tool_use/tool_calls
                tool_result_message: tool_result message(s) with fetch results
        """
        if response_format == "openai":
            return WebFetchTransformation._build_openai_messages(
                tool_calls, fetch_results
            )
        else:
            return WebFetchTransformation._build_anthropic_messages(
                tool_calls, fetch_results, thinking_blocks
            )

    @staticmethod
    def _build_anthropic_messages(
        tool_calls: List[Dict],
        fetch_results: List[str],
        thinking_blocks: Optional[List[Dict]] = None,
    ) -> Tuple[Dict, Dict]:
        """Build Anthropic-format assistant and user messages with tool results"""

        # Build assistant message with tool_use blocks
        assistant_content = []

        # Preserve thinking blocks first
        if thinking_blocks:
            assistant_content.extend(thinking_blocks)

        for i, tool_call in enumerate(tool_calls):
            assistant_content.append(
                {
                    "type": "tool_use",
                    "id": tool_call["id"],
                    "name": tool_call["name"],
                    "input": tool_call.get("input", {}),
                }
            )

        assistant_message = {
            "role": "assistant",
            "content": assistant_content,
        }

        # Build user message with tool_result blocks
        tool_result_content = []
        for i, result in enumerate(fetch_results):
            tool_call = tool_calls[i] if i < len(tool_calls) else {}
            tool_result_content.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call.get("id", "unknown"),
                    "content": result,
                }
            )

        user_message = {
            "role": "user",
            "content": tool_result_content,
        }

        return assistant_message, user_message

    @staticmethod
    def _build_openai_messages(
        tool_calls: List[Dict],
        fetch_results: List[str],
    ) -> Tuple[Dict, List[Dict]]:
        """Build OpenAI-format assistant and tool messages"""

        # Build assistant message with tool_calls
        openai_tool_calls = []
        for i, tool_call in enumerate(tool_calls):
            function_data = tool_call.get("function", {})
            input_data = tool_call.get("input", {})
            arguments = function_data.get("arguments", "")

            # If arguments is empty or not a string, use input
            if not arguments and input_data:
                import json

                arguments = json.dumps(input_data)
            elif not isinstance(arguments, str):
                import json

                arguments = json.dumps(arguments)

            openai_tool_calls.append(
                {
                    "id": tool_call.get("id", f"call_{i}"),
                    "type": "function",
                    "function": {
                        "name": LITELLM_WEB_FETCH_TOOL_NAME,
                        "arguments": arguments,
                    },
                }
            )

        assistant_message = {
            "role": "assistant",
            "content": None,
            "tool_calls": openai_tool_calls,
        }

        # Build tool result messages
        tool_messages = []
        for i, result in enumerate(fetch_results):
            tool_call = tool_calls[i] if i < len(tool_calls) else {}
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", f"call_{i}"),
                    "content": result,
                }
            )

        return assistant_message, tool_messages

    @staticmethod
    def format_fetch_response(fetch_response: WebFetchResponse) -> str:
        """
        Format a WebFetchResponse into a text string for LLM consumption.

        Args:
            fetch_response: The fetch response object

        Returns:
            Formatted string with title and content
        """
        parts = []
        if fetch_response.title:
            parts.append(f"Title: {fetch_response.title}")
        parts.append(f"URL: {fetch_response.url}")
        parts.append("")
        parts.append(fetch_response.content)
        return "\n".join(parts)
