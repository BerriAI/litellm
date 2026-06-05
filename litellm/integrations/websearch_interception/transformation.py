"""
WebSearch Tool Transformation

Transforms between Anthropic/OpenAI tool_use format and LiteLLM search format.
"""

import json
from typing import Any, Dict, List, Optional, Tuple, Union

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
            verbose_logger.warning("WebSearchInterception: Unexpected streaming response, skipping interception")
            return False, []

        # Parse non-streaming response based on format
        if response_format == "responses":
            return WebSearchTransformation._detect_from_responses_api_response(response)
        if response_format == "openai":
            return WebSearchTransformation._detect_from_openai_response(response)
        return WebSearchTransformation._detect_from_non_streaming_response(response)

    @staticmethod
    def _detect_from_responses_api_response(
        response: Any,
    ) -> Tuple[bool, List[Dict]]:
        """Parse OpenAI Responses-API response for ``litellm_web_search`` function calls.

        The Responses API returns ``output`` as a list of items; tool calls
        appear as ``{"type": "function_call", "call_id": "...", "name": "...",
        "arguments": "<json string>"}``. Pre-request conversion replaces all
        web-search tools with the LiteLLM standard ``litellm_web_search``
        function tool, so the only name we need to recognize here is the
        standard one. ``call_id`` is preserved as ``id`` so the agentic loop
        can pair it with a ``function_call_output`` item.
        """
        if isinstance(response, dict):
            output = response.get("output", []) or []
        else:
            output = getattr(response, "output", None) or []

        tool_calls: List[Dict] = []
        for item in output:
            if isinstance(item, dict):
                item_type = item.get("type")
                item_name = item.get("name")
                call_id = item.get("call_id") or item.get("id")
                arguments = item.get("arguments")
            else:
                item_type = getattr(item, "type", None)
                item_name = getattr(item, "name", None)
                call_id = getattr(item, "call_id", None) or getattr(item, "id", None)
                arguments = getattr(item, "arguments", None)

            if item_type != "function_call":
                continue
            if item_name != LITELLM_WEB_SEARCH_TOOL_NAME:
                continue

            if isinstance(arguments, str):
                try:
                    parsed_arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    verbose_logger.warning(
                        "WebSearchInterception: Failed to parse Responses-API "
                        f"function_call arguments: {arguments}"
                    )
                    parsed_arguments = {}
            elif isinstance(arguments, dict):
                parsed_arguments = arguments
            else:
                parsed_arguments = {}

            tool_calls.append(
                {
                    "id": call_id,
                    "call_id": call_id,
                    "type": "function_call",
                    "name": item_name,
                    "input": parsed_arguments,
                    "arguments": parsed_arguments,
                }
            )
            verbose_logger.debug(
                f"WebSearchInterception: Found Responses-API function_call "
                f"name={item_name} call_id={call_id}"
            )

        return len(tool_calls) > 0, tool_calls

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
                verbose_logger.debug("WebSearchInterception: Response has no content attribute")
                return False, []
            content = response.content or []

        if not content:
            verbose_logger.debug("WebSearchInterception: Response has empty content")
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

            # Detect tool_use blocks that came from interception. After
            # pre-request conversion the model always sees
            # ``litellm_web_search``; the bare ``web_search`` entry handles
            # callers that bypass our pre-request hooks (e.g. direct
            # litellm.acompletion). "WebSearch" is intentionally omitted —
            # see is_web_search_tool for the Cowork rationale.
            if block_type == "tool_use" and block_name in (
                LITELLM_WEB_SEARCH_TOOL_NAME,
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
                verbose_logger.debug(f"WebSearchInterception: Found {block_name} tool_use with id={tool_call['id']}")

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
                verbose_logger.debug("WebSearchInterception: Response has no choices attribute")
                return False, []
            choices = response.choices or []

        if not choices:
            verbose_logger.debug("WebSearchInterception: Response has empty choices")
            return False, []

        # Get first choice's message
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message", {})
        else:
            message = getattr(first_choice, "message", None)

        if not message:
            verbose_logger.debug("WebSearchInterception: First choice has no message")
            return False, []

        # Get tool_calls from message
        if isinstance(message, dict):
            openai_tool_calls = message.get("tool_calls", [])
        else:
            openai_tool_calls = getattr(message, "tool_calls", None) or []

        if not openai_tool_calls:
            verbose_logger.debug("WebSearchInterception: Message has no tool_calls")
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
                function_arguments = (
                    function.get("arguments") if isinstance(function, dict) else getattr(function, "arguments", None)
                )
            else:
                tool_id = getattr(tool_call, "id", None)
                tool_type = getattr(tool_call, "type", None)
                function = getattr(tool_call, "function", None)
                function_name = getattr(function, "name", None) if function else None
                function_arguments = getattr(function, "arguments", None) if function else None

            # Detect function-style web search tool_calls. ``WebSearch`` is
            # intentionally omitted — see is_web_search_tool for the Cowork
            # rationale (clients ship their own client-side ``WebSearch`` and
            # we must not hijack it).
            if tool_type == "function" and function_name in (
                LITELLM_WEB_SEARCH_TOOL_NAME,
                "web_search",
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
                verbose_logger.debug(f"WebSearchInterception: Found {function_name} tool_call with id={tool_id}")

        return len(tool_calls) > 0, tool_calls

    @staticmethod
    def transform_response(
        tool_calls: List[Dict],
        search_results: List[str],
        response_format: str = "anthropic",
        thinking_blocks: Optional[List[Dict]] = None,
    ) -> Tuple[Dict, Union[Dict, List[Dict]]]:
        """
        Transform LiteLLM search results to Anthropic/OpenAI tool_result format.

        Builds the assistant and user/tool messages needed for the agentic loop
        follow-up request.

        Args:
            tool_calls: List of tool_use/tool_calls dicts from transform_request
            search_results: List of search result strings (one per tool_call)
            response_format: Response format - "anthropic" or "openai" (default: "anthropic")
            thinking_blocks: Optional list of thinking/redacted_thinking blocks
                from the model's response. When present, prepended to the
                assistant message content (required by Anthropic API when
                thinking is enabled).

        Returns:
            (assistant_message, user_or_tool_messages):
                For Anthropic: assistant_message with tool_use blocks, user_message with tool_result blocks
                For OpenAI: assistant_message with tool_calls, tool_messages list with tool results
        """
        if response_format == "openai":
            return WebSearchTransformation._transform_response_openai(tool_calls, search_results)
        else:
            return WebSearchTransformation._transform_response_anthropic(
                tool_calls, search_results, thinking_blocks=thinking_blocks
            )

    @staticmethod
    def _transform_response_anthropic(
        tool_calls: List[Dict],
        search_results: List[str],
        thinking_blocks: Optional[List[Dict]] = None,
    ) -> Tuple[Dict, Dict]:
        """Transform to Anthropic format (single user message with tool_result blocks)"""
        # Build assistant message content
        assistant_content: List[Dict] = []

        # Prepend thinking blocks if present.
        # When extended thinking is enabled, Anthropic requires the assistant
        # message to start with thinking/redacted_thinking blocks before any
        # tool_use blocks. Same pattern as anthropic_messages_pt in factory.py.
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
                        "arguments": (json.dumps(tc["input"]) if isinstance(tc["input"], dict) else str(tc["input"])),
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
    def build_web_search_tool_result_block(
        tool_use_id: str,
        search_response: Optional[SearchResponse],
    ) -> Dict[str, Any]:
        """
        Build an Anthropic-native ``web_search_tool_result`` content block.

        Native Anthropic clients (Claude Desktop, the Anthropic SDK, the
        Anthropic Console) expect search-tool results to be returned as
        structured ``web_search_tool_result`` blocks so that citations and
        source links can be rendered. The agentic loop currently feeds the
        model a flat text blob in the follow-up call (which is correct — the
        model needs readable evidence). This helper produces the *additional*
        block that should accompany the model's text reply when the original
        request used a native ``web_search_*`` tool.

        Spec reference:
        https://docs.anthropic.com/en/api/web-search-tool

        Args:
            tool_use_id: The ``tool_use_id`` the model emitted on the first
                turn. Must match exactly so the client can pair the result
                with its tool_use block.
            search_response: Structured ``SearchResponse`` from
                ``litellm.asearch()``. If None or empty, the block is still
                emitted with an empty result list (signals "search ran, no
                results" rather than "search did not run").
        """
        items: List[Dict[str, Any]] = []
        if search_response is not None:
            results = getattr(search_response, "results", None) or []
            for r in results:
                url = getattr(r, "url", "") or ""
                title = getattr(r, "title", "") or ""
                page_age = getattr(r, "date", None) or getattr(r, "last_updated", None)
                items.append(
                    {
                        "type": "web_search_result",
                        "url": url,
                        "title": title,
                        "page_age": page_age,
                        "encrypted_content": "",
                    }
                )
        return {
            "type": "web_search_tool_result",
            "tool_use_id": tool_use_id,
            "content": items,
        }

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
                [f"Title: {r.title}\nURL: {r.url}\nSnippet: {r.snippet}" for r in result.results]
            )
        else:
            search_result_text = str(result)

        return search_result_text
