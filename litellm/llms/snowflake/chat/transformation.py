"""
Support for Snowflake REST API
"""

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx

from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ChatCompletionMessageToolCall, Function, ModelResponse

from ...openai_like.chat.transformation import OpenAIGPTConfig

from ..utils import SnowflakeBaseConfig


if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class SnowflakeConfig(SnowflakeBaseConfig, OpenAIGPTConfig):
    """
    Reference: https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-llm-rest-api

    Snowflake Cortex LLM REST API supports function calling with specific models (e.g., Claude 3.5 Sonnet).
    This config handles transformation between OpenAI format and Snowflake's tool_spec format.
    """

    @classmethod
    def get_config(cls):
        return super().get_config()

    def _transform_tool_calls_from_snowflake_to_openai(
        self, content_list: List[Dict[str, Any]]
    ) -> Tuple[str, Optional[List[ChatCompletionMessageToolCall]]]:
        """
        Transform Snowflake tool calls to OpenAI format.

        Args:
            content_list: Snowflake's content_list array containing text and tool_use items

        Returns:
            Tuple of (text_content, tool_calls)

        Snowflake format in content_list:
        {
          "type": "tool_use",
          "tool_use": {
            "tool_use_id": "tooluse_...",
            "name": "get_weather",
            "input": {"location": "Paris"}
          }
        }

        OpenAI format (returned tool_calls):
        ChatCompletionMessageToolCall(
            id="tooluse_...",
            type="function",
            function=Function(name="get_weather", arguments='{"location": "Paris"}')
        )
        """
        text_content = ""
        tool_calls: List[ChatCompletionMessageToolCall] = []

        for idx, content_item in enumerate(content_list):
            if content_item.get("type") == "text":
                text_content += content_item.get("text", "")

            ## TOOL CALLING
            elif content_item.get("type") == "tool_use":
                tool_use_data = content_item.get("tool_use", {})
                tool_call = ChatCompletionMessageToolCall(
                    id=tool_use_data.get("tool_use_id", ""),
                    type="function",
                    function=Function(
                        name=tool_use_data.get("name", ""),
                        arguments=json.dumps(tool_use_data.get("input", {})),
                    ),
                )
                tool_calls.append(tool_call)

        return text_content, tool_calls if tool_calls else None

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        response_json = raw_response.json()

        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response=response_json,
            additional_args={"complete_input_dict": request_data},
        )

        ## RESPONSE TRANSFORMATION
        # Snowflake returns content_list (not content) with tool_use objects
        # We need to transform this to OpenAI's format with content + tool_calls
        if "choices" in response_json and len(response_json["choices"]) > 0:
            choice = response_json["choices"][0]
            if "message" in choice and "content_list" in choice["message"]:
                content_list = choice["message"]["content_list"]
                (
                    text_content,
                    tool_calls,
                ) = self._transform_tool_calls_from_snowflake_to_openai(content_list)

                # Update the choice message with OpenAI format
                choice["message"]["content"] = text_content
                if tool_calls:
                    choice["message"]["tool_calls"] = tool_calls

                # Remove Snowflake-specific content_list
                del choice["message"]["content_list"]

        returned_response = ModelResponse(**response_json)

        returned_response.model = "snowflake/" + (returned_response.model or "")

        if model is not None:
            returned_response._hidden_params["model"] = model
        return returned_response

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        If api_base is not provided, use the default DeepSeek /chat/completions endpoint.
        """

        api_base = self._get_api_base(api_base, optional_params)

        return f"{api_base}/cortex/inference:complete"

    def _transform_messages(
        self, messages: List[AllMessageValues], model: str, is_async: bool = False
    ) -> List[AllMessageValues]:
        """
        Transform OpenAI messages to Snowflake format.

        Key transformations:
        1. Assistant messages with tool_calls -> content_list with tool_use blocks
        2. Tool messages (role: "tool") -> User messages with content_list containing tool_results

        Snowflake uses a format similar to Anthropic/Bedrock where:
        - tool_use blocks are in assistant message content_list
        - tool_results are in user message content_list (not role: "tool")
        """
        # Build a map of tool_call_id -> tool_call for looking up function names
        tool_calls_map: Dict[str, Dict[str, Any]] = {}
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "assistant":
                for tc in message.get("tool_calls") or []:
                    if isinstance(tc, dict):
                        tool_calls_map[tc.get("id", "")] = tc

        transformed: List[Dict[str, Any]] = []
        pending_tool_messages: List[Dict[str, Any]] = []

        for message in messages:
            if not isinstance(message, dict):
                continue

            role = message.get("role", "")

            # Flush pending tool messages before any non-tool message
            if role != "tool" and pending_tool_messages:
                transformed.append(
                    self._convert_tool_messages_to_user_message(
                        pending_tool_messages, tool_calls_map
                    )
                )
                pending_tool_messages = []

            if role == "tool":
                # Collect tool messages to combine into a single user message
                pending_tool_messages.append(message)

            elif role == "assistant" and message.get("tool_calls"):
                # Transform assistant message with tool_calls to content_list format
                transformed.append(self._convert_assistant_tool_message(message))

            else:
                # Pass through other messages as-is
                transformed.append(message)

        # Flush any remaining tool messages
        if pending_tool_messages:
            transformed.append(
                self._convert_tool_messages_to_user_message(
                    pending_tool_messages, tool_calls_map
                )
            )

        return transformed  # type: ignore

    def _convert_assistant_tool_message(
        self, message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert assistant message with tool_calls to Snowflake's content_list format.

        OpenAI format:
        {
            "role": "assistant",
            "content": "I'll check that for you.",
            "tool_calls": [{"id": "...", "function": {"name": "...", "arguments": "..."}}]
        }

        Snowflake format:
        {
            "role": "assistant",
            "content_list": [
                {"type": "text", "text": "I'll check that for you."},
                {"type": "tool_use", "tool_use": {"tool_use_id": "...", "name": "...", "input": {...}}}
            ]
        }
        """
        content_list: List[Dict[str, Any]] = []

        # Add text content if present
        text_content = message.get("content")
        if isinstance(text_content, list):
            # Flatten multipart content to a single string
            text_content = " ".join(
                part.get("text", "") for part in text_content if isinstance(part, dict)
            )
        if text_content:
            content_list.append({"type": "text", "text": text_content})

        # Add tool_use blocks
        for tool_call in message.get("tool_calls") or []:
            if isinstance(tool_call, dict):
                function = tool_call.get("function", {})
                # Parse arguments from JSON string to dict
                arguments_str = function.get("arguments", "{}")
                try:
                    arguments = json.loads(arguments_str) if arguments_str else {}
                except (json.JSONDecodeError, TypeError):
                    # TypeError if arguments is not a string (e.g., already a dict)
                    arguments = arguments_str if isinstance(arguments_str, dict) else {}

                content_list.append({
                    "type": "tool_use",
                    "tool_use": {
                        "tool_use_id": tool_call.get("id", ""),
                        "name": function.get("name", ""),
                        "input": arguments,
                    },
                })

        return {"role": "assistant", "content_list": content_list}

    def _convert_tool_messages_to_user_message(
        self,
        tool_messages: List[Dict[str, Any]],
        tool_calls_map: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Convert tool result messages to a single Snowflake user message with tool_results.

        OpenAI format (multiple messages):
        [
            {"role": "tool", "tool_call_id": "...", "content": "result1"},
            {"role": "tool", "tool_call_id": "...", "content": "result2"}
        ]

        Snowflake format (single user message):
        {
            "role": "user",
            "content_list": [
                {"type": "tool_results", "tool_results": {"tool_use_id": "...", "name": "...", "content": [{"type": "text", "text": "result1"}]}},
                {"type": "tool_results", "tool_results": {"tool_use_id": "...", "name": "...", "content": [{"type": "text", "text": "result2"}]}}
            ]
        }
        """
        content_list: List[Dict[str, Any]] = []

        for tool_msg in tool_messages:
            tool_call_id = tool_msg.get("tool_call_id", "")
            tool_call = tool_calls_map.get(tool_call_id, {})
            function = tool_call.get("function", {})
            function_name = function.get("name", "")

            # Get content - could be string, list, or None
            content = tool_msg.get("content")
            if content is None:
                content = "null"
            elif isinstance(content, list):
                # Flatten OpenAI multipart tool content to a plain string
                content = " ".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            elif not isinstance(content, str):
                content = str(content)

            content_list.append({
                "type": "tool_results",
                "tool_results": {
                    "tool_use_id": tool_call_id,
                    "name": function_name,
                    "content": [{"type": "text", "text": content}],
                },
            })

        return {"role": "user", "content_list": content_list}

    def _transform_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform OpenAI tool format to Snowflake tool format.

        Args:
            tools: List of tools in OpenAI format

        Returns:
            List of tools in Snowflake format

        OpenAI format:
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "...",
                "parameters": {...}
            }
        }

        Snowflake format:
        {
            "tool_spec": {
                "type": "generic",
                "name": "get_weather",
                "description": "...",
                "input_schema": {...}
            }
        }
        """
        snowflake_tools: List[Dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") == "function":
                function = tool.get("function", {})
                snowflake_tool: Dict[str, Any] = {
                    "tool_spec": {
                        "type": "generic",
                        "name": function.get("name"),
                        "input_schema": function.get(
                            "parameters",
                            {"type": "object", "properties": {}},
                        ),
                    }
                }
                # Add description if present
                if "description" in function:
                    snowflake_tool["tool_spec"]["description"] = function["description"]

                snowflake_tools.append(snowflake_tool)

        return snowflake_tools

    def _transform_tool_choice(
        self, tool_choice: Union[str, Dict[str, Any]]
    ) -> Union[str, Dict[str, Any]]:
        """
        Transform OpenAI tool_choice format to Snowflake format.

        Args:
            tool_choice: Tool choice in OpenAI format (str or dict)

        Returns:
            Tool choice in Snowflake format

        OpenAI format:
        {"type": "function", "function": {"name": "get_weather"}}

        Snowflake format:
        {"type": "tool", "name": ["get_weather"]}

        Note: String values ("auto", "required", "none") pass through unchanged.
        """
        if isinstance(tool_choice, str):
            # "auto", "required", "none" pass through as-is
            return tool_choice

        if isinstance(tool_choice, dict):
            if tool_choice.get("type") == "function":
                function_name = tool_choice.get("function", {}).get("name")
                if function_name:
                    return {
                        "type": "tool",
                        "name": [function_name],  # Snowflake expects array
                    }

        return tool_choice

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        stream: bool = optional_params.pop("stream", None) or False
        extra_body = optional_params.pop("extra_body", {})

        ## TOOL CALLING
        # Transform tools from OpenAI format to Snowflake's tool_spec format
        tools = optional_params.pop("tools", None)
        if tools:
            optional_params["tools"] = self._transform_tools(tools)

        # Transform tool_choice from OpenAI format to Snowflake's tool name array format
        tool_choice = optional_params.pop("tool_choice", None)
        if tool_choice:
            optional_params["tool_choice"] = self._transform_tool_choice(tool_choice)

        # Transform messages from OpenAI format to Snowflake format.
        # This handles role: "tool" -> role: "user" with tool_results content_list
        # and assistant messages with tool_calls -> content_list with tool_use blocks.
        # Note: We call _transform_messages here directly because Snowflake builds
        # its own request dict (doesn't delegate to super().transform_request()).
        # This is intentional - Snowflake routes through base_llm_http_handler,
        # not openai_like handler, so there's no double-transformation risk.
        transformed_messages = self._transform_messages(messages, model=model)

        return {
            "model": model,
            "messages": transformed_messages,
            "stream": stream,
            **optional_params,
            **extra_body,
        }
