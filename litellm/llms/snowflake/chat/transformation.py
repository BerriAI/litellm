"""
Support for Snowflake REST API
"""

import json
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import httpx

from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
)
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Delta,
    Function,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)
from litellm.types.utils import _generate_id

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

        return {
            "model": model,
            "messages": messages,
            "stream": stream,
            **optional_params,
            **extra_body,
        }

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], ModelResponse],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ) -> Any:
        """
        Return a Snowflake-specific streaming handler that transforms
        Snowflake's streaming format to OpenAI-compatible format.

        Snowflake streams tool calls using content_block_start/delta/stop events,
        which differs from OpenAI's delta.tool_calls format and requires custom handling.
        """
        return SnowflakeStreamingHandler(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )


class SnowflakeStreamingHandler(BaseModelResponseIterator):
    """
    Streaming handler for Snowflake Cortex LLM REST API.

    Snowflake streams tool calls using content_block events rather than OpenAI's
    delta.tool_calls format. This handler transforms those events to OpenAI-compatible
    streaming chunks.

    Snowflake streaming format (tool calls):
        - content_block_start: {"type": "content_block_start", "content_block": {"type": "tool_use", "id": "...", "name": "..."}}
        - content_block_delta: {"type": "content_block_delta", "delta": {"type": "input_json_delta", "partial_json": "..."}}
        - content_block_stop: {"type": "content_block_stop"}

    OpenAI streaming format (tool calls):
        - {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "...", "function": {"name": "...", "arguments": "..."}}]}}]}
    """

    def __init__(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], ModelResponse],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ):
        super().__init__(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )
        # Track tool call index across chunks
        self.tool_index = -1
        # Track current content block type (text or tool_use)
        self.current_content_block_type: Optional[str] = None
        # Generate consistent response ID for the stream
        self.response_id = _generate_id()

    def chunk_parser(self, chunk: dict) -> ModelResponseStream:
        """
        Parse a Snowflake streaming chunk and transform it to OpenAI format.

        Handles both:
        1. OpenAI-compatible format (choices with delta) - pass through
        2. Snowflake content_block format - transform to OpenAI format
        """
        # Check if this is OpenAI-compatible format (has choices)
        if "choices" in chunk:
            return self._handle_openai_format_chunk(chunk)

        # Otherwise, handle Snowflake's content_block format
        return self._handle_content_block_chunk(chunk)

    def _handle_openai_format_chunk(self, chunk: dict) -> ModelResponseStream:
        """Handle chunks that are already in OpenAI-compatible format."""
        return ModelResponseStream(
            id=chunk.get("id", self.response_id),
            object="chat.completion.chunk",
            created=chunk.get("created"),
            model=chunk.get("model"),
            choices=chunk.get("choices", []),
        )

    def _handle_content_block_chunk(self, chunk: dict) -> ModelResponseStream:
        """
        Handle Snowflake's content_block streaming chunks.

        Event types:
        - message_start: Start of message with usage info
        - content_block_start: Start of a content block (text or tool_use)
        - content_block_delta: Delta update for the current block
        - content_block_stop: End of current content block
        - message_delta: End of message with finish_reason
        """
        type_chunk = chunk.get("type", "")
        text = ""
        tool_use: Optional[ChatCompletionToolCallChunk] = None
        finish_reason: Optional[str] = None
        usage: Optional[Usage] = None

        if type_chunk == "message_start":
            # Extract usage from message_start if available
            message = chunk.get("message", {})
            if "usage" in message:
                usage_data = message["usage"]
                usage = Usage(
                    prompt_tokens=usage_data.get("input_tokens", 0),
                    completion_tokens=usage_data.get("output_tokens", 0),
                    total_tokens=usage_data.get("input_tokens", 0)
                    + usage_data.get("output_tokens", 0),
                )

        elif type_chunk == "content_block_start":
            content_block = chunk.get("content_block", {})
            self.current_content_block_type = content_block.get("type")

            if self.current_content_block_type == "text":
                # Text block start - may have initial text
                text = content_block.get("text", "")

            elif self.current_content_block_type == "tool_use":
                # Tool use block start - extract id and name
                self.tool_index += 1
                tool_id = content_block.get("id", "")
                tool_name = content_block.get("name", "")

                tool_use = ChatCompletionToolCallChunk(
                    id=tool_id,
                    type="function",
                    function=ChatCompletionToolCallFunctionChunk(
                        name=tool_name,
                        arguments="",
                    ),
                    index=self.tool_index,
                )

        elif type_chunk == "content_block_delta":
            delta = chunk.get("delta", {})
            delta_type = delta.get("type", "")

            if delta_type == "text_delta":
                text = delta.get("text", "")

            elif delta_type == "input_json_delta":
                # Tool argument delta - only emit if in tool_use block
                if self.current_content_block_type == "tool_use":
                    partial_json = delta.get("partial_json", "")
                    tool_use = ChatCompletionToolCallChunk(
                        id=None,
                        type="function",
                        function=ChatCompletionToolCallFunctionChunk(
                            name=None,
                            arguments=partial_json,
                        ),
                        index=self.tool_index,
                    )

        elif type_chunk == "content_block_stop":
            # Reset current content block tracking
            self.current_content_block_type = None

        elif type_chunk == "message_delta":
            # End of message - extract finish_reason and final usage
            delta = chunk.get("delta", {})
            stop_reason = delta.get("stop_reason")
            if stop_reason:
                # Map Anthropic stop reasons to OpenAI finish reasons
                if stop_reason == "end_turn":
                    finish_reason = "stop"
                elif stop_reason == "tool_use":
                    finish_reason = "tool_calls"
                elif stop_reason == "max_tokens":
                    finish_reason = "length"
                else:
                    finish_reason = stop_reason

            # Extract final usage if available.
            # Note: message_delta only contains output_tokens (completion counts).
            # Prompt tokens are reported in message_start, not here.
            if "usage" in chunk:
                usage_data = chunk["usage"]
                usage = Usage(
                    prompt_tokens=0,  # Not available in message_delta
                    completion_tokens=usage_data.get("output_tokens", 0),
                    total_tokens=usage_data.get("output_tokens", 0),
                )

        # Build the response chunk
        return ModelResponseStream(
            id=self.response_id,
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        content=text if text else None,
                        tool_calls=[tool_use] if tool_use is not None else None,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            usage=usage,
        )
