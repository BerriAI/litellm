"""
Chat provider mapping chat completion requests to /responses API and back to chat format
"""
import json

import httpx
import time
from typing import Any, Dict, List, Optional, Union, Iterator, AsyncIterator

from openai.types.responses.response_create_params import ResponseCreateParamsBase
from openai.types.responses.response import Response as ResponsesAPIResponse

from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.responses.litellm_completion_transformation.transformation import LiteLLMCompletionResponsesConfig
from litellm.types.utils import ModelResponse, Choices, Usage, GenericStreamingChunk, ModelResponseStream
from litellm.constants import OPENAI_CHAT_COMPLETION_PARAMS
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams, ChatCompletionUsageBlock, \
    ChatCompletionToolCallChunk, ChatCompletionToolCallFunctionChunk

from litellm._logging import verbose_logger

class ChatConfig(BaseConfig):
    """
    Provider config for chat that uses the /responses API under the hood.
    Transforms chat completion requests into Responses API requests and vice versa.
    """
    def get_supported_openai_params(self, model: str) -> List[str]:  # noqa: U100
        # Support standard OpenAI chat parameters plus responses API specific ones
        base_params = list(OPENAI_CHAT_COMPLETION_PARAMS)
        responses_specific_params = ["previous_response_id", "instructions"]
        return base_params + responses_specific_params

    def validate_environment(
        self,
        headers: Dict[str, Any],
        model: str,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # Ensure proper authentication headers for OpenAI Responses API
        verbose_logger.debug(f"Chat provider validate_environment: model={model}, api_key={'***' if api_key else None}, headers_auth={'present' if headers.get('Authorization') else 'missing'}")
        
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            verbose_logger.debug(f"Chat provider: Set Authorization header from provided api_key")
        elif not headers.get("Authorization"):
            # Try to get API key from environment or other sources
            import os
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                headers["Authorization"] = f"Bearer {openai_key}"
                verbose_logger.debug(f"Chat provider: Set Authorization header from OPENAI_API_KEY environment variable")
            else:
                verbose_logger.debug("Warning: No API key found for chat provider. Set OPENAI_API_KEY or provide api_key in config.")
        else:
            verbose_logger.debug(f"Chat provider: Using existing Authorization header")
        
        # Ensure proper content type
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
            
        verbose_logger.debug(f"Chat provider final headers: {list(headers.keys())}")
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        **kwargs,
    ) -> str:
        # Default to OpenAI if no api_base provided
        if not api_base:
            api_base = "https://api.openai.com/v1"
            verbose_logger.debug(f"Warning: No api_base provided for chat provider, defaulting to: {api_base}")
        
        return api_base.rstrip("/") + "/responses"

    def transform_request(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        optional_params: Dict[str, Any],
        litellm_params: Dict[str, Any],
        headers: Dict[str, Any],  # noqa: U100
    ) -> Dict[str, Any]:
        # Convert chat completion messages back to responses API input format
        input_items: List[Any] = []
        
        # Process messages to extract system instructions and convert to responses format
        instructions = None
        converted_messages = []
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")
            tool_call_id = msg.get("tool_call_id")
            
            if role == "system":
                # Extract system message as instructions
                instructions = content
            elif role == "tool":
                # Convert tool message to function call output format
                input_items.append({
                    "type": "function_call_output",
                    "call_id": tool_call_id,
                    "output": content
                })
            elif role == "assistant" and tool_calls:
                # Convert assistant message with tool calls
                # {
                #     "type": "function_call",
                #     "id": "fc_12345xyz",
                #     "call_id": "call_12345xyz",
                #     "name": "get_weather",
                #     "arguments": "{\"latitude\":48.8566,\"longitude\":2.3522}"
                # }
                for tool_call in tool_calls:
                    function = tool_call.get("function")
                    if function:
                        input_items.append({
                            "type": "function_call",
                            "call_id": tool_call["id"],
                            "name": function["name"],
                            "arguments": function["arguments"],
                        })
                    else:
                        raise ValueError(f"tool call not supported: {tool_call}")
            else:
                # Regular user/assistant message
                input_items.append({
                    "type": "message", 
                    "role": role,
                    "content": self._convert_content_to_responses_format(content)
                })
        
        # Build responses API request using the reverse transformation logic
        responses_api_request = ResponsesAPIOptionalRequestParams()
        
        # Set instructions if we found a system message
        if instructions:
            responses_api_request["instructions"] = instructions
            
        # Map optional parameters
        for key, value in optional_params.items():
            if value is None:
                continue
            if key in ("max_tokens", "max_completion_tokens"):
                responses_api_request["max_output_tokens"] = value
            elif key == "tools":
                # Convert chat completion tools to responses API tools format
                responses_api_request["tools"] = self._convert_tools_to_responses_format(value)
            elif key in ["temperature", "top_p", "tool_choice", "parallel_tool_calls", "user", "stream"]:
                responses_api_request[key] = value
            elif key == "metadata":
                responses_api_request["metadata"] = value
            elif key == "previous_response_id":
                # Support for responses API session management
                responses_api_request["previous_response_id"] = value
        
        # Get stream parameter from litellm_params if not in optional_params
        stream = optional_params.get("stream") or litellm_params.get("stream", False)
        verbose_logger.debug(f"Chat provider: Stream parameter: {stream}")
        
        # Ensure stream is properly set in the request
        if stream:
            responses_api_request["stream"] = True
        
        # Handle session management if previous_response_id is provided
        previous_response_id = optional_params.get("previous_response_id")
        if previous_response_id:
            # Use the existing session handler for responses API
            verbose_logger.debug(f"Chat provider: Warning ignoring previous response ID: {previous_response_id}")

        # Convert back to responses API format for the actual request
        # Strip chat/ prefix if present for the underlying API call
        api_model = model
        if model.startswith("chat/"):
            api_model = model[5:]  # Remove "chat/" prefix
            verbose_logger.debug(f"Chat provider: Stripped chat/ prefix from model: {model} -> {api_model}")
        else:
            verbose_logger.debug(f"Chat provider: Using model as-is: {api_model}")
        
        request_data = {
            "model": api_model,
            "input": input_items,
        }
        
        verbose_logger.debug(f"Chat provider: Final request model={api_model}, input_items={len(input_items)}")
        
        # Add non-None values from responses_api_request
        for key, value in responses_api_request.items():
            if value is not None:
                if key == "instructions" and instructions:
                    request_data["instructions"] = instructions
                else:
                    request_data[key] = value
        
        verbose_logger.debug(f"Chat provider: Final request data keys: {list(request_data.keys())}")
        verbose_logger.debug(f"Chat provider: Stream in request: {request_data.get('stream', False)}")
        
        return request_data

    def transform_response(
        self,
        model: str,  # noqa: U100
        raw_response: httpx.Response,
        model_response: ModelResponse,  # noqa: U100
        logging_obj: Any,  # noqa: U100
        request_data: Dict[str, Any],
        messages: List[Any],  # noqa: U100
        optional_params: Dict[str, Any],  # noqa: U100
        litellm_params: Dict[str, Any],  # noqa: U100
        encoding: Any,  # noqa: U100
        api_key: Optional[str] = None,  # noqa: U100
        json_mode: Optional[bool] = None,  # noqa: U100
    ) -> ModelResponse:
        # Parse Responses API response and convert to chat ModelResponse
        try:
            resp_json = raw_response.json()
        except Exception as e:
            verbose_logger.debug(f"Error parsing JSON response from responses API: {e}")
            verbose_logger.debug(f"Raw response text: {raw_response.text}")
            import traceback
            traceback.print_exc()
            raise httpx.HTTPError(f"Invalid JSON from responses API: {raw_response.text}")
        
        # Use the existing transformation logic to convert responses API to chat completion
        responses_api_response = ResponsesAPIResponse.model_validate(raw_response.json())

        # Transform the responses API response to chat completion format
        verbose_logger.debug(f"Chat provider: Raw response from openai: {resp_json}")
        verbose_logger.debug(f"Chat provider: responses_api_request: {responses_api_response}")

        # Inverse‐transform back to the original ModelResponse
        transformed = LiteLLMCompletionResponsesConfig.transform_responses_api_response_to_chat_completion_response(
            responses_api_response=responses_api_response,
            model_response=model_response
        )

        verbose_logger.debug(f"Chat provider: transformed {transformed}")
        return transformed

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], ModelResponse],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ) -> Any:
        return OpenAiResponsesToChatCompletionStreamIterator(streaming_response, sync_stream, json_mode)

    def map_openai_params(
        self,
        non_default_params: Dict[str, Any],
        optional_params: Dict[str, Any],
        model: str,
        drop_params: bool,
    ) -> Dict[str, Any]:
        # Pass through all non-default parameters into optional params
        for key, value in non_default_params.items():
            optional_params[key] = value
        return optional_params

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Union[dict, "httpx.Headers"],  # noqa: F821
    ):
        # Return a BaseLLMException for chat errors
        from litellm.llms.base_llm.chat.transformation import BaseLLMException
        return BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
    
    def _convert_content_to_responses_format(self, content: Union[str, List[Any]]) -> List[Dict[str, Any]]:
        """Convert chat completion content to responses API format"""
        verbose_logger.debug(f"Chat provider: Converting content to responses format - input type: {type(content)}")
        
        if isinstance(content, str):
            result = [{"type": "input_text", "text": content}]
            verbose_logger.debug(f"Chat provider: String content -> {result}")
            return result
        elif isinstance(content, list):
            result = []
            for i, item in enumerate(content):
                verbose_logger.debug(f"Chat provider: Processing content item {i}: {type(item)} = {item}")
                if isinstance(item, str):
                    converted = {"type": "input_text", "text": item}
                    result.append(converted)
                    verbose_logger.debug(f"Chat provider:   -> {converted}")
                elif isinstance(item, dict):
                    # Handle multimodal content
                    original_type = item.get("type")
                    if original_type == "text":
                        converted = {"type": "input_text", "text": item.get("text", "")}
                        result.append(converted)
                        verbose_logger.debug(f"Chat provider:   text -> {converted}")
                    elif original_type == "image_url":
                        # Map to responses API image format
                        converted = {"type": "input_image", "image_url": item.get("image_url", {})}
                        result.append(converted)
                        verbose_logger.debug(f"Chat provider:   image_url -> {converted}")
                    else:
                        # Try to map other types to responses API format
                        item_type = original_type or "input_text"
                        if item_type == "image":
                            converted = {"type": "input_image", **item}
                            result.append(converted)
                            verbose_logger.debug(f"Chat provider:   image -> {converted}")
                        elif item_type in ["input_text", "input_image", "output_text", "refusal", "input_file", "computer_screenshot", "summary_text"]:
                            # Already in responses API format
                            result.append(item)
                            verbose_logger.debug(f"Chat provider:   passthrough -> {item}")
                        else:
                            # Default to input_text for unknown types
                            converted = {"type": "input_text", "text": str(item.get("text", item))}
                            result.append(converted)
                            verbose_logger.debug(f"Chat provider:   unknown({original_type}) -> {converted}")
            verbose_logger.debug(f"Chat provider: Final converted content: {result}")
            return result
        else:
            result = [{"type": "input_text", "text": str(content)}]
            verbose_logger.debug(f"Chat provider: Other content type -> {result}")
            return result
    
    def _convert_tools_to_responses_format(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert chat completion tools to responses API tools format"""
        responses_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                function = tool.get("function", {})
                responses_tools.append({
                    "type": "function",
                    "name": function.get("name", ""),
                    "description": function.get("description", ""),
                    "parameters": function.get("parameters", {}),
                    "strict": function.get("strict", False)
                })
        return responses_tools
    
    def _map_responses_status_to_finish_reason(self, status: Optional[str]) -> str:
        """Map responses API status to chat completion finish_reason"""
        if not status:
            return "stop"
        
        status_mapping = {
            "completed": "stop",
            "incomplete": "length", 
            "failed": "stop",
            "cancelled": "stop"
        }
        
        return status_mapping.get(status, "stop")



class OpenAiResponsesToChatCompletionStreamIterator(BaseModelResponseIterator):

    def __init__(
            self, streaming_response, sync_stream: bool, json_mode: Optional[bool] = False
    ):
        super().__init__(streaming_response, sync_stream, json_mode)

    def _handle_string_chunk(
        self, str_line: str
    ) -> Union[GenericStreamingChunk, ModelResponseStream]:
        if not str_line or str_line.startswith("event:"):
            # ignore.
            return GenericStreamingChunk(
                text="",
                tool_use=None,
                is_finished=False,
                finish_reason="",
                usage=None
            )
        index = str_line.find("data:")
        if index != -1:
            str_line = str_line[index + 5:]

        return self.chunk_parser(json.loads(str_line))

    def chunk_parser(self, chunk: dict) -> GenericStreamingChunk:
        # Transform responses API streaming chunk to chat completion format
        verbose_logger.debug(f"Chat provider: transform_streaming_response called with chunk: {chunk}")
        parsed_chunk = chunk

        if not parsed_chunk:
            raise ValueError("Chat provider: Empty parsed_chunk")

        if not isinstance(parsed_chunk, dict):
            raise ValueError(f"Chat provider: Invalid chunk type {type(parsed_chunk)}")

        # Handle different event types from responses API
        event_type = parsed_chunk.get("type")
        verbose_logger.debug(f"Chat provider: Processing event type: {event_type}")

        if event_type == "response.created":
            # Initial response creation event
            verbose_logger.debug(f"Chat provider: response.created -> {chunk}")
            return GenericStreamingChunk(
                text="",
                tool_use=None,
                is_finished=False,
                finish_reason="",
                usage=None
            )
        elif event_type == "response.output_item.added":
            # New output item added
            output_item = parsed_chunk.get("item", {})
            if output_item.get("type") == "function_call":
                return GenericStreamingChunk(
                    text="",

                    tool_use=ChatCompletionToolCallChunk(
                        id=output_item.get("call_id"),
                        index=0,
                        type="function",
                        function=ChatCompletionToolCallFunctionChunk(
                            name=parsed_chunk.get("name", None),
                            arguments=parsed_chunk.get("arguments", "")
                        )
                    ),
                    is_finished=False,
                    finish_reason="",
                    usage=None
                )
            elif output_item.get("type") == "message":
                pass
            elif output_item.get("type") == "reasoning":
                pass
            else:
                raise ValueError(f"Chat provider: Invalid output_item  {output_item}")
        elif event_type == "response.function_call_arguments.delta":
            content_part: Optional[str] = parsed_chunk.get("delta", None)
            if content_part:
                return GenericStreamingChunk(
                    text="",
                    tool_use=ChatCompletionToolCallChunk(
                        id=None,
                        index=0,
                        type="function",
                        function=ChatCompletionToolCallFunctionChunk(
                            name=None,
                            arguments=content_part
                        )
                    ),
                    is_finished=False,
                    finish_reason="",
                    usage=None
                )
            else:
                raise ValueError(f"Chat provider: Invalid function argument delta {parsed_chunk}")
        elif event_type == "response.output_item.done":
            # New output item added
            output_item = parsed_chunk.get("item", {})
            if output_item.get("type") == "function_call":
                return GenericStreamingChunk(
                    text="",
                    tool_use=ChatCompletionToolCallChunk(
                        id=output_item.get("call_id"),
                        index=0,
                        type="function",
                        function=ChatCompletionToolCallFunctionChunk(
                            name=parsed_chunk.get("name", None),
                            arguments="" # responses API sends everything again, we don't
                        )
                    ),
                    is_finished=True,
                    finish_reason="tool_calls",
                    usage=None
                )
            elif output_item.get("type") == "message":
                return GenericStreamingChunk(
                    finish_reason="stop",
                    is_finished=True,
                    usage=None,
                    text=""
                )
            elif output_item.get("type") == "reasoning":
                pass
            else:
                raise ValueError(f"Chat provider: Invalid output_item  {output_item}")

        elif event_type == "response.output_text.delta":
            # Content part added to output
            content_part: Optional[str] = parsed_chunk.get("delta", None)
            if content_part is not None:
                return GenericStreamingChunk(
                    text=content_part,
                    tool_use=None,
                    is_finished=False,
                    finish_reason="",
                    usage=None
                )
            else:
                raise ValueError(f"Chat provider: Invalid text delta {parsed_chunk}")
        else:
            pass
        # For any unhandled event types, create a minimal valid chunk or skip
        verbose_logger.debug(f"Chat provider: Unhandled event type '{event_type}', creating empty chunk")

        # Return a minimal valid chunk for unknown events
        return GenericStreamingChunk(
            text="",
            tool_use=None,
            is_finished=False,
            finish_reason="",
            usage=None
        )

